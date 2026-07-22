//! N-port microwave networks and network-parameter conversions.
//!
//! [`Network`] stores frequency-dependent scattering matrices, reference
//! impedances, metadata, and optional noise parameters. This module also
//! provides connection, cascading, interpolation, mixed-mode, and conversion
//! operations used throughout the crate.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::constants::ZERO;
use crate::math::{
    RationalInterpolator, inverse_fft_centered, left_solve, right_solve, unwrap_radians,
};
use crate::{Error, Frequency, Result};

const NOISE_PARAMETER_EQUALITY_TOLERANCE: f64 = 1.0e-12;

/// A frequency-dependent N-port microwave network.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Network {
    /// Network frequency axis.
    pub frequency: Frequency,
    /// Scattering matrices with shape `(frequency, output port, input port)`.
    pub s: Array3<Complex64>,
    /// Per-frequency, per-port reference impedances.
    pub z0: Array2<Complex64>,
    /// Optional human-readable network name.
    pub name: Option<String>,
    /// Free-form comments carried with the network.
    pub comments: String,
    /// Optional names for individual ports.
    pub port_names: Vec<String>,
    /// Additional string-valued variables or metadata.
    pub variables: BTreeMap<String, String>,
    /// Scattering-wave definition used by `s`.
    pub s_definition: SParameterDefinition,
    /// Optional two-port noise parameters.
    pub noise: Option<NoiseParameters>,
    /// Single-ended, differential, or common designation for each port.
    #[serde(default)]
    pub port_modes: Vec<PortMode>,
    /// Optional propagation constants associated with the ports.
    #[serde(default)]
    pub propagation_constants: Option<Array2<Complex64>>,
}

/// Touchstone single-ended, differential, or common-mode port designation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum PortMode {
    /// Ordinary single-ended port.
    #[default]
    SingleEnded,
    /// Differential-mode port.
    Differential,
    /// Common-mode port.
    Common,
}

/// Four-parameter representation of two-port noise data.
///
/// Equality uses scale-aware floating-point comparisons so equivalent values
/// remain equal after conversion between Cartesian and polar representations.
///
/// Origin: `skrf.network.Network.set_noise_a`, `nfmin_db`, `g_opt`, and `rn`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoiseParameters {
    /// Frequencies at which the noise parameters are defined.
    pub frequency: Frequency,
    /// Minimum noise figure in decibels.
    pub minimum_noise_figure_db: Array1<f64>,
    /// Optimum source reflection coefficient $\Gamma_\mathrm{opt}$.
    pub optimal_reflection: Array1<Complex64>,
    /// Equivalent noise resistance in ohms.
    pub equivalent_noise_resistance: Array1<f64>,
}

impl NoiseParameters {
    /// Creates validated two-port noise parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when the parameter arrays do not match the frequency
    /// axis or contain invalid values.
    pub fn new(
        frequency: Frequency,
        minimum_noise_figure_db: Array1<f64>,
        optimal_reflection: Array1<Complex64>,
        equivalent_noise_resistance: Array1<f64>,
    ) -> Result<Self> {
        let points = frequency.points();
        if minimum_noise_figure_db.len() != points
            || optimal_reflection.len() != points
            || equivalent_noise_resistance.len() != points
        {
            return Err(Error::IncompatibleShape(format!(
                "noise frequency has {points} points but parameter lengths are {}, {}, and {}",
                minimum_noise_figure_db.len(),
                optimal_reflection.len(),
                equivalent_noise_resistance.len()
            )));
        }
        if minimum_noise_figure_db
            .iter()
            .any(|value| !value.is_finite())
            || optimal_reflection
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            || equivalent_noise_resistance
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(Error::Unsupported(
                "noise parameters must be finite and resistance must be non-negative".to_owned(),
            ));
        }
        Ok(Self {
            frequency,
            minimum_noise_figure_db,
            optimal_reflection,
            equivalent_noise_resistance,
        })
    }
}

impl PartialEq for NoiseParameters {
    fn eq(&self, other: &Self) -> bool {
        self.frequency == other.frequency
            && real_arrays_equal(
                &self.minimum_noise_figure_db,
                &other.minimum_noise_figure_db,
            )
            && complex_arrays_equal(&self.optimal_reflection, &other.optimal_reflection)
            && real_arrays_equal(
                &self.equivalent_noise_resistance,
                &other.equivalent_noise_resistance,
            )
    }
}

fn real_arrays_equal(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| scalars_equal(*left, *right))
}

fn complex_arrays_equal(left: &Array1<Complex64>, right: &Array1<Complex64>) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            scalars_equal(left.re, right.re) && scalars_equal(left.im, right.im)
        })
}

fn scalars_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= NOISE_PARAMETER_EQUALITY_TOLERANCE * scale
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
/// Definition used to normalize scattering waves for complex impedances.
pub enum SParameterDefinition {
    /// Kurokawa power waves.
    #[default]
    Power,
    /// Pseudo waves.
    Pseudo,
    /// Traveling waves.
    Traveling,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Interpolation method for frequency-dependent network data.
pub enum InterpolationMode {
    /// Linear interpolation of real and imaginary parts.
    #[default]
    CartesianLinear,
    /// Linear interpolation of magnitude and unwrapped phase.
    PolarLinear,
    /// Cubic interpolation.
    Cubic,
    /// Rational interpolation with a selected polynomial degree.
    Rational {
        /// Numerator and denominator degree used by the rational interpolator.
        degree: usize,
    },
}

impl Network {
    /// Creates a network from scattering matrices and reference impedances.
    ///
    /// # Errors
    ///
    /// Returns an error when the scattering matrices are not non-empty and
    /// square, or when their dimensions disagree with the frequency axis or
    /// reference impedances.
    pub fn new(frequency: Frequency, s: Array3<Complex64>, z0: Array2<Complex64>) -> Result<Self> {
        let (frequency_points, output_ports, input_ports) = s.dim();
        if output_ports == 0 || output_ports != input_ports {
            return Err(Error::IncompatibleShape(format!(
                "S parameters must contain non-empty square port matrices, got {:?}",
                s.dim()
            )));
        }
        if frequency.points() != frequency_points {
            return Err(Error::IncompatibleShape(format!(
                "frequency has {} points but S parameters have {frequency_points}",
                frequency.points()
            )));
        }
        if z0.dim() != (frequency_points, output_ports) {
            return Err(Error::IncompatibleShape(format!(
                "reference impedance must have shape ({frequency_points}, {output_ports}), got {:?}",
                z0.dim()
            )));
        }

        Ok(Self {
            frequency,
            s,
            z0,
            name: None,
            comments: String::new(),
            port_names: Vec::new(),
            variables: BTreeMap::new(),
            s_definition: SParameterDefinition::Power,
            noise: None,
            port_modes: vec![PortMode::SingleEnded; output_ports],
            propagation_constants: None,
        })
    }

    /// Creates a network from impedance matrices.
    ///
    /// # Errors
    ///
    /// Returns an error when conversion to scattering parameters fails or the
    /// resulting network dimensions are inconsistent.
    pub fn from_impedance(
        frequency: Frequency,
        impedance: &Array3<Complex64>,
        z0: Array2<Complex64>,
        definition: SParameterDefinition,
    ) -> Result<Self> {
        let scattering = z_to_s(impedance, &z0, definition)?;
        let mut network = Self::new(frequency, scattering, z0)?;
        network.s_definition = definition;
        Ok(network)
    }

    /// Creates a network from admittance matrices.
    ///
    /// # Errors
    ///
    /// Returns an error when conversion to scattering parameters fails or the
    /// resulting network dimensions are inconsistent.
    pub fn from_admittance(
        frequency: Frequency,
        admittance: &Array3<Complex64>,
        z0: Array2<Complex64>,
        definition: SParameterDefinition,
    ) -> Result<Self> {
        let scattering = y_to_s(admittance, &z0, definition)?;
        let mut network = Self::new(frequency, scattering, z0)?;
        network.s_definition = definition;
        Ok(network)
    }

    /// Returns the number of ports.
    #[must_use]
    pub fn ports(&self) -> usize {
        self.s.dim().1
    }

    /// Returns the number of frequency points.
    #[must_use]
    pub fn frequency_points(&self) -> usize {
        self.frequency.points()
    }

    /// Converts scattering parameters to impedance parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when the parameter dimensions or reference impedances
    /// are invalid, or when a required linear solve fails.
    pub fn impedance(&self) -> Result<Array3<Complex64>> {
        s_to_z(&self.s, &self.z0, self.s_definition)
    }

    /// Converts scattering parameters to admittance parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when the parameter dimensions or reference impedances
    /// are invalid, or when a required linear solve fails.
    pub fn admittance(&self) -> Result<Array3<Complex64>> {
        s_to_y(&self.s, &self.z0, self.s_definition)
    }

    /// Converts scattering parameters to hybrid $H$ parameters.
    ///
    /// # Errors
    ///
    /// Returns an error unless the network is a valid two-port whose
    /// scattering parameters can be converted to hybrid parameters.
    pub fn hybrid(&self) -> Result<Array3<Complex64>> {
        s_to_h(&self.s, &self.z0, self.s_definition)
    }

    /// Converts scattering parameters to inverse-hybrid $G$ parameters.
    ///
    /// # Errors
    ///
    /// Returns an error unless the network is a valid two-port whose
    /// scattering parameters can be converted to inverse-hybrid parameters.
    pub fn inverse_hybrid(&self) -> Result<Array3<Complex64>> {
        s_to_g(&self.s, &self.z0, self.s_definition)
    }

    /// Converts scattering parameters to scattering-transfer $T$ parameters.
    ///
    /// # Errors
    ///
    /// Returns an error unless the network is a two-port network.
    pub fn scattering_transfer(&self) -> Result<Array3<Complex64>> {
        s_to_t(&self.s)
    }

    /// Converts a two-port network to ABCD parameters.
    ///
    /// # Errors
    ///
    /// Returns an error unless the network and its reference impedances form a
    /// valid two-port data set.
    pub fn abcd(&self) -> Result<Array3<Complex64>> {
        s_to_abcd(&self.s, &self.z0)
    }

    /// Returns scattering-parameter magnitudes $|S|$.
    #[must_use]
    pub fn s_magnitude(&self) -> Array3<f64> {
        self.s.mapv(Complex64::norm)
    }

    /// Returns scattering magnitudes in decibels, $20\log_{10}|S|$.
    #[must_use]
    pub fn s_db(&self) -> Array3<f64> {
        self.s.mapv(|value| 20.0 * value.norm().log10())
    }

    /// Returns power-style scattering values, $10\log_{10}|S|$.
    #[must_use]
    pub fn s_db10(&self) -> Array3<f64> {
        self.s.mapv(|value| 10.0 * value.norm().log10())
    }

    /// Returns scattering phases in radians.
    #[must_use]
    pub fn s_phase_radians(&self) -> Array3<f64> {
        self.s.mapv(Complex64::arg)
    }

    /// Returns scattering phases in degrees.
    #[must_use]
    pub fn s_phase_degrees(&self) -> Array3<f64> {
        self.s.mapv(|value| value.arg().to_degrees())
    }

    /// Returns unwrapped scattering phases in radians along frequency.
    #[must_use]
    pub fn s_phase_unwrapped_radians(&self) -> Array3<f64> {
        let mut phase = self.s_phase_radians();
        for output in 0..self.ports() {
            for input in 0..self.ports() {
                let unwrapped = unwrap_radians(&Array1::from_iter(
                    (0..self.frequency_points()).map(|point| phase[(point, output, input)]),
                ));
                for point in 0..self.frequency_points() {
                    phase[(point, output, input)] = unwrapped[point];
                }
            }
        }
        phase
    }

    /// Returns the real parts of the scattering parameters.
    #[must_use]
    pub fn s_real(&self) -> Array3<f64> {
        self.s.mapv(|value| value.re)
    }

    /// Returns the imaginary parts of the scattering parameters.
    #[must_use]
    pub fn s_imaginary(&self) -> Array3<f64> {
        self.s.mapv(|value| value.im)
    }

    /// Returns voltage standing-wave ratio $(1+|S|)/(1-|S|)$.
    #[must_use]
    pub fn s_vswr(&self) -> Array3<f64> {
        self.s
            .mapv(|value| (1.0 + value.norm()) / (1.0 - value.norm()))
    }

    /// Returns centered inverse-FFT time-domain scattering data.
    ///
    /// # Errors
    ///
    /// This implementation currently produces an `Ok` value unconditionally;
    /// the result type is retained for API compatibility.
    pub fn s_time(&self) -> Result<Array3<Complex64>> {
        let mut time = Array3::zeros(self.s.dim());
        for output in 0..self.ports() {
            for input in 0..self.ports() {
                let transformed = inverse_fft_centered(&Array1::from_iter(
                    (0..self.frequency_points()).map(|point| self.s[(point, output, input)]),
                ));
                for point in 0..self.frequency_points() {
                    time[(point, output, input)] = transformed[point];
                }
            }
        }
        Ok(time)
    }

    /// Returns time-domain scattering magnitudes in decibels.
    ///
    /// # Errors
    ///
    /// Returns an error if conversion to time-domain scattering data fails.
    pub fn s_time_db(&self) -> Result<Array3<f64>> {
        Ok(self.s_time()?.mapv(|value| 20.0 * value.norm().log10()))
    }

    /// Applies a frequency-domain window to every scattering trace.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested window cannot be generated for the
    /// network's number of frequency points.
    pub fn windowed(&self, window: &crate::time::Window, normalize: bool) -> Result<Self> {
        let samples = crate::time::window_samples(window, self.frequency_points())?;
        let scale = if normalize {
            let point_count = u32::try_from(self.frequency_points()).map_err(|_| {
                Error::Unsupported(
                    "window point count exceeds the supported normalization range".to_owned(),
                )
            })?;
            f64::from(point_count) / samples.iter().sum::<f64>()
        } else {
            1.0
        };
        let mut result = self.clone();
        for point in 0..self.frequency_points() {
            for output in 0..self.ports() {
                for input in 0..self.ports() {
                    result.s[(point, output, input)] *= samples[point] * scale;
                }
            }
        }
        Ok(result)
    }

    /// Calculates the impulse response and its time axis.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid port indices, an unsupported window, or a
    /// frequency axis that cannot produce a time axis.
    pub fn impulse_response(
        &self,
        output: usize,
        input: usize,
        window: Option<&crate::time::Window>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        self.validate_port_pair(output, input)?;
        let transformed = if let Some(window) = window {
            self.windowed(window, true)?.s_time()?
        } else {
            self.s_time()?
        };
        Ok((
            self.frequency.time()?,
            Array1::from_iter(
                (0..self.frequency_points()).map(|point| transformed[(point, output, input)].re),
            ),
        ))
    }

    /// Calculates the step response and its time axis.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying impulse response cannot be
    /// calculated.
    pub fn step_response(
        &self,
        output: usize,
        input: usize,
        window: Option<&crate::time::Window>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let (time, impulse) = self.impulse_response(output, input, window)?;
        let step = if time.len() > 1 {
            time[1] - time[0]
        } else {
            1.0
        };
        let mut accumulated = 0.0;
        Ok((
            time,
            Array1::from_iter(impulse.iter().map(|value| {
                accumulated += value * step;
                accumulated
            })),
        ))
    }

    /// Tests whether all singular values of $S$ are at most one within tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error when `tolerance` is negative or non-finite.
    pub fn is_passive(&self, tolerance: f64) -> Result<bool> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(Error::Unsupported(
                "passivity tolerance must be finite and non-negative".to_owned(),
            ));
        }
        Ok((0..self.frequency_points())
            .all(|point| scattering_spectral_norm(&self.s, point) <= 1.0 + tolerance))
    }

    /// Tests whether $S_{ij}=S_{ji}$ within tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error when `tolerance` is negative or non-finite, or when the
    /// scattering matrices are not square.
    pub fn is_reciprocal(&self, tolerance: f64) -> Result<bool> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(Error::Unsupported(
                "reciprocity tolerance must be finite and non-negative".to_owned(),
            ));
        }
        Ok(reciprocity(&self.s)?
            .iter()
            .all(|difference| *difference <= tolerance))
    }

    /// Tests whether all ports have symmetric scattering behavior.
    ///
    /// # Errors
    ///
    /// Returns an error when `tolerance` is negative or non-finite, or when the
    /// network's ports cannot be flipped in pairs.
    pub fn is_symmetric(&self, tolerance: f64) -> Result<bool> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(Error::Unsupported(
                "symmetry tolerance must be finite and non-negative".to_owned(),
            ));
        }
        let flipped = self.flipped()?;
        Ok(self
            .s
            .iter()
            .zip(flipped.s.iter())
            .all(|(left, right)| (*left - *right).norm() <= tolerance))
    }

    /// Returns a per-frequency aggregate scattering error against another network.
    ///
    /// # Errors
    ///
    /// Returns an error when the networks have different frequency axes or
    /// scattering-matrix dimensions.
    pub fn scattering_error(&self, other: &Self) -> Result<Array1<f64>> {
        if self.frequency != other.frequency || self.s.dim() != other.s.dim() {
            return Err(Error::IncompatibleShape(
                "scattering error requires matching frequency and port shapes".to_owned(),
            ));
        }
        let matrix_elements = self
            .ports()
            .checked_mul(self.ports())
            .and_then(|count| u32::try_from(count).ok())
            .ok_or_else(|| {
                Error::Unsupported(
                    "port count exceeds the supported error-normalization range".to_owned(),
                )
            })?;
        let normalization = f64::from(matrix_elements);
        Ok(Array1::from_iter((0..self.frequency_points()).map(
            |point| {
                ((0..self.ports())
                    .flat_map(|output| {
                        (0..self.ports()).map(move |input| {
                            (self.s[(point, output, input)] - other.s[(point, output, input)])
                                .norm_sqr()
                        })
                    })
                    .sum::<f64>()
                    / normalization)
                    .sqrt()
            },
        )))
    }

    /// Tests whether $S^\dagger S=I$ within tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error when `tolerance` is negative or non-finite.
    pub fn is_lossless(&self, tolerance: f64) -> Result<bool> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(Error::Unsupported(
                "lossless tolerance must be finite and non-negative".to_owned(),
            ));
        }
        Ok((0..self.frequency_points()).all(|point| {
            (0..self.ports()).all(|row| {
                (0..self.ports()).all(|column| {
                    let gram = (0..self.ports())
                        .map(|port| {
                            self.s[(point, port, row)].conj() * self.s[(point, port, column)]
                        })
                        .sum::<Complex64>();
                    let expected = if row == column { 1.0 } else { 0.0 };
                    (gram - expected).norm() <= tolerance
                })
            })
        }))
    }

    /// Returns group delay $-d\arg(S)/d\omega$ in seconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the network has fewer than two frequency points.
    pub fn group_delay(&self) -> Result<Array3<f64>> {
        if self.frequency_points() < 2 {
            return Err(Error::InvalidFrequency(
                "group delay requires at least two frequency points".to_owned(),
            ));
        }
        let phase = self.s_phase_unwrapped_radians();
        let mut delay = Array3::zeros(self.s.dim());
        for output in 0..self.ports() {
            for input in 0..self.ports() {
                for point in 0..self.frequency_points() {
                    let (left, right) = if point == 0 {
                        (0, 1)
                    } else if point + 1 == self.frequency_points() {
                        (point - 1, point)
                    } else {
                        (point - 1, point + 1)
                    };
                    let delta_omega = std::f64::consts::TAU
                        * (self.frequency.values_hz()[right] - self.frequency.values_hz()[left]);
                    delay[(point, output, input)] = -(phase[(right, output, input)]
                        - phase[(left, output, input)])
                        / delta_omega;
                }
            }
        }
        Ok(delay)
    }

    /// Returns Rollet's two-port stability factor $K$.
    ///
    /// # Errors
    ///
    /// Returns an error unless the network is a two-port network.
    pub fn stability_factor(&self) -> Result<Array1<f64>> {
        if self.ports() != 2 {
            return Err(Error::IncompatibleShape(
                "stability factor requires a two-port network".to_owned(),
            ));
        }
        Ok(Array1::from_iter((0..self.frequency_points()).map(
            |point| {
                let s11 = self.s[(point, 0, 0)];
                let s12 = self.s[(point, 0, 1)];
                let s21 = self.s[(point, 1, 0)];
                let s22 = self.s[(point, 1, 1)];
                let determinant = s11 * s22 - s12 * s21;
                (1.0 - s11.norm_sqr() - s22.norm_sqr() + determinant.norm_sqr())
                    / (2.0 * (s12 * s21).norm())
            },
        )))
    }

    /// Returns maximum stable gain for a two-port network.
    ///
    /// # Errors
    ///
    /// Returns an error unless the network is a two-port network.
    pub fn maximum_stable_gain(&self) -> Result<Array1<f64>> {
        if self.ports() != 2 {
            return Err(Error::IncompatibleShape(
                "maximum stable gain requires a two-port network".to_owned(),
            ));
        }
        Ok(Array1::from_iter((0..self.frequency_points()).map(
            |point| self.s[(point, 1, 0)].norm() / self.s[(point, 0, 1)].norm(),
        )))
    }

    /// Returns maximum available or stable gain according to the stability factor.
    ///
    /// # Errors
    ///
    /// Returns an error unless the network is a two-port network.
    pub fn maximum_gain(&self) -> Result<Array1<f64>> {
        let stability = self.stability_factor()?;
        let stable_gain = self.maximum_stable_gain()?;
        Ok(Array1::from_iter((0..self.frequency_points()).map(
            |point| {
                let clipped = stability[point].max(1.0);
                stable_gain[point] / (clipped + (clipped * clipped - 1.0).sqrt())
            },
        )))
    }

    /// Returns Mason's unilateral gain.
    ///
    /// # Errors
    ///
    /// Returns an error unless the network is a two-port network.
    pub fn unilateral_gain(&self) -> Result<Array1<f64>> {
        let stability = self.stability_factor()?;
        let stable_gain = self.maximum_stable_gain()?;
        Ok(Array1::from_iter((0..self.frequency_points()).map(
            |point| {
                let ratio = self.s[(point, 1, 0)] / self.s[(point, 0, 1)];
                (ratio - Complex64::new(1.0, 0.0)).norm_sqr()
                    / 2.0f64.mul_add(-ratio.re, 2.0 * stability[point] * stable_gain[point])
            },
        )))
    }

    /// Selects and reorders a subset of ports.
    ///
    /// # Errors
    ///
    /// Returns an error when `ports` is empty or contains an out-of-range port
    /// index.
    pub fn subnetwork(&self, ports: &[usize]) -> Result<Self> {
        if ports.is_empty() || ports.iter().any(|port| *port >= self.ports()) {
            return Err(Error::InvalidPort {
                port: ports.iter().copied().max().unwrap_or(0),
                ports: self.ports(),
            });
        }
        let mut result = self.clone();
        result.s = Array3::from_shape_fn(
            (self.frequency_points(), ports.len(), ports.len()),
            |(point, output, input)| self.s[(point, ports[output], ports[input])],
        );
        result.z0 =
            Array2::from_shape_fn((self.frequency_points(), ports.len()), |(point, port)| {
                self.z0[(point, ports[port])]
            });
        result.port_modes = ports.iter().map(|port| self.port_modes[*port]).collect();
        result.port_names = if self.port_names.len() == self.ports() {
            ports
                .iter()
                .map(|port| self.port_names[*port].clone())
                .collect()
        } else {
            Vec::new()
        };
        Ok(result)
    }

    /// Crops the network to an inclusive frequency interval in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite or reversed bounds, when the interval
    /// contains no samples, or when the cropped frequency axis is invalid.
    pub fn cropped(&self, start_hz: f64, stop_hz: f64) -> Result<Self> {
        if !start_hz.is_finite() || !stop_hz.is_finite() || start_hz > stop_hz {
            return Err(Error::InvalidFrequency(
                "crop bounds must be finite and increasing".to_owned(),
            ));
        }
        let indexes = self
            .frequency
            .values_hz()
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (*value >= start_hz && *value <= stop_hz).then_some(index))
            .collect::<Vec<_>>();
        if indexes.is_empty() {
            return Err(Error::InvalidFrequency(
                "crop range does not contain frequency samples".to_owned(),
            ));
        }
        let mut result = self.clone();
        result.frequency = Frequency::from_hz(Array1::from_iter(
            indexes
                .iter()
                .map(|index| self.frequency.values_hz()[*index]),
        ))?;
        result.s = Array3::from_shape_fn(
            (indexes.len(), self.ports(), self.ports()),
            |(point, output, input)| self.s[(indexes[point], output, input)],
        );
        result.z0 = Array2::from_shape_fn((indexes.len(), self.ports()), |(point, port)| {
            self.z0[(indexes[point], port)]
        });
        Ok(result)
    }

    /// Adds a phase delay to one port.
    ///
    /// # Errors
    ///
    /// Returns an error when `port` is out of range or `phase_degrees` is not
    /// finite.
    pub fn delayed_port(&self, port: usize, phase_degrees: f64) -> Result<Self> {
        if port >= self.ports() {
            return Err(Error::InvalidPort {
                port,
                ports: self.ports(),
            });
        }
        if !phase_degrees.is_finite() {
            return Err(Error::Unsupported("delay phase must be finite".to_owned()));
        }
        let factor = Complex64::from_polar(1.0, -phase_degrees.to_radians());
        let mut result = self.clone();
        for point in 0..self.frequency_points() {
            for other in 0..self.ports() {
                result.s[(point, port, other)] *= factor;
                result.s[(point, other, port)] *= factor;
            }
        }
        Ok(result)
    }

    /// Rotates every scattering parameter by a common phase.
    ///
    /// # Errors
    ///
    /// Returns an error when `phase_degrees` is not finite.
    pub fn rotated(&self, phase_degrees: f64) -> Result<Self> {
        if !phase_degrees.is_finite() {
            return Err(Error::Unsupported(
                "rotation phase must be finite".to_owned(),
            ));
        }
        let factor = Complex64::from_polar(1.0, -phase_degrees.to_radians());
        let mut result = self.clone();
        result.s.mapv_inplace(|value| value * factor);
        Ok(result)
    }

    /// Adds independent Gaussian magnitude and phase noise to scattering data.
    ///
    /// # Errors
    ///
    /// Returns an error when either deviation cannot define a normal
    /// distribution.
    pub fn with_added_polar_noise(
        &self,
        magnitude_deviation: f64,
        phase_deviation_degrees: f64,
        flatband: bool,
    ) -> Result<Self> {
        let shape = if flatband {
            (1, self.ports(), self.ports())
        } else {
            self.s.dim()
        };
        let magnitude =
            crate::math::random_normal_like(&Array3::from_elem(shape, magnitude_deviation))?;
        let phase =
            crate::math::random_normal_like(&Array3::from_elem(shape, phase_deviation_degrees))?;
        let mut result = self.clone();
        for point in 0..self.frequency_points() {
            let noise_point = if flatband { 0 } else { point };
            for output in 0..self.ports() {
                for input in 0..self.ports() {
                    let value = self.s[(point, output, input)];
                    result.s[(point, output, input)] = Complex64::from_polar(
                        value.norm() + magnitude[(noise_point, output, input)],
                        value.arg() + phase[(noise_point, output, input)].to_radians(),
                    );
                }
            }
        }
        Ok(result)
    }

    /// Multiplies scattering data by Gaussian polar perturbations.
    ///
    /// # Errors
    ///
    /// Returns an error when either deviation cannot define a normal
    /// distribution.
    pub fn with_multiplicative_noise(
        &self,
        magnitude_deviation: f64,
        phase_deviation_degrees: f64,
    ) -> Result<Self> {
        let magnitude =
            crate::math::random_normal_like(&Array3::from_elem(self.s.dim(), magnitude_deviation))?;
        let phase = crate::math::random_normal_like(&Array3::from_elem(
            self.s.dim(),
            phase_deviation_degrees,
        ))?;
        let mut result = self.clone();
        for (index, value) in result.s.indexed_iter_mut() {
            *value *= Complex64::from_polar(1.0 + magnitude[index], phase[index].to_radians());
        }
        Ok(result)
    }

    /// Adds a small complex perturbation to avoid exact singularities.
    ///
    /// # Errors
    ///
    /// Returns an error when `amount` is not finite.
    pub fn nudged(&self, amount: f64) -> Result<Self> {
        if !amount.is_finite() {
            return Err(Error::Unsupported("nudge amount must be finite".to_owned()));
        }
        let mut result = self.clone();
        result.s.mapv_inplace(|value| value + amount);
        Ok(result)
    }

    fn validate_port_pair(&self, output: usize, input: usize) -> Result<()> {
        if output >= self.ports() {
            return Err(Error::InvalidPort {
                port: output,
                ports: self.ports(),
            });
        }
        if input >= self.ports() {
            return Err(Error::InvalidPort {
                port: input,
                ports: self.ports(),
            });
        }
        Ok(())
    }

    /// Returns whether noise parameters are attached.
    #[must_use]
    pub const fn is_noisy(&self) -> bool {
        self.noise.is_some()
    }

    /// Port of `Network.set_noise_a`, retaining the standard four noise
    /// parameters rather than materializing the equivalent correlation matrix.
    /// Attaches validated two-port noise parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when the parameter arrays do not match the frequency
    /// axis or contain invalid values.
    pub fn set_noise_parameters(
        &mut self,
        frequency: Frequency,
        minimum_noise_figure_db: Array1<f64>,
        optimal_reflection: Array1<Complex64>,
        equivalent_noise_resistance: Array1<f64>,
    ) -> Result<()> {
        self.noise = Some(NoiseParameters::new(
            frequency,
            minimum_noise_figure_db,
            optimal_reflection,
            equivalent_noise_resistance,
        )?);
        Ok(())
    }

    /// Port of `Network.nf` for a constant complex source impedance.
    /// Returns noise factor for a specified source impedance.
    ///
    /// # Errors
    ///
    /// Returns an error when noise parameters are absent, the source impedance
    /// is zero, or its admittance does not have positive conductance.
    pub fn noise_factor(&self, source_impedance: Complex64) -> Result<Array1<f64>> {
        let noise = self.noise.as_ref().ok_or_else(|| {
            Error::Unsupported("network does not contain noise parameters".to_owned())
        })?;
        if source_impedance.norm_sqr() <= f64::EPSILON {
            return Err(Error::Unsupported(
                "source impedance must be non-zero for a noise-factor calculation".to_owned(),
            ));
        }
        let source_admittance = Complex64::new(1.0, 0.0) / source_impedance;
        if source_admittance.re <= 0.0 {
            return Err(Error::Unsupported(
                "source impedance must have positive conductance".to_owned(),
            ));
        }
        let reference = self.z0[(0, 0)];
        Ok(Array1::from_iter((0..noise.frequency.points()).map(
            |index| {
                let gamma = noise.optimal_reflection[index];
                let optimal_impedance = reference * (Complex64::new(1.0, 0.0) + gamma)
                    / (Complex64::new(1.0, 0.0) - gamma);
                let optimal_admittance = Complex64::new(1.0, 0.0) / optimal_impedance;
                let minimum = 10.0_f64.powf(noise.minimum_noise_figure_db[index] / 10.0);
                (noise.equivalent_noise_resistance[index] / source_admittance.re)
                    .mul_add((source_admittance - optimal_admittance).norm_sqr(), minimum)
            },
        )))
    }

    /// Returns minimum noise factor in linear units.
    ///
    /// # Errors
    ///
    /// Returns an error when the network has no noise parameters.
    pub fn minimum_noise_factor(&self) -> Result<Array1<f64>> {
        let noise = self.noise.as_ref().ok_or_else(|| {
            Error::Unsupported("network does not contain noise parameters".to_owned())
        })?;
        Ok(noise
            .minimum_noise_figure_db
            .mapv(|value| 10.0_f64.powf(value / 10.0)))
    }

    /// Returns the optimum source impedance for minimum noise.
    ///
    /// # Errors
    ///
    /// Returns an error when the network has no noise parameters.
    pub fn optimal_noise_impedance(&self) -> Result<Array1<Complex64>> {
        let noise = self.noise.as_ref().ok_or_else(|| {
            Error::Unsupported("network does not contain noise parameters".to_owned())
        })?;
        let reference = self.z0[(0, 0)];
        Ok(noise.optimal_reflection.mapv(|gamma| {
            reference * (Complex64::new(1.0, 0.0) + gamma) / (Complex64::new(1.0, 0.0) - gamma)
        }))
    }

    /// Returns the optimum source admittance for minimum noise.
    ///
    /// # Errors
    ///
    /// Returns an error when the network has no noise parameters.
    pub fn optimal_noise_admittance(&self) -> Result<Array1<Complex64>> {
        Ok(self
            .optimal_noise_impedance()?
            .mapv(|impedance| Complex64::new(1.0, 0.0) / impedance))
    }

    /// Returns noise figure in decibels for a specified source impedance.
    ///
    /// # Errors
    ///
    /// Returns an error when noise parameters are absent or the source
    /// impedance is invalid for a noise-factor calculation.
    pub fn noise_figure_db(&self, source_impedance: Complex64) -> Result<Array1<f64>> {
        Ok(self
            .noise_factor(source_impedance)?
            .mapv(|factor| 10.0 * factor.log10()))
    }

    /// Returns source or load stability-circle points on the reflection plane.
    ///
    /// # Errors
    ///
    /// Returns an error unless the network is a two-port, `target_port` is zero
    /// or one, and at least two circle points are requested.
    pub fn stability_circle(&self, target_port: usize, points: usize) -> Result<Array2<Complex64>> {
        if self.ports() != 2 || target_port >= 2 || points < 2 {
            return Err(Error::IncompatibleShape(
                "stability circles require a two-port, target port 0 or 1, and at least two points"
                    .to_owned(),
            ));
        }
        let point_values = (0..points)
            .map(|point| {
                u32::try_from(point).map(f64::from).map_err(|_| {
                    Error::Unsupported(
                        "stability-circle point count exceeds the supported range".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let point_intervals = point_values[points - 1];
        Ok(Array2::from_shape_fn(
            (self.frequency_points(), points),
            |(frequency, point)| {
                let s11 = self.s[(frequency, 0, 0)];
                let s12 = self.s[(frequency, 0, 1)];
                let s21 = self.s[(frequency, 1, 0)];
                let s22 = self.s[(frequency, 1, 1)];
                let determinant = s11 * s22 - s12 * s21;
                let (parameter, opposite) = if target_port == 0 {
                    (s11, s22)
                } else {
                    (s22, s11)
                };
                let denominator = parameter.norm_sqr() - determinant.norm_sqr();
                let center = (parameter - determinant * opposite.conj()).conj() / denominator;
                let radius = (s12 * s21).norm() / denominator.abs();
                center
                    + Complex64::from_polar(
                        radius,
                        std::f64::consts::TAU * point_values[point] / point_intervals,
                    )
            },
        ))
    }

    /// Returns constant operating- or available-gain circle points.
    ///
    /// # Errors
    ///
    /// Returns an error unless the network is a two-port, `target_port` is zero
    /// or one, `gain_db` is finite, and at least two points are requested.
    pub fn gain_circle(
        &self,
        target_port: usize,
        gain_db: f64,
        points: usize,
    ) -> Result<Array2<Complex64>> {
        if self.ports() != 2 || target_port >= 2 || points < 2 || !gain_db.is_finite() {
            return Err(Error::IncompatibleShape(
                "gain circles require a two-port, finite gain, target port 0 or 1, and at least two points"
                    .to_owned(),
            ));
        }
        let point_values = (0..points)
            .map(|point| {
                u32::try_from(point).map(f64::from).map_err(|_| {
                    Error::Unsupported(
                        "gain-circle point count exceeds the supported range".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let point_intervals = point_values[points - 1];
        let requested = 10.0_f64.powf(gain_db / 10.0);
        Ok(Array2::from_shape_fn(
            (self.frequency_points(), points),
            |(frequency, point)| {
                let reflection = self.s[(frequency, target_port, target_port)];
                let gain_factor = (requested * (1.0 - reflection.norm_sqr())).min(1.0);
                let denominator = (1.0 - gain_factor).mul_add(-reflection.norm_sqr(), 1.0);
                let center = gain_factor * reflection.conj() / denominator;
                let radius =
                    (1.0 - gain_factor).sqrt() * (1.0 - reflection.norm_sqr()) / denominator;
                center
                    + Complex64::from_polar(
                        radius.abs(),
                        std::f64::consts::TAU * point_values[point] / point_intervals,
                    )
            },
        ))
    }

    /// Returns constant-noise-figure circle points.
    ///
    /// # Errors
    ///
    /// Returns an error when noise parameters are absent, `noise_figure_db` is
    /// not finite, or fewer than two points are requested.
    pub fn noise_figure_circle(
        &self,
        noise_figure_db: f64,
        points: usize,
    ) -> Result<Array2<Complex64>> {
        let noise = self.noise.as_ref().ok_or_else(|| {
            Error::Unsupported("network does not contain noise parameters".to_owned())
        })?;
        if points < 2 || !noise_figure_db.is_finite() {
            return Err(Error::Unsupported(
                "noise circles require finite figure and at least two points".to_owned(),
            ));
        }
        let point_values = (0..points)
            .map(|point| {
                u32::try_from(point).map(f64::from).map_err(|_| {
                    Error::Unsupported(
                        "noise-circle point count exceeds the supported range".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let point_intervals = point_values[points - 1];
        let requested = 10.0_f64.powf(noise_figure_db / 10.0);
        let minimum = self.minimum_noise_factor()?;
        Ok(Array2::from_shape_fn(
            (noise.frequency.points(), points),
            |(frequency, point)| {
                let optimum = noise.optimal_reflection[frequency];
                let reference = self.z0[(frequency.min(self.frequency_points() - 1), 0)]
                    .re
                    .abs();
                let normalized_resistance =
                    noise.equivalent_noise_resistance[frequency] / reference;
                let n = (requested - minimum[frequency]) * (1.0 + optimum).norm_sqr()
                    / (4.0 * normalized_resistance);
                let center = optimum / (n + 1.0);
                let radius =
                    (n * n + n * (1.0 - optimum.norm_sqr())).max(0.0).sqrt() / (n + 1.0).abs();
                center
                    + Complex64::from_polar(
                        radius,
                        std::f64::consts::TAU * point_values[point] / point_intervals,
                    )
            },
        ))
    }

    /// Interpolates the network using Cartesian linear interpolation.
    ///
    /// # Errors
    ///
    /// Returns an error when either frequency axis is unsuitable for
    /// interpolation, a target lies outside the source range, or the resulting
    /// network dimensions are invalid.
    pub fn interpolate(&self, frequency: &Frequency) -> Result<Self> {
        if self.frequency_points() < 2 {
            return Err(Error::InvalidFrequency(
                "interpolation requires at least two source frequency points".to_owned(),
            ));
        }
        if !self.frequency.is_monotonic_increasing() || !frequency.is_monotonic_increasing() {
            return Err(Error::InvalidFrequency(
                "interpolation requires monotonically increasing frequency axes".to_owned(),
            ));
        }
        let source_start = self.frequency.start().ok_or_else(|| {
            Error::InvalidFrequency(
                "interpolation requires at least two source frequency points".to_owned(),
            )
        })?;
        let source_stop = self.frequency.stop().ok_or_else(|| {
            Error::InvalidFrequency(
                "interpolation requires at least two source frequency points".to_owned(),
            )
        })?;
        if frequency
            .values_hz()
            .iter()
            .any(|value| *value < source_start || *value > source_stop)
        {
            return Err(Error::InvalidFrequency(
                "interpolation target is outside the source frequency range".to_owned(),
            ));
        }

        let ports = self.ports();
        let mut s = Array3::zeros((frequency.points(), ports, ports));
        let mut z0 = Array2::zeros((frequency.points(), ports));
        for (target_index, target) in frequency.values_hz().iter().copied().enumerate() {
            let upper = self
                .frequency
                .values_hz()
                .iter()
                .position(|source| *source >= target)
                .ok_or_else(|| {
                    Error::InvalidFrequency(
                        "interpolation target is outside the source frequency range".to_owned(),
                    )
                })?;
            let (lower, upper, fraction) = if upper == 0 {
                (0, 0, 0.0)
            } else if self.frequency.values_hz()[upper].total_cmp(&target).is_eq() {
                (upper, upper, 0.0)
            } else {
                let lower = upper - 1;
                let low_frequency = self.frequency.values_hz()[lower];
                let high_frequency = self.frequency.values_hz()[upper];
                (
                    lower,
                    upper,
                    (target - low_frequency) / (high_frequency - low_frequency),
                )
            };
            for output_port in 0..ports {
                for input_port in 0..ports {
                    s[(target_index, output_port, input_port)] = interpolate_complex(
                        self.s[(lower, output_port, input_port)],
                        self.s[(upper, output_port, input_port)],
                        fraction,
                    );
                }
                z0[(target_index, output_port)] = interpolate_complex(
                    self.z0[(lower, output_port)],
                    self.z0[(upper, output_port)],
                    fraction,
                );
            }
        }

        let mut network = Self::new(frequency.clone(), s, z0)?;
        network.copy_metadata_from(self);
        Ok(network)
    }

    /// Interpolates in Cartesian, polar, or Floater-Hormann rational form.
    /// Interpolates the network using the selected [`InterpolationMode`].
    ///
    /// # Errors
    ///
    /// Returns an error when the frequency axes are unsuitable for
    /// interpolation or rational-interpolator construction fails.
    pub fn interpolate_with_mode(
        &self,
        frequency: &Frequency,
        mode: InterpolationMode,
    ) -> Result<Self> {
        if mode == InterpolationMode::CartesianLinear {
            return self.interpolate(frequency);
        }
        let mut result = self.interpolate(frequency)?;
        for output in 0..self.ports() {
            for input in 0..self.ports() {
                let source = Array1::from_iter(
                    (0..self.frequency_points()).map(|point| self.s[(point, output, input)]),
                );
                let values = match mode {
                    InterpolationMode::CartesianLinear => unreachable!(),
                    InterpolationMode::PolarLinear => {
                        let magnitude = source.mapv(Complex64::norm);
                        let phase = unwrap_radians(&source.mapv(Complex64::arg));
                        Array1::from_iter(frequency.values_hz().iter().map(|target| {
                            let magnitude =
                                linear_sample(self.frequency.values_hz(), &magnitude, *target);
                            let phase = linear_sample(self.frequency.values_hz(), &phase, *target);
                            Complex64::from_polar(magnitude, phase)
                        }))
                    }
                    InterpolationMode::Cubic => {
                        Array1::from_iter(frequency.values_hz().iter().map(|target| {
                            cubic_sample(self.frequency.values_hz(), &source, *target)
                        }))
                    }
                    InterpolationMode::Rational { degree } => RationalInterpolator::new(
                        self.frequency.values_hz(),
                        &source,
                        degree,
                        f64::EPSILON,
                        true,
                    )?
                    .evaluate(frequency.values_hz()),
                };
                for point in 0..frequency.points() {
                    result.s[(point, output, input)] = values[point];
                }
            }
        }
        Ok(result)
    }

    /// Extrapolates a measured network to DC on a uniform frequency grid.
    ///
    /// Origin: `skrf.network.Network.extrapolate_to_dc`. Rust uses linear
    /// magnitude/unwrapped-phase interpolation for a deterministic dependency-free API.
    /// Extrapolates the network to DC, optionally using supplied DC scattering data.
    ///
    /// # Errors
    ///
    /// Returns an error for insufficient frequency points, an invalid output
    /// count or DC matrix shape, or a failure to construct or interpolate the
    /// extended frequency grid.
    pub fn extrapolate_to_dc(
        &self,
        points: Option<usize>,
        dc_s_parameters: Option<Array2<Complex64>>,
    ) -> Result<Self> {
        if self.frequency_points() < 2 {
            return Err(Error::InvalidFrequency(
                "DC extrapolation requires at least two frequency points".to_owned(),
            ));
        }
        if self.frequency.values_hz()[0] == 0.0 {
            return Ok(self.clone());
        }
        let requested_points = points.map_or_else(
            || {
                let step = self.frequency.values_hz()[1] - self.frequency.values_hz()[0];
                let extrapolated_points = (self.frequency.values_hz()[0] / step)
                    .round()
                    .max(1.0)
                    .to_usize()
                    .ok_or_else(|| {
                        Error::InvalidFrequency(
                            "DC extrapolation point count is outside the supported range"
                                .to_owned(),
                        )
                    })?;
                self.frequency_points()
                    .checked_add(extrapolated_points)
                    .ok_or_else(|| {
                        Error::InvalidFrequency(
                            "DC extrapolation point count overflowed".to_owned(),
                        )
                    })
            },
            Ok,
        )?;
        if requested_points < 2 {
            return Err(Error::InvalidFrequency(
                "DC extrapolation requires at least two output points".to_owned(),
            ));
        }
        let ports = self.ports();
        let dc = if let Some(dc) = dc_s_parameters {
            if dc.dim() != (ports, ports) {
                return Err(Error::IncompatibleShape(format!(
                    "DC S parameters have shape {:?}, expected ({ports}, {ports})",
                    dc.dim()
                )));
            }
            dc
        } else {
            let first_frequency = self.frequency.values_hz()[0];
            let second_frequency = self.frequency.values_hz()[1];
            Array2::from_shape_fn((ports, ports), |(row, column)| {
                let first = self.s[(0, row, column)];
                let second = self.s[(1, row, column)];
                let fraction = -first_frequency / (second_frequency - first_frequency);
                let magnitude = fraction.mul_add(second.norm() - first.norm(), first.norm());
                let mut phase_delta = second.arg() - first.arg();
                while phase_delta.total_cmp(&std::f64::consts::PI).is_gt() {
                    phase_delta = 2.0f64.mul_add(-std::f64::consts::PI, phase_delta);
                }
                while phase_delta.total_cmp(&-std::f64::consts::PI).is_lt() {
                    phase_delta = 2.0f64.mul_add(std::f64::consts::PI, phase_delta);
                }
                Complex64::from_polar(magnitude, fraction.mul_add(phase_delta, first.arg()))
            })
        };
        let mut frequencies = Vec::with_capacity(self.frequency_points() + 1);
        frequencies.push(0.0);
        frequencies.extend(self.frequency.values_hz().iter().copied());
        let mut s = Array3::zeros((self.frequency_points() + 1, ports, ports));
        let mut z0 = Array2::zeros((self.frequency_points() + 1, ports));
        for row in 0..ports {
            for column in 0..ports {
                s[(0, row, column)] = dc[(row, column)];
                for point in 0..self.frequency_points() {
                    s[(point + 1, row, column)] = self.s[(point, row, column)];
                }
            }
            z0[(0, row)] = self.z0[(0, row)];
            for point in 0..self.frequency_points() {
                z0[(point + 1, row)] = self.z0[(point, row)];
            }
        }
        let mut extended = Self::new(Frequency::from_hz(Array1::from(frequencies))?, s, z0)?;
        extended.copy_metadata_from(self);
        extended.noise = None;
        extended.propagation_constants = None;
        let target = Frequency::new(
            0.0,
            self.frequency.values_hz()[self.frequency_points() - 1],
            requested_points,
            crate::FrequencyUnit::Hz,
            crate::SweepType::Linear,
        )?;
        let mut result = extended.interpolate(&target)?;
        for value in result.s.index_axis_mut(ndarray::Axis(0), 0) {
            *value = Complex64::new(value.re, 0.0);
        }
        Ok(result)
    }

    /// Cascades this network with another network.
    ///
    /// # Errors
    ///
    /// Returns an error unless both networks are compatible two-ports, or when
    /// conversion between scattering and chain matrices fails.
    pub fn cascade(&self, other: &Self) -> Result<Self> {
        validate_two_port_pair(self, other)?;
        let points = self.frequency_points();
        let mut s = Array3::zeros((points, 2, 2));
        let mut z0 = Array2::zeros((points, 2));
        for point in 0..points {
            let left = scattering_to_chain(&self.s, point)?;
            let right = scattering_to_chain(&other.s, point)?;
            let cascaded = multiply_two_by_two(left, right);
            write_chain_as_scattering(cascaded, &mut s, point)?;
            z0[(point, 0)] = self.z0[(point, 0)];
            z0[(point, 1)] = other.z0[(point, 1)];
        }
        let mut network = Self::new(self.frequency.clone(), s, z0)?;
        network.name = match (&self.name, &other.name) {
            (Some(left), Some(right)) => Some(format!("{left}**{right}")),
            _ => None,
        };
        Ok(network)
    }

    /// Element-wise complex addition of two compatible scattering matrices.
    ///
    /// Origin: `skrf.network.Network.__add__`.
    /// Adds aligned scattering matrices element by element.
    ///
    /// # Errors
    ///
    /// Returns an error when the networks have incompatible frequency axes or
    /// scattering-matrix dimensions.
    pub fn add_elementwise(&self, other: &Self) -> Result<Self> {
        self.elementwise_binary(other, |left, right| left + right)
    }

    /// Element-wise complex subtraction of two compatible scattering matrices.
    ///
    /// Origin: `skrf.network.Network.__sub__`.
    /// Subtracts aligned scattering matrices element by element.
    ///
    /// # Errors
    ///
    /// Returns an error when the networks have incompatible frequency axes or
    /// scattering-matrix dimensions.
    pub fn subtract_elementwise(&self, other: &Self) -> Result<Self> {
        self.elementwise_binary(other, |left, right| left - right)
    }

    /// Element-wise complex multiplication of two compatible scattering matrices.
    ///
    /// Origin: `skrf.network.Network.__mul__`.
    /// Multiplies aligned scattering matrices element by element.
    ///
    /// # Errors
    ///
    /// Returns an error when the networks have incompatible frequency axes or
    /// scattering-matrix dimensions.
    pub fn multiply_elementwise(&self, other: &Self) -> Result<Self> {
        self.elementwise_binary(other, |left, right| left * right)
    }

    /// Element-wise complex division of two compatible scattering matrices.
    ///
    /// Origin: `skrf.network.Network.__truediv__`.
    /// Divides aligned scattering matrices element by element.
    ///
    /// # Errors
    ///
    /// Returns an error when the networks have incompatible frequency axes or
    /// scattering-matrix dimensions.
    pub fn divide_elementwise(&self, other: &Self) -> Result<Self> {
        self.elementwise_binary(other, |left, right| left / right)
    }

    /// Raises every scattering value to a real power.
    ///
    /// Origin: the numeric branch of `skrf.network.Network.__pow__`.
    /// Raises every scattering value to a real power.
    ///
    /// # Errors
    ///
    /// Returns an error when `exponent` is not finite.
    pub fn elementwise_power(&self, exponent: f64) -> Result<Self> {
        if !exponent.is_finite() {
            return Err(Error::Unsupported(
                "network exponent must be finite".to_owned(),
            ));
        }
        let mut result = self.clone();
        result.s.mapv_inplace(|value| value.powf(exponent));
        Ok(result)
    }

    /// Removes one left fixture and an optional right fixture from a two-port network.
    ///
    /// Origin: `skrf.network.Network.__floordiv__`.
    /// De-embeds left and optional right fixture networks.
    ///
    /// # Errors
    ///
    /// Returns an error when a fixture cannot be inverted or the networks are
    /// not compatible two-ports for cascading.
    pub fn deembed(&self, left_fixture: &Self, right_fixture: Option<&Self>) -> Result<Self> {
        let mut deembedded = left_fixture.inverse()?.cascade(self)?;
        if let Some(right_fixture) = right_fixture {
            deembedded = deembedded.cascade(&right_fixture.inverse()?)?;
        }
        let mut result = self.clone();
        result.s = deembedded.s;
        Ok(result)
    }

    /// Connects one port of this network to one port of another network.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible networks or invalid ports, when no
    /// external port remains, or when connection construction or its linear
    /// solve fails.
    pub fn connect(&self, port: usize, other: &Self, other_port: usize) -> Result<Self> {
        validate_connection_networks(self, port, other, other_port)?;
        let left_ports = self.ports();
        let right_ports = other.ports();
        let combined_ports = left_ports + right_ports;
        let internal = [port, left_ports + other_port];
        let external = (0..combined_ports)
            .filter(|index| !internal.contains(index))
            .collect::<Vec<_>>();
        if external.is_empty() {
            return Err(Error::Unsupported(
                "connecting two one-port Networks has no external ports".to_owned(),
            ));
        }
        let points = self.frequency_points();
        let mut scattering = Array3::zeros((points, external.len(), external.len()));
        let mut z0 = Array2::zeros((points, external.len()));
        for point in 0..points {
            let combined = combined_scattering(self, other, point);
            let connection = connection_scattering(
                combined_reference(self, other, point, internal[0]),
                combined_reference(self, other, point, internal[1]),
            )?;
            let internal_scattering = [
                combined[internal[0] * combined_ports + internal[0]],
                combined[internal[0] * combined_ports + internal[1]],
                combined[internal[1] * combined_ports + internal[0]],
                combined[internal[1] * combined_ports + internal[1]],
            ];
            let product = multiply_two_by_two(connection, internal_scattering);
            let system = [
                Complex64::new(1.0, 0.0) - product[0],
                -product[1],
                -product[2],
                Complex64::new(1.0, 0.0) - product[3],
            ];
            for (input, external_input) in external.iter().enumerate() {
                let internal_from_external = [
                    combined[internal[0] * combined_ports + external_input],
                    combined[internal[1] * combined_ports + external_input],
                ];
                let right = [
                    connection[0] * internal_from_external[0]
                        + connection[1] * internal_from_external[1],
                    connection[2] * internal_from_external[0]
                        + connection[3] * internal_from_external[1],
                ];
                let internal_incident = solve_two_by_two(system, right)?;
                for (output, external_output) in external.iter().enumerate() {
                    scattering[(point, output, input)] = combined
                        [external_output * combined_ports + external_input]
                        + combined[external_output * combined_ports + internal[0]]
                            * internal_incident[0]
                        + combined[external_output * combined_ports + internal[1]]
                            * internal_incident[1];
                }
            }
            for (output, combined_port) in external.iter().enumerate() {
                z0[(point, output)] = combined_reference(self, other, point, *combined_port);
            }
        }
        let mut network = Self::new(self.frequency.clone(), scattering, z0)?;
        network.name = match (&self.name, &other.name) {
            (Some(left), Some(right)) => Some(format!("{left}-{right}")),
            _ => None,
        };
        network.s_definition = self.s_definition;
        Ok(network)
    }

    /// Port of `skrf.network.innerconnect`.
    /// Connects two ports within the same network and removes them.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid or identical ports, when no external port
    /// would remain, or when connection construction or its linear solve
    /// fails.
    pub fn inner_connect(&self, first_port: usize, second_port: usize) -> Result<Self> {
        if first_port >= self.ports() {
            return Err(Error::InvalidPort {
                port: first_port,
                ports: self.ports(),
            });
        }
        if second_port >= self.ports() {
            return Err(Error::InvalidPort {
                port: second_port,
                ports: self.ports(),
            });
        }
        if first_port == second_port || self.ports() <= 2 {
            return Err(Error::Unsupported(
                "inner connection requires two distinct ports and at least one external port"
                    .to_owned(),
            ));
        }
        let points = self.frequency_points();
        let internal = [first_port, second_port];
        let external = (0..self.ports())
            .filter(|index| !internal.contains(index))
            .collect::<Vec<_>>();
        let mut scattering = Array3::zeros((points, external.len(), external.len()));
        let mut z0 = Array2::zeros((points, external.len()));
        for point in 0..points {
            let connection =
                connection_scattering(self.z0[(point, first_port)], self.z0[(point, second_port)])?;
            let internal_scattering = [
                self.s[(point, first_port, first_port)],
                self.s[(point, first_port, second_port)],
                self.s[(point, second_port, first_port)],
                self.s[(point, second_port, second_port)],
            ];
            let product = multiply_two_by_two(connection, internal_scattering);
            let system = [
                Complex64::new(1.0, 0.0) - product[0],
                -product[1],
                -product[2],
                Complex64::new(1.0, 0.0) - product[3],
            ];
            for (input, external_input) in external.iter().enumerate() {
                let right = [
                    connection[0] * self.s[(point, first_port, *external_input)]
                        + connection[1] * self.s[(point, second_port, *external_input)],
                    connection[2] * self.s[(point, first_port, *external_input)]
                        + connection[3] * self.s[(point, second_port, *external_input)],
                ];
                let internal_incident = solve_two_by_two(system, right)?;
                for (output, external_output) in external.iter().enumerate() {
                    scattering[(point, output, input)] = self.s
                        [(point, *external_output, *external_input)]
                        + self.s[(point, *external_output, first_port)] * internal_incident[0]
                        + self.s[(point, *external_output, second_port)] * internal_incident[1];
                }
            }
            for (output, port) in external.iter().enumerate() {
                z0[(point, output)] = self.z0[(point, *port)];
            }
        }
        let mut network = Self::new(self.frequency.clone(), scattering, z0)?;
        network.name.clone_from(&self.name);
        network.s_definition = self.s_definition;
        Ok(network)
    }

    /// Renormalizes scattering data to new port impedances and wave definition.
    ///
    /// # Errors
    ///
    /// Returns an error when `z0` has the wrong shape or conversion through
    /// impedance parameters fails.
    pub fn renormalize(
        &mut self,
        z0: Array2<Complex64>,
        definition: SParameterDefinition,
    ) -> Result<()> {
        if z0.dim() != self.z0.dim() {
            return Err(Error::IncompatibleShape(format!(
                "new reference impedance has shape {:?}, expected {:?}",
                z0.dim(),
                self.z0.dim()
            )));
        }
        let impedance = s_to_z(&self.s, &self.z0, self.s_definition)?;
        self.s = z_to_s(&impedance, &z0, definition)?;
        self.z0 = z0;
        self.s_definition = definition;
        Ok(())
    }

    /// Returns the cascading inverse of a two-port network.
    ///
    /// # Errors
    ///
    /// Returns an error unless this is a two-port, or when a chain matrix is
    /// singular or cannot be converted to or from scattering parameters.
    pub fn inverse(&self) -> Result<Self> {
        if self.ports() != 2 {
            return Err(Error::Unsupported(
                "network inversion currently requires a two-port network".to_owned(),
            ));
        }
        let points = self.frequency_points();
        let mut s = Array3::zeros((points, 2, 2));
        let mut z0 = Array2::zeros((points, 2));
        for point in 0..points {
            let chain = scattering_to_chain(&self.s, point)?;
            let determinant = chain[0] * chain[3] - chain[1] * chain[2];
            if determinant.norm_sqr() <= f64::EPSILON {
                return Err(Error::Unsupported(
                    "cannot invert a singular chain matrix".to_owned(),
                ));
            }
            let inverse = [
                chain[3] / determinant,
                -chain[1] / determinant,
                -chain[2] / determinant,
                chain[0] / determinant,
            ];
            write_chain_as_scattering(inverse, &mut s, point)?;
            z0[(point, 0)] = self.z0[(point, 1)];
            z0[(point, 1)] = self.z0[(point, 0)];
        }
        let mut network = Self::new(self.frequency.clone(), s, z0)?;
        network.name = self.name.as_ref().map(|name| format!("{name}-inverse"));
        Ok(network)
    }

    /// Port of `skrf.network.Network.flipped`.
    /// Reverses a two-port network's port orientation.
    ///
    /// # Errors
    ///
    /// Returns an error when the network has an odd number of ports.
    pub fn flipped(&self) -> Result<Self> {
        let ports = self.ports();
        let half = ports / 2;
        let mut network = self.clone();
        network.s = flip_ports(&self.s)?;
        network.z0 = Array2::from_shape_fn(self.z0.dim(), |(point, port)| {
            self.z0[(point, (port + half) % ports)]
        });
        if self.port_names.len() == ports {
            network.port_names = (0..ports)
                .map(|port| self.port_names[(port + half) % ports].clone())
                .collect();
        }
        network.name = self.name.as_ref().map(|name| format!("{name}-flipped"));
        Ok(network)
    }

    /// Return a copy with ports reordered, where each output position names
    /// the corresponding old zero-based port index.
    ///
    /// Port of `Network.renumber` used by `skrf.media.device.DualCoupler`.
    /// Returns a network whose ports follow the supplied permutation.
    ///
    /// # Errors
    ///
    /// Returns an error when `order` has the wrong length or is not a complete
    /// permutation of the network ports.
    pub fn renumbered(&self, order: &[usize]) -> Result<Self> {
        let ports = self.ports();
        if order.len() != ports {
            return Err(Error::IncompatibleShape(format!(
                "port order contains {} entries for a {ports}-port network",
                order.len()
            )));
        }
        let mut sorted = order.to_vec();
        sorted.sort_unstable();
        if sorted != (0..ports).collect::<Vec<_>>() {
            return Err(Error::Unsupported(
                "port order must be a permutation of every network port".to_owned(),
            ));
        }
        let mut network = self.clone();
        network.s = Array3::from_shape_fn(self.s.dim(), |(point, output, input)| {
            self.s[(point, order[output], order[input])]
        });
        network.z0 =
            Array2::from_shape_fn(self.z0.dim(), |(point, port)| self.z0[(point, order[port])]);
        if self.port_names.len() == ports {
            network.port_names = order
                .iter()
                .map(|port| self.port_names[*port].clone())
                .collect();
        }
        if self.port_modes.len() == ports {
            network.port_modes = order.iter().map(|port| self.port_modes[*port]).collect();
        }
        Ok(network)
    }

    /// Reads a Touchstone network file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or its Touchstone data
    /// cannot be parsed into a network.
    pub fn read_touchstone(path: impl AsRef<Path>) -> Result<Self> {
        crate::io::Touchstone::from_path(path)?.network()
    }

    /// Writes the network as a Touchstone file.
    ///
    /// # Errors
    ///
    /// Returns an error when the network cannot be represented as Touchstone
    /// data or the output file cannot be written.
    pub fn write_touchstone(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::io::write_touchstone(
            self,
            path,
            crate::io::TouchstoneParameter::Scattering,
            crate::io::TouchstoneFormat::RealImaginary,
        )
    }

    fn elementwise_binary(
        &self,
        other: &Self,
        operation: impl Fn(Complex64, Complex64) -> Complex64,
    ) -> Result<Self> {
        if self.frequency != other.frequency || self.s.dim() != other.s.dim() {
            return Err(Error::IncompatibleShape(
                "element-wise network operations require matching frequency and port shapes"
                    .to_owned(),
            ));
        }
        let mut result = self.clone();
        result.s = Array3::from_shape_fn(self.s.dim(), |index| {
            operation(self.s[index], other.s[index])
        });
        Ok(result)
    }

    /// Converts the first `2 * pairs` single-ended ports into differential and common modes.
    /// Converts paired single-ended ports to differential and common modes.
    ///
    /// # Errors
    ///
    /// Returns an error when `pairs` is zero or exceeds the available ports, or
    /// when a pair's reference impedances are unequal.
    pub fn single_ended_to_mixed_mode(&self, pairs: usize) -> Result<Self> {
        if pairs == 0 || 2 * pairs > self.ports() {
            return Err(Error::IncompatibleShape(format!(
                "{pairs} mixed-mode pairs do not fit {} ports",
                self.ports()
            )));
        }
        let ports = self.ports();
        let scale = 1.0 / 2.0_f64.sqrt();
        let mut transform = Array2::<Complex64>::zeros((ports, ports));
        for pair in 0..pairs {
            transform[(pair, pair)] = Complex64::new(scale, 0.0);
            transform[(pair, pair + pairs)] = Complex64::new(-scale, 0.0);
            transform[(pair + pairs, pair)] = Complex64::new(scale, 0.0);
            transform[(pair + pairs, pair + pairs)] = Complex64::new(scale, 0.0);
        }
        for port in 2 * pairs..ports {
            transform[(port, port)] = Complex64::new(1.0, 0.0);
        }
        let mut result = self.clone();
        result.s = similarity_transform(&self.s, &transform, false);
        for point in 0..self.frequency_points() {
            for pair in 0..pairs {
                let first = self.z0[(point, pair)];
                let second = self.z0[(point, pair + pairs)];
                if (first - second).norm() > 1.0e-9 * first.norm().max(1.0) {
                    return Err(Error::Unsupported(
                        "mixed-mode conversion currently requires equal pair impedances".to_owned(),
                    ));
                }
                result.z0[(point, pair)] = 2.0 * first;
                result.z0[(point, pair + pairs)] = first / 2.0;
            }
        }
        result.port_modes = (0..ports)
            .map(|port| {
                if port < pairs {
                    PortMode::Differential
                } else if port < 2 * pairs {
                    PortMode::Common
                } else {
                    PortMode::SingleEnded
                }
            })
            .collect();
        Ok(result)
    }

    /// Converts differential and common modes back to paired single-ended ports.
    ///
    /// # Errors
    ///
    /// Returns an error when `pairs` is zero or exceeds the available ports, or
    /// when differential and common reference impedances are incompatible.
    pub fn mixed_mode_to_single_ended(&self, pairs: usize) -> Result<Self> {
        if pairs == 0 || 2 * pairs > self.ports() {
            return Err(Error::IncompatibleShape(format!(
                "{pairs} mixed-mode pairs do not fit {} ports",
                self.ports()
            )));
        }
        let ports = self.ports();
        let scale = 1.0 / 2.0_f64.sqrt();
        let mut transform = Array2::<Complex64>::zeros((ports, ports));
        for pair in 0..pairs {
            transform[(pair, pair)] = Complex64::new(scale, 0.0);
            transform[(pair, pair + pairs)] = Complex64::new(-scale, 0.0);
            transform[(pair + pairs, pair)] = Complex64::new(scale, 0.0);
            transform[(pair + pairs, pair + pairs)] = Complex64::new(scale, 0.0);
        }
        for port in 2 * pairs..ports {
            transform[(port, port)] = Complex64::new(1.0, 0.0);
        }
        let mut result = self.clone();
        result.s = similarity_transform(&self.s, &transform, true);
        for point in 0..self.frequency_points() {
            for pair in 0..pairs {
                let differential = self.z0[(point, pair)] / 2.0;
                let common = self.z0[(point, pair + pairs)] * 2.0;
                if (differential - common).norm() > 1.0e-9 * differential.norm().max(1.0) {
                    return Err(Error::Unsupported(
                        "mixed-mode impedances are not a compatible differential/common pair"
                            .to_owned(),
                    ));
                }
                result.z0[(point, pair)] = differential;
                result.z0[(point, pair + pairs)] = differential;
            }
        }
        result.port_modes = vec![PortMode::SingleEnded; ports];
        Ok(result)
    }

    fn copy_metadata_from(&mut self, source: &Self) {
        self.name.clone_from(&source.name);
        self.comments.clone_from(&source.comments);
        self.port_names.clone_from(&source.port_names);
        self.variables.clone_from(&source.variables);
        self.s_definition = source.s_definition;
        self.noise.clone_from(&source.noise);
        self.port_modes.clone_from(&source.port_modes);
        self.propagation_constants
            .clone_from(&source.propagation_constants);
    }
}

fn similarity_transform(
    values: &Array3<Complex64>,
    transform: &Array2<Complex64>,
    inverse: bool,
) -> Array3<Complex64> {
    let (points, ports, _) = values.dim();
    Array3::from_shape_fn((points, ports, ports), |(point, row, column)| {
        let mut value = Complex64::new(0.0, 0.0);
        for inner_row in 0..ports {
            for inner_column in 0..ports {
                let left = if inverse {
                    transform[(inner_row, row)]
                } else {
                    transform[(row, inner_row)]
                };
                let right = if inverse {
                    transform[(column, inner_column)]
                } else {
                    transform[(inner_column, column)]
                };
                value += left * values[(point, inner_row, inner_column)] * right;
            }
        }
        value
    })
}

fn scattering_spectral_norm(scattering: &Array3<Complex64>, point: usize) -> f64 {
    let ports = scattering.dim().1;
    let port_count = ports.to_f64().unwrap_or(f64::INFINITY);
    let mut vector = vec![Complex64::new(1.0 / port_count.sqrt(), 0.0); ports];
    let mut singular = 0.0;
    for _ in 0..32 {
        let transformed = (0..ports)
            .map(|row| {
                (0..ports)
                    .map(|column| scattering[(point, row, column)] * vector[column])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        singular = transformed
            .iter()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt();
        let adjoint = (0..ports)
            .map(|column| {
                (0..ports)
                    .map(|row| scattering[(point, row, column)].conj() * transformed[row])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        let norm = adjoint.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        for (value, next) in vector.iter_mut().zip(adjoint) {
            *value = next / norm;
        }
    }
    singular
}

/// Cascades a non-empty sequence from left to right.
///
/// Origin: `skrf.network.cascade_list`.
/// Cascades a non-empty sequence of networks from left to right.
///
/// # Errors
///
/// Returns an error when `networks` is empty or adjacent networks cannot be
/// cascaded as compatible two-ports.
pub fn cascade_list(networks: &[Network]) -> Result<Network> {
    let (first, remaining) = networks.split_first().ok_or_else(|| {
        Error::IncompatibleShape("cascade requires at least one network".to_owned())
    })?;
    remaining
        .iter()
        .try_fold(first.clone(), |combined, network| combined.cascade(network))
}

/// Connects one selected port from each network at a common intersection.
///
/// Origin: `skrf.network.parallelconnect`.
/// Connects selected ports of multiple networks in parallel at one junction.
///
/// # Errors
///
/// Returns an error for mismatched input lists, incompatible frequencies or
/// invalid ports, or when the equivalent circuit cannot be constructed.
pub fn parallel_connect(networks: &[Network], selected_ports: &[usize]) -> Result<Network> {
    if networks.len() < 2 || networks.len() != selected_ports.len() {
        return Err(Error::IncompatibleShape(
            "parallel connection requires matching lists of at least two networks and ports"
                .to_owned(),
        ));
    }
    let frequency = networks[0].frequency.clone();
    if networks
        .iter()
        .zip(selected_ports)
        .any(|(network, port)| network.frequency != frequency || *port >= network.ports())
    {
        return Err(Error::IncompatibleShape(
            "parallel-connected networks must share frequency and use valid ports".to_owned(),
        ));
    }
    let mut components = Vec::with_capacity(networks.len());
    for (index, network) in networks.iter().enumerate() {
        let mut component = network.clone();
        component.name = Some(format!("parallel_component_{index}"));
        components.push(Arc::new(component));
    }
    let mut connections = vec![
        components
            .iter()
            .zip(selected_ports)
            .map(|(network, port)| crate::circuit::CircuitConnection::new(network.clone(), *port))
            .collect::<Vec<_>>(),
    ];
    for (component_index, component) in components.iter().enumerate() {
        for port in 0..component.ports() {
            if port == selected_ports[component_index] {
                continue;
            }
            let mut external = Network::new(
                frequency.clone(),
                Array3::zeros((frequency.points(), 1, 1)),
                Array2::from_shape_fn((frequency.points(), 1), |(point, _)| {
                    component.z0[(point, port)]
                }),
            )?;
            external.name = Some(format!("parallel_external_{component_index}_{port}"));
            connections.push(vec![
                crate::circuit::CircuitConnection::new(component.clone(), port),
                crate::circuit::CircuitConnection::external(Arc::new(external), 0),
            ]);
        }
    }
    crate::circuit::Circuit::new(connections)?.network()
}

/// Interpolates two networks onto their common frequency samples.
///
/// Origin: `skrf.network.overlap`.
/// Crops two networks to their common frequency interval.
///
/// # Errors
///
/// Returns an error when the frequency axes do not overlap or either network
/// cannot be interpolated onto the common samples.
pub fn overlap(first: &Network, second: &Network) -> Result<(Network, Network)> {
    let frequency = first.frequency.overlap(&second.frequency)?;
    Ok((
        first.interpolate(&frequency)?,
        second.interpolate(&frequency)?,
    ))
}

/// Joins ordered, non-overlapping frequency ranges.
///
/// Origin: `skrf.network.stitch`.
/// Concatenates two compatible networks along the frequency axis.
///
/// # Errors
///
/// Returns an error when port counts or wave definitions differ, frequency
/// ranges overlap or are reversed, or the combined network is invalid.
pub fn stitch(first: &Network, second: &Network) -> Result<Network> {
    if first.ports() != second.ports() || first.s_definition != second.s_definition {
        return Err(Error::IncompatibleShape(
            "stitched networks must share ports and wave definition".to_owned(),
        ));
    }
    if first.frequency.values_hz().last() >= second.frequency.values_hz().first() {
        return Err(Error::InvalidFrequency(
            "stitched frequency ranges must be ordered and non-overlapping".to_owned(),
        ));
    }
    let first_points = first.frequency_points();
    let points = first_points + second.frequency_points();
    let ports = first.ports();
    let frequency = Frequency::from_hz(Array1::from_iter(
        first
            .frequency
            .values_hz()
            .iter()
            .chain(second.frequency.values_hz())
            .copied(),
    ))?;
    let s = Array3::from_shape_fn((points, ports, ports), |(point, output, input)| {
        if point < first_points {
            first.s[(point, output, input)]
        } else {
            second.s[(point - first_points, output, input)]
        }
    });
    let z0 = Array2::from_shape_fn((points, ports), |(point, port)| {
        if point < first_points {
            first.z0[(point, port)]
        } else {
            second.z0[(point - first_points, port)]
        }
    });
    let mut result = Network::new(frequency, s, z0)?;
    result.name.clone_from(&first.name);
    result.s_definition = first.s_definition;
    Ok(result)
}

/// Builds a block-diagonal N-port from frequency-compatible networks.
///
/// Origin: `skrf.network.concat_ports`.
/// Combines aligned networks into one block-diagonal multiport network.
///
/// # Errors
///
/// Returns an error when `networks` is empty, frequency axes differ, or the
/// combined network dimensions are invalid.
pub fn concatenate_ports(networks: &[Network]) -> Result<Network> {
    let first = networks.first().ok_or_else(|| {
        Error::IncompatibleShape("port concatenation requires networks".to_owned())
    })?;
    if networks
        .iter()
        .any(|network| network.frequency != first.frequency)
    {
        return Err(Error::InvalidFrequency(
            "port-concatenated networks must share frequency".to_owned(),
        ));
    }
    let ports = networks.iter().map(Network::ports).sum();
    let mut s = Array3::zeros((first.frequency_points(), ports, ports));
    let mut z0 = Array2::zeros((first.frequency_points(), ports));
    let mut offset = 0;
    for network in networks {
        for point in 0..first.frequency_points() {
            for output in 0..network.ports() {
                z0[(point, offset + output)] = network.z0[(point, output)];
                for input in 0..network.ports() {
                    s[(point, offset + output, offset + input)] = network.s[(point, output, input)];
                }
            }
        }
        offset += network.ports();
    }
    Network::new(first.frequency.clone(), s, z0)
}

/// Element-wise complex average of compatible networks.
///
/// Origin: `skrf.network.average`.
/// Returns the elementwise complex average of aligned networks.
///
/// # Errors
///
/// Returns an error when `networks` is empty or their frequency axes, port
/// counts, or reference impedances differ.
pub fn average(networks: &[Network]) -> Result<Network> {
    let first = networks.first().ok_or_else(|| {
        Error::IncompatibleShape("average requires at least one network".to_owned())
    })?;
    if networks.iter().any(|network| {
        network.frequency != first.frequency
            || network.ports() != first.ports()
            || network.z0 != first.z0
    }) {
        return Err(Error::IncompatibleShape(
            "averaged networks must share frequency, ports, and impedance".to_owned(),
        ));
    }
    let mut result = first.clone();
    result.s.fill(Complex64::new(0.0, 0.0));
    for network in networks {
        result.s += &network.s;
    }
    let network_count = networks.len().to_f64().ok_or_else(|| {
        Error::Unsupported("network count exceeds the supported averaging range".to_owned())
    })?;
    result.s.mapv_inplace(|value| value / network_count);
    Ok(result)
}

/// Population standard deviation of complex S-parameter distance from the mean.
///
/// Origin: `skrf.network.stdev`.
/// Returns sample standard deviation of scattering values across aligned networks.
///
/// # Errors
///
/// Returns an error when `networks` is empty or their frequency axes, port
/// counts, or reference impedances differ.
pub fn scattering_standard_deviation(networks: &[Network]) -> Result<Array3<f64>> {
    let mean = average(networks)?;
    let network_count = networks.len().to_f64().ok_or_else(|| {
        Error::Unsupported("network count exceeds the supported deviation range".to_owned())
    })?;
    Ok(Array3::from_shape_fn(mean.s.dim(), |index| {
        (networks
            .iter()
            .map(|network| (network.s[index] - mean.s[index]).norm_sqr())
            .sum::<f64>()
            / network_count)
            .sqrt()
    }))
}

/// Places two one-port reflections on the diagonal of a two-port network.
///
/// Origin: `skrf.network.two_port_reflect`.
/// Builds a two-port reflect standard from one or two one-port networks.
///
/// # Errors
///
/// Returns an error unless both inputs are frequency-compatible one-ports, or
/// when the resulting two-port dimensions are invalid.
pub fn two_port_reflect(first: &Network, second: Option<&Network>) -> Result<Network> {
    let second = second.unwrap_or(first);
    if first.ports() != 1 || second.ports() != 1 || first.frequency != second.frequency {
        return Err(Error::IncompatibleShape(
            "two-port reflect requires frequency-compatible one-port networks".to_owned(),
        ));
    }
    let points = first.frequency_points();
    let mut s = Array3::zeros((points, 2, 2));
    let mut z0 = Array2::zeros((points, 2));
    for point in 0..points {
        s[(point, 0, 0)] = first.s[(point, 0, 0)];
        s[(point, 1, 1)] = second.s[(point, 0, 0)];
        z0[(point, 0)] = first.z0[(point, 0)];
        z0[(point, 1)] = second.z0[(point, 0)];
    }
    Network::new(first.frequency.clone(), s, z0)
}

/// Embeds a two-port into selected ports of an otherwise zero N-port.
///
/// Origin: `skrf.network.twoport_to_nport`.
/// Maps two-port measurements into specified entries of an N-port network.
///
/// # Errors
///
/// Returns an error unless `network` is a two-port and the distinct target
/// ports fit the requested N-port, or when the output dimensions are invalid.
pub fn two_port_to_nport(
    network: &Network,
    first_port: usize,
    second_port: usize,
    ports: usize,
) -> Result<Network> {
    if network.ports() != 2
        || ports < 2
        || first_port >= ports
        || second_port >= ports
        || first_port == second_port
    {
        return Err(Error::IncompatibleShape(
            "two-port embedding requires distinct valid target ports".to_owned(),
        ));
    }
    let mut s = Array3::zeros((network.frequency_points(), ports, ports));
    let default_z0 = network.z0[(0, 0)];
    let mut z0 = Array2::from_elem((network.frequency_points(), ports), default_z0);
    let mapped = [first_port, second_port];
    for point in 0..network.frequency_points() {
        for output in 0..2 {
            z0[(point, mapped[output])] = network.z0[(point, output)];
            for input in 0..2 {
                s[(point, mapped[output], mapped[input])] = network.s[(point, output, input)];
            }
        }
    }
    Network::new(network.frequency.clone(), s, z0)
}

/// Assembles the diagonal of an N-port from one-port networks.
///
/// Origin: `skrf.network.n_oneports_2_nport`.
/// Places one-port networks on the diagonal of an N-port network.
///
/// # Errors
///
/// Returns an error when `networks` is empty, an input is not a
/// frequency-compatible one-port, or the output dimensions are invalid.
pub fn one_ports_to_nport(networks: &[Network]) -> Result<Network> {
    let first = networks.first().ok_or_else(|| {
        Error::IncompatibleShape("N-port assembly requires one-port networks".to_owned())
    })?;
    if networks
        .iter()
        .any(|network| network.ports() != 1 || network.frequency != first.frequency)
    {
        return Err(Error::IncompatibleShape(
            "N-port assembly requires frequency-compatible one-port networks".to_owned(),
        ));
    }
    let ports = networks.len();
    let mut s = Array3::zeros((first.frequency_points(), ports, ports));
    let mut z0 = Array2::zeros((first.frequency_points(), ports));
    for (port, network) in networks.iter().enumerate() {
        for point in 0..first.frequency_points() {
            s[(point, port, port)] = network.s[(point, 0, 0)];
            z0[(point, port)] = network.z0[(point, 0)];
        }
    }
    Network::new(first.frequency.clone(), s, z0)
}

/// Reconstructs an N-port from explicitly indexed two-port measurements.
///
/// This typed mapping replaces upstream filename parsing in
/// `skrf.network.n_twoports_2_nport`.
/// Assembles an N-port network from indexed two-port measurements.
///
/// # Errors
///
/// Returns an error when measurements are absent or have incompatible port
/// mappings, frequencies, or shapes, or when the output dimensions are invalid.
pub fn two_port_measurements_to_nport(
    measurements: &[(usize, usize, Network)],
    ports: usize,
) -> Result<Network> {
    let (_, _, first) = measurements.first().ok_or_else(|| {
        Error::IncompatibleShape("N-port reconstruction requires measurements".to_owned())
    })?;
    if ports < 2
        || measurements.iter().any(|(left, right, network)| {
            *left >= ports
                || *right >= ports
                || left == right
                || network.ports() != 2
                || network.frequency != first.frequency
        })
    {
        return Err(Error::IncompatibleShape(
            "N-port reconstruction has incompatible measurement mapping".to_owned(),
        ));
    }
    let mut s = Array3::zeros((first.frequency_points(), ports, ports));
    let mut z0 = Array2::from_elem((first.frequency_points(), ports), first.z0[(0, 0)]);
    for (left, right, network) in measurements {
        let mapped = [*left, *right];
        for point in 0..first.frequency_points() {
            for output in 0..2 {
                z0[(point, mapped[output])] = network.z0[(point, output)];
                for input in 0..2 {
                    s[(point, mapped[output], mapped[input])] = network.s[(point, output, input)];
                }
            }
        }
    }
    Network::new(first.frequency.clone(), s, z0)
}

/// Port of `skrf.network.s2z`.
/// Converts scattering matrices to impedance matrices.
///
/// # Errors
///
/// Returns an error for incompatible parameter or reference-impedance shapes,
/// invalid reference impedances, or a failed linear solve.
pub fn s_to_z(
    scattering: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
    definition: SParameterDefinition,
) -> Result<Array3<Complex64>> {
    let (frequencies, ports, input_ports) = scattering.dim();
    validate_parameter_shapes(scattering, reference_impedance)?;
    debug_assert_eq!(ports, input_ports);
    let z0 = adjusted_reference_impedance(reference_impedance);
    let mut left = Array3::zeros((frequencies, ports, ports));
    let mut right = Array3::zeros((frequencies, ports, ports));

    for frequency in 0..frequencies {
        for row in 0..ports {
            for column in 0..ports {
                let identity = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let s = scattering[(frequency, row, column)];
                match definition {
                    SParameterDefinition::Power => {
                        let column_scale = 1.0 / (2.0 * z0[(frequency, column)].re.sqrt());
                        left[(frequency, row, column)] = (identity - s) * column_scale;
                        right[(frequency, row, column)] = (s * z0[(frequency, column)]
                            + if row == column {
                                z0[(frequency, row)].conj()
                            } else {
                                Complex64::new(0.0, 0.0)
                            })
                            * column_scale;
                    }
                    SParameterDefinition::Pseudo => {
                        let row_scale =
                            z0[(frequency, row)].re.sqrt() / z0[(frequency, row)].norm();
                        let column_scale =
                            z0[(frequency, column)].re.sqrt() / z0[(frequency, column)].norm();
                        let transformed = s * column_scale / row_scale;
                        left[(frequency, row, column)] = identity - transformed;
                        right[(frequency, row, column)] =
                            (identity + transformed) * z0[(frequency, column)];
                    }
                    SParameterDefinition::Traveling => {
                        left[(frequency, row, column)] = identity - s;
                        right[(frequency, row, column)] =
                            (identity + s) * z0[(frequency, column)].sqrt();
                    }
                }
            }
        }
    }

    let mut impedance = left_solve(&left, &right)?;
    if definition == SParameterDefinition::Traveling {
        for frequency in 0..frequencies {
            for row in 0..ports {
                let scale = z0[(frequency, row)].sqrt();
                for column in 0..ports {
                    impedance[(frequency, row, column)] *= scale;
                }
            }
        }
    }
    Ok(impedance)
}

/// Port of `skrf.network.z2s`.
/// Converts impedance matrices to scattering matrices.
///
/// # Errors
///
/// Returns an error for incompatible parameter or reference-impedance shapes,
/// invalid reference impedances, or a failed linear solve.
pub fn z_to_s(
    impedance: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
    definition: SParameterDefinition,
) -> Result<Array3<Complex64>> {
    let (frequencies, ports, _) = impedance.dim();
    validate_parameter_shapes(impedance, reference_impedance)?;
    let z0 = adjusted_reference_impedance(reference_impedance);
    let mut left = Array3::zeros((frequencies, ports, ports));
    let mut right = Array3::zeros((frequencies, ports, ports));

    for frequency in 0..frequencies {
        for row in 0..ports {
            for column in 0..ports {
                let identity = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let z = impedance[(frequency, row, column)];
                match definition {
                    SParameterDefinition::Power => {
                        let scale = 1.0 / (2.0 * z0[(frequency, row)].re.sqrt());
                        left[(frequency, row, column)] = (z + if row == column {
                            z0[(frequency, row)]
                        } else {
                            Complex64::new(0.0, 0.0)
                        }) * scale;
                        right[(frequency, row, column)] = (z - if row == column {
                            z0[(frequency, row)].conj()
                        } else {
                            Complex64::new(0.0, 0.0)
                        }) * scale;
                    }
                    SParameterDefinition::Pseudo => {
                        let scale = z0[(frequency, row)].re.sqrt() / z0[(frequency, row)].norm();
                        left[(frequency, row, column)] = (z + if row == column {
                            z0[(frequency, row)]
                        } else {
                            Complex64::new(0.0, 0.0)
                        }) * scale;
                        right[(frequency, row, column)] = (z - if row == column {
                            z0[(frequency, row)]
                        } else {
                            Complex64::new(0.0, 0.0)
                        }) * scale;
                    }
                    SParameterDefinition::Traveling => {
                        let normalized = Complex64::new(1.0, 0.0) / z0[(frequency, row)].sqrt() * z
                            / z0[(frequency, column)].sqrt();
                        left[(frequency, row, column)] = normalized + identity;
                        right[(frequency, row, column)] = normalized - identity;
                    }
                }
            }
        }
    }
    right_solve(&left, &right)
}

/// Port of `skrf.network.z2y`.
/// Converts impedance matrices to admittance matrices by inversion.
///
/// # Errors
///
/// Returns an error when an impedance matrix is not square or cannot be inverted.
pub fn z_to_y(impedance: &Array3<Complex64>) -> Result<Array3<Complex64>> {
    invert_parameter_matrices(impedance)
}

/// Port of `skrf.network.y2z`.
/// Converts admittance matrices to impedance matrices by inversion.
///
/// # Errors
///
/// Returns an error when an admittance matrix is not square or cannot be inverted.
pub fn y_to_z(admittance: &Array3<Complex64>) -> Result<Array3<Complex64>> {
    invert_parameter_matrices(admittance)
}

/// Port of `skrf.network.s2y`.
/// Converts scattering matrices to admittance matrices.
///
/// # Errors
///
/// Returns an error for incompatible parameter or reference-impedance shapes, invalid reference
/// impedances, or a failed matrix inversion or linear solve.
pub fn s_to_y(
    scattering: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
    definition: SParameterDefinition,
) -> Result<Array3<Complex64>> {
    z_to_y(&s_to_z(scattering, reference_impedance, definition)?)
}

/// Port of `skrf.network.y2s`.
/// Converts admittance matrices to scattering matrices.
///
/// # Errors
///
/// Returns an error for incompatible parameter or reference-impedance shapes, invalid reference
/// impedances, or a failed matrix inversion or linear solve.
pub fn y_to_s(
    admittance: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
    definition: SParameterDefinition,
) -> Result<Array3<Complex64>> {
    z_to_s(&y_to_z(admittance)?, reference_impedance, definition)
}

/// Port of `skrf.network.z2h` for two-port hybrid parameters.
/// Converts two-port impedance matrices to hybrid $H$ parameters.
///
/// # Errors
///
/// Returns an error unless every impedance matrix is 2-by-2 and has a nonzero `Z22` element.
pub fn z_to_h(impedance: &Array3<Complex64>) -> Result<Array3<Complex64>> {
    let (frequencies, rows, columns) = impedance.dim();
    if rows != 2 || columns != 2 {
        return Err(Error::IncompatibleShape(format!(
            "hybrid conversion requires 2x2 matrices, got {rows}x{columns}"
        )));
    }
    let mut hybrid = Array3::zeros((frequencies, 2, 2));
    for point in 0..frequencies {
        let z22 = impedance[(point, 1, 1)];
        if z22.norm_sqr() <= f64::EPSILON {
            return Err(Error::Unsupported(format!(
                "impedance matrix at frequency index {point} has zero Z22"
            )));
        }
        hybrid[(point, 0, 0)] = (impedance[(point, 0, 0)] * z22
            - impedance[(point, 1, 0)] * impedance[(point, 0, 1)])
            / z22;
        hybrid[(point, 0, 1)] = impedance[(point, 0, 1)] / z22;
        hybrid[(point, 1, 0)] = -impedance[(point, 1, 0)] / z22;
        hybrid[(point, 1, 1)] = Complex64::new(1.0, 0.0) / z22;
    }
    Ok(hybrid)
}

/// Port of `skrf.network.h2z` for two-port hybrid parameters.
/// Converts two-port hybrid $H$ parameters to impedance matrices.
///
/// # Errors
///
/// Returns an error unless every hybrid matrix is 2-by-2 and has a nonzero `H22` element.
pub fn h_to_z(hybrid: &Array3<Complex64>) -> Result<Array3<Complex64>> {
    z_to_h(hybrid)
}

/// Port of `skrf.network.h2s`.
/// Converts hybrid $H$ parameters to scattering matrices.
///
/// # Errors
///
/// Returns an error for invalid two-port hybrid matrices, incompatible reference-impedance
/// shapes, invalid reference impedances, or a failed linear solve.
pub fn h_to_s(
    hybrid: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
    definition: SParameterDefinition,
) -> Result<Array3<Complex64>> {
    z_to_s(&h_to_z(hybrid)?, reference_impedance, definition)
}

/// Port of `skrf.network.s2h`.
/// Converts scattering matrices to hybrid $H$ parameters.
///
/// # Errors
///
/// Returns an error for invalid scattering or reference-impedance shapes, invalid reference
/// impedances, a failed linear solve, or a converted impedance matrix with zero `Z22`.
pub fn s_to_h(
    scattering: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
    definition: SParameterDefinition,
) -> Result<Array3<Complex64>> {
    z_to_h(&s_to_z(scattering, reference_impedance, definition)?)
}

/// Port of `skrf.network.g2s`.
/// Converts inverse-hybrid $G$ parameters to scattering matrices.
///
/// # Errors
///
/// Returns an error for non-square or singular inverse-hybrid matrices, invalid two-port hybrid
/// matrices, incompatible reference-impedance shapes, invalid reference impedances, or a failed
/// linear solve.
pub fn g_to_s(
    inverse_hybrid: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
    definition: SParameterDefinition,
) -> Result<Array3<Complex64>> {
    h_to_s(
        &invert_parameter_matrices(inverse_hybrid)?,
        reference_impedance,
        definition,
    )
}

/// Port of `skrf.network.s2g`.
/// Converts scattering matrices to inverse-hybrid $G$ parameters.
///
/// # Errors
///
/// Returns an error for invalid scattering or reference-impedance shapes, invalid reference
/// impedances, a failed matrix inversion or linear solve, or a converted admittance matrix with
/// zero `Y22`.
pub fn s_to_g(
    scattering: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
    definition: SParameterDefinition,
) -> Result<Array3<Complex64>> {
    z_to_h(&s_to_y(scattering, reference_impedance, definition)?)
}

/// Port of `skrf.network.passivity`.
/// Returns the passivity metric $\sqrt{S^\dagger S}$ at each frequency.
///
/// # Errors
///
/// Returns an error unless the scattering data contains square matrices with at least two ports.
pub fn passivity(scattering: &Array3<Complex64>) -> Result<Array3<Complex64>> {
    let (frequencies, rows, columns) = scattering.dim();
    if rows <= 1 || rows != columns {
        return Err(Error::IncompatibleShape(format!(
            "passivity requires square multi-port matrices, got {rows}x{columns}"
        )));
    }
    let mut metric = Array3::zeros((frequencies, rows, columns));
    for point in 0..frequencies {
        for row in 0..rows {
            for column in 0..columns {
                let mut gram = Complex64::new(0.0, 0.0);
                for port in 0..rows {
                    gram +=
                        scattering[(point, port, row)].conj() * scattering[(point, port, column)];
                }
                metric[(point, row, column)] = gram.sqrt();
            }
        }
    }
    Ok(metric)
}

/// Port of `skrf.network.reciprocity`.
/// Returns elementwise reciprocity error $|S-S^T|$.
///
/// # Errors
///
/// Returns an error unless the scattering data contains square matrices with at least two ports.
pub fn reciprocity(scattering: &Array3<Complex64>) -> Result<Array3<f64>> {
    let (frequencies, rows, columns) = scattering.dim();
    if rows <= 1 || rows != columns {
        return Err(Error::IncompatibleShape(format!(
            "reciprocity requires square multi-port matrices, got {rows}x{columns}"
        )));
    }
    Ok(Array3::from_shape_fn(
        (frequencies, rows, columns),
        |(point, row, column)| {
            (scattering[(point, row, column)] - scattering[(point, column, row)]).norm()
        },
    ))
}

/// Port of `skrf.network.flip` for batched even-port scattering matrices.
/// Reverses the port order of scattering matrices.
///
/// # Errors
///
/// Returns an error unless the scattering data contains nonempty, square matrices with an even
/// number of ports.
pub fn flip_ports(scattering: &Array3<Complex64>) -> Result<Array3<Complex64>> {
    let (frequencies, rows, columns) = scattering.dim();
    if rows == 0 || rows != columns || rows % 2 != 0 {
        return Err(Error::IncompatibleShape(format!(
            "port flipping requires 2n-by-2n matrices, got {rows}x{columns}"
        )));
    }
    let half = rows / 2;
    Ok(Array3::from_shape_fn(
        (frequencies, rows, columns),
        |(point, row, column)| scattering[(point, (row + half) % rows, (column + half) % columns)],
    ))
}

/// Port of `skrf.network.s2s_active`.
/// Returns active scattering parameters for a specified port-excitation vector.
///
/// # Errors
///
/// Returns an error unless the scattering data contains nonempty square matrices and the
/// excitation vector contains one value per port.
pub fn active_s(
    scattering: &Array3<Complex64>,
    excitation: &Array1<Complex64>,
) -> Result<Array2<Complex64>> {
    let (frequencies, rows, columns) = scattering.dim();
    if rows == 0 || rows != columns || excitation.len() != rows {
        return Err(Error::IncompatibleShape(format!(
            "active parameters received scattering shape {:?} and excitation length {}",
            scattering.dim(),
            excitation.len()
        )));
    }
    let epsilon = Complex64::new(1.0e-12, 0.0);
    Ok(Array2::from_shape_fn(
        (frequencies, rows),
        |(point, output)| {
            let denominator = if excitation[output].norm_sqr() == 0.0 {
                epsilon
            } else {
                excitation[output]
            };
            let mut outgoing = Complex64::new(0.0, 0.0);
            for input in 0..columns {
                let incident = if excitation[input].norm_sqr() == 0.0 {
                    epsilon
                } else {
                    excitation[input]
                };
                outgoing += scattering[(point, output, input)] * incident;
            }
            outgoing / denominator
        },
    ))
}

/// Port of `skrf.network.s2z_active`.
/// Returns active port impedances for a specified excitation vector.
///
/// # Errors
///
/// Returns an error when the scattering and reference-impedance shapes are incompatible, or when
/// the scattering matrices and excitation vector do not describe the same nonzero port count.
pub fn active_z(
    scattering: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
    excitation: &Array1<Complex64>,
) -> Result<Array2<Complex64>> {
    validate_parameter_shapes(scattering, reference_impedance)?;
    let active = active_s(scattering, excitation)?;
    let one = Complex64::new(1.0, 0.0);
    Ok(Array2::from_shape_fn(active.dim(), |(point, port)| {
        reference_impedance[(point, port)] * (one + active[(point, port)])
            / (one - active[(point, port)])
    }))
}

/// Port of `skrf.network.s2y_active`.
/// Returns active port admittances for a specified excitation vector.
///
/// # Errors
///
/// Returns an error when the scattering and reference-impedance shapes are incompatible, or when
/// the scattering matrices and excitation vector do not describe the same nonzero port count.
pub fn active_y(
    scattering: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
    excitation: &Array1<Complex64>,
) -> Result<Array2<Complex64>> {
    validate_parameter_shapes(scattering, reference_impedance)?;
    let active = active_s(scattering, excitation)?;
    let one = Complex64::new(1.0, 0.0);
    Ok(Array2::from_shape_fn(active.dim(), |(point, port)| {
        (one - active[(point, port)])
            / (reference_impedance[(point, port)] * (one + active[(point, port)]))
    }))
}

/// Port of `skrf.network.s2vswr_active`.
/// Returns active VSWR for a specified excitation vector.
///
/// # Errors
///
/// Returns an error unless the scattering data contains nonempty square matrices and the
/// excitation vector contains one value per port.
pub fn active_vswr(
    scattering: &Array3<Complex64>,
    excitation: &Array1<Complex64>,
) -> Result<Array2<f64>> {
    let active = active_s(scattering, excitation)?;
    Ok(active.mapv(|value| (1.0 + value.norm()) / (1.0 - value.norm())))
}

/// Port of `skrf.network.s2t` for two-port scattering transfer parameters.
/// Converts even-port scattering matrices to scattering-transfer matrices.
///
/// # Errors
///
/// Returns an error unless every scattering matrix is 2-by-2 and has nonzero forward
/// transmission.
pub fn s_to_t(scattering: &Array3<Complex64>) -> Result<Array3<Complex64>> {
    let (points, rows, columns) = scattering.dim();
    if rows != 2 || columns != 2 {
        return Err(Error::IncompatibleShape(format!(
            "scattering transfer conversion requires 2x2 matrices, got {:?}",
            scattering.dim()
        )));
    }
    let mut transfer = Array3::zeros((points, 2, 2));
    for point in 0..points {
        let chain = scattering_to_chain(scattering, point)?;
        transfer[(point, 0, 0)] = chain[3];
        transfer[(point, 0, 1)] = chain[2];
        transfer[(point, 1, 0)] = chain[1];
        transfer[(point, 1, 1)] = chain[0];
    }
    Ok(transfer)
}

/// Port of `skrf.network.t2s` for two-port scattering transfer parameters.
/// Converts scattering-transfer matrices to scattering matrices.
///
/// # Errors
///
/// Returns an error unless every transfer matrix is 2-by-2 and has a nonzero leading chain
/// element.
pub fn t_to_s(transfer: &Array3<Complex64>) -> Result<Array3<Complex64>> {
    let (points, rows, columns) = transfer.dim();
    if rows != 2 || columns != 2 {
        return Err(Error::IncompatibleShape(format!(
            "scattering conversion requires 2x2 transfer matrices, got {:?}",
            transfer.dim()
        )));
    }
    let mut scattering = Array3::zeros((points, 2, 2));
    for point in 0..points {
        write_chain_as_scattering(
            [
                transfer[(point, 1, 1)],
                transfer[(point, 1, 0)],
                transfer[(point, 0, 1)],
                transfer[(point, 0, 0)],
            ],
            &mut scattering,
            point,
        )?;
    }
    Ok(scattering)
}

/// Port of `skrf.network.s2a` for equal, real two-port references.
/// Converts two-port scattering matrices to ABCD matrices.
///
/// # Errors
///
/// Returns an error unless the scattering matrices and references describe equal, positive-real
/// two-port impedances and every matrix has nonzero forward transmission.
pub fn s_to_abcd(
    scattering: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
) -> Result<Array3<Complex64>> {
    validate_equal_real_two_port_reference(scattering, reference_impedance)?;
    let points = scattering.dim().0;
    let mut abcd = Array3::zeros((points, 2, 2));
    for point in 0..points {
        let s11 = scattering[(point, 0, 0)];
        let s12 = scattering[(point, 0, 1)];
        let s21 = scattering[(point, 1, 0)];
        let s22 = scattering[(point, 1, 1)];
        if s21.norm_sqr() <= f64::EPSILON {
            return Err(Error::Unsupported(
                "ABCD conversion requires non-zero forward transmission".to_owned(),
            ));
        }
        let z0 = reference_impedance[(point, 0)].re;
        let two_s21 = 2.0 * s21;
        let one = Complex64::new(1.0, 0.0);
        abcd[(point, 0, 0)] = ((one + s11) * (one - s22) + s12 * s21) / two_s21;
        abcd[(point, 0, 1)] = z0 * ((one + s11) * (one + s22) - s12 * s21) / two_s21;
        abcd[(point, 1, 0)] = ((one - s11) * (one - s22) - s12 * s21) / (z0 * two_s21);
        abcd[(point, 1, 1)] = ((one - s11) * (one + s22) + s12 * s21) / two_s21;
    }
    Ok(abcd)
}

/// Port of `skrf.network.a2s` for equal, real two-port references.
/// Converts two-port ABCD matrices to scattering matrices.
///
/// # Errors
///
/// Returns an error unless the ABCD matrices and references describe equal, positive-real
/// two-port impedances and the conversion denominator is nonzero at every frequency.
pub fn abcd_to_s(
    abcd: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
) -> Result<Array3<Complex64>> {
    validate_equal_real_two_port_reference(abcd, reference_impedance)?;
    let points = abcd.dim().0;
    let mut scattering = Array3::zeros((points, 2, 2));
    for point in 0..points {
        let a = abcd[(point, 0, 0)];
        let b = abcd[(point, 0, 1)];
        let c = abcd[(point, 1, 0)];
        let d = abcd[(point, 1, 1)];
        let z0 = reference_impedance[(point, 0)].re;
        let denominator = a + b / z0 + c * z0 + d;
        if denominator.norm_sqr() <= f64::EPSILON {
            return Err(Error::Unsupported(
                "ABCD-to-scattering conversion has a zero denominator".to_owned(),
            ));
        }
        scattering[(point, 0, 0)] = (a + b / z0 - c * z0 - d) / denominator;
        scattering[(point, 1, 0)] = 2.0 / denominator;
        scattering[(point, 0, 1)] = 2.0 * (a * d - b * c) / denominator;
        scattering[(point, 1, 1)] = (-a + b / z0 - c * z0 + d) / denominator;
    }
    Ok(scattering)
}

fn validate_parameter_shapes(
    parameters: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
) -> Result<()> {
    let (frequencies, rows, columns) = parameters.dim();
    if rows == 0 || rows != columns || reference_impedance.dim() != (frequencies, rows) {
        return Err(Error::IncompatibleShape(format!(
            "parameter matrices {:?} require reference impedance shape ({frequencies}, {rows}), got {:?}",
            parameters.dim(),
            reference_impedance.dim()
        )));
    }
    Ok(())
}

fn validate_equal_real_two_port_reference(
    parameters: &Array3<Complex64>,
    reference_impedance: &Array2<Complex64>,
) -> Result<()> {
    if parameters.dim().1 != 2
        || parameters.dim().2 != 2
        || reference_impedance.dim() != (parameters.dim().0, 2)
    {
        return Err(Error::IncompatibleShape(format!(
            "two-port conversion received parameter shape {:?} and reference shape {:?}",
            parameters.dim(),
            reference_impedance.dim()
        )));
    }
    for point in 0..parameters.dim().0 {
        let left = reference_impedance[(point, 0)];
        let right = reference_impedance[(point, 1)];
        if left.im != 0.0 || right.im != 0.0 || left.re <= 0.0 || left != right {
            return Err(Error::Unsupported(
                "ABCD conversion currently requires equal positive real port references".to_owned(),
            ));
        }
    }
    Ok(())
}

fn adjusted_reference_impedance(reference_impedance: &Array2<Complex64>) -> Array2<Complex64> {
    reference_impedance.mapv(|mut value| {
        if value.re == 0.0 {
            value.re += ZERO;
        }
        value
    })
}

fn invert_parameter_matrices(parameters: &Array3<Complex64>) -> Result<Array3<Complex64>> {
    let (frequencies, rows, columns) = parameters.dim();
    if rows == 0 || rows != columns {
        return Err(Error::IncompatibleShape(format!(
            "matrix inversion requires square matrices, got {:?}",
            parameters.dim()
        )));
    }
    let mut identity = Array3::zeros((frequencies, rows, columns));
    for frequency in 0..frequencies {
        for port in 0..rows {
            identity[(frequency, port, port)] = Complex64::new(1.0, 0.0);
        }
    }
    left_solve(parameters, &identity)
}

fn interpolate_complex(left: Complex64, right: Complex64, fraction: f64) -> Complex64 {
    left + (right - left) * fraction
}

fn linear_sample(x: &Array1<f64>, y: &Array1<f64>, target: f64) -> f64 {
    let upper = x
        .iter()
        .position(|source| *source >= target)
        .unwrap_or_else(|| x.len().saturating_sub(1));
    if upper == 0 || x[upper].total_cmp(&target).is_eq() {
        return y[upper];
    }
    let lower = upper - 1;
    y[lower] + (y[upper] - y[lower]) * (target - x[lower]) / (x[upper] - x[lower])
}

fn cubic_sample(x: &Array1<f64>, y: &Array1<Complex64>, target: f64) -> Complex64 {
    if x.len() < 4 {
        let real = linear_sample(x, &y.mapv(|value| value.re), target);
        let imaginary = linear_sample(x, &y.mapv(|value| value.im), target);
        return Complex64::new(real, imaginary);
    }
    let upper = x
        .iter()
        .position(|source| *source >= target)
        .unwrap_or_else(|| x.len().saturating_sub(1));
    let start = upper.saturating_sub(2).min(x.len() - 4);
    (start..start + 4)
        .map(|sample| {
            let weight = (start..start + 4)
                .filter(|other| *other != sample)
                .map(|other| (target - x[other]) / (x[sample] - x[other]))
                .product::<f64>();
            y[sample] * weight
        })
        .sum()
}

fn validate_two_port_pair(left: &Network, right: &Network) -> Result<()> {
    if left.ports() != 2 || right.ports() != 2 {
        return Err(Error::Unsupported(
            "cascade currently requires two two-port networks".to_owned(),
        ));
    }
    if left.frequency != right.frequency {
        return Err(Error::InvalidFrequency(
            "cascaded networks must share the same frequency axis".to_owned(),
        ));
    }
    Ok(())
}

fn validate_connection_networks(
    left: &Network,
    left_port: usize,
    right: &Network,
    right_port: usize,
) -> Result<()> {
    if left_port >= left.ports() {
        return Err(Error::InvalidPort {
            port: left_port,
            ports: left.ports(),
        });
    }
    if right_port >= right.ports() {
        return Err(Error::InvalidPort {
            port: right_port,
            ports: right.ports(),
        });
    }
    if left.frequency != right.frequency {
        return Err(Error::InvalidFrequency(
            "connected Networks must share the same frequency axis".to_owned(),
        ));
    }
    if left.s_definition != right.s_definition {
        return Err(Error::Unsupported(
            "connected Networks must use the same scattering definition".to_owned(),
        ));
    }
    Ok(())
}

fn combined_scattering(left: &Network, right: &Network, point: usize) -> Vec<Complex64> {
    let ports = left.ports() + right.ports();
    let mut combined = vec![Complex64::new(0.0, 0.0); ports * ports];
    for output in 0..left.ports() {
        for input in 0..left.ports() {
            combined[output * ports + input] = left.s[(point, output, input)];
        }
    }
    for output in 0..right.ports() {
        for input in 0..right.ports() {
            combined[(left.ports() + output) * ports + left.ports() + input] =
                right.s[(point, output, input)];
        }
    }
    combined
}

fn combined_reference(
    left: &Network,
    right: &Network,
    point: usize,
    combined_port: usize,
) -> Complex64 {
    if combined_port < left.ports() {
        left.z0[(point, combined_port)]
    } else {
        right.z0[(point, combined_port - left.ports())]
    }
}

fn connection_scattering(left_z0: Complex64, right_z0: Complex64) -> Result<[Complex64; 4]> {
    if left_z0.norm_sqr() <= f64::EPSILON || right_z0.norm_sqr() <= f64::EPSILON {
        return Err(Error::Unsupported(
            "connected port reference impedances must be non-zero".to_owned(),
        ));
    }
    let left_admittance = Complex64::new(1.0, 0.0) / left_z0;
    let right_admittance = Complex64::new(1.0, 0.0) / right_z0;
    let total = left_admittance + right_admittance;
    if total.norm_sqr() <= f64::EPSILON {
        return Err(Error::Unsupported(
            "connected ports have zero total reference admittance".to_owned(),
        ));
    }
    let transmission = 2.0 * (left_admittance * right_admittance).sqrt() / total;
    Ok([
        2.0 * left_admittance / total - 1.0,
        transmission,
        transmission,
        2.0 * right_admittance / total - 1.0,
    ])
}

fn solve_two_by_two(matrix: [Complex64; 4], right: [Complex64; 2]) -> Result<[Complex64; 2]> {
    let determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2];
    if determinant.norm_sqr() <= f64::EPSILON {
        return Err(Error::Unsupported(
            "network connection produced a singular internal system".to_owned(),
        ));
    }
    Ok([
        (matrix[3] * right[0] - matrix[1] * right[1]) / determinant,
        (-matrix[2] * right[0] + matrix[0] * right[1]) / determinant,
    ])
}

fn scattering_to_chain(s: &Array3<Complex64>, point: usize) -> Result<[Complex64; 4]> {
    let s11 = s[(point, 0, 0)];
    let s12 = s[(point, 0, 1)];
    let s21 = s[(point, 1, 0)];
    let s22 = s[(point, 1, 1)];
    if s21.norm_sqr() <= f64::EPSILON {
        return Err(Error::Unsupported(
            "chain conversion requires non-zero forward transmission".to_owned(),
        ));
    }
    Ok([
        Complex64::new(1.0, 0.0) / s21,
        -s22 / s21,
        s11 / s21,
        s12 - s11 * s22 / s21,
    ])
}

fn write_chain_as_scattering(
    chain: [Complex64; 4],
    s: &mut Array3<Complex64>,
    point: usize,
) -> Result<()> {
    if chain[0].norm_sqr() <= f64::EPSILON {
        return Err(Error::Unsupported(
            "scattering conversion requires a non-zero chain leading element".to_owned(),
        ));
    }
    s[(point, 0, 0)] = chain[2] / chain[0];
    s[(point, 1, 0)] = Complex64::new(1.0, 0.0) / chain[0];
    s[(point, 0, 1)] = chain[3] - chain[2] * chain[1] / chain[0];
    s[(point, 1, 1)] = -chain[1] / chain[0];
    Ok(())
}

fn multiply_two_by_two(left: [Complex64; 4], right: [Complex64; 4]) -> [Complex64; 4] {
    [
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
    ]
}
