//! De-embedding algorithms.
//!
//! Origin: `skrf/calibration/deembedding.py`.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use rustfft::FftPlanner;

use super::ensure_nonzero;
use crate::network::{concatenate_ports, s_to_y, s_to_z, y_to_s, y_to_z, z_to_s, z_to_y};
use crate::{Error, Frequency, Network, Result};
/// Origin: `skrf/calibration/deembedding.py::Deembedding`.
pub trait Deembedding {
    fn deembed(&self, network: &Network) -> Result<Network>;
}

/// Removes forward and reverse VNA switch-term loading from a two-port measurement.
///
/// Common IEEE 370 helpers originating from `skrf/calibration/deembedding.py::IEEEP370`.
#[derive(Clone, Copy, Debug, Default)]
pub struct IeeeP370;

impl IeeeP370 {
    pub fn extrapolate_to_dc(network: &Network) -> Result<Network> {
        if network.frequency_points() < 2 {
            return Err(Error::InvalidFrequency(
                "IEEE 370 DC extrapolation requires at least two samples".to_owned(),
            ));
        }
        let source = if network.frequency.values_hz()[0] == 0.0 {
            let frequency = Frequency::from_hz(Array1::from_iter(
                network.frequency.values_hz().iter().skip(1).copied(),
            ))?;
            network.interpolate(&frequency)?
        } else {
            network.clone()
        };
        source.extrapolate_to_dc(None, None)
    }

    pub fn add_dc(network: &Network) -> Result<Network> {
        network.extrapolate_to_dc(None, None)
    }

    pub fn thru(network: &Network) -> Result<Network> {
        if network.ports() != 2 {
            return Err(Error::IncompatibleShape(
                "IEEE 370 thru construction requires a two-port network".to_owned(),
            ));
        }
        let mut thru = network.clone();
        for point in 0..thru.frequency_points() {
            thru.s[(point, 0, 0)] = Complex64::new(0.0, 0.0);
            thru.s[(point, 1, 1)] = Complex64::new(0.0, 0.0);
            thru.s[(point, 0, 1)] = Complex64::new(1.0, 0.0);
            thru.s[(point, 1, 0)] = Complex64::new(1.0, 0.0);
        }
        Ok(thru)
    }

    pub fn com_receiver_noise_filter(
        frequencies: &[f64],
        receiver_frequency: f64,
    ) -> Result<Array1<Complex64>> {
        if !receiver_frequency.is_finite() || receiver_frequency <= 0.0 {
            return Err(Error::InvalidFrequency(
                "COM receiver frequency must be finite and positive".to_owned(),
            ));
        }
        Ok(Array1::from_iter(frequencies.iter().map(|frequency| {
            let ratio = frequency / receiver_frequency;
            Complex64::new(
                1.0 - 3.414_214 * ratio.powi(2) + ratio.powi(4),
                2.613_126 * (ratio - ratio.powi(3)),
            )
            .inv()
        })))
    }

    pub fn make_step(impulse: &[f64]) -> Array1<f64> {
        let mut sum = 0.0;
        Array1::from_iter(impulse.iter().map(|value| {
            sum += value;
            sum
        }))
    }

    /// Symmetrically interpolates a scalar S-parameter to DC from its first
    /// frequency samples.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370.dc_interp`.
    pub fn dc_interp(values: &[Complex64], frequencies_hz: &[f64]) -> Result<f64> {
        if values.len() != frequencies_hz.len() || values.len() < 2 {
            return Err(Error::IncompatibleShape(
                "IEEE 370 DC interpolation requires equal S-parameter and frequency arrays with at least two samples"
                    .to_owned(),
            ));
        }
        if frequencies_hz
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(Error::InvalidFrequency(
                "IEEE 370 DC interpolation requires finite positive frequencies".to_owned(),
            ));
        }
        let sample_count = values.len().min(9);
        let scale = frequencies_hz[0];
        let mut samples = Vec::with_capacity(sample_count * 2);
        for index in (0..sample_count).rev() {
            samples.push((-frequencies_hz[index] / scale, values[index].conj()));
        }
        for index in 0..sample_count {
            samples.push((frequencies_hz[index] / scale, values[index]));
        }
        let interpolated = samples
            .iter()
            .enumerate()
            .map(|(index, (x, value))| {
                let weight = samples
                    .iter()
                    .enumerate()
                    .filter(|(other, _)| *other != index)
                    .map(|(_, (other_x, _))| -other_x / (x - other_x))
                    .product::<f64>();
                *value * weight
            })
            .sum::<Complex64>();
        Ok(interpolated.re)
    }

    /// Iteratively extracts a reflective DC value using the IEEE 370 COM
    /// receiver-filter time-domain constraint.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370.DC`.
    pub fn dc(values: &[Complex64], frequencies_hz: &[f64], allowed_error: f64) -> Result<f64> {
        if values.len() != frequencies_hz.len() || values.len() < 2 {
            return Err(Error::IncompatibleShape(
                "IEEE 370 reflective DC extraction requires equal arrays with at least two samples"
                    .to_owned(),
            ));
        }
        if !allowed_error.is_finite() || allowed_error <= 0.0 {
            return Err(Error::Unsupported(
                "IEEE 370 reflective DC tolerance must be finite and positive".to_owned(),
            ));
        }
        let delta_frequency = frequencies_hz[1] - frequencies_hz[0];
        if !delta_frequency.is_finite() || delta_frequency <= 0.0 {
            return Err(Error::InvalidFrequency(
                "IEEE 370 reflective DC extraction requires increasing frequencies".to_owned(),
            ));
        }
        let receiver_frequency = frequencies_hz[frequencies_hz.len() - 1] / 2.0;
        let filter = Self::com_receiver_noise_filter(frequencies_hz, receiver_frequency)?;
        let output_length = values.len() * 2;
        let time_index = (0..output_length)
            .min_by(|left, right| {
                let left_time = -1.0 / delta_frequency
                    + 2.0 / delta_frequency * *left as f64 / output_length as f64;
                let right_time = -1.0 / delta_frequency
                    + 2.0 / delta_frequency * *right as f64 / output_length as f64;
                (left_time + 3.0e-9)
                    .abs()
                    .total_cmp(&(right_time + 3.0e-9).abs())
            })
            .ok_or_else(|| {
                Error::IncompatibleShape(
                    "IEEE 370 reflective DC extraction requires samples".to_owned(),
                )
            })?;
        let response_at = |dc_value: f64| -> Result<f64> {
            let mut spectrum = Vec::with_capacity(values.len() + 1);
            spectrum.push(Complex64::new(dc_value, 0.0));
            spectrum.extend(
                values
                    .iter()
                    .zip(filter.iter())
                    .map(|(value, filter)| value * filter),
            );
            let impulse = crate::time::irfft(&Array1::from_vec(spectrum), output_length)?;
            let shifted = fft_shift_real(&impulse.to_vec());
            Ok(Self::make_step(&shifted)[time_index])
        };

        let mut dc_value = 0.002;
        for _ in 0..100 {
            let response = response_at(dc_value)?;
            if response.abs() <= allowed_error {
                return Ok(dc_value);
            }
            let offset_response = response_at(dc_value + 0.001)?;
            let slope = (offset_response - response) / 0.001;
            if !slope.is_finite() || slope.abs() <= f64::EPSILON {
                return Err(Error::Unsupported(
                    "IEEE 370 reflective DC iteration has a singular slope".to_owned(),
                ));
            }
            dc_value -= response / slope;
            if !dc_value.is_finite() {
                return Err(Error::Unsupported(
                    "IEEE 370 reflective DC iteration diverged".to_owned(),
                ));
            }
        }
        Err(Error::Unsupported(
            "IEEE 370 reflective DC extraction did not converge".to_owned(),
        ))
    }

    /// Reconstructs the time-domain impedance step response of a one-port
    /// reflection spectrum.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370.getz`.
    pub fn getz(
        values: &[Complex64],
        frequencies_hz: &[f64],
        reference_impedance: f64,
    ) -> Result<Array1<f64>> {
        if !reference_impedance.is_finite() || reference_impedance <= 0.0 {
            return Err(Error::Unsupported(
                "IEEE 370 impedance reconstruction requires positive finite reference impedance"
                    .to_owned(),
            ));
        }
        let dc_value = Self::dc(values, frequencies_hz, 1.0e-10)?;
        let mut spectrum = Vec::with_capacity(values.len() + 1);
        spectrum.push(Complex64::new(dc_value, 0.0));
        spectrum.extend_from_slice(values);
        let output_length = values.len() * 2;
        let impulse = crate::time::irfft(&Array1::from_vec(spectrum), output_length)?;
        let shifted = fft_shift_real(&impulse.to_vec());
        let step = Self::make_step(&shifted);
        let shifted_impedance = Array1::from_iter(
            step.iter()
                .map(|value| -reference_impedance * (value + 1.0) / (value - 1.0)),
        );
        Ok(ifft_shift_real(&shifted_impedance.to_vec()))
    }

    pub fn make_transmission_line(
        line_impedance: f64,
        reference_impedance: f64,
        propagation: &[Complex64],
        length: f64,
    ) -> Result<Array3<Complex64>> {
        if !line_impedance.is_finite()
            || !reference_impedance.is_finite()
            || line_impedance <= 0.0
            || reference_impedance <= 0.0
            || !length.is_finite()
        {
            return Err(Error::Unsupported(
                "IEEE 370 transmission-line inputs must be finite and impedances positive"
                    .to_owned(),
            ));
        }
        let mut scattering = Array3::zeros((propagation.len(), 2, 2));
        for (point, gamma) in propagation.iter().enumerate() {
            let sinh = (*gamma * length).sinh();
            let cosh = (*gamma * length).cosh();
            let denominator = (line_impedance.powi(2) + reference_impedance.powi(2)) * sinh
                + 2.0 * reference_impedance * line_impedance * cosh;
            ensure_nonzero(denominator, "IEEE 370 transmission line is singular")?;
            let reflection =
                (line_impedance.powi(2) - reference_impedance.powi(2)) * sinh / denominator;
            let transmission = 2.0 * reference_impedance * line_impedance / denominator;
            scattering[(point, 0, 0)] = reflection;
            scattering[(point, 1, 1)] = reflection;
            scattering[(point, 0, 1)] = transmission;
            scattering[(point, 1, 0)] = transmission;
        }
        Ok(scattering)
    }

    /// Exact-name wrapper for `IEEEP370.makeTL`.
    pub fn make_tl(
        line_impedance: f64,
        reference_impedance: f64,
        propagation: &[Complex64],
        length: f64,
    ) -> Result<Array3<Complex64>> {
        Self::make_transmission_line(line_impedance, reference_impedance, propagation, length)
    }

    /// Enforces or removes the IEEE 370 Nyquist-rate-point port delays.
    ///
    /// Passing `None` computes and applies the delays. Passing a delay vector
    /// removes those delays; `port` restricts removal to one port.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370.NRP`.
    pub fn enforce_nyquist_rate_point(
        network: &Network,
        delays: Option<&Array1<f64>>,
        port: Option<usize>,
    ) -> Result<(Network, Array1<f64>)> {
        if network.frequency_points() == 0 {
            return Err(Error::InvalidFrequency(
                "Nyquist-rate-point enforcement requires frequency samples".to_owned(),
            ));
        }
        if let Some(port) = port {
            if port >= network.ports() {
                return Err(Error::InvalidPort {
                    port,
                    ports: network.ports(),
                });
            }
        }
        let maximum_frequency = network.frequency.values_hz()[network.frequency_points() - 1];
        if maximum_frequency <= 0.0 {
            return Err(Error::InvalidFrequency(
                "Nyquist-rate-point enforcement requires a positive maximum frequency".to_owned(),
            ));
        }
        let computed = if let Some(delays) = delays {
            if delays.len() != network.ports() || delays.iter().any(|delay| !delay.is_finite()) {
                return Err(Error::IncompatibleShape(
                    "Nyquist-rate-point delays must provide one finite value per port".to_owned(),
                ));
            }
            delays.clone()
        } else {
            Array1::from_iter((0..network.ports()).map(|port| {
                let phase = network.s[(network.frequency_points() - 1, port, port)].arg();
                let correction = if phase < -std::f64::consts::FRAC_PI_2 {
                    -std::f64::consts::PI - phase
                } else if phase > std::f64::consts::FRAC_PI_2 {
                    std::f64::consts::PI - phase
                } else {
                    -phase
                };
                -correction / (2.0 * std::f64::consts::PI * maximum_frequency)
            }))
        };
        let sign = if delays.is_some() { 1.0 } else { -1.0 };
        let mut result = network.clone();
        for point in 0..network.frequency_points() {
            let frequency = network.frequency.values_hz()[point];
            let factors = Array1::from_iter((0..network.ports()).map(|index| {
                if port.is_none() || port == Some(index) {
                    Complex64::from_polar(
                        1.0,
                        sign * std::f64::consts::PI * frequency * computed[index],
                    )
                } else {
                    Complex64::new(1.0, 0.0)
                }
            }));
            for output in 0..network.ports() {
                for input in 0..network.ports() {
                    result.s[(point, output, input)] *= factors[output] * factors[input];
                }
            }
        }
        Ok((result, computed))
    }

    /// Exact-name wrapper for `IEEEP370.NRP`.
    pub fn nrp(
        network: &Network,
        delays: Option<&Array1<f64>>,
        port: Option<usize>,
    ) -> Result<(Network, Array1<f64>)> {
        Self::enforce_nyquist_rate_point(network, delays, port)
    }

    /// Shifts one network port by an integer number of time samples.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370.shiftOnePort`.
    pub fn shift_one_port(network: &Network, samples: isize, port: usize) -> Result<Network> {
        if port >= network.ports() {
            return Err(Error::InvalidPort {
                port,
                ports: network.ports(),
            });
        }
        let points = network.frequency_points();
        let mut result = network.clone();
        for point in 0..points {
            let omega = std::f64::consts::PI * (point + 1) as f64 / points as f64;
            let factor = Complex64::from_polar(1.0, -(samples as f64) * omega / 2.0);
            for other in 0..network.ports() {
                result.s[(point, port, other)] *= factor;
                result.s[(point, other, port)] *= factor;
            }
        }
        Ok(result)
    }

    /// Shifts every network port by an integer number of time samples.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370.shiftNPoints`.
    pub fn shift_points(network: &Network, samples: isize) -> Result<Network> {
        let mut result = network.clone();
        for port in 0..network.ports() {
            result = Self::shift_one_port(&result, samples, port)?;
        }
        Ok(result)
    }

    /// Exact-name wrapper for `IEEEP370.shiftNPoints`.
    pub fn shift_n_points(network: &Network, samples: isize) -> Result<Network> {
        Self::shift_points(network, samples)
    }

    /// Peels a lossless time sample from both sides repeatedly and returns the
    /// remaining network plus the accumulated left and right error boxes.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370.peelNPointsLossless`.
    pub fn peel_n_points_lossless(
        network: &Network,
        samples: usize,
        reference_impedance: f64,
    ) -> Result<(Network, Network, Network)> {
        if network.ports() != 2 || samples == 0 {
            return Err(Error::IncompatibleShape(
                "lossless IEEE 370 peeling requires a two-port network and at least one sample"
                    .to_owned(),
            ));
        }
        let points = network.frequency_points();
        let propagation = Array1::from_iter((0..points).map(|point| {
            Complex64::new(
                0.0,
                std::f64::consts::PI * (point + 1) as f64 / points as f64 / 2.0,
            )
        }));
        let mut remaining = network.clone();
        let mut left_box: Option<Network> = None;
        let mut right_box: Option<Network> = None;
        for _ in 0..samples {
            let left_reflection =
                Array1::from_iter((0..points).map(|point| remaining.s[(point, 0, 0)]));
            let right_reflection =
                Array1::from_iter((0..points).map(|point| remaining.s[(point, 1, 1)]));
            let left_impedance = Self::getz(
                &left_reflection.to_vec(),
                &remaining.frequency.values_hz().to_vec(),
                reference_impedance,
            )?[0];
            let right_impedance = Self::getz(
                &right_reflection.to_vec(),
                &remaining.frequency.values_hz().to_vec(),
                reference_impedance,
            )?[0];
            let left_line = ieee_p370_line_network(
                network,
                left_impedance,
                reference_impedance,
                &propagation.to_vec(),
            )?;
            let right_line = ieee_p370_line_network(
                network,
                right_impedance,
                reference_impedance,
                &propagation.to_vec(),
            )?;
            remaining = remaining.deembed(&left_line, Some(&right_line))?;
            left_box = Some(match left_box {
                Some(accumulated) => accumulated.cascade(&left_line)?,
                None => left_line,
            });
            right_box = Some(match right_box {
                Some(accumulated) => right_line.cascade(&accumulated)?,
                None => right_line,
            });
        }
        let left_box = left_box.ok_or_else(|| {
            Error::IncompatibleShape("lossless IEEE 370 peeling requires samples".to_owned())
        })?;
        let right_box = right_box.ok_or_else(|| {
            Error::IncompatibleShape("lossless IEEE 370 peeling requires samples".to_owned())
        })?;
        Ok((remaining, left_box, right_box))
    }
}

fn fft_shift_real(values: &[f64]) -> Vec<f64> {
    let split = values.len().div_ceil(2);
    values[split..]
        .iter()
        .chain(values[..split].iter())
        .copied()
        .collect()
}

fn ifft_shift_real(values: &[f64]) -> Array1<f64> {
    let split = values.len() / 2;
    Array1::from_iter(
        values[split..]
            .iter()
            .chain(values[..split].iter())
            .copied(),
    )
}

fn ieee_p370_line_network(
    template: &Network,
    line_impedance: f64,
    reference_impedance: f64,
    propagation: &[Complex64],
) -> Result<Network> {
    let scattering = IeeeP370::make_tl(line_impedance, reference_impedance, propagation, 1.0)?;
    let z0 = Array2::from_elem(
        (template.frequency_points(), 2),
        Complex64::new(reference_impedance, 0.0),
    );
    Network::new(template.frequency.clone(), scattering, z0)
}

/// Frequency-domain traces used by IEEE 370 fixture electrical requirements.
///
/// Origin: `skrf/calibration/deembedding.py::IEEEP370_FER`.
#[derive(Clone, Debug, PartialEq)]
pub struct FixtureElectricalRequirements {
    pub insertion_loss_forward_db: Array1<f64>,
    pub insertion_loss_reverse_db: Array1<f64>,
    pub return_loss_port1_db: Array1<f64>,
    pub return_loss_port2_db: Array1<f64>,
    pub insertion_minus_return_port1_db: Array1<f64>,
    pub insertion_minus_return_port2_db: Array1<f64>,
    pub differential_to_common_forward_db: Option<Array1<f64>>,
    pub differential_to_common_reverse_db: Option<Array1<f64>>,
}

/// Backend-neutral IEEE 370 fixture electrical requirement calculations.
///
/// The upstream class renders these values with Matplotlib. The Rust port
/// returns the traces so any plotting backend can apply the standard limits.
///
/// Origin: `skrf/calibration/deembedding.py::IEEEP370_FER`.
#[derive(Clone, Copy, Debug, Default)]
pub struct IeeeP370FixtureElectricalRequirements;

impl IeeeP370FixtureElectricalRequirements {
    pub const FER1_MINIMUM_A_DB: f64 = -10.0;
    pub const FER1_MINIMUM_BC_DB: f64 = -15.0;
    pub const FER2_MAXIMUM_A_DB: f64 = -20.0;
    pub const FER2_MAXIMUM_B_DB: f64 = -10.0;
    pub const FER2_MAXIMUM_C_DB: f64 = -6.0;
    pub const FER3_MINIMUM_A_DB: f64 = 5.0;
    pub const FER3_MINIMUM_BC_DB: f64 = 0.0;
    pub const FER5_RELATIVE_A: f64 = 0.025;
    pub const FER5_RELATIVE_B: f64 = 0.05;
    pub const FER5_RELATIVE_C: f64 = 0.1;
    pub const FER6_MAXIMUM_DB: f64 = -15.0;

    /// Origin: `IEEEP370_FER.plot_fd_se_fer` without renderer side effects.
    pub fn single_ended(network: &Network) -> Result<FixtureElectricalRequirements> {
        if network.ports() != 2 {
            return Err(Error::IncompatibleShape(
                "single-ended IEEE 370 FER requires a two-port network".to_owned(),
            ));
        }
        Ok(ieee_p370_fer_traces(network, false))
    }

    /// Origin: `IEEEP370_FER.plot_fd_mm_fer` without renderer side effects.
    pub fn mixed_mode(network: &Network) -> Result<FixtureElectricalRequirements> {
        if network.ports() != 4 {
            return Err(Error::IncompatibleShape(
                "mixed-mode IEEE 370 FER requires a four-port network".to_owned(),
            ));
        }
        Ok(ieee_p370_fer_traces(
            &network.single_ended_to_mixed_mode(2)?,
            true,
        ))
    }
}

fn ieee_p370_fer_traces(network: &Network, mixed_mode: bool) -> FixtureElectricalRequirements {
    let db = |output: usize, input: usize| {
        Array1::from_iter((0..network.frequency_points()).map(|point| {
            20.0 * network.s[(point, output, input)]
                .norm()
                .max(f64::MIN_POSITIVE)
                .log10()
        }))
    };
    let insertion_loss_forward_db = db(1, 0);
    let insertion_loss_reverse_db = db(0, 1);
    let return_loss_port1_db = db(0, 0);
    let return_loss_port2_db = db(1, 1);
    let insertion_minus_return_port1_db = &insertion_loss_forward_db - &return_loss_port1_db;
    let insertion_minus_return_port2_db = &insertion_loss_reverse_db - &return_loss_port2_db;
    let (differential_to_common_forward_db, differential_to_common_reverse_db) = if mixed_mode {
        (
            Some(&db(2, 0) - &insertion_loss_forward_db),
            Some(&db(3, 1) - &insertion_loss_reverse_db),
        )
    } else {
        (None, None)
    };
    FixtureElectricalRequirements {
        insertion_loss_forward_db,
        insertion_loss_reverse_db,
        return_loss_port1_db,
        return_loss_port2_db,
        insertion_minus_return_port1_db,
        insertion_minus_return_port2_db,
        differential_to_common_forward_db,
        differential_to_common_reverse_db,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityEvaluation {
    Poor,
    Inconclusive,
    Acceptable,
    Good,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityMetric {
    pub value_percent: f64,
    pub evaluation: QualityEvaluation,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrequencyDomainQuality {
    pub causality: QualityMetric,
    pub passivity: QualityMetric,
    pub reciprocity: QualityMetric,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixedModeFrequencyDomainQuality {
    pub differential: FrequencyDomainQuality,
    pub common: FrequencyDomainQuality,
}

/// Origin: `skrf/calibration/deembedding.py::IEEEP370_FD_QM`.
#[derive(Clone, Copy, Debug, Default)]
pub struct IeeeP370FrequencyDomainQuality;

impl IeeeP370FrequencyDomainQuality {
    pub fn check_causality(network: &Network) -> f64 {
        let points = network.frequency_points();
        if points < 3 {
            return 100.0;
        }
        let mut minimum: f64 = 100.0;
        for output in 0..network.ports() {
            for input in 0..network.ports() {
                let first = network.s[(0, output, input)];
                if (1..points).all(|point| network.s[(point, output, input)] == first) {
                    continue;
                }
                let mut total = 0.0;
                let mut positive = 0.0;
                for point in 0..points - 2 {
                    let current =
                        network.s[(point + 1, output, input)] - network.s[(point, output, input)];
                    let next = network.s[(point + 2, output, input)]
                        - network.s[(point + 1, output, input)];
                    let rotation = next.re * current.im - next.im * current.re;
                    if rotation > 0.0 {
                        positive += rotation;
                    }
                    total += rotation.abs();
                }
                let metric = if total == 0.0 {
                    0.0
                } else {
                    (positive / total).max(0.0) * 100.0
                };
                minimum = minimum.min(metric);
            }
        }
        minimum
    }

    pub fn check_passivity(network: &Network) -> Result<f64> {
        if network.ports() == 1 {
            return Err(Error::Unsupported(
                "IEEE 370 passivity metric is undefined for one-port networks".to_owned(),
            ));
        }
        let points = network.frequency_points();
        let penalty = (0..points)
            .map(|point| {
                let norm = calibration_spectral_norm(&network.s, point);
                if norm > 1.000_01 {
                    (norm - 1.000_01) / 0.1
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        Ok(((points as f64 - penalty).max(0.0) / points as f64) * 100.0)
    }

    pub fn check_reciprocity(network: &Network) -> Result<f64> {
        if network.ports() == 1 {
            return Err(Error::Unsupported(
                "IEEE 370 reciprocity metric is undefined for one-port networks".to_owned(),
            ));
        }
        let ports = network.ports();
        let points = network.frequency_points();
        let penalty = (0..points)
            .map(|point| {
                let difference = (0..ports)
                    .flat_map(|row| (0..ports).map(move |column| (row, column)))
                    .map(|(row, column)| {
                        (network.s[(point, row, column)] - network.s[(point, column, row)]).norm()
                    })
                    .sum::<f64>()
                    / (ports * (ports - 1)) as f64;
                if difference > 1.0e-6 {
                    (difference - 1.0e-6) / 0.1
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        Ok(((points as f64 - penalty).max(0.0) / points as f64) * 100.0)
    }

    pub fn check_single_ended(network: &Network) -> Result<FrequencyDomainQuality> {
        let causality = Self::check_causality(network);
        let passivity = Self::check_passivity(network)?;
        let reciprocity = Self::check_reciprocity(network)?;
        Ok(FrequencyDomainQuality {
            causality: QualityMetric {
                value_percent: causality,
                evaluation: evaluate_causality(causality),
            },
            passivity: QualityMetric {
                value_percent: passivity,
                evaluation: evaluate_passivity_or_reciprocity(passivity),
            },
            reciprocity: QualityMetric {
                value_percent: reciprocity,
                evaluation: evaluate_passivity_or_reciprocity(reciprocity),
            },
        })
    }

    /// Checks the differential and common sub-networks of a four-port network.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370_FD_QM.check_mm_quality`.
    pub fn check_mixed_mode(network: &Network) -> Result<MixedModeFrequencyDomainQuality> {
        if network.ports() != 4 {
            return Err(Error::IncompatibleShape(
                "IEEE 370 mixed-mode quality requires a four-port network".to_owned(),
            ));
        }
        let mixed_mode = network.single_ended_to_mixed_mode(2)?;
        Ok(MixedModeFrequencyDomainQuality {
            differential: Self::check_single_ended(&mixed_mode.subnetwork(&[0, 1])?)?,
            common: Self::check_single_ended(&mixed_mode.subnetwork(&[2, 3])?)?,
        })
    }
}

/// Core signal helpers from `skrf/calibration/deembedding.py::IEEEP370_TD_QM`.
#[derive(Clone, Copy, Debug)]
pub struct IeeeP370TimeDomainQuality {
    pub data_rate: f64,
    pub samples_per_unit_interval: usize,
    pub rise_time_fraction: f64,
    pub pulse_shape: usize,
    pub extrapolation: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeDomainQualityMetric {
    pub value_millivolts: f64,
    pub evaluation: QualityEvaluation,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeDomainQuality {
    pub causality: TimeDomainQualityMetric,
    pub passivity: TimeDomainQualityMetric,
    pub reciprocity: TimeDomainQualityMetric,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixedModeTimeDomainQuality {
    pub differential: TimeDomainQuality,
    pub common: TimeDomainQuality,
}

impl IeeeP370TimeDomainQuality {
    pub fn new(
        data_rate: f64,
        samples_per_unit_interval: usize,
        rise_time_fraction: f64,
        pulse_shape: usize,
        extrapolation: usize,
    ) -> Result<Self> {
        if !data_rate.is_finite() || data_rate <= 0.0 {
            return Err(Error::InvalidFrequency(
                "IEEE 370 data rate must be finite and positive".to_owned(),
            ));
        }
        if samples_per_unit_interval == 0 {
            return Err(Error::Unsupported(
                "IEEE 370 samples per unit interval must be positive".to_owned(),
            ));
        }
        if !rise_time_fraction.is_finite() || rise_time_fraction <= 0.0 {
            return Err(Error::Unsupported(
                "IEEE 370 rise-time fraction must be finite and positive".to_owned(),
            ));
        }
        if !(1..=3).contains(&pulse_shape) {
            return Err(Error::Unsupported(
                "IEEE 370 pulse shape must be 1, 2, or 3".to_owned(),
            ));
        }
        if !(1..=2).contains(&extrapolation) {
            return Err(Error::Unsupported(
                "IEEE 370 extrapolation mode must be 1 or 2".to_owned(),
            ));
        }
        Ok(Self {
            data_rate,
            samples_per_unit_interval,
            rise_time_fraction,
            pulse_shape,
            extrapolation,
        })
    }

    pub fn add_conjugates(values: &[Complex64]) -> Array1<Complex64> {
        let mut output = Vec::with_capacity(values.len().saturating_mul(2).saturating_sub(1));
        output.extend_from_slice(values);
        output.extend(values.iter().skip(1).rev().map(|value| value.conj()));
        Array1::from_vec(output)
    }

    pub fn create_reciprocal(network: &Network) -> Network {
        let mut reciprocal = network.clone();
        for point in 0..network.frequency_points() {
            for row in 0..network.ports() {
                for column in 0..network.ports() {
                    reciprocal.s[(point, row, column)] = network.s[(point, column, row)];
                }
            }
        }
        reciprocal
    }

    /// Clips singular values above one at every frequency sample.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370_TD_QM.create_passive`.
    pub fn create_passive(network: &Network) -> Result<Network> {
        let ports = network.ports();
        let mut passive = network.clone();
        for point in 0..network.frequency_points() {
            let matrix = faer::Mat::<Complex64>::from_fn(ports, ports, |row, column| {
                network.s[(point, row, column)]
            });
            let decomposition = matrix
                .svd()
                .map_err(|error| Error::Unsupported(format!("SVD failed: {error:?}")))?;
            let singular = decomposition.S().column_vector();
            let left = decomposition.U();
            let right = decomposition.V();
            for row in 0..ports {
                for column in 0..ports {
                    passive.s[(point, row, column)] = (0..ports)
                        .map(|index| {
                            left[(row, index)]
                                * singular[index].re.min(1.0)
                                * right[(column, index)].conj()
                        })
                        .sum();
                }
            }
        }
        Ok(passive)
    }

    /// Extrapolates to DC on a harmonic frequency axis for TD reconstruction.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370_TD_QM.extrapolate_to_dc`.
    pub fn extrapolate_to_dc(network: &Network) -> Result<Network> {
        IeeeP370::extrapolate_to_dc(network)
    }

    /// Computes application-based IEEE 370 time-domain quality metrics.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370_TD_QM.check_se_quality`.
    pub fn check_single_ended(&self, network: &Network) -> Result<TimeDomainQuality> {
        if network.ports() < 2 || network.frequency_points() < 2 {
            return Err(Error::IncompatibleShape(
                "IEEE 370 time-domain quality requires a multiport network with at least two frequency samples"
                    .to_owned(),
            ));
        }
        let original = Self::extrapolate_to_dc(network)?;
        let causal = ieee_p370_causal_model(&original)?;
        let passive = Self::create_passive(&original)?;
        let reciprocal = Self::create_reciprocal(&original);
        let original_response = self.application_response(&original)?;
        let causality = ieee_p370_response_difference(
            &self.application_response(&causal)?,
            &original_response,
            self.samples_per_unit_interval,
        )?;
        let passivity = ieee_p370_response_difference(
            &self.application_response(&passive)?,
            &original_response,
            self.samples_per_unit_interval,
        )?;
        let reciprocity = ieee_p370_response_difference(
            &self.application_response(&reciprocal)?,
            &original_response,
            self.samples_per_unit_interval,
        )?;
        Ok(TimeDomainQuality {
            causality: time_domain_metric(causality),
            passivity: time_domain_metric(passivity),
            reciprocity: time_domain_metric(reciprocity),
        })
    }

    /// Checks differential and common modes of a four-port network.
    ///
    /// Origin: `skrf/calibration/deembedding.py::IEEEP370_TD_QM.check_mm_quality`.
    pub fn check_mixed_mode(&self, network: &Network) -> Result<MixedModeTimeDomainQuality> {
        if network.ports() != 4 {
            return Err(Error::IncompatibleShape(
                "IEEE 370 mixed-mode time-domain quality requires a four-port network".to_owned(),
            ));
        }
        let mixed_mode = network.single_ended_to_mixed_mode(2)?;
        Ok(MixedModeTimeDomainQuality {
            differential: self.check_single_ended(&mixed_mode.subnetwork(&[0, 1])?)?,
            common: self.check_single_ended(&mixed_mode.subnetwork(&[2, 3])?)?,
        })
    }

    fn application_response(&self, network: &Network) -> Result<Array3<f64>> {
        let points = network.frequency_points();
        let length = 2 * points - 1;
        let frequencies = network.frequency.values_hz();
        let step = frequencies[1] - frequencies[0];
        let time_step = 1.0 / (2.0 * frequencies[points - 1] + step);
        let pulse = ieee_p370_pulse_spectrum(
            length,
            time_step,
            self.data_rate,
            self.rise_time_fraction,
            self.pulse_shape,
        );
        let cutoff = 1.5 * self.data_rate;
        let sigma = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
        let rise_time = 1.0 / self.data_rate * 1000.0 * self.rise_time_fraction;
        let filter_frequency = 320.0 / rise_time;
        let filter =
            Array1::from_iter(frequencies.iter().map(|frequency| match self.pulse_shape {
                1 => Complex64::new(1.0, 0.0),
                2 => Complex64::new(1.0, *frequency / filter_frequency).inv(),
                _ => Complex64::new(
                    (-2.0 * std::f64::consts::PI.powi(2) * frequency.powi(2) * sigma.powi(2)).exp(),
                    0.0,
                ),
            }));
        let mut response = Array3::zeros((length, network.ports(), network.ports()));
        for output in 0..network.ports() {
            for input in 0..network.ports() {
                let mut positive = (0..points)
                    .map(|point| network.s[(point, output, input)] * filter[point])
                    .collect::<Vec<_>>();
                positive[0] = Complex64::new(positive[0].re, 0.0);
                let spectrum = Self::add_conjugates(&positive)
                    .iter()
                    .zip(pulse.iter())
                    .map(|(value, pulse)| value * pulse)
                    .collect::<Vec<_>>();
                let time = calibration_fft(&spectrum, true);
                for index in 0..length {
                    response[(index, output, input)] = time[index].re;
                }
            }
        }
        Ok(response)
    }

    pub fn align_signals(first: &[f64], second: &[f64]) -> Result<isize> {
        if first.len() != second.len() || first.is_empty() {
            return Err(Error::IncompatibleShape(
                "IEEE 370 signal alignment requires equal non-empty arrays".to_owned(),
            ));
        }
        let search = 1000.min(first.len().div_ceil(10));
        let mut best_shift = 0;
        let mut best_error = f64::INFINITY;
        for shift in -(search as isize)..=search as isize {
            let error = first
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let shifted =
                        (index as isize - shift).rem_euclid(second.len() as isize) as usize;
                    (value - second[shifted]).powi(2)
                })
                .sum::<f64>();
            if error < best_error {
                best_error = error;
                best_shift = shift;
            }
        }
        Ok(best_shift)
    }
}

fn ieee_p370_causal_model(network: &Network) -> Result<Network> {
    let points = network.frequency_points();
    let frequencies = network.frequency.values_hz();
    let mut causal = network.clone();
    for output in 0..network.ports() {
        for input in 0..network.ports() {
            let values = (0..points)
                .map(|point| {
                    let value = network.s[(point, output, input)];
                    if value.norm() <= 1.0e-12 {
                        Complex64::new(1.0e-5, 0.0)
                    } else {
                        value
                    }
                })
                .collect::<Vec<_>>();
            let complete = IeeeP370TimeDomainQuality::add_conjugates(&values);
            let logarithmic_magnitude = complete
                .iter()
                .map(|value| Complex64::new(value.norm().ln(), 0.0))
                .collect::<Vec<_>>();
            let mut magnitude_time = calibration_fft(&logarithmic_magnitude, true);
            for value in magnitude_time.iter_mut().skip(points) {
                *value = -*value;
            }
            magnitude_time
                .iter_mut()
                .for_each(|value| *value *= Complex64::i());
            let enforced_phase = calibration_fft(&magnitude_time, false);
            let phase = crate::math::unwrap_radians(&Array1::from_iter(
                values.iter().map(|value| -value.arg()),
            ));
            let delay = frequencies
                .iter()
                .zip(phase.iter())
                .filter(|(frequency, _)| **frequency > 0.0)
                .map(|(frequency, phase)| phase / (2.0 * std::f64::consts::PI * frequency))
                .filter(|value| value.is_finite() && *value >= 0.0)
                .fold(f64::INFINITY, f64::min);
            let delay = if delay.is_finite() { delay } else { 0.0 };
            for point in 0..points {
                causal.s[(point, output, input)] = Complex64::from_polar(
                    values[point].norm(),
                    -enforced_phase[point].re
                        - 2.0 * std::f64::consts::PI * frequencies[point] * delay,
                );
            }
        }
    }
    Ok(causal)
}

fn ieee_p370_response_difference(
    model: &Array3<f64>,
    original: &Array3<f64>,
    samples_per_unit_interval: usize,
) -> Result<f64> {
    if model.dim() != original.dim() {
        return Err(Error::IncompatibleShape(
            "IEEE 370 time-domain comparisons require equal response shapes".to_owned(),
        ));
    }
    let (samples, ports, _) = model.dim();
    let samples_per_ui = samples_per_unit_interval.min(samples).max(1);
    let mut differences = Array3::zeros((1, ports, ports));
    for output in 0..ports {
        for input in 0..ports {
            let peak = (0..samples)
                .max_by(|left, right| {
                    original[(*left, output, input)].total_cmp(&original[(*right, output, input)])
                })
                .unwrap_or(0);
            let mut maximum: f64 = 0.0;
            for phase in 0..samples_per_ui {
                let sum = (phase..samples)
                    .step_by(samples_per_ui)
                    .filter(|index| index.abs_diff(peak) <= 31 * samples_per_ui)
                    .map(|index| {
                        (model[(index, output, input)] - original[(index, output, input)]).abs()
                    })
                    .sum::<f64>();
                maximum = maximum.max(sum);
            }
            differences[(0, output, input)] = Complex64::new(maximum, 0.0);
        }
    }
    Ok(500.0 * calibration_spectral_norm(&differences, 0))
}

fn calibration_fft(values: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let mut output = values.to_vec();
    let mut planner = FftPlanner::new();
    if inverse {
        planner.plan_fft_inverse(output.len()).process(&mut output);
        let scale = output.len() as f64;
        output.iter_mut().for_each(|value| *value /= scale);
    } else {
        planner.plan_fft_forward(output.len()).process(&mut output);
    }
    output
}

fn ieee_p370_pulse_spectrum(
    length: usize,
    time_step: f64,
    data_rate: f64,
    rise_time_fraction: f64,
    pulse_shape: usize,
) -> Vec<Complex64> {
    let mut pulse = vec![0.0; length];
    if pulse_shape == 1 {
        let half = (length - 1) / 2;
        let sigma =
            rise_time_fraction / (data_rate * ((-0.2_f64.ln()).sqrt() - (-0.8_f64.ln()).sqrt()));
        let start = (1.5 / (data_rate * time_step)).round() as isize - 1;
        for (index, value) in pulse.iter_mut().enumerate() {
            let source = (index as isize + half as isize - start).rem_euclid(length as isize);
            let time = (source - half as isize) as f64 * time_step;
            *value = (-(time / sigma).powi(2)).exp();
        }
    } else {
        let high = (1.0 / (data_rate * time_step)).round().max(1.0) as usize;
        let rise = (high as f64 * 1.4 * rise_time_fraction).round().max(1.0) as usize;
        for (index, value) in pulse.iter_mut().enumerate() {
            *value = if index < rise {
                index as f64 / rise as f64
            } else if index < high + rise {
                1.0
            } else if index < high + 2 * rise {
                1.0 - (index - high - rise) as f64 / rise as f64
            } else {
                0.0
            };
        }
    }
    calibration_fft(
        &pulse
            .into_iter()
            .map(|value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>(),
        false,
    )
}

fn time_domain_metric(value_millivolts: f64) -> TimeDomainQualityMetric {
    let evaluation = if value_millivolts >= 15.0 {
        QualityEvaluation::Poor
    } else if value_millivolts >= 10.0 {
        QualityEvaluation::Inconclusive
    } else if value_millivolts >= 5.0 {
        QualityEvaluation::Acceptable
    } else {
        QualityEvaluation::Good
    };
    TimeDomainQualityMetric {
        value_millivolts: (value_millivolts * 10.0).round() / 10.0,
        evaluation,
    }
}

fn evaluate_causality(value: f64) -> QualityEvaluation {
    if value <= 20.0 {
        QualityEvaluation::Poor
    } else if value <= 50.0 {
        QualityEvaluation::Inconclusive
    } else if value <= 80.0 {
        QualityEvaluation::Acceptable
    } else {
        QualityEvaluation::Good
    }
}

fn evaluate_passivity_or_reciprocity(value: f64) -> QualityEvaluation {
    if value <= 80.0 {
        QualityEvaluation::Poor
    } else if value <= 99.0 {
        QualityEvaluation::Inconclusive
    } else if value <= 99.9 {
        QualityEvaluation::Acceptable
    } else {
        QualityEvaluation::Good
    }
}

fn calibration_spectral_norm(scattering: &Array3<Complex64>, point: usize) -> f64 {
    let ports = scattering.dim().1;
    let mut vector = vec![Complex64::new(1.0 / (ports as f64).sqrt(), 0.0); ports];
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
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let adjoint = (0..ports)
            .map(|column| {
                (0..ports)
                    .map(|row| scattering[(point, row, column)].conj() * transformed[row])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        let norm = adjoint
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        for (value, next) in vector.iter_mut().zip(adjoint) {
            *value = next / norm;
        }
    }
    singular
}

/// IEEE 370 single-ended non-zero-crossing 2x-thru fixture extraction.
///
/// This ports the upstream impedance-transform split selected by
/// `use_z_instead_ifft`, which is deterministic and does not require a plotting backend.
#[derive(Clone, Debug)]
pub struct IeeeP370SeNzc2xThru {
    pub two_x_thru: Network,
    pub side1: Network,
    pub side2: Network,
    pub name: Option<String>,
}

impl IeeeP370SeNzc2xThru {
    pub fn new(two_x_thru: Network, name: Option<String>) -> Result<Self> {
        validate_two_port_dummy(&two_x_thru)?;
        let (side1, side2) = Self::split_two_x_thru(&two_x_thru)?;
        Ok(Self {
            two_x_thru,
            side1,
            side2,
            name,
        })
    }

    pub fn split_two_x_thru(two_x_thru: &Network) -> Result<(Network, Network)> {
        validate_two_port_dummy(two_x_thru)?;
        let impedance = two_x_thru.impedance()?;
        let points = two_x_thru.frequency_points();
        let mut left = Array3::zeros((points, 2, 2));
        let mut right = Array3::zeros((points, 2, 2));
        for point in 0..points {
            left[(point, 0, 0)] = impedance[(point, 0, 0)] + impedance[(point, 1, 0)];
            left[(point, 0, 1)] = 2.0 * impedance[(point, 1, 0)];
            left[(point, 1, 0)] = 2.0 * impedance[(point, 1, 0)];
            left[(point, 1, 1)] = 2.0 * impedance[(point, 1, 0)];
            right[(point, 0, 0)] = 2.0 * impedance[(point, 0, 1)];
            right[(point, 0, 1)] = 2.0 * impedance[(point, 0, 1)];
            right[(point, 1, 0)] = 2.0 * impedance[(point, 0, 1)];
            right[(point, 1, 1)] = impedance[(point, 1, 1)] + impedance[(point, 0, 1)];
        }
        let side1 = Network::from_impedance(
            two_x_thru.frequency.clone(),
            &left,
            two_x_thru.z0.clone(),
            two_x_thru.s_definition,
        )?;
        let side2 = Network::from_impedance(
            two_x_thru.frequency.clone(),
            &right,
            two_x_thru.z0.clone(),
            two_x_thru.s_definition,
        )?
        .flipped()?;
        Ok((side1, side2))
    }

    /// Builds an NZC fixture using the upstream time-gated reflection split.
    pub fn new_time_gated(
        two_x_thru: Network,
        forced_split_impedance: Option<f64>,
        name: Option<String>,
    ) -> Result<Self> {
        validate_two_port_dummy(&two_x_thru)?;
        let (side1, side2) =
            Self::split_two_x_thru_time_gated(&two_x_thru, forced_split_impedance)?;
        Ok(Self {
            two_x_thru,
            side1,
            side2,
            name,
        })
    }

    /// Origin: `IEEEP370_SE_NZC_2xThru.split2xthru` with
    /// `use_z_instead_ifft=False`.
    pub fn split_two_x_thru_time_gated(
        two_x_thru: &Network,
        forced_split_impedance: Option<f64>,
    ) -> Result<(Network, Network)> {
        validate_two_port_dummy(two_x_thru)?;
        if two_x_thru.frequency_points() < 2 {
            return Err(Error::InvalidFrequency(
                "time-gated IEEE 370 NZC extraction requires at least two samples".to_owned(),
            ));
        }
        let original_frequency = two_x_thru.frequency.clone();
        let original_values = original_frequency.values_hz();
        let has_dc = original_values[0] == 0.0;
        let mut working = if has_dc {
            if original_values.len() < 3 {
                return Err(Error::InvalidFrequency(
                    "time-gated IEEE 370 NZC extraction requires two non-DC samples".to_owned(),
                ));
            }
            let frequency =
                Frequency::from_hz(Array1::from_iter(original_values.iter().skip(1).copied()))?;
            two_x_thru.interpolate(&frequency)?
        } else {
            two_x_thru.clone()
        };
        let working_values = working.frequency.values_hz();
        let working_step = working_values[1] - working_values[0];
        let harmonic =
            (working_step - working_values[0]).abs() <= 1.0e-6 * working_values[0].abs().max(1.0);
        let mut interpolated = false;
        if !harmonic {
            let projected_points = (working_values[working_values.len() - 1] / working_values[0])
                .round()
                .clamp(2.0, 10_000.0) as usize;
            let step = working_values[working_values.len() - 1] / projected_points as f64;
            let frequency = Frequency::from_hz(Array1::from_iter(
                (1..=projected_points).map(|index| step * index as f64),
            ))?;
            working = working.interpolate(&frequency)?;
            interpolated = true;
        }
        if has_dc || interpolated {
            let (mut side1, mut side2) =
                Self::split_two_x_thru_time_gated(&working, forced_split_impedance)?;
            if has_dc {
                side1 = IeeeP370::add_dc(&side1)?;
                side2 = IeeeP370::add_dc(&side2)?;
            }
            return Ok((
                side1.interpolate(&original_frequency)?,
                side2.interpolate(&original_frequency)?,
            ));
        }
        let frequencies = working.frequency.values_hz();
        let step = frequencies[1] - frequencies[0];
        debug_assert!((step - frequencies[0]).abs() <= 1.0e-6 * frequencies[0].abs().max(1.0));
        let two_x_thru = &working;
        let with_dc = IeeeP370::extrapolate_to_dc(two_x_thru)?;
        let transmission = (0..with_dc.frequency_points())
            .map(|point| with_dc.s[(point, 1, 0)])
            .collect::<Vec<_>>();
        let transmission_impulse = ieee_p370_real_impulse(&transmission)?;
        let midpoint = transmission_impulse
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let impedance_profile = ieee_p370_impedance_profile(two_x_thru, 0)?;
        let split_impedance = if let Some(impedance) = forced_split_impedance {
            if !impedance.is_finite() || impedance <= 0.0 {
                return Err(Error::Unsupported(
                    "forced IEEE 370 split impedance must be finite and positive".to_owned(),
                ));
            }
            impedance
        } else if midpoint == 0 {
            impedance_profile[0]
        } else {
            0.5 * (impedance_profile[midpoint - 1] + impedance_profile[midpoint])
        };

        let split_reference =
            Array2::from_elem(two_x_thru.z0.dim(), Complex64::new(split_impedance, 0.0));
        let mut renormalized = two_x_thru.clone();
        renormalized.renormalize(split_reference.clone(), two_x_thru.s_definition)?;
        let e001 = ieee_p370_gate_reflection(&renormalized, 0, midpoint)?;
        let e002 = ieee_p370_gate_reflection(&renormalized, 1, midpoint)?;
        let points = two_x_thru.frequency_points();
        let mut e111 = Array1::zeros(points);
        let mut e112 = Array1::zeros(points);
        let mut e01 = Array1::zeros(points);
        let mut e10 = Array1::zeros(points);
        for point in 0..points {
            let s11 = renormalized.s[(point, 0, 0)];
            let s21 = renormalized.s[(point, 1, 0)];
            let s12 = renormalized.s[(point, 0, 1)];
            let s22 = renormalized.s[(point, 1, 1)];
            ensure_nonzero(s12, "time-gated NZC reverse transmission is zero")?;
            ensure_nonzero(s21, "time-gated NZC forward transmission is zero")?;
            e111[point] = (s22 - e002[point]) / s12;
            e112[point] = (s11 - e001[point]) / s21;
            let coupling = Complex64::new(1.0, 0.0) - e111[point] * e112[point];
            e01[point] = (s21 * coupling).sqrt();
            e10[point] = (s12 * coupling).sqrt();
            if point > 0 {
                if (-e01[point] - e01[point - 1]).norm() < (e01[point] - e01[point - 1]).norm() {
                    e01[point] = -e01[point];
                }
                if (-e10[point] - e10[point - 1]).norm() < (e10[point] - e10[point - 1]).norm() {
                    e10[point] = -e10[point];
                }
            }
        }
        let mut left = Array3::zeros((points, 2, 2));
        let mut right = Array3::zeros((points, 2, 2));
        for point in 0..points {
            left[(point, 0, 0)] = e001[point];
            left[(point, 0, 1)] = e01[point];
            left[(point, 1, 0)] = e01[point];
            left[(point, 1, 1)] = e111[point];
            right[(point, 0, 0)] = e112[point];
            right[(point, 0, 1)] = e10[point];
            right[(point, 1, 0)] = e10[point];
            right[(point, 1, 1)] = e002[point];
        }
        let mut side1 = Network::new(two_x_thru.frequency.clone(), left, split_reference.clone())?;
        let mut side2 = Network::new(two_x_thru.frequency.clone(), right, split_reference)?;
        side1.renormalize(two_x_thru.z0.clone(), two_x_thru.s_definition)?;
        side2.renormalize(two_x_thru.z0.clone(), two_x_thru.s_definition)?;
        Ok((side1, side2.flipped()?))
    }
}

fn ieee_p370_real_impulse(spectrum: &[Complex64]) -> Result<Vec<f64>> {
    if spectrum.len() < 2 {
        return Err(Error::InvalidFrequency(
            "IEEE 370 real impulse reconstruction requires at least two bins".to_owned(),
        ));
    }
    let mut spectrum = spectrum.to_vec();
    spectrum[0] = Complex64::new(spectrum[0].re, 0.0);
    if let Some(last) = spectrum.last_mut() {
        *last = Complex64::new(last.re, 0.0);
    }
    let output_length = 2 * (spectrum.len() - 1);
    let mut impulse = crate::time::irfft(&Array1::from_vec(spectrum), output_length)?.to_vec();
    let split = impulse.len().div_ceil(2);
    impulse.rotate_left(split);
    Ok(impulse)
}

fn ieee_p370_impedance_profile(network: &Network, port: usize) -> Result<Vec<f64>> {
    let with_dc = IeeeP370::extrapolate_to_dc(network)?;
    let reflection = (0..with_dc.frequency_points())
        .map(|point| with_dc.s[(point, port, port)])
        .collect::<Vec<_>>();
    let impulse = ieee_p370_real_impulse(&reflection)?;
    let reference = network.z0[(0, port)].re;
    let mut cumulative = 0.0;
    impulse
        .into_iter()
        .map(|value| {
            cumulative += value;
            let denominator = 1.0 - cumulative;
            if denominator.abs() <= f64::EPSILON {
                Err(Error::Unsupported(
                    "IEEE 370 impedance profile is singular".to_owned(),
                ))
            } else {
                Ok(reference * (1.0 + cumulative) / denominator)
            }
        })
        .collect()
}

fn ieee_p370_gate_reflection(
    network: &Network,
    port: usize,
    midpoint: usize,
) -> Result<Array1<Complex64>> {
    let with_dc = IeeeP370::extrapolate_to_dc(network)?;
    let reflection = (0..with_dc.frequency_points())
        .map(|point| with_dc.s[(point, port, port)])
        .collect::<Vec<_>>();
    let mut gated = ieee_p370_real_impulse(&reflection)?;
    for value in gated.iter_mut().skip(midpoint) {
        *value = 0.0;
    }
    let split = gated.len() / 2;
    gated.rotate_left(split);
    let spectrum = calibration_fft(
        &gated
            .into_iter()
            .map(|value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>(),
        false,
    );
    Ok(Array1::from_iter(
        spectrum
            .into_iter()
            .skip(1)
            .take(network.frequency_points()),
    ))
}

impl Deembedding for IeeeP370SeNzc2xThru {
    fn deembed(&self, network: &Network) -> Result<Network> {
        let target = if network.frequency == self.two_x_thru.frequency {
            network.clone()
        } else {
            network.interpolate(&self.two_x_thru.frequency)?
        };
        self.side1
            .inverse()?
            .cascade(&target)?
            .cascade(&self.side2.flipped()?.inverse()?)
    }
}

/// IEEE 370 single-ended impedance-corrected 2x-thru fixture extraction.
///
/// Origin: `skrf/calibration/deembedding.py::IEEEP370_SE_ZC_2xThru`.
#[derive(Clone, Debug)]
pub struct IeeeP370SeZc2xThru {
    pub two_x_thru: Network,
    pub fixture_dut_fixture: Network,
    pub side1: Network,
    pub side2: Network,
    pub propagation: Array1<Complex64>,
    pub pullback1: usize,
    pub pullback2: usize,
    pub name: Option<String>,
}

impl IeeeP370SeZc2xThru {
    pub fn new(
        two_x_thru: Network,
        fixture_dut_fixture: Network,
        name: Option<String>,
    ) -> Result<Self> {
        Self::with_pullbacks(two_x_thru, fixture_dut_fixture, 0, 0, name)
    }

    pub fn with_pullbacks(
        two_x_thru: Network,
        fixture_dut_fixture: Network,
        pullback1: usize,
        pullback2: usize,
        name: Option<String>,
    ) -> Result<Self> {
        validate_two_port_dummy(&two_x_thru)?;
        validate_two_port_dummy(&fixture_dut_fixture)?;
        if two_x_thru.frequency_points() < 2 {
            return Err(Error::InvalidFrequency(
                "IEEE 370 ZC extraction requires at least two frequency samples".to_owned(),
            ));
        }
        let original_two_x_thru = two_x_thru;
        let original_frequency = original_two_x_thru.frequency.clone();
        let aligned_fixture = if fixture_dut_fixture.frequency == original_frequency {
            fixture_dut_fixture
        } else {
            fixture_dut_fixture.interpolate(&original_frequency)?
        };
        let original_values = original_frequency.values_hz();
        let has_dc = original_values[0] == 0.0;
        let mut working_two_x_thru = if has_dc {
            if original_values.len() < 3 {
                return Err(Error::InvalidFrequency(
                    "IEEE 370 ZC extraction requires two non-DC samples".to_owned(),
                ));
            }
            let frequency =
                Frequency::from_hz(Array1::from_iter(original_values.iter().skip(1).copied()))?;
            original_two_x_thru.interpolate(&frequency)?
        } else {
            original_two_x_thru.clone()
        };
        let mut working_fixture = if has_dc {
            aligned_fixture.interpolate(&working_two_x_thru.frequency)?
        } else {
            aligned_fixture.clone()
        };
        let working_values = working_two_x_thru.frequency.values_hz();
        let working_step = working_values[1] - working_values[0];
        let harmonic =
            (working_step - working_values[0]).abs() <= 1.0e-6 * working_values[0].abs().max(1.0);
        let mut interpolated = false;
        if !harmonic {
            let projected_points = (working_values[working_values.len() - 1] / working_values[0])
                .round()
                .clamp(2.0, 10_000.0) as usize;
            let step = working_values[working_values.len() - 1] / projected_points as f64;
            let frequency = Frequency::from_hz(Array1::from_iter(
                (1..=projected_points).map(|index| step * index as f64),
            ))?;
            working_two_x_thru = working_two_x_thru.interpolate(&frequency)?;
            working_fixture = working_fixture.interpolate(&frequency)?;
            interpolated = true;
        }
        if has_dc || interpolated {
            let extracted = Self::with_pullbacks(
                working_two_x_thru,
                working_fixture,
                pullback1,
                pullback2,
                name.clone(),
            )?;
            let mut side1 = extracted.side1;
            let mut side2 = extracted.side2;
            if has_dc {
                side1 = IeeeP370::add_dc(&side1)?;
                side2 = IeeeP370::add_dc(&side2)?;
            }
            side1 = side1.interpolate(&original_frequency)?;
            side2 = side2.interpolate(&original_frequency)?;
            return Ok(Self {
                propagation: ieee_p370_zc_propagation(&original_two_x_thru)?,
                two_x_thru: original_two_x_thru,
                fixture_dut_fixture: aligned_fixture,
                side1,
                side2,
                pullback1,
                pullback2,
                name,
            });
        }
        let propagation = ieee_p370_zc_propagation(&original_two_x_thru)?;
        let (side1, side2) = ieee_p370_zc_split(
            &original_two_x_thru,
            &aligned_fixture,
            &propagation,
            pullback1,
            pullback2,
        )?;
        Ok(Self {
            two_x_thru: original_two_x_thru,
            fixture_dut_fixture: aligned_fixture,
            side1,
            side2,
            propagation,
            pullback1,
            pullback2,
            name,
        })
    }
}

impl Deembedding for IeeeP370SeZc2xThru {
    fn deembed(&self, network: &Network) -> Result<Network> {
        let target = if network.frequency == self.two_x_thru.frequency {
            network.clone()
        } else {
            network.interpolate(&self.two_x_thru.frequency)?
        };
        self.side1
            .inverse()?
            .cascade(&target)?
            .cascade(&self.side2.flipped()?.inverse()?)
    }
}

fn ieee_p370_zc_propagation(two_x_thru: &Network) -> Result<Array1<Complex64>> {
    let phase = Array1::from_iter(
        (0..two_x_thru.frequency_points()).map(|point| two_x_thru.s[(point, 1, 0)].arg()),
    );
    let phase = crate::math::unwrap_radians(&phase);
    let mut propagation = Array1::zeros(two_x_thru.frequency_points());
    for point in 0..two_x_thru.frequency_points() {
        let transmission = two_x_thru.s[(point, 1, 0)].norm_sqr();
        let reflection = two_x_thru.s[(point, 0, 0)].norm_sqr();
        let available = 1.0 - reflection;
        if available <= f64::EPSILON || transmission <= f64::EPSILON {
            return Err(Error::Unsupported(
                "IEEE 370 ZC propagation cannot be recovered from a singular 2x-thru".to_owned(),
            ));
        }
        let attenuation = transmission / available;
        let alpha = 10.0 * attenuation.log10() / -8.686;
        propagation[point] = Complex64::new(alpha, -phase[point]);
    }
    Ok(propagation)
}

fn ieee_p370_zc_split(
    two_x_thru: &Network,
    fixture_dut_fixture: &Network,
    propagation: &Array1<Complex64>,
    pullback1: usize,
    pullback2: usize,
) -> Result<(Network, Network)> {
    let with_dc = IeeeP370::extrapolate_to_dc(two_x_thru)?;
    let mut transmission =
        Array1::from_iter((0..with_dc.frequency_points()).map(|point| with_dc.s[(point, 1, 0)]));
    if let Some(last) = transmission.last_mut() {
        *last = Complex64::new(last.re, 0.0);
    }
    let impulse = crate::time::irfft(
        &transmission,
        two_x_thru.frequency_points().saturating_mul(2),
    )?;
    let midpoint = impulse
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let side1_segments = midpoint.saturating_sub(pullback1).saturating_add(1);
    let side2_segments = midpoint.saturating_sub(pullback2).saturating_add(1);
    let segment_length = 1.0 / (2.0 * (midpoint + 1) as f64);
    let reference_impedance = fixture_dut_fixture.z0[(0, 0)].re;
    if !reference_impedance.is_finite() || reference_impedance <= 0.0 {
        return Err(Error::Unsupported(
            "IEEE 370 ZC extraction requires a positive real reference impedance".to_owned(),
        ));
    }

    let mut remaining = fixture_dut_fixture.clone();
    let mut side1 = IeeeP370::thru(fixture_dut_fixture)?;
    let mut side2 = IeeeP370::thru(fixture_dut_fixture)?;
    for segment in 0..side1_segments.max(side2_segments) {
        let left = if segment < side1_segments {
            ieee_p370_zc_segment(
                &remaining,
                0,
                reference_impedance,
                propagation,
                segment_length,
            )?
        } else {
            IeeeP370::thru(fixture_dut_fixture)?
        };
        let right = if segment < side2_segments {
            ieee_p370_zc_segment(
                &remaining,
                1,
                reference_impedance,
                propagation,
                segment_length,
            )?
        } else {
            IeeeP370::thru(fixture_dut_fixture)?
        };
        if segment < side1_segments {
            side1 = side1.cascade(&left)?;
        }
        if segment < side2_segments {
            side2 = side2.cascade(&right)?;
        }
        remaining = left
            .inverse()?
            .cascade(&remaining)?
            .cascade(&right.inverse()?)?;
    }
    Ok((side1, side2.flipped()?))
}

fn ieee_p370_zc_segment(
    network: &Network,
    port: usize,
    reference_impedance: f64,
    propagation: &Array1<Complex64>,
    segment_length: f64,
) -> Result<Network> {
    let reflection = (0..network.frequency_points())
        .map(|point| network.s[(point, port, port)])
        .collect::<Vec<_>>();
    let line_impedance = ieee_p370_initial_impedance(&reflection, reference_impedance)?;
    let scattering = IeeeP370::make_transmission_line(
        line_impedance,
        reference_impedance,
        &propagation.to_vec(),
        segment_length,
    )?;
    let mut segment = network.clone();
    segment.s = scattering;
    Ok(segment)
}

fn ieee_p370_initial_impedance(reflection: &[Complex64], reference_impedance: f64) -> Result<f64> {
    if reflection.len() < 2 {
        return Err(Error::InvalidFrequency(
            "IEEE 370 impedance reconstruction requires at least two samples".to_owned(),
        ));
    }
    let mut spectrum = Vec::with_capacity(reflection.len() + 1);
    spectrum.push(Complex64::new(reflection[0].re, 0.0));
    spectrum.extend_from_slice(reflection);
    if let Some(last) = spectrum.last_mut() {
        *last = Complex64::new(last.re, 0.0);
    }
    let impulse = crate::time::irfft(
        &Array1::from_vec(spectrum),
        reflection.len().saturating_mul(2),
    )?;
    let split = impulse.len() / 2;
    let mut cumulative = 0.0;
    let mut step_at_zero = 0.0;
    for (index, value) in impulse
        .iter()
        .skip(split)
        .chain(impulse.iter().take(split))
        .enumerate()
    {
        cumulative += value;
        if index == split {
            step_at_zero = cumulative;
            break;
        }
    }
    let denominator = 1.0 - step_at_zero;
    if denominator.abs() <= f64::EPSILON {
        return Err(Error::Unsupported(
            "IEEE 370 time-domain impedance is singular".to_owned(),
        ));
    }
    let impedance = reference_impedance * (1.0 + step_at_zero) / denominator;
    if !impedance.is_finite() || impedance <= 0.0 {
        return Err(Error::Unsupported(
            "IEEE 370 time-domain impedance is not positive and finite".to_owned(),
        ));
    }
    Ok(impedance)
}

/// Port ordering accepted by the IEEE 370 mixed-mode fixture algorithms.
///
/// Origin: `skrf/calibration/deembedding.py::PortOrderT`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IeeeP370PortOrder {
    /// Front-to-back (odd/even): left ports 0/2 and right ports 1/3.
    First,
    /// Left-to-right (sequential): left ports 0/1 and right ports 2/3.
    #[default]
    Second,
    /// Left ports 0/1 and reversed right ports 3/2.
    Third,
}

/// IEEE 370 mixed-mode non-zero-crossing 2x-thru fixture extraction.
///
/// The fixture is transformed to differential/common mode, each uncoupled
/// modal fixture is split with the single-ended NZC algorithm, and the two
/// modal error boxes are recombined. De-embedding uses balanced multiport
/// connections, preserving differential/common conversion terms in the DUT.
///
/// Origin: `skrf/calibration/deembedding.py::IEEEP370_MM_NZC_2xThru`.
#[derive(Clone, Debug)]
pub struct IeeeP370MmNzc2xThru {
    pub two_x_thru: Network,
    pub side1: Network,
    pub side2: Network,
    pub port_order: IeeeP370PortOrder,
    pub name: Option<String>,
    differential_side1: Network,
    differential_side2: Network,
    common_side1: Network,
    common_side2: Network,
}

impl IeeeP370MmNzc2xThru {
    pub fn new(
        two_x_thru: Network,
        port_order: IeeeP370PortOrder,
        name: Option<String>,
    ) -> Result<Self> {
        if two_x_thru.ports() != 4 {
            return Err(Error::IncompatibleShape(
                "IEEE 370 mixed-mode 2x-thru must be a four-port network".to_owned(),
            ));
        }
        let normalized = ieee_p370_to_second_port_order(&two_x_thru, port_order)?;
        let mixed_mode = normalized.single_ended_to_mixed_mode(2)?;
        let differential = IeeeP370SeNzc2xThru::new(
            mixed_mode.subnetwork(&[0, 1])?,
            name.as_ref().map(|value| format!("{value}-differential")),
        )?;
        let common = IeeeP370SeNzc2xThru::new(
            mixed_mode.subnetwork(&[2, 3])?,
            name.as_ref().map(|value| format!("{value}-common")),
        )?;
        let side1_mixed = concatenate_ports(&[differential.side1.clone(), common.side1.clone()])?;
        let side2_mixed = concatenate_ports(&[differential.side2.clone(), common.side2.clone()])?;
        let side1 = ieee_p370_from_second_port_order(
            &side1_mixed.mixed_mode_to_single_ended(2)?,
            port_order,
        )?;
        let side2 = ieee_p370_from_second_port_order(
            &side2_mixed.mixed_mode_to_single_ended(2)?,
            port_order,
        )?;
        Ok(Self {
            two_x_thru,
            side1,
            side2,
            port_order,
            name,
            differential_side1: differential.side1,
            differential_side2: differential.side2,
            common_side1: common.side1,
            common_side2: common.side2,
        })
    }

    pub fn new_time_gated(
        two_x_thru: Network,
        port_order: IeeeP370PortOrder,
        forced_differential_impedance: Option<f64>,
        forced_common_impedance: Option<f64>,
        name: Option<String>,
    ) -> Result<Self> {
        if two_x_thru.ports() != 4 {
            return Err(Error::IncompatibleShape(
                "IEEE 370 mixed-mode 2x-thru must be a four-port network".to_owned(),
            ));
        }
        let normalized = ieee_p370_to_second_port_order(&two_x_thru, port_order)?;
        let mixed_mode = normalized.single_ended_to_mixed_mode(2)?;
        let differential = IeeeP370SeNzc2xThru::new_time_gated(
            mixed_mode.subnetwork(&[0, 1])?,
            forced_differential_impedance,
            name.as_ref().map(|value| format!("{value}-differential")),
        )?;
        let common = IeeeP370SeNzc2xThru::new_time_gated(
            mixed_mode.subnetwork(&[2, 3])?,
            forced_common_impedance,
            name.as_ref().map(|value| format!("{value}-common")),
        )?;
        let side1_mixed = concatenate_ports(&[differential.side1.clone(), common.side1.clone()])?;
        let side2_mixed = concatenate_ports(&[differential.side2.clone(), common.side2.clone()])?;
        let side1 = ieee_p370_from_second_port_order(
            &side1_mixed.mixed_mode_to_single_ended(2)?,
            port_order,
        )?;
        let side2 = ieee_p370_from_second_port_order(
            &side2_mixed.mixed_mode_to_single_ended(2)?,
            port_order,
        )?;
        Ok(Self {
            two_x_thru,
            side1,
            side2,
            port_order,
            name,
            differential_side1: differential.side1,
            differential_side2: differential.side2,
            common_side1: common.side1,
            common_side2: common.side2,
        })
    }
}

impl Deembedding for IeeeP370MmNzc2xThru {
    fn deembed(&self, network: &Network) -> Result<Network> {
        if network.ports() != 4 {
            return Err(Error::IncompatibleShape(
                "IEEE 370 mixed-mode de-embedding requires a four-port network".to_owned(),
            ));
        }
        let target = if network.frequency == self.two_x_thru.frequency {
            network.clone()
        } else {
            network.interpolate(&self.two_x_thru.frequency)?
        };
        let target = ieee_p370_to_second_port_order(&target, self.port_order)?
            .single_ended_to_mixed_mode(2)?;
        let left_inverse = concatenate_ports(&[
            self.differential_side1.inverse()?,
            self.common_side1.inverse()?,
        ])?;
        let right_inverse = concatenate_ports(&[
            self.differential_side2.flipped()?.inverse()?,
            self.common_side2.flipped()?.inverse()?,
        ])?;
        let corrected = cascade_ieee_p370_modes(
            &cascade_ieee_p370_modes(&left_inverse, &target)?,
            &right_inverse,
        )?
        .mixed_mode_to_single_ended(2)?;
        ieee_p370_from_second_port_order(&corrected, self.port_order)
    }
}

/// IEEE 370 mixed-mode impedance-corrected 2x-thru fixture extraction.
///
/// Origin: `skrf/calibration/deembedding.py::IEEEP370_MM_ZC_2xThru`.
#[derive(Clone, Debug)]
pub struct IeeeP370MmZc2xThru {
    pub two_x_thru: Network,
    pub fixture_dut_fixture: Network,
    pub side1: Network,
    pub side2: Network,
    pub port_order: IeeeP370PortOrder,
    pub name: Option<String>,
    differential: IeeeP370SeZc2xThru,
    common: IeeeP370SeZc2xThru,
}

impl IeeeP370MmZc2xThru {
    pub fn new(
        two_x_thru: Network,
        fixture_dut_fixture: Network,
        port_order: IeeeP370PortOrder,
        name: Option<String>,
    ) -> Result<Self> {
        if two_x_thru.ports() != 4 || fixture_dut_fixture.ports() != 4 {
            return Err(Error::IncompatibleShape(
                "IEEE 370 mixed-mode ZC extraction requires four-port networks".to_owned(),
            ));
        }
        let aligned_fixture = if fixture_dut_fixture.frequency == two_x_thru.frequency {
            fixture_dut_fixture
        } else {
            fixture_dut_fixture.interpolate(&two_x_thru.frequency)?
        };
        let mixed_two_x_thru = ieee_p370_to_second_port_order(&two_x_thru, port_order)?
            .single_ended_to_mixed_mode(2)?;
        let mixed_fixture = ieee_p370_to_second_port_order(&aligned_fixture, port_order)?
            .single_ended_to_mixed_mode(2)?;
        let differential = IeeeP370SeZc2xThru::new(
            mixed_two_x_thru.subnetwork(&[0, 1])?,
            mixed_fixture.subnetwork(&[0, 1])?,
            name.as_ref().map(|value| format!("{value}-differential")),
        )?;
        let common = IeeeP370SeZc2xThru::new(
            mixed_two_x_thru.subnetwork(&[2, 3])?,
            mixed_fixture.subnetwork(&[2, 3])?,
            name.as_ref().map(|value| format!("{value}-common")),
        )?;
        let side1_mixed = concatenate_ports(&[differential.side1.clone(), common.side1.clone()])?;
        let side2_mixed = concatenate_ports(&[differential.side2.clone(), common.side2.clone()])?;
        let side1 = ieee_p370_from_second_port_order(
            &side1_mixed.mixed_mode_to_single_ended(2)?,
            port_order,
        )?;
        let side2 = ieee_p370_from_second_port_order(
            &side2_mixed.mixed_mode_to_single_ended(2)?,
            port_order,
        )?;
        Ok(Self {
            two_x_thru,
            fixture_dut_fixture: aligned_fixture,
            side1,
            side2,
            port_order,
            name,
            differential,
            common,
        })
    }
}

impl Deembedding for IeeeP370MmZc2xThru {
    fn deembed(&self, network: &Network) -> Result<Network> {
        if network.ports() != 4 {
            return Err(Error::IncompatibleShape(
                "IEEE 370 mixed-mode ZC de-embedding requires a four-port network".to_owned(),
            ));
        }
        let target = if network.frequency == self.two_x_thru.frequency {
            network.clone()
        } else {
            network.interpolate(&self.two_x_thru.frequency)?
        };
        let target = ieee_p370_to_second_port_order(&target, self.port_order)?
            .single_ended_to_mixed_mode(2)?;
        let left_inverse = concatenate_ports(&[
            self.differential.side1.inverse()?,
            self.common.side1.inverse()?,
        ])?;
        let right_inverse = concatenate_ports(&[
            self.differential.side2.flipped()?.inverse()?,
            self.common.side2.flipped()?.inverse()?,
        ])?;
        let corrected = cascade_ieee_p370_modes(
            &cascade_ieee_p370_modes(&left_inverse, &target)?,
            &right_inverse,
        )?
        .mixed_mode_to_single_ended(2)?;
        ieee_p370_from_second_port_order(&corrected, self.port_order)
    }
}

fn cascade_ieee_p370_modes(left: &Network, right: &Network) -> Result<Network> {
    let cascade_order = [0, 2, 1, 3];
    cascade_balanced(
        &left.renumbered(&cascade_order)?,
        &right.renumbered(&cascade_order)?,
    )?
    .renumbered(&cascade_order)
}

fn cascade_balanced(left: &Network, right: &Network) -> Result<Network> {
    if left.frequency != right.frequency
        || left.ports() != right.ports()
        || left.ports() < 2
        || left.ports() % 2 != 0
    {
        return Err(Error::IncompatibleShape(
            "balanced cascade requires equal, even port counts and matching frequencies".to_owned(),
        ));
    }
    if left.ports() == 2 {
        return left.cascade(right);
    }
    let half = left.ports() / 2;
    let mut connected = left.connect(half, right, 0)?;
    for remaining in (1..half).rev() {
        connected = connected.inner_connect(half, half + remaining)?;
    }
    Ok(connected)
}

fn ieee_p370_to_second_port_order(
    network: &Network,
    port_order: IeeeP370PortOrder,
) -> Result<Network> {
    match port_order {
        IeeeP370PortOrder::First => network.renumbered(&[0, 2, 1, 3]),
        IeeeP370PortOrder::Second => Ok(network.clone()),
        IeeeP370PortOrder::Third => network.renumbered(&[0, 1, 3, 2]),
    }
}

fn ieee_p370_from_second_port_order(
    network: &Network,
    port_order: IeeeP370PortOrder,
) -> Result<Network> {
    ieee_p370_to_second_port_order(network, port_order)
}

/// Origin: `skrf/calibration/deembedding.py::Open`.
#[derive(Clone, Debug)]
pub struct Open {
    pub open: Network,
    pub name: Option<String>,
}

impl Open {
    pub fn new(open: Network, name: Option<String>) -> Self {
        Self { open, name }
    }
}

impl Deembedding for Open {
    fn deembed(&self, network: &Network) -> Result<Network> {
        let open = align_dummy(&self.open, network)?;
        let corrected = s_to_y(&network.s, &network.z0, network.s_definition)?
            - s_to_y(&open.s, &open.z0, open.s_definition)?;
        network_from_y(network, corrected)
    }
}

/// Origin: `skrf/calibration/deembedding.py::Short`.
#[derive(Clone, Debug)]
pub struct Short {
    pub short: Network,
    pub name: Option<String>,
}

impl Short {
    pub fn new(short: Network, name: Option<String>) -> Self {
        Self { short, name }
    }
}

impl Deembedding for Short {
    fn deembed(&self, network: &Network) -> Result<Network> {
        let short = align_dummy(&self.short, network)?;
        let corrected = s_to_z(&network.s, &network.z0, network.s_definition)?
            - s_to_z(&short.s, &short.z0, short.s_definition)?;
        network_from_z(network, corrected)
    }
}

/// Origin: `skrf/calibration/deembedding.py::OpenShort`.
#[derive(Clone, Debug)]
pub struct OpenShort {
    pub open: Network,
    pub short: Network,
    pub name: Option<String>,
}

impl OpenShort {
    pub fn new(open: Network, short: Network, name: Option<String>) -> Result<Self> {
        validate_dummy_pair(&open, &short)?;
        Ok(Self { open, short, name })
    }
}

impl Deembedding for OpenShort {
    fn deembed(&self, network: &Network) -> Result<Network> {
        let open = align_dummy(&self.open, network)?;
        let short = align_dummy(&self.short, network)?;
        let open_y = s_to_y(&open.s, &open.z0, open.s_definition)?;
        let deembedded_short_y = s_to_y(&short.s, &short.z0, short.s_definition)? - &open_y;
        let deembedded_short_z = y_to_z(&deembedded_short_y)?;
        let parallel_corrected_y = s_to_y(&network.s, &network.z0, network.s_definition)? - open_y;
        let corrected_z = y_to_z(&parallel_corrected_y)? - deembedded_short_z;
        network_from_z(network, corrected_z)
    }
}

/// Origin: `skrf/calibration/deembedding.py::ShortOpen`.
#[derive(Clone, Debug)]
pub struct ShortOpen {
    pub short: Network,
    pub open: Network,
    pub name: Option<String>,
}

impl ShortOpen {
    pub fn new(short: Network, open: Network, name: Option<String>) -> Result<Self> {
        validate_dummy_pair(&short, &open)?;
        Ok(Self { short, open, name })
    }
}

impl Deembedding for ShortOpen {
    fn deembed(&self, network: &Network) -> Result<Network> {
        let short = align_dummy(&self.short, network)?;
        let open = align_dummy(&self.open, network)?;
        let short_z = s_to_z(&short.s, &short.z0, short.s_definition)?;
        let deembedded_open_z = s_to_z(&open.s, &open.z0, open.s_definition)? - &short_z;
        let deembedded_open_y = z_to_y(&deembedded_open_z)?;
        let series_corrected_z = s_to_z(&network.s, &network.z0, network.s_definition)? - short_z;
        let corrected_y = z_to_y(&series_corrected_z)? - deembedded_open_y;
        network_from_y(network, corrected_y)
    }
}

/// Origin: `skrf/calibration/deembedding.py::SplitPi`.
#[derive(Clone, Debug)]
pub struct SplitPi {
    pub thru: Network,
    pub name: Option<String>,
}

impl SplitPi {
    pub fn new(thru: Network, name: Option<String>) -> Result<Self> {
        validate_two_port_dummy(&thru)?;
        Ok(Self { thru, name })
    }
}

impl Deembedding for SplitPi {
    fn deembed(&self, network: &Network) -> Result<Network> {
        let thru = align_dummy(&self.thru, network)?;
        validate_two_port_dummy(&thru)?;
        let y = s_to_y(&thru.s, &thru.z0, thru.s_definition)?;
        let mut left_y = y.clone();
        for point in 0..thru.frequency_points() {
            left_y[(point, 0, 0)] =
                (y[(point, 0, 0)] - y[(point, 1, 0)] + y[(point, 1, 1)] - y[(point, 0, 1)]) / 2.0;
            left_y[(point, 0, 1)] = y[(point, 1, 0)] + y[(point, 0, 1)];
            left_y[(point, 1, 0)] = left_y[(point, 0, 1)];
            left_y[(point, 1, 1)] = -left_y[(point, 0, 1)];
        }
        let left = network_from_y(&thru, left_y)?;
        let right = left.flipped()?;
        left.inverse()?.cascade(network)?.cascade(&right.inverse()?)
    }
}

/// Origin: `skrf/calibration/deembedding.py::SplitTee`.
#[derive(Clone, Debug)]
pub struct SplitTee {
    pub thru: Network,
    pub name: Option<String>,
}

impl SplitTee {
    pub fn new(thru: Network, name: Option<String>) -> Result<Self> {
        validate_two_port_dummy(&thru)?;
        Ok(Self { thru, name })
    }
}

impl Deembedding for SplitTee {
    fn deembed(&self, network: &Network) -> Result<Network> {
        let thru = align_dummy(&self.thru, network)?;
        validate_two_port_dummy(&thru)?;
        let z = s_to_z(&thru.s, &thru.z0, thru.s_definition)?;
        let mut left_z = z.clone();
        for point in 0..thru.frequency_points() {
            left_z[(point, 0, 0)] =
                (z[(point, 0, 0)] + z[(point, 1, 0)] + z[(point, 1, 1)] + z[(point, 0, 1)]) / 2.0;
            left_z[(point, 0, 1)] = z[(point, 1, 0)] + z[(point, 0, 1)];
            left_z[(point, 1, 0)] = left_z[(point, 0, 1)];
            left_z[(point, 1, 1)] = left_z[(point, 0, 1)];
        }
        let left = network_from_z(&thru, left_z)?;
        let right = left.flipped()?;
        left.inverse()?.cascade(network)?.cascade(&right.inverse()?)
    }
}

/// Origin: `skrf/calibration/deembedding.py::AdmittanceCancel`.
#[derive(Clone, Debug)]
pub struct AdmittanceCancel {
    pub thru: Network,
    pub name: Option<String>,
}

impl AdmittanceCancel {
    pub fn new(thru: Network, name: Option<String>) -> Result<Self> {
        validate_two_port_dummy(&thru)?;
        Ok(Self { thru, name })
    }
}

impl Deembedding for AdmittanceCancel {
    fn deembed(&self, network: &Network) -> Result<Network> {
        let thru = align_dummy(&self.thru, network)?;
        let half = network.cascade(&thru.inverse()?)?;
        let flipped = half.flipped()?;
        let average = (s_to_y(&half.s, &half.z0, half.s_definition)?
            + s_to_y(&flipped.s, &flipped.z0, flipped.s_definition)?)
            / 2.0;
        network_from_y(network, average)
    }
}

/// Origin: `skrf/calibration/deembedding.py::ImpedanceCancel`.
#[derive(Clone, Debug)]
pub struct ImpedanceCancel {
    pub thru: Network,
    pub name: Option<String>,
}

impl ImpedanceCancel {
    pub fn new(thru: Network, name: Option<String>) -> Result<Self> {
        validate_two_port_dummy(&thru)?;
        Ok(Self { thru, name })
    }
}

impl Deembedding for ImpedanceCancel {
    fn deembed(&self, network: &Network) -> Result<Network> {
        let thru = align_dummy(&self.thru, network)?;
        let half = network.cascade(&thru.inverse()?)?;
        let flipped = half.flipped()?;
        let average = (s_to_z(&half.s, &half.z0, half.s_definition)?
            + s_to_z(&flipped.s, &flipped.z0, flipped.s_definition)?)
            / 2.0;
        network_from_z(network, average)
    }
}

fn align_dummy(dummy: &Network, target: &Network) -> Result<Network> {
    if dummy.ports() != target.ports() {
        return Err(Error::IncompatibleShape(format!(
            "dummy has {} ports but target has {}",
            dummy.ports(),
            target.ports()
        )));
    }
    if dummy.frequency == target.frequency {
        Ok(dummy.clone())
    } else {
        dummy.interpolate(&target.frequency)
    }
}

fn validate_dummy_pair(first: &Network, second: &Network) -> Result<()> {
    if first.ports() != second.ports() {
        return Err(Error::IncompatibleShape(
            "de-embedding dummies must have the same number of ports".to_owned(),
        ));
    }
    if first.frequency != second.frequency {
        return Err(Error::InvalidFrequency(
            "de-embedding dummies must share a frequency axis".to_owned(),
        ));
    }
    Ok(())
}

fn validate_two_port_dummy(dummy: &Network) -> Result<()> {
    if dummy.ports() != 2 {
        return Err(Error::IncompatibleShape(
            "this de-embedding method requires a two-port dummy".to_owned(),
        ));
    }
    Ok(())
}

fn network_from_y(template: &Network, admittance: Array3<Complex64>) -> Result<Network> {
    let mut corrected = template.clone();
    corrected.s = y_to_s(&admittance, &corrected.z0, corrected.s_definition)?;
    Ok(corrected)
}

fn network_from_z(template: &Network, impedance: Array3<Complex64>) -> Result<Network> {
    let mut corrected = template.clone();
    corrected.s = z_to_s(&impedance, &corrected.z0, corrected.s_definition)?;
    Ok(corrected)
}
