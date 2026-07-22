//! Ideal microwave-device network generators.

use ndarray::Array1;
use num_complex::Complex64;

use super::Media;
use crate::{Error, Network, Result};

/// Common behavior of microwave-device generators.
pub trait Device {
    /// Return the network representation of the device.
    ///
    /// # Errors
    ///
    /// Returns an error if the device network cannot be constructed.
    fn network(&self) -> Result<Network>;
}

/// Ideal reciprocal matched symmetric directional coupler.
///
/// Port assignment is insertion, transmit, coupled, and isolated. A three-port
/// device terminates the fourth port in a match.
#[derive(Clone, Debug)]
pub struct MatchedSymmetricCoupler<M> {
    /// Medium defining frequency and reference impedance.
    pub media: M,
    /// Complex coupled-arm transmission coefficient.
    pub coupling: Array1<Complex64>,
    /// Complex through-arm transmission coefficient.
    pub transmission: Array1<Complex64>,
    /// Complex isolated-arm leakage coefficient.
    pub isolation: Array1<Complex64>,
    /// Exposed port count, either three or four.
    pub ports: usize,
}

impl<M: Media> MatchedSymmetricCoupler<M> {
    /// Construct a matched symmetric coupler from coupling or transmission.
    ///
    /// The missing magnitude is derived from $|c|^2+|t|^2=1$. Phase arguments
    /// are in degrees.
    ///
    /// # Errors
    ///
    /// Returns an error if `ports` is not three or four, neither coefficient is
    /// supplied, or a supplied coefficient array has an incompatible length or
    /// contains an invalid value.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        media: M,
        coupling: Option<Array1<Complex64>>,
        transmission: Option<Array1<Complex64>>,
        transmission_phase_degrees: f64,
        phase_difference_degrees: f64,
        ports: usize,
    ) -> Result<Self> {
        if ports != 3 && ports != 4 {
            return Err(Error::Unsupported(
                "a matched symmetric coupler must have three or four ports".to_owned(),
            ));
        }
        if coupling.is_none() && transmission.is_none() {
            return Err(Error::Unsupported(
                "coupling or transmission must be supplied".to_owned(),
            ));
        }
        let points = media.frequency().points();
        let transmission_phase =
            Complex64::from_polar(1.0, transmission_phase_degrees.to_radians());
        let coupling_phase = Complex64::from_polar(
            1.0,
            (transmission_phase_degrees + phase_difference_degrees).to_radians(),
        );
        let (coupling, transmission) = if let Some(coupling) = coupling {
            validate_values(&coupling, points, "coupling")?;
            let transmission = coupling.mapv(|value| {
                Complex64::new((1.0 - value.norm_sqr()).max(0.0).sqrt(), 0.0) * transmission_phase
            });
            (coupling.mapv(|value| value * coupling_phase), transmission)
        } else {
            let transmission = transmission.ok_or_else(|| {
                Error::Unsupported("coupling or transmission must be supplied".to_owned())
            })?;
            validate_values(&transmission, points, "transmission")?;
            let coupling = transmission.mapv(|value| {
                Complex64::new((1.0 - value.norm_sqr()).max(0.0).sqrt(), 0.0) * coupling_phase
            });
            (
                coupling,
                transmission.mapv(|value| value * transmission_phase),
            )
        };
        Ok(Self {
            media,
            coupling,
            transmission,
            isolation: Array1::zeros(points),
            ports,
        })
    }

    /// Construct a zero-phase coupler from a linear coupling magnitude.
    ///
    /// # Errors
    ///
    /// Returns an error if `ports` is not three or four or `coupling` is not
    /// finite or has a magnitude greater than one.
    pub fn from_coupling(media: M, coupling: f64, ports: usize) -> Result<Self> {
        let points = media.frequency().points();
        Self::new(
            media,
            Some(Array1::from_elem(points, Complex64::new(coupling, 0.0))),
            None,
            0.0,
            0.0,
            ports,
        )
    }

    /// Construct a coupler from coupling in dB and phase offset in degrees.
    ///
    /// Coupling sign is ignored and converted as $10^{-|dB|/20}$.
    ///
    /// # Errors
    ///
    /// Returns an error if `ports` is not three or four or `coupling_db`
    /// converts to an invalid coupling magnitude.
    pub fn from_db_degrees(
        media: M,
        coupling_db: f64,
        phase_difference_degrees: f64,
        ports: usize,
    ) -> Result<Self> {
        let magnitude = 10.0_f64.powf(-coupling_db.abs() / 20.0);
        let points = media.frequency().points();
        Self::new(
            media,
            Some(Array1::from_elem(points, Complex64::new(magnitude, 0.0))),
            None,
            0.0,
            phase_difference_degrees,
            ports,
        )
    }
}

impl<M: Media> Device for MatchedSymmetricCoupler<M> {
    fn network(&self) -> Result<Network> {
        let points = self.media.frequency().points();
        let mut network = self.media.match_network(4, None)?;
        for point in 0..points {
            let transmission = self.transmission[point];
            let coupling = self.coupling[point];
            let isolation = self.isolation[point];
            for (output, input) in [(0, 1), (1, 0), (3, 2), (2, 3)] {
                network.s[(point, output, input)] = transmission;
            }
            for (output, input) in [(0, 2), (2, 0), (3, 1), (1, 3)] {
                network.s[(point, output, input)] = coupling;
            }
            for (output, input) in [(0, 3), (3, 0), (1, 2), (2, 1)] {
                network.s[(point, output, input)] = isolation;
            }
        }
        if self.ports == 3 {
            network.connect(3, &self.media.match_network(1, None)?, 0)
        } else {
            Ok(network)
        }
    }
}

/// Ideal 3 dB coupler with configurable common and differential phase.
#[derive(Clone, Debug)]
pub struct Hybrid<M>(MatchedSymmetricCoupler<M>);

impl<M: Media> Hybrid<M> {
    /// Construct an equal-split four-port hybrid.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing matched symmetric coupler cannot be
    /// constructed.
    pub fn new(
        media: M,
        transmission_phase_degrees: f64,
        phase_difference_degrees: f64,
    ) -> Result<Self> {
        let points = media.frequency().points();
        Ok(Self(MatchedSymmetricCoupler::new(
            media,
            Some(Array1::from_elem(
                points,
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            )),
            None,
            transmission_phase_degrees,
            phase_difference_degrees,
            4,
        )?))
    }
}

impl<M: Media> Device for Hybrid<M> {
    fn network(&self) -> Result<Network> {
        self.0.network()
    }
}

/// Ideal 3 dB quadrature hybrid with a $-90^\circ$ coupled-arm offset.
#[derive(Clone, Debug)]
pub struct QuadratureHybrid<M>(MatchedSymmetricCoupler<M>);

impl<M: Media> QuadratureHybrid<M> {
    /// Construct a quadrature hybrid with a common transmission phase.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing matched symmetric coupler cannot be
    /// constructed.
    pub fn new(media: M, transmission_phase_degrees: f64) -> Result<Self> {
        let points = media.frequency().points();
        Ok(Self(MatchedSymmetricCoupler::new(
            media,
            Some(Array1::from_elem(
                points,
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            )),
            None,
            transmission_phase_degrees,
            -90.0,
            4,
        )?))
    }
}

impl<M: Media> Device for QuadratureHybrid<M> {
    fn network(&self) -> Result<Network> {
        self.0.network()
    }
}

/// Ideal 180-degree hybrid for in-phase/out-of-phase combining or division.
///
/// Port order is sum $(A+B)$, input A, input B, and difference $(A-B)$.
/// See [hybrid couplers](http://www.microwaves101.com/encyclopedias/hybrid-couplers).
#[derive(Clone, Debug)]
pub struct Hybrid180<M> {
    /// Medium defining frequency and reference impedance.
    pub media: M,
    /// Exposed port count, either three or four.
    pub ports: usize,
}

impl<M: Media> Hybrid180<M> {
    /// Construct a three- or four-port 180-degree hybrid.
    ///
    /// # Errors
    ///
    /// Returns an error if `ports` is not three or four.
    pub fn new(media: M, ports: usize) -> Result<Self> {
        if ports != 3 && ports != 4 {
            return Err(Error::Unsupported(
                "a 180-degree hybrid must have three or four ports".to_owned(),
            ));
        }
        Ok(Self { media, ports })
    }
}

impl<M: Media> Device for Hybrid180<M> {
    fn network(&self) -> Result<Network> {
        let mut network = self.media.match_network(4, None)?;
        let negative = Complex64::new(0.0, -std::f64::consts::FRAC_1_SQRT_2);
        let positive = Complex64::new(0.0, std::f64::consts::FRAC_1_SQRT_2);
        for point in 0..self.media.frequency().points() {
            for (output, input) in [(0, 1), (1, 0), (2, 0), (0, 2), (3, 2), (2, 3)] {
                network.s[(point, output, input)] = negative;
            }
            for (output, input) in [(3, 1), (1, 3)] {
                network.s[(point, output, input)] = positive;
            }
        }
        if self.ports == 3 {
            network.connect(3, &self.media.match_network(1, None)?, 0)
        } else {
            Ok(network)
        }
    }
}

/// Pair of back-to-back three-port directional couplers.
///
/// Ports are the insertion ports of couplers 1 and 2 followed by their
/// respective coupled ports.
#[derive(Clone, Debug)]
pub struct DualCoupler<M> {
    /// First directional coupler.
    pub first: MatchedSymmetricCoupler<M>,
    /// Second directional coupler.
    pub second: MatchedSymmetricCoupler<M>,
}

impl<M: Media + Clone> DualCoupler<M> {
    /// Construct a dual coupler; the second coupling defaults to the first.
    ///
    /// # Errors
    ///
    /// Returns an error if either coupling is not finite or has a magnitude
    /// greater than one.
    pub fn new(media: M, first_coupling: f64, second_coupling: Option<f64>) -> Result<Self> {
        Ok(Self {
            first: MatchedSymmetricCoupler::from_coupling(media.clone(), first_coupling, 3)?,
            second: MatchedSymmetricCoupler::from_coupling(
                media,
                second_coupling.unwrap_or(first_coupling),
                3,
            )?,
        })
    }
}

impl<M: Media> Device for DualCoupler<M> {
    fn network(&self) -> Result<Network> {
        self.first
            .network()?
            .connect(1, &self.second.network()?, 1)?
            .renumbered(&[0, 2, 1, 3])
    }
}

fn validate_values(values: &Array1<Complex64>, points: usize, name: &str) -> Result<()> {
    if values.len() != points {
        return Err(Error::IncompatibleShape(format!(
            "coupler {name} has {} values for {points} frequency points",
            values.len()
        )));
    }
    if values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite() || value.norm() > 1.0)
    {
        return Err(Error::Unsupported(format!(
            "coupler {name} values must be finite with magnitude no greater than one"
        )));
    }
    Ok(())
}
