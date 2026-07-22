//! Transmission lines defined by distributed impedance and admittance.

use std::fmt;
use std::path::Path;

use ndarray::Array1;
use num_complex::Complex64;

use crate::{Error, Frequency, Network, Result};

use super::media::{DefinedGammaZ0, LengthUnit, Media};
/// Transmission-line medium defined by distributed RLGC values.
///
/// | Quantity | Symbol | Field/method |
/// | --- | --- | --- |
/// | Resistance | $R'$ | [`Self::resistance_per_meter`] |
/// | Inductance | $L'$ | [`Self::inductance_per_meter`] |
/// | Conductance | $G'$ | [`Self::conductance_per_meter`] |
/// | Capacitance | $C'$ | [`Self::capacitance_per_meter`] |
/// | Impedance | $Z'=R'+j\omega L'$ | [`Self::distributed_impedance`] |
/// | Admittance | $Y'=G'+j\omega C'$ | [`Self::distributed_admittance`] |
#[derive(Clone, Debug)]
pub struct DistributedCircuit {
    /// Frequency band.
    pub frequency: Frequency,
    /// Distributed resistance $R'$ in ohms per meter.
    pub resistance_per_meter: Array1<f64>,
    /// Distributed conductance $G'$ in siemens per meter.
    pub conductance_per_meter: Array1<f64>,
    /// Distributed inductance $L'$ in henries per meter.
    pub inductance_per_meter: Array1<f64>,
    /// Distributed capacitance $C'$ in farads per meter.
    pub capacitance_per_meter: Array1<f64>,
    /// Optional port impedance used to renormalize generated networks.
    pub port_z0: Option<Array1<Complex64>>,
}

impl DistributedCircuit {
    /// Construct a distributed circuit from frequency-dependent RLGC arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if an RLGC or port-impedance array has the wrong length.
    pub fn new(
        frequency: Frequency,
        resistance_per_meter: Array1<f64>,
        conductance_per_meter: Array1<f64>,
        inductance_per_meter: Array1<f64>,
        capacitance_per_meter: Array1<f64>,
        port_z0: Option<Array1<Complex64>>,
    ) -> Result<Self> {
        let points = frequency.points();
        for (name, length) in [
            ("resistance", resistance_per_meter.len()),
            ("conductance", conductance_per_meter.len()),
            ("inductance", inductance_per_meter.len()),
            ("capacitance", capacitance_per_meter.len()),
        ] {
            if length != points {
                return Err(Error::IncompatibleShape(format!(
                    "distributed {name} has {length} values for {points} frequency points"
                )));
            }
        }
        if port_z0
            .as_ref()
            .is_some_and(|values| values.len() != points)
        {
            return Err(Error::IncompatibleShape(
                "distributed-circuit port impedance must match the frequency length".to_owned(),
            ));
        }
        Ok(Self {
            frequency,
            resistance_per_meter,
            conductance_per_meter,
            inductance_per_meter,
            capacitance_per_meter,
            port_z0,
        })
    }

    /// Construct a distributed circuit from scalar RLGC values.
    ///
    /// # Errors
    ///
    /// Returns an error if the generated RLGC arrays are incompatible with the frequency axis.
    pub fn from_scalars(
        frequency: Frequency,
        resistance_per_meter: f64,
        conductance_per_meter: f64,
        inductance_per_meter: f64,
        capacitance_per_meter: f64,
    ) -> Result<Self> {
        let points = frequency.points();
        Self::new(
            frequency,
            Array1::from_elem(points, resistance_per_meter),
            Array1::from_elem(points, conductance_per_meter),
            Array1::from_elem(points, inductance_per_meter),
            Array1::from_elem(points, capacitance_per_meter),
            None,
        )
    }

    /// Return distributed impedance $Z'=R'+j\omega L'$ in ohms per meter.
    #[must_use]
    pub fn distributed_impedance(&self) -> Array1<Complex64> {
        Array1::from_shape_fn(self.frequency.points(), |point| {
            let mut value = Complex64::new(
                self.resistance_per_meter[point],
                self.frequency.angular()[point] * self.inductance_per_meter[point],
            );
            if value.im == 0.0 {
                value.im = 1.0e-12;
            }
            value
        })
    }

    /// Return distributed admittance $Y'=G'+j\omega C'$ in siemens per meter.
    #[must_use]
    pub fn distributed_admittance(&self) -> Array1<Complex64> {
        Array1::from_shape_fn(self.frequency.points(), |point| {
            let mut value = Complex64::new(
                self.conductance_per_meter[point],
                self.frequency.angular()[point] * self.capacitance_per_meter[point],
            );
            if value.im == 0.0 {
                value.im = 1.0e-12;
            }
            value
        })
    }

    /// Recover distributed RLGC values from an existing medium.
    ///
    /// Uses $Z'=\gamma Z_{0}$ and $Y'=\gamma/Z_{0}$; zero frequency is rejected.
    ///
    /// # Errors
    ///
    /// Returns an error for zero frequency, zero characteristic impedance, or incompatible arrays.
    pub fn from_media<M: Media>(media: &M) -> Result<Self> {
        let angular = media.frequency().angular();
        if angular.iter().any(|value| *value == 0.0) {
            return Err(Error::InvalidFrequency(
                "distributed parameters cannot be recovered at zero frequency".to_owned(),
            ));
        }
        let gamma = media.propagation_constant()?;
        let z0 = media.characteristic_impedance()?;
        if z0.iter().any(|value| value.norm_sqr() == 0.0) {
            return Err(Error::Unsupported(
                "distributed parameters require non-zero characteristic impedance".to_owned(),
            ));
        }
        let admittance =
            Array1::from_shape_fn(media.frequency().points(), |point| gamma[point] / z0[point]);
        let impedance =
            Array1::from_shape_fn(media.frequency().points(), |point| gamma[point] * z0[point]);
        Self::new(
            media.frequency().clone(),
            impedance.mapv(|value| value.re),
            admittance.mapv(|value| value.re),
            Array1::from_shape_fn(media.frequency().points(), |point| {
                impedance[point].im / angular[point]
            }),
            Array1::from_shape_fn(media.frequency().points(), |point| {
                admittance[point].im / angular[point]
            }),
            media.port_impedance().cloned(),
        )
    }

    /// Read a distributed circuit from a media CSV file.
    ///
    /// The expected columns contain frequency, $Z_{0}$, $\gamma$, and optional
    /// port impedance as real/imaginary pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if the CSV cannot be read or its medium data is invalid.
    pub fn from_csv(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_media(&DefinedGammaZ0::from_csv(path)?)
    }

    /// Write frequency, $Z_{0}$, $\gamma$, and port impedance to CSV.
    ///
    /// # Errors
    ///
    /// Returns an error if the medium cannot be evaluated or the CSV cannot be written.
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        self.as_defined()?.write_csv(path)
    }

    fn as_defined(&self) -> Result<DefinedGammaZ0> {
        DefinedGammaZ0::new(
            self.frequency.clone(),
            self.propagation_constant()?,
            self.characteristic_impedance()?,
            self.port_z0.clone(),
        )
    }
}

impl Media for DistributedCircuit {
    fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    /// Return propagation constant $\gamma=\sqrt{Z'Y'}$.
    fn propagation_constant(&self) -> Result<Array1<Complex64>> {
        let impedance = self.distributed_impedance();
        let admittance = self.distributed_admittance();
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            (impedance[point] * admittance[point]).sqrt()
        }))
    }

    /// Return characteristic impedance $Z_{0}=\sqrt{Z'/Y'}$.
    fn characteristic_impedance(&self) -> Result<Array1<Complex64>> {
        let impedance = self.distributed_impedance();
        let admittance = self.distributed_admittance();
        if admittance.iter().any(|value| value.norm_sqr() == 0.0) {
            return Err(Error::Unsupported(
                "distributed admittance must be non-zero".to_owned(),
            ));
        }
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            (impedance[point] / admittance[point]).sqrt()
        }))
    }

    fn port_impedance(&self) -> Option<&Array1<Complex64>> {
        self.port_z0.as_ref()
    }

    fn line(&self, length: f64, unit: LengthUnit) -> Result<Network> {
        self.as_defined()?.line(length, unit)
    }

    fn thru(&self) -> Result<Network> {
        self.as_defined()?.thru()
    }

    fn load(&self, reflection_coefficient: Complex64) -> Result<Network> {
        self.as_defined()?.load(reflection_coefficient)
    }

    fn open(&self) -> Result<Network> {
        self.as_defined()?.open()
    }

    fn short(&self) -> Result<Network> {
        self.as_defined()?.short()
    }
}

impl fmt::Display for DistributedCircuit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format_parameter = |values: &Array1<f64>| match (values.first(), values.last()) {
            (Some(first), Some(last)) if values.len() > 1 => {
                format!("{first:.2e}, ..., {last:.2e}")
            }
            (Some(value), _) => format!("{value:.2e}"),
            _ => "empty".to_owned(),
        };
        write!(
            formatter,
            "Distributed Circuit Media.  {}-{} {}.  {} points\nL'= {}, C'= {}, R'= {}, G'= {}",
            self.frequency.start_scaled().unwrap_or_default(),
            self.frequency.stop_scaled().unwrap_or_default(),
            self.frequency.unit().symbol(),
            self.frequency.points(),
            format_parameter(&self.inductance_per_meter),
            format_parameter(&self.capacitance_per_meter),
            format_parameter(&self.resistance_per_meter),
            format_parameter(&self.conductance_per_meter),
        )
    }
}
