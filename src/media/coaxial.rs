//! Coaxial transmission lines defined by electrical or physical properties.

use super::media::{DefinedGammaZ0, LengthUnit, Media};
use super::{
    Array1, Complex64, DistributedCircuit, Error, FREE_SPACE_PERMEABILITY, FREE_SPACE_PERMITTIVITY,
    Frequency, Network, Result, SPEED_OF_LIGHT, db_to_nepers, fmt,
};
/// Unit used to specify coaxial attenuation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttenuationUnit {
    /// Decibels per meter.
    DecibelsPerMeter,
    /// Decibels per 100 meters.
    DecibelsPerHundredMeters,
    /// Decibels per foot.
    DecibelsPerFoot,
    /// Decibels per 100 feet.
    DecibelsPerHundredFeet,
    /// Nepers per meter.
    NepersPerMeter,
    /// Nepers per foot.
    NepersPerFoot,
}

/// A coaxial transmission line defined by conductor geometry and dielectric properties.
///
/// Diameter, relative permittivity, loss tangent, and conductivity may vary
/// with frequency, but all arrays must have the same length.
///
/// See Pozar, *Microwave Engineering* (Wiley, 2009).
#[derive(Clone, Debug)]
pub struct Coaxial {
    /// Frequency band.
    pub frequency: Frequency,
    /// Inner-conductor diameter in meters.
    pub inner_diameter: Array1<f64>,
    /// Outer-conductor inside diameter in meters.
    pub outer_diameter: Array1<f64>,
    /// Dielectric relative permittivity.
    pub relative_permittivity: Array1<f64>,
    /// Dielectric loss tangent.
    pub loss_tangent: Array1<f64>,
    /// Conductor conductivity in siemens per meter; infinity is lossless.
    pub conductivity: Array1<f64>,
    /// Optional port impedance used to renormalize generated networks.
    pub port_z0: Option<Array1<Complex64>>,
    /// Optional override for the calculated characteristic impedance.
    pub characteristic_impedance_override: Option<Array1<Complex64>>,
}

impl Coaxial {
    /// Construct a coaxial line from frequency-dependent physical properties.
    ///
    /// # Errors
    ///
    /// Returns an error if array lengths or physical property values are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frequency: Frequency,
        inner_diameter: Array1<f64>,
        outer_diameter: Array1<f64>,
        relative_permittivity: Array1<f64>,
        loss_tangent: Array1<f64>,
        conductivity: Array1<f64>,
        port_z0: Option<Array1<Complex64>>,
        characteristic_impedance_override: Option<Array1<Complex64>>,
    ) -> Result<Self> {
        let points = frequency.points();
        for (name, values) in [
            ("inner diameter", &inner_diameter),
            ("outer diameter", &outer_diameter),
            ("relative permittivity", &relative_permittivity),
            ("loss tangent", &loss_tangent),
            ("conductivity", &conductivity),
        ] {
            if values.len() != points {
                return Err(Error::IncompatibleShape(format!(
                    "coaxial {name} has {} values for {points} frequency points",
                    values.len()
                )));
            }
        }
        for point in 0..points {
            if !inner_diameter[point].is_finite()
                || !outer_diameter[point].is_finite()
                || inner_diameter[point] <= 0.0
                || outer_diameter[point] <= inner_diameter[point]
            {
                return Err(Error::Unsupported(
                    "coaxial diameters must be finite and satisfy 0 < inner < outer".to_owned(),
                ));
            }
            if !relative_permittivity[point].is_finite() || relative_permittivity[point] <= 0.0 {
                return Err(Error::Unsupported(
                    "coaxial relative permittivity must be positive and finite".to_owned(),
                ));
            }
            if !loss_tangent[point].is_finite() || loss_tangent[point] < 0.0 {
                return Err(Error::Unsupported(
                    "coaxial loss tangent must be finite and non-negative".to_owned(),
                ));
            }
            if conductivity[point].is_nan() || conductivity[point] <= 0.0 {
                return Err(Error::Unsupported(
                    "coaxial conductivity must be positive".to_owned(),
                ));
            }
        }
        for (name, values) in [
            ("port impedance", port_z0.as_ref()),
            (
                "characteristic-impedance override",
                characteristic_impedance_override.as_ref(),
            ),
        ] {
            if values.is_some_and(|values| values.len() != points) {
                return Err(Error::IncompatibleShape(format!(
                    "coaxial {name} must match the frequency length"
                )));
            }
        }
        Ok(Self {
            frequency,
            inner_diameter,
            outer_diameter,
            relative_permittivity,
            loss_tangent,
            conductivity,
            port_z0,
            characteristic_impedance_override,
        })
    }

    /// Construct a coaxial line from scalar properties repeated over frequency.
    ///
    /// # Errors
    ///
    /// Returns an error if any scalar physical property is invalid.
    pub fn from_scalars(
        frequency: Frequency,
        inner_diameter: f64,
        outer_diameter: f64,
        relative_permittivity: f64,
        loss_tangent: f64,
        conductivity: f64,
        port_impedance: Option<Complex64>,
    ) -> Result<Self> {
        let points = frequency.points();
        Self::new(
            frequency,
            Array1::from_elem(points, inner_diameter),
            Array1::from_elem(points, outer_diameter),
            Array1::from_elem(points, relative_permittivity),
            Array1::from_elem(points, loss_tangent),
            Array1::from_elem(points, conductivity),
            port_impedance.map(|value| Array1::from_elem(points, value)),
            None,
        )
    }

    /// Construct a line from attenuation, velocity factor, and impedance.
    ///
    /// Attenuation is converted from [`AttenuationUnit`] to nepers per meter.
    /// Phase constant is $\beta=2\pi f/(c\,\mathrm{VF})$.
    ///
    /// See [velocity factor](https://www.microwaves101.com/encyclopedias/light-phase-and-group-velocities).
    ///
    /// # Errors
    ///
    /// Returns an error if input lengths or velocity-factor values are invalid.
    pub fn from_attenuation_and_velocity_factor(
        frequency: Frequency,
        attenuation: Array1<f64>,
        unit: AttenuationUnit,
        velocity_factor: Array1<f64>,
        characteristic_impedance: Complex64,
        port_z0: Option<Array1<Complex64>>,
    ) -> Result<DefinedGammaZ0> {
        let points = frequency.points();
        if attenuation.len() != points || velocity_factor.len() != points {
            return Err(Error::IncompatibleShape(
                "coaxial attenuation and velocity factor must match the frequency length"
                    .to_owned(),
            ));
        }
        if velocity_factor
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::Unsupported(
                "coaxial velocity factor must be positive and finite".to_owned(),
            ));
        }
        let feet_per_meter = 1.0 / 0.3048;
        let alpha = attenuation.mapv_into(|value| match unit {
            AttenuationUnit::DecibelsPerMeter => db_to_nepers(value),
            AttenuationUnit::DecibelsPerHundredMeters => db_to_nepers(value) / 100.0,
            AttenuationUnit::DecibelsPerFoot => db_to_nepers(value) * feet_per_meter,
            AttenuationUnit::DecibelsPerHundredFeet => db_to_nepers(value) * feet_per_meter / 100.0,
            AttenuationUnit::NepersPerMeter => value,
            AttenuationUnit::NepersPerFoot => value * feet_per_meter,
        });
        let frequencies_hz = frequency.values_hz().clone();
        let gamma = Array1::from_shape_fn(points, move |point| {
            Complex64::new(
                alpha[point],
                std::f64::consts::TAU * frequencies_hz[point]
                    / (SPEED_OF_LIGHT * velocity_factor[point]),
            )
        });
        DefinedGammaZ0::new(
            frequency,
            gamma,
            Array1::from_elem(points, characteristic_impedance),
            port_z0,
        )
    }

    /// Construct a lossless coax from characteristic impedance and outer diameter.
    ///
    /// The inner diameter is derived for the specified relative permittivity.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested impedance or derived geometry is invalid.
    pub fn from_characteristic_impedance_and_outer_diameter(
        frequency: Frequency,
        characteristic_impedance: f64,
        outer_diameter: f64,
        relative_permittivity: f64,
    ) -> Result<Self> {
        if !characteristic_impedance.is_finite() || characteristic_impedance <= 0.0 {
            return Err(Error::Unsupported(
                "coaxial characteristic impedance must be positive and finite".to_owned(),
            ));
        }
        let exponent = 2.0
            * std::f64::consts::PI
            * characteristic_impedance
            * (FREE_SPACE_PERMITTIVITY * relative_permittivity / FREE_SPACE_PERMEABILITY).sqrt();
        let inner_diameter = outer_diameter / exponent.exp();
        Self::from_scalars(
            frequency,
            inner_diameter,
            outer_diameter,
            relative_permittivity,
            0.0,
            f64::INFINITY,
            None,
        )
    }

    /// Return inner-conductor radius in meters.
    #[must_use]
    pub fn inner_radius(&self) -> Array1<f64> {
        &self.inner_diameter / 2.0
    }

    /// Return outer-conductor radius in meters.
    #[must_use]
    pub fn outer_radius(&self) -> Array1<f64> {
        &self.outer_diameter / 2.0
    }

    /// Return conductor surface resistance in ohms per square.
    #[must_use]
    pub fn surface_resistivity(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.frequency.points(), |point| {
            if self.conductivity[point].is_infinite() {
                0.0
            } else {
                (std::f64::consts::PI * self.frequency.values_hz()[point] * FREE_SPACE_PERMEABILITY
                    / self.conductivity[point])
                    .sqrt()
            }
        })
    }

    /// Return distributed series resistance $R$ in ohms per meter.
    ///
    /// DC uses the inner-conductor cross section; nonzero frequency includes
    /// skin effect in both conductors. See [coax attenuation](https://www.microwaves101.com/encyclopedias/a-more-exact-coax-attenuation-solution).
    #[must_use]
    pub fn resistance_per_meter(&self) -> Array1<f64> {
        let inner = self.inner_radius();
        let outer = self.outer_radius();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            let conductivity = self.conductivity[point];
            if conductivity.is_infinite() {
                return 0.0;
            }
            let resistivity = 1.0 / conductivity;
            let frequency = self.frequency.values_hz()[point];
            if frequency == 0.0 {
                return resistivity / (std::f64::consts::PI * inner[point].powi(2));
            }
            let depth = (resistivity
                / (std::f64::consts::PI * frequency * FREE_SPACE_PERMEABILITY))
                .sqrt()
                .min(1.0e6);
            let inner_denominator = std::f64::consts::TAU
                * depth
                    .powi(2)
                    .mul_add((-inner[point] / depth).exp_m1(), depth * inner[point]);
            let inner_resistance = resistivity / inner_denominator;
            let outer_resistance = resistivity / (std::f64::consts::TAU * depth * outer[point]);
            inner_resistance + outer_resistance
        })
    }

    /// Return distributed inductance $L$ in henries per meter.
    #[must_use]
    pub fn inductance_per_meter(&self) -> Array1<f64> {
        let inner = self.inner_radius();
        let outer = self.outer_radius();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            FREE_SPACE_PERMEABILITY / std::f64::consts::TAU * (outer[point] / inner[point]).ln()
        })
    }

    /// Return distributed capacitance $C$ in farads per meter.
    #[must_use]
    pub fn capacitance_per_meter(&self) -> Array1<f64> {
        let inner = self.inner_radius();
        let outer = self.outer_radius();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            std::f64::consts::TAU * FREE_SPACE_PERMITTIVITY * self.relative_permittivity[point]
                / (outer[point] / inner[point]).ln()
        })
    }

    /// Return distributed shunt conductance $G$ in siemens per meter.
    #[must_use]
    pub fn conductance_per_meter(&self) -> Array1<f64> {
        let inner = self.inner_radius();
        let outer = self.outer_radius();
        let angular = self.frequency.angular();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            std::f64::consts::TAU
                * angular[point]
                * FREE_SPACE_PERMITTIVITY
                * self.relative_permittivity[point]
                * self.loss_tangent[point]
                / (outer[point] / inner[point]).ln()
        })
    }

    fn distributed_circuit(&self) -> Result<DistributedCircuit> {
        DistributedCircuit::new(
            self.frequency.clone(),
            self.resistance_per_meter(),
            self.conductance_per_meter(),
            self.inductance_per_meter(),
            self.capacitance_per_meter(),
            self.port_z0.clone(),
        )
    }

    fn as_defined(&self) -> Result<DefinedGammaZ0> {
        let circuit = self.distributed_circuit()?;
        let characteristic_impedance =
            if let Some(impedance) = &self.characteristic_impedance_override {
                impedance.clone()
            } else {
                circuit.characteristic_impedance()?
            };
        DefinedGammaZ0::new(
            self.frequency.clone(),
            circuit.propagation_constant()?,
            characteristic_impedance,
            self.port_z0.clone(),
        )
    }
}

impl fmt::Display for Coaxial {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let scaled = self.frequency.scaled();
        let start = scaled.first().copied().unwrap_or_default();
        let stop = scaled.last().copied().unwrap_or_default();
        write!(
            formatter,
            "Coaxial Media. {start}-{stop} {}, {} points. Dint = {:.2} mm, Dout = {:.2} mm",
            self.frequency.unit().symbol(),
            self.frequency.points(),
            self.inner_diameter.first().copied().unwrap_or_default() * 1.0e3,
            self.outer_diameter.first().copied().unwrap_or_default() * 1.0e3
        )
    }
}

impl Media for Coaxial {
    fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    fn propagation_constant(&self) -> Result<Array1<Complex64>> {
        self.distributed_circuit()?.propagation_constant()
    }

    fn characteristic_impedance(&self) -> Result<Array1<Complex64>> {
        if let Some(impedance) = &self.characteristic_impedance_override {
            Ok(impedance.clone())
        } else {
            self.distributed_circuit()?.characteristic_impedance()
        }
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
