use super::media::*;
use super::*;
/// A single mode of a homogeneously filled circular waveguide.
///
/// Origin: `skrf/media/circularWaveguide.py::CircularWaveguide`.
#[derive(Clone, Debug)]
pub struct CircularWaveguide {
    pub frequency: Frequency,
    pub radius: Array1<f64>,
    pub mode: WaveguideMode,
    pub azimuthal_mode_index: usize,
    pub radial_mode_index: usize,
    pub relative_permittivity: Array1<f64>,
    pub relative_permeability: Array1<f64>,
    pub resistivity: Option<Array1<f64>>,
    pub port_z0: Option<Array1<Complex64>>,
    pub characteristic_impedance_override: Option<Array1<Complex64>>,
}

impl CircularWaveguide {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frequency: Frequency,
        radius: Array1<f64>,
        mode: WaveguideMode,
        azimuthal_mode_index: usize,
        radial_mode_index: usize,
        relative_permittivity: Array1<f64>,
        relative_permeability: Array1<f64>,
        resistivity: Option<Array1<f64>>,
        port_z0: Option<Array1<Complex64>>,
        characteristic_impedance_override: Option<Array1<Complex64>>,
    ) -> Result<Self> {
        let points = frequency.points();
        if radial_mode_index == 0 {
            return Err(Error::Unsupported(
                "circular-waveguide radial mode indices are one-based".to_owned(),
            ));
        }
        for (name, values) in [
            ("radius", &radius),
            ("relative permittivity", &relative_permittivity),
            ("relative permeability", &relative_permeability),
        ] {
            if values.len() != points {
                return Err(Error::IncompatibleShape(format!(
                    "circular-waveguide {name} has {} values for {points} frequency points",
                    values.len()
                )));
            }
            if values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(Error::Unsupported(format!(
                    "circular-waveguide {name} must be positive and finite"
                )));
            }
        }
        if resistivity
            .as_ref()
            .is_some_and(|values| values.len() != points)
        {
            return Err(Error::IncompatibleShape(
                "circular-waveguide resistivity must match the frequency length".to_owned(),
            ));
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
                    "circular-waveguide {name} must match the frequency length"
                )));
            }
        }
        if resistivity.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        }) {
            return Err(Error::Unsupported(
                "circular-waveguide resistivity must be positive and finite".to_owned(),
            ));
        }
        Ok(Self {
            frequency,
            radius,
            mode,
            azimuthal_mode_index,
            radial_mode_index,
            relative_permittivity,
            relative_permeability,
            resistivity,
            port_z0,
            characteristic_impedance_override,
        })
    }

    pub fn dominant_mode(frequency: Frequency, radius: f64) -> Result<Self> {
        let points = frequency.points();
        Self::new(
            frequency,
            Array1::from_elem(points, radius),
            WaveguideMode::TransverseElectric,
            1,
            1,
            Array1::ones(points),
            Array1::ones(points),
            None,
            None,
            None,
        )
    }

    /// Resolves a sidewall conductor name or alias through `skrf.data.materials`.
    pub fn set_resistivity_material(&mut self, material: &str) -> Result<()> {
        let properties = crate::data::MATERIALS
            .get(material.to_ascii_lowercase().as_str())
            .ok_or_else(|| Error::Unsupported(format!("unknown material `{material}`")))?;
        let resistivity = properties.resistivity_ohm_meter.ok_or_else(|| {
            Error::Unsupported(format!("material `{material}` does not define resistivity"))
        })?;
        self.resistivity = Some(Array1::from_elem(self.frequency.points(), resistivity));
        Ok(())
    }

    pub fn from_characteristic_impedance(
        frequency: Frequency,
        characteristic_impedance: f64,
        specification_frequency_hz: f64,
        relative_permittivity: f64,
        relative_permeability: f64,
    ) -> Result<Self> {
        if !characteristic_impedance.is_finite()
            || characteristic_impedance <= 0.0
            || !specification_frequency_hz.is_finite()
            || specification_frequency_hz <= 0.0
        {
            return Err(Error::Unsupported(
                "waveguide impedance and specification frequency must be positive and finite"
                    .to_owned(),
            ));
        }
        let root = bessel_j_zero(1, 1, true)?;
        let permittivity = FREE_SPACE_PERMITTIVITY * relative_permittivity;
        let permeability = FREE_SPACE_PERMEABILITY * relative_permeability;
        let angular = std::f64::consts::TAU * specification_frequency_hz;
        let k0_squared = angular.powi(2) * permittivity * permeability;
        let beta_squared = (angular * permeability / characteristic_impedance).powi(2);
        if k0_squared <= beta_squared {
            return Err(Error::Unsupported(
                "requested TE impedance does not produce a propagating circular waveguide"
                    .to_owned(),
            ));
        }
        let radius = root / (k0_squared - beta_squared).sqrt();
        let points = frequency.points();
        Self::new(
            frequency,
            Array1::from_elem(points, radius),
            WaveguideMode::TransverseElectric,
            1,
            1,
            Array1::from_elem(points, relative_permittivity),
            Array1::from_elem(points, relative_permeability),
            None,
            None,
            None,
        )
    }

    pub fn permittivity(&self) -> Array1<f64> {
        &self.relative_permittivity * FREE_SPACE_PERMITTIVITY
    }

    pub fn permeability(&self) -> Array1<f64> {
        &self.relative_permeability * FREE_SPACE_PERMEABILITY
    }

    pub fn characteristic_wavenumber(&self) -> Array1<f64> {
        let permittivity = self.permittivity();
        let permeability = self.permeability();
        let angular = self.frequency.angular();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            angular[point] * (permittivity[point] * permeability[point]).sqrt()
        })
    }

    pub fn modal_root(&self) -> Result<f64> {
        bessel_j_zero(
            self.azimuthal_mode_index,
            self.radial_mode_index,
            self.mode == WaveguideMode::TransverseElectric,
        )
    }

    pub fn cutoff_wavenumber(&self) -> Result<Array1<f64>> {
        let root = self.modal_root()?;
        Ok(self.radius.mapv(|radius| root / radius))
    }

    pub fn cutoff_frequency(&self) -> Result<Array1<f64>> {
        let cutoff = self.cutoff_wavenumber()?;
        let permittivity = self.permittivity();
        let permeability = self.permeability();
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            cutoff[point]
                / (std::f64::consts::TAU * (permittivity[point] * permeability[point]).sqrt())
        }))
    }

    pub fn normalized_frequency(&self) -> Result<Array1<f64>> {
        let cutoff = self.cutoff_frequency()?;
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            self.frequency.values_hz()[point] / cutoff[point]
        }))
    }

    pub fn guide_wavelength(&self) -> Result<Array1<Complex64>> {
        Ok(self
            .propagation_constant()?
            .mapv(|value| Complex64::new(0.0, std::f64::consts::TAU) / value))
    }

    pub fn cutoff_wavelength(&self) -> Result<Array1<f64>> {
        let cutoff = self.cutoff_frequency()?;
        let permittivity = self.permittivity();
        let permeability = self.permeability();
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            1.0 / ((permittivity[point] * permeability[point]).sqrt() * cutoff[point])
        }))
    }

    pub fn conductor_attenuation(&self) -> Result<Array1<f64>> {
        let Some(resistivity) = &self.resistivity else {
            return Ok(Array1::zeros(self.frequency.points()));
        };
        if self.mode != WaveguideMode::TransverseElectric
            || self.azimuthal_mode_index != 1
            || self.radial_mode_index != 1
        {
            return Err(Error::Unsupported(
                "circular-waveguide conductor loss is implemented for TE11 only".to_owned(),
            ));
        }
        let normalized = self.normalized_frequency()?;
        if normalized.iter().any(|value| *value <= 1.0) {
            return Err(Error::Unsupported(
                "circular-waveguide conductor loss is defined only above cutoff".to_owned(),
            ));
        }
        let angular = self.frequency.angular();
        let permittivity = self.permittivity();
        let root = self.modal_root()?;
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            let inverse_normalized_squared = normalized[point].powi(-2);
            1.0 / self.radius[point]
                * (angular[point] * permittivity[point] * resistivity[point] / 2.0).sqrt()
                * (inverse_normalized_squared + 1.0 / (root.powi(2) - 1.0))
                / (1.0 - inverse_normalized_squared).sqrt()
        }))
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

impl Media for CircularWaveguide {
    fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    fn propagation_constant(&self) -> Result<Array1<Complex64>> {
        let k0 = self.characteristic_wavenumber();
        let cutoff = self.cutoff_wavenumber()?;
        let attenuation = self.conductor_attenuation()?;
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            if k0[point] > cutoff[point] {
                Complex64::new(
                    attenuation[point],
                    (k0[point].powi(2) - cutoff[point].powi(2)).sqrt(),
                )
            } else if k0[point] < cutoff[point] {
                Complex64::new((cutoff[point].powi(2) - k0[point].powi(2)).sqrt(), 0.0)
            } else {
                Complex64::new(attenuation[point], 0.0)
            }
        }))
    }

    fn characteristic_impedance(&self) -> Result<Array1<Complex64>> {
        if let Some(impedance) = &self.characteristic_impedance_override {
            return Ok(impedance.clone());
        }
        let gamma = self.propagation_constant()?;
        let angular = self.frequency.angular();
        let permittivity = self.permittivity();
        let permeability = self.permeability();
        if angular.iter().any(|value| *value == 0.0)
            || gamma.iter().any(|value| value.norm_sqr() == 0.0)
        {
            return Err(Error::Unsupported(
                "circular-waveguide impedance is singular at zero frequency or exact cutoff"
                    .to_owned(),
            ));
        }
        Ok(Array1::from_shape_fn(
            self.frequency.points(),
            |point| match self.mode {
                WaveguideMode::TransverseElectric => {
                    Complex64::new(0.0, angular[point] * permeability[point]) / gamma[point]
                }
                WaveguideMode::TransverseMagnetic => {
                    Complex64::new(0.0, -1.0) * gamma[point]
                        / (angular[point] * permittivity[point])
                }
            },
        ))
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

impl fmt::Display for CircularWaveguide {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let radius = match (self.radius.first(), self.radius.last()) {
            (Some(first), Some(last)) if self.radius.len() > 1 && first != last => {
                format!("{first:.2e}, ..., {last:.2e}")
            }
            (Some(value), _) => format!("{value:.2e}"),
            _ => "empty".to_owned(),
        };
        write!(
            formatter,
            "Circular Waveguide Media.  {}-{} {}.  {} points\n r= {radius}m",
            self.frequency.start_scaled().unwrap_or_default(),
            self.frequency.stop_scaled().unwrap_or_default(),
            self.frequency.unit().symbol(),
            self.frequency.points(),
        )
    }
}
