use super::media::*;
use super::*;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaveguideMode {
    TransverseElectric,
    TransverseMagnetic,
}

/// A single mode of a homogeneously filled rectangular waveguide.
///
/// Origin: `skrf/media/rectangularWaveguide.py::RectangularWaveguide`.
#[derive(Clone, Debug)]
pub struct RectangularWaveguide {
    pub frequency: Frequency,
    pub width: f64,
    pub height: f64,
    pub mode: WaveguideMode,
    pub horizontal_mode_index: usize,
    pub vertical_mode_index: usize,
    pub relative_permittivity: Array1<f64>,
    pub relative_permeability: Array1<f64>,
    pub resistivity: Option<Array1<f64>>,
    pub roughness: Option<Array1<f64>>,
    pub port_z0: Option<Array1<Complex64>>,
    pub characteristic_impedance_override: Option<Array1<Complex64>>,
}

impl RectangularWaveguide {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frequency: Frequency,
        width: f64,
        height: Option<f64>,
        mode: WaveguideMode,
        horizontal_mode_index: usize,
        vertical_mode_index: usize,
        relative_permittivity: Array1<f64>,
        relative_permeability: Array1<f64>,
        resistivity: Option<Array1<f64>>,
        roughness: Option<Array1<f64>>,
        port_z0: Option<Array1<Complex64>>,
        characteristic_impedance_override: Option<Array1<Complex64>>,
    ) -> Result<Self> {
        let height = height.unwrap_or(width / 2.0);
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return Err(Error::Unsupported(
                "waveguide width and height must be positive and finite".to_owned(),
            ));
        }
        if horizontal_mode_index == 0 && vertical_mode_index == 0 {
            return Err(Error::Unsupported(
                "waveguide mode indices cannot both be zero".to_owned(),
            ));
        }
        if mode == WaveguideMode::TransverseMagnetic
            && (horizontal_mode_index == 0 || vertical_mode_index == 0)
        {
            return Err(Error::Unsupported(
                "rectangular-waveguide TM modes require two non-zero indices".to_owned(),
            ));
        }
        let points = frequency.points();
        for (name, values) in [
            ("relative permittivity", &relative_permittivity),
            ("relative permeability", &relative_permeability),
        ] {
            if values.len() != points {
                return Err(Error::IncompatibleShape(format!(
                    "waveguide {name} has {} values for {points} frequency points",
                    values.len()
                )));
            }
            if values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(Error::Unsupported(format!(
                    "waveguide {name} must be positive and finite"
                )));
            }
        }
        for (name, values) in [
            ("resistivity", resistivity.as_ref().map(Array1::len)),
            ("roughness", roughness.as_ref().map(Array1::len)),
            ("port impedance", port_z0.as_ref().map(Array1::len)),
            (
                "characteristic-impedance override",
                characteristic_impedance_override.as_ref().map(Array1::len),
            ),
        ] {
            if values.is_some_and(|length| length != points) {
                return Err(Error::IncompatibleShape(format!(
                    "waveguide {name} must match the frequency length"
                )));
            }
        }
        if resistivity.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        }) {
            return Err(Error::Unsupported(
                "waveguide resistivity must be positive and finite".to_owned(),
            ));
        }
        if roughness.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        }) {
            return Err(Error::Unsupported(
                "waveguide roughness must be positive and finite".to_owned(),
            ));
        }
        if roughness.is_some() && resistivity.is_none() {
            return Err(Error::Unsupported(
                "waveguide roughness requires finite wall resistivity".to_owned(),
            ));
        }
        Ok(Self {
            frequency,
            width,
            height,
            mode,
            horizontal_mode_index,
            vertical_mode_index,
            relative_permittivity,
            relative_permeability,
            resistivity,
            roughness,
            port_z0,
            characteristic_impedance_override,
        })
    }

    pub fn dominant_mode(frequency: Frequency, width: f64) -> Result<Self> {
        let points = frequency.points();
        Self::new(
            frequency,
            width,
            None,
            WaveguideMode::TransverseElectric,
            1,
            0,
            Array1::ones(points),
            Array1::ones(points),
            None,
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
        let permittivity = FREE_SPACE_PERMITTIVITY * relative_permittivity;
        let permeability = FREE_SPACE_PERMEABILITY * relative_permeability;
        let angular = std::f64::consts::TAU * specification_frequency_hz;
        let k0_squared = angular.powi(2) * permittivity * permeability;
        let beta_squared = (angular * permeability / characteristic_impedance).powi(2);
        if k0_squared <= beta_squared {
            return Err(Error::Unsupported(
                "requested TE impedance does not produce a propagating waveguide".to_owned(),
            ));
        }
        let width = std::f64::consts::PI / (k0_squared - beta_squared).sqrt();
        let points = frequency.points();
        Self::new(
            frequency,
            width,
            None,
            WaveguideMode::TransverseElectric,
            1,
            0,
            Array1::from_elem(points, relative_permittivity),
            Array1::from_elem(points, relative_permeability),
            None,
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

    pub fn horizontal_wavenumber(&self) -> f64 {
        self.horizontal_mode_index as f64 * std::f64::consts::PI / self.width
    }

    pub fn vertical_wavenumber(&self) -> f64 {
        self.vertical_mode_index as f64 * std::f64::consts::PI / self.height
    }

    pub fn cutoff_wavenumber(&self) -> f64 {
        self.horizontal_wavenumber()
            .hypot(self.vertical_wavenumber())
    }

    pub fn cutoff_frequency(&self) -> Array1<f64> {
        let permittivity = self.permittivity();
        let permeability = self.permeability();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            self.cutoff_wavenumber()
                / (std::f64::consts::TAU * (permittivity[point] * permeability[point]).sqrt())
        })
    }

    pub fn normalized_frequency(&self) -> Array1<f64> {
        let cutoff = self.cutoff_frequency();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            self.frequency.values_hz()[point] / cutoff[point]
        })
    }

    pub fn guide_wavelength(&self) -> Result<Array1<Complex64>> {
        let gamma = self.propagation_constant()?;
        Ok(gamma.mapv(|value| Complex64::new(0.0, std::f64::consts::TAU) / value))
    }

    pub fn cutoff_wavelength(&self) -> Array1<f64> {
        let permittivity = self.permittivity();
        let permeability = self.permeability();
        let cutoff = self.cutoff_frequency();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            1.0 / ((permittivity[point] * permeability[point]).sqrt() * cutoff[point])
        })
    }

    pub fn effective_resistivity(&self) -> Result<Option<Array1<f64>>> {
        let Some(resistivity) = &self.resistivity else {
            return Ok(None);
        };
        let Some(roughness) = &self.roughness else {
            return Ok(Some(resistivity.clone()));
        };
        if self
            .frequency
            .values_hz()
            .iter()
            .any(|frequency| *frequency <= 0.0)
        {
            return Err(Error::InvalidFrequency(
                "waveguide roughness correction requires positive frequencies".to_owned(),
            ));
        }
        Ok(Some(Array1::from_shape_fn(
            self.frequency.points(),
            |point| {
                let depth = (resistivity[point]
                    / (std::f64::consts::PI
                        * self.frequency.values_hz()[point]
                        * self.relative_permeability[point]
                        * FREE_SPACE_PERMEABILITY))
                    .sqrt();
                let roughness_factor = 1.0 + (-(depth / (2.0 * roughness[point])).powf(1.6)).exp();
                resistivity[point] * roughness_factor.powi(2)
            },
        )))
    }

    pub fn conductor_attenuation(&self) -> Result<Array1<f64>> {
        let Some(resistivity) = self.effective_resistivity()? else {
            return Ok(Array1::zeros(self.frequency.points()));
        };
        let normalized = self.normalized_frequency();
        if normalized.iter().any(|value| *value <= 1.0) {
            return Err(Error::Unsupported(
                "waveguide conductor attenuation is defined only above cutoff".to_owned(),
            ));
        }
        let angular = self.frequency.angular();
        let permittivity = self.permittivity();
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            let inverse_normalized_squared = normalized[point].powi(-2);
            1.0 / self.height
                * (angular[point] * permittivity[point] * resistivity[point] / 2.0).sqrt()
                * (1.0 + 2.0 * self.height / self.width * inverse_normalized_squared)
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

impl Media for RectangularWaveguide {
    fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    fn propagation_constant(&self) -> Result<Array1<Complex64>> {
        let k0 = self.characteristic_wavenumber();
        let cutoff = self.cutoff_wavenumber();
        let attenuation = self.conductor_attenuation()?;
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            if k0[point] > cutoff {
                Complex64::new(
                    attenuation[point],
                    (k0[point].powi(2) - cutoff.powi(2)).sqrt(),
                )
            } else if k0[point] < cutoff {
                Complex64::new((cutoff.powi(2) - k0[point].powi(2)).sqrt(), 0.0)
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
                "waveguide impedance is singular at zero frequency or exact cutoff".to_owned(),
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

impl fmt::Display for RectangularWaveguide {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Rectangular Waveguide Media.  {}-{} {}.  {} points\n a= {:.2e}m, b= {:.2e}m",
            self.frequency.start_scaled().unwrap_or_default(),
            self.frequency.stop_scaled().unwrap_or_default(),
            self.frequency.unit().symbol(),
            self.frequency.points(),
            self.width,
            self.height,
        )
    }
}
