//! Single-mode propagation in a homogeneously filled rectangular waveguide.

use super::media::{DefinedGammaZ0, LengthUnit, Media};
use super::{
    Array1, Complex64, Error, FREE_SPACE_PERMEABILITY, FREE_SPACE_PERMITTIVITY, Frequency, Network,
    Result, fmt,
};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Electromagnetic mode family in a rectangular waveguide.
pub enum WaveguideMode {
    /// Transverse-electric mode ($`E_z=0`$).
    TransverseElectric,
    /// Transverse-magnetic mode ($`H_z=0`$).
    TransverseMagnetic,
}

/// A single mode of a homogeneously filled rectangular waveguide.
///
/// Mode indices $(m,n)$ determine the cutoff wavenumber
/// $$`k_c=\sqrt{(m\pi/a)^2+(n\pi/b)^2`}.$$
#[derive(Clone, Debug)]
pub struct RectangularWaveguide {
    /// Frequencies at which the waveguide is evaluated.
    pub frequency: Frequency,
    /// Broad-wall dimension $a$ in meters.
    pub width: f64,
    /// Narrow-wall dimension $b$ in meters.
    pub height: f64,
    /// TE or TM mode family.
    pub mode: WaveguideMode,
    /// Horizontal mode index $m$.
    pub horizontal_mode_index: u32,
    /// Vertical mode index $n$.
    pub vertical_mode_index: u32,
    /// Filling material's relative permittivity.
    pub relative_permittivity: Array1<f64>,
    /// Filling material's relative permeability.
    pub relative_permeability: Array1<f64>,
    /// Optional wall resistivity in $\Omega\,\mathrm{m}$.
    pub resistivity: Option<Array1<f64>>,
    /// Optional RMS wall roughness in meters.
    pub roughness: Option<Array1<f64>>,
    /// Optional port-renormalization impedance.
    pub port_z0: Option<Array1<Complex64>>,
    /// Optional characteristic-impedance override.
    pub characteristic_impedance_override: Option<Array1<Complex64>>,
}

impl RectangularWaveguide {
    /// Creates a rectangular-waveguide mode from geometry and material properties.
    ///
    /// If `height` is omitted it defaults to half the width. TM modes require
    /// both mode indices to be non-zero.
    ///
    /// # Errors
    ///
    /// Returns an error when the geometry or mode indices are invalid, when a
    /// material-property array has the wrong length or invalid values, or when
    /// roughness is supplied without valid wall resistivity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frequency: Frequency,
        width: f64,
        height: Option<f64>,
        mode: WaveguideMode,
        horizontal_mode_index: u32,
        vertical_mode_index: u32,
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

    /// Creates an air-filled dominant $\mathrm{TE}_{10}$ waveguide.
    ///
    /// # Errors
    ///
    /// Returns an error when `width` is not positive and finite or when the
    /// generated material arrays are incompatible with `frequency`.
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

    /// Sets wall resistivity from a named entry in [`crate::data::MATERIALS`].
    ///
    /// # Errors
    ///
    /// Returns an error when `material` is unknown or has no resistivity value.
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

    /// Derives the broad-wall width from a desired $\mathrm{TE}_{10}$ impedance.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested impedance or specification frequency
    /// is invalid, the requested mode cannot propagate, or the derived
    /// waveguide fails construction validation.
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

    /// Returns absolute permittivity $`\varepsilon=\varepsilon_0\varepsilon_r`$.
    #[must_use]
    pub fn permittivity(&self) -> Array1<f64> {
        &self.relative_permittivity * FREE_SPACE_PERMITTIVITY
    }

    /// Returns absolute permeability $`\mu=\mu_0\mu_r`$.
    #[must_use]
    pub fn permeability(&self) -> Array1<f64> {
        &self.relative_permeability * FREE_SPACE_PERMEABILITY
    }

    /// Returns the material wavenumber $`k_0=\omega\sqrt{\mu\varepsilon}`$.
    #[must_use]
    pub fn characteristic_wavenumber(&self) -> Array1<f64> {
        let permittivity = self.permittivity();
        let permeability = self.permeability();
        let angular = self.frequency.angular();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            angular[point] * (permittivity[point] * permeability[point]).sqrt()
        })
    }

    /// Returns $`k_x=m\pi/a`$.
    #[must_use]
    pub fn horizontal_wavenumber(&self) -> f64 {
        f64::from(self.horizontal_mode_index) * std::f64::consts::PI / self.width
    }

    /// Returns $`k_y=n\pi/b`$.
    #[must_use]
    pub fn vertical_wavenumber(&self) -> f64 {
        f64::from(self.vertical_mode_index) * std::f64::consts::PI / self.height
    }

    /// Returns cutoff wavenumber $`k_c=\sqrt{k_x^2+k_y^2}`$.
    #[must_use]
    pub fn cutoff_wavenumber(&self) -> f64 {
        self.horizontal_wavenumber()
            .hypot(self.vertical_wavenumber())
    }

    /// Returns cutoff frequency $`f_c=k_c/(2\pi\sqrt{\mu\varepsilon})`$.
    #[must_use]
    pub fn cutoff_frequency(&self) -> Array1<f64> {
        let permittivity = self.permittivity();
        let permeability = self.permeability();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            self.cutoff_wavenumber()
                / (std::f64::consts::TAU * (permittivity[point] * permeability[point]).sqrt())
        })
    }

    /// Returns normalized frequency $`f/f_c`$.
    #[must_use]
    pub fn normalized_frequency(&self) -> Array1<f64> {
        let cutoff = self.cutoff_frequency();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            self.frequency.values_hz()[point] / cutoff[point]
        })
    }

    /// Returns guide wavelength $`\lambda_g=2\pi/\beta`$.
    ///
    /// # Errors
    ///
    /// Returns an error when the propagation constant cannot be evaluated for
    /// the configured frequencies or wall properties.
    pub fn guide_wavelength(&self) -> Result<Array1<Complex64>> {
        let gamma = self.propagation_constant()?;
        Ok(gamma.mapv(|value| Complex64::new(0.0, std::f64::consts::TAU) / value))
    }

    /// Returns cutoff wavelength $`\lambda_c=2\pi/k_c`$.
    #[must_use]
    pub fn cutoff_wavelength(&self) -> Array1<f64> {
        let permittivity = self.permittivity();
        let permeability = self.permeability();
        let cutoff = self.cutoff_frequency();
        Array1::from_shape_fn(self.frequency.points(), |point| {
            1.0 / ((permittivity[point] * permeability[point]).sqrt() * cutoff[point])
        })
    }

    /// Returns wall resistivity corrected for surface roughness.
    ///
    /// # Errors
    ///
    /// Returns an error when roughness correction is requested at a
    /// non-positive frequency.
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

    /// Returns conductor attenuation in nepers per meter.
    ///
    /// # Errors
    ///
    /// Returns an error when effective resistivity cannot be evaluated or when
    /// any configured frequency is at or below cutoff.
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
                * (2.0 * self.height / self.width).mul_add(inverse_normalized_squared, 1.0)
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
    /// Returns the waveguide frequency axis.
    fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    /// Returns the propagation constant, including wall loss when configured.
    fn propagation_constant(&self) -> Result<Array1<Complex64>> {
        let k0 = self.characteristic_wavenumber();
        let cutoff = self.cutoff_wavenumber();
        let attenuation = self.conductor_attenuation()?;
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            if k0[point] > cutoff {
                Complex64::new(
                    attenuation[point],
                    k0[point].mul_add(k0[point], -cutoff.powi(2)).sqrt(),
                )
            } else if k0[point] < cutoff {
                Complex64::new(cutoff.mul_add(cutoff, -k0[point].powi(2)).sqrt(), 0.0)
            } else {
                Complex64::new(attenuation[point], 0.0)
            }
        }))
    }

    /// Returns TE or TM characteristic impedance, unless overridden.
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

    /// Returns the optional port-renormalization impedance.
    fn port_impedance(&self) -> Option<&Array1<Complex64>> {
        self.port_z0.as_ref()
    }

    /// Creates a matched waveguide section of the requested length.
    fn line(&self, length: f64, unit: LengthUnit) -> Result<Network> {
        self.as_defined()?.line(length, unit)
    }

    /// Creates a zero-length through network.
    fn thru(&self) -> Result<Network> {
        self.as_defined()?.thru()
    }

    /// Creates a one-port load with the supplied reflection coefficient.
    fn load(&self, reflection_coefficient: Complex64) -> Result<Network> {
        self.as_defined()?.load(reflection_coefficient)
    }

    /// Creates an ideal open circuit.
    fn open(&self) -> Result<Network> {
        self.as_defined()?.open()
    }

    /// Creates an ideal short circuit.
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
