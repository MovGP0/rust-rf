//! Coplanar-waveguide media with dielectric dispersion and conductor loss.

use super::media::{DefinedGammaZ0, LengthUnit, Media};
use super::{
    Array1, Complex64, DielectricDispersionModel, Error, FREE_SPACE_PERMEABILITY,
    FREE_SPACE_PERMITTIVITY, Frequency, Network, Result, SPEED_OF_LIGHT,
    complete_elliptic_integral_first_kind, fmt,
};
/// Compatibility adjustments for external CPW models.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CpwCompatibilityMode {
    /// Native rust-rf behavior.
    #[default]
    Native,
    /// Match Qucs behavior, including real quasi-static impedance.
    Qucs,
    /// Match Keysight ADS dispersion behavior where implemented.
    Ads,
}

type CpwQuasiStatic = (Array1<Complex64>, Array1<Complex64>, f64, f64, f64);

/// Coplanar waveguide with optional conductor backing and frequency dispersion.
///
/// Geometry is defined by center-strip width, gap, substrate height, and
/// optional conductor thickness. The implementation includes quasi-static
/// Ghione/Naldi models, wideband Djordjevic-Svensson dielectric dispersion,
/// Frankel/Gevorgian frequency dispersion, and Wheeler-rule losses.
///
/// Technical background is available in the [Qucs technical documentation](http://qucs.sourceforge.net/docs/technical.pdf).
#[derive(Clone, Debug)]
pub struct Cpw {
    /// Frequency band.
    pub frequency: Frequency,
    /// Center-conductor width in meters.
    pub width: f64,
    /// Gap width in meters.
    pub gap: f64,
    /// Substrate height in meters.
    pub substrate_height: f64,
    /// Conductor thickness in meters; `None` disables thickness correction.
    pub thickness: Option<f64>,
    /// Substrate relative permittivity at the model specification frequency.
    pub relative_permittivity: f64,
    /// Dielectric loss tangent at the model specification frequency.
    pub loss_tangent: f64,
    /// Conductor resistivity in ohm-meters.
    pub resistivity: Option<f64>,
    /// Dielectric frequency-dispersion model.
    pub dispersion_model: DielectricDispersionModel,
    /// Whether the substrate backside is metal rather than air.
    pub has_metal_backside: bool,
    /// External-simulator compatibility behavior.
    pub compatibility_mode: CpwCompatibilityMode,
    /// Optional port impedance used to renormalize generated networks.
    pub port_z0: Option<Array1<Complex64>>,
    /// Optional override for characteristic impedance.
    pub characteristic_impedance_override: Option<Array1<Complex64>>,
}

impl Cpw {
    /// Construct a coplanar-waveguide medium.
    ///
    /// Geometric and dielectric inputs must be finite and physically valid.
    /// Conductor loss requires both nonzero thickness and resistivity.
    ///
    /// # Errors
    ///
    /// Returns an error when geometry, dielectric, thickness, or resistivity
    /// inputs are invalid, conductor loss lacks resistivity, or an impedance
    /// array does not match the frequency axis.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frequency: Frequency,
        width: f64,
        gap: f64,
        substrate_height: f64,
        thickness: Option<f64>,
        relative_permittivity: f64,
        loss_tangent: f64,
        resistivity: Option<f64>,
        dispersion_model: DielectricDispersionModel,
        has_metal_backside: bool,
        compatibility_mode: CpwCompatibilityMode,
        port_z0: Option<Array1<Complex64>>,
        characteristic_impedance_override: Option<Array1<Complex64>>,
    ) -> Result<Self> {
        for (name, value) in [
            ("width", width),
            ("gap", gap),
            ("substrate height", substrate_height),
            ("relative permittivity", relative_permittivity),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::Unsupported(format!(
                    "CPW {name} must be positive and finite"
                )));
            }
        }
        if thickness.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(Error::Unsupported(
                "CPW thickness must be finite and non-negative".to_owned(),
            ));
        }
        if !loss_tangent.is_finite() || loss_tangent < 0.0 {
            return Err(Error::Unsupported(
                "CPW loss tangent must be finite and non-negative".to_owned(),
            ));
        }
        if resistivity.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            return Err(Error::Unsupported(
                "CPW resistivity must be positive and finite".to_owned(),
            ));
        }
        if thickness.is_some_and(|value| value > 0.0) && resistivity.is_none() {
            return Err(Error::Unsupported(
                "CPW conductor loss requires resistivity when thickness is non-zero".to_owned(),
            ));
        }
        let points = frequency.points();
        for (name, values) in [
            ("port impedance", port_z0.as_ref()),
            (
                "characteristic-impedance override",
                characteristic_impedance_override.as_ref(),
            ),
        ] {
            if values.is_some_and(|values| values.len() != points) {
                return Err(Error::IncompatibleShape(format!(
                    "CPW {name} must match the frequency length"
                )));
            }
        }
        Ok(Self {
            frequency,
            width,
            gap,
            substrate_height,
            thickness,
            relative_permittivity,
            loss_tangent,
            resistivity,
            dispersion_model,
            has_metal_backside,
            compatibility_mode,
            port_z0,
            characteristic_impedance_override,
        })
    }

    /// Construct an unbacked, lossless, frequency-invariant CPW.
    ///
    /// # Errors
    ///
    /// Returns an error when the geometry or relative permittivity is invalid.
    pub fn lossless(
        frequency: Frequency,
        width: f64,
        gap: f64,
        substrate_height: f64,
        relative_permittivity: f64,
    ) -> Result<Self> {
        Self::new(
            frequency,
            width,
            gap,
            substrate_height,
            None,
            relative_permittivity,
            0.0,
            None,
            DielectricDispersionModel::FrequencyInvariant,
            false,
            CpwCompatibilityMode::Native,
            None,
            None,
        )
    }

    /// Set conductor resistivity from a named material or alias.
    ///
    /// # Errors
    ///
    /// Returns an error when `material` is unknown or has no resistivity value.
    pub fn set_resistivity_material(&mut self, material: &str) -> Result<()> {
        let properties = crate::data::MATERIALS
            .get(material.to_ascii_lowercase().as_str())
            .ok_or_else(|| Error::Unsupported(format!("unknown material `{material}`")))?;
        self.resistivity = Some(properties.resistivity_ohm_meter.ok_or_else(|| {
            Error::Unsupported(format!("material `{material}` does not define resistivity"))
        })?);
        Ok(())
    }

    fn elliptic_ratio(modulus: f64) -> Result<f64> {
        if !modulus.is_finite() || !(0.0..1.0).contains(&modulus) || modulus == 0.0 {
            return Err(Error::Unsupported(
                "CPW elliptic modulus must satisfy 0 < k < 1".to_owned(),
            ));
        }
        if modulus < (0.5_f64).sqrt() {
            let complementary = modulus.mul_add(-modulus, 1.0).sqrt();
            Ok(std::f64::consts::PI
                / (2.0 * (1.0 + complementary.sqrt()) / (1.0 - complementary.sqrt())).ln())
        } else {
            Ok((2.0 * (1.0 + modulus.sqrt()) / (1.0 - modulus.sqrt())).ln() / std::f64::consts::PI)
        }
    }

    /// Calculate frequency-dependent complex permittivity and loss tangent.
    ///
    /// The Djordjevic-Svensson option provides a causal wideband dielectric
    /// response; the frequency-invariant option repeats the specified values.
    ///
    /// # Errors
    ///
    /// Returns an error when the Djordjevic-Svensson logarithmic slope is
    /// singular for the configured frequency bounds.
    pub fn dielectric_properties(&self) -> Result<(Array1<Complex64>, Array1<f64>)> {
        match self.dispersion_model {
            DielectricDispersionModel::FrequencyInvariant => Ok((
                Array1::from_elem(
                    self.frequency.points(),
                    Complex64::new(
                        self.relative_permittivity,
                        -self.relative_permittivity * self.loss_tangent,
                    ),
                ),
                Array1::from_elem(self.frequency.points(), self.loss_tangent),
            )),
            DielectricDispersionModel::DjordjevicSvensson {
                low_frequency_hz,
                high_frequency_hz,
                specification_frequency_hz,
            } => {
                let k = (Complex64::new(high_frequency_hz, specification_frequency_hz)
                    / Complex64::new(low_frequency_hz, specification_frequency_hz))
                .ln();
                if k.im == 0.0 {
                    return Err(Error::Unsupported(
                        "CPW dielectric logarithmic slope is singular".to_owned(),
                    ));
                }
                let permittivity = Array1::from_shape_fn(self.frequency.points(), |point| {
                    let frequency = self.frequency.values_hz()[point];
                    let frequency_log = (Complex64::new(high_frequency_hz, frequency)
                        / Complex64::new(low_frequency_hz, frequency))
                    .ln();
                    let slope = -self.loss_tangent * self.relative_permittivity / k.im;
                    let infinite =
                        self.relative_permittivity * (1.0 + self.loss_tangent * k.re / k.im);
                    infinite + slope * frequency_log
                });
                let tangent = permittivity.mapv(|value| -value.im / value.re);
                Ok((permittivity, tangent))
            }
        }
    }

    fn quasi_static_for(
        &self,
        substrate_permittivity: &Array1<Complex64>,
    ) -> Result<CpwQuasiStatic> {
        let free_space_impedance = (FREE_SPACE_PERMEABILITY / FREE_SPACE_PERMITTIVITY).sqrt();
        let conductor_width = self.width;
        let total_width = 2.0f64.mul_add(self.gap, self.width);
        let modulus = conductor_width / total_width;
        let elliptic = complete_elliptic_integral_first_kind(modulus)?;
        let complementary_elliptic =
            complete_elliptic_integral_first_kind(modulus.mul_add(-modulus, 1.0).sqrt())?;
        let primary_ratio = Self::elliptic_ratio(modulus)?;

        let (mut impedance_factor, mut effective) = if self.has_metal_backside {
            let backside_modulus =
                (std::f64::consts::PI * conductor_width / (4.0 * self.substrate_height)).tanh()
                    / (std::f64::consts::PI * total_width / (4.0 * self.substrate_height)).tanh();
            let backside_ratio = Self::elliptic_ratio(backside_modulus)?;
            let total_ratio = 1.0 / (primary_ratio + backside_ratio);
            (
                free_space_impedance / 2.0 * total_ratio,
                substrate_permittivity
                    .mapv(|value| 1.0 + backside_ratio * total_ratio * (value - 1.0)),
            )
        } else {
            let substrate_modulus =
                (std::f64::consts::PI * conductor_width / (4.0 * self.substrate_height)).sinh()
                    / (std::f64::consts::PI * total_width / (4.0 * self.substrate_height)).sinh();
            let substrate_ratio = Self::elliptic_ratio(substrate_modulus)?;
            (
                free_space_impedance / (4.0 * primary_ratio),
                substrate_permittivity
                    .mapv(|value| 1.0 + (value - 1.0) / 2.0 * substrate_ratio / primary_ratio),
            )
        };

        if let Some(thickness) = self.thickness.filter(|value| *value > 0.0) {
            let correction = 1.25 * thickness / std::f64::consts::PI
                * (1.0 + (4.0 * std::f64::consts::PI * self.width / thickness).ln());
            let effective_modulus =
                modulus + modulus.mul_add(-modulus, 1.0) * correction / (2.0 * self.gap);
            let effective_ratio = Self::elliptic_ratio(effective_modulus)?;
            if self.has_metal_backside {
                let backside_modulus = (std::f64::consts::PI * conductor_width
                    / (4.0 * self.substrate_height))
                    .tanh()
                    / (std::f64::consts::PI * total_width / (4.0 * self.substrate_height)).tanh();
                let backside_ratio = Self::elliptic_ratio(backside_modulus)?;
                impedance_factor =
                    free_space_impedance / (2.0 * (effective_ratio + backside_ratio));
            } else {
                impedance_factor = free_space_impedance / (4.0 * effective_ratio);
            }
            effective.mapv_inplace(|value| {
                value
                    - 0.7 * (value - 1.0) * thickness
                        / self.gap
                        / (primary_ratio + 0.7 * thickness / self.gap)
            });
        }
        let impedance = effective.mapv(|value| impedance_factor / value.sqrt());
        Ok((
            impedance,
            effective,
            modulus,
            elliptic,
            complementary_elliptic,
        ))
    }

    /// Calculate quasi-static impedance and effective permittivity.
    ///
    /// Air- and metal-backed models use the Ghione/Naldi filling factors; a
    /// first-order Gupta thickness correction is applied when requested.
    ///
    /// # Errors
    ///
    /// Returns an error when dielectric dispersion is singular or a derived
    /// elliptic modulus is outside its valid range.
    pub fn quasi_static_characteristics(&self) -> Result<(Array1<Complex64>, Array1<Complex64>)> {
        let (permittivity, _) = self.dielectric_properties()?;
        let input = if self.compatibility_mode == CpwCompatibilityMode::Qucs {
            permittivity.mapv(|value| Complex64::new(value.re, 0.0))
        } else {
            permittivity
        };
        let (impedance, effective, _, _, _) = self.quasi_static_for(&input)?;
        Ok((impedance, effective))
    }

    /// Calculate frequency-dependent impedance and effective permittivity.
    ///
    /// Native/Qucs modes apply the Frankel-Gevorgian dispersion model. ADS
    /// compatibility retains the quasi-static result.
    ///
    /// # Errors
    ///
    /// Returns an error when dielectric dispersion is singular or the
    /// quasi-static elliptic calculations are invalid.
    pub fn frequency_dependent_characteristics(
        &self,
    ) -> Result<(Array1<Complex64>, Array1<Complex64>)> {
        let (permittivity, _) = self.dielectric_properties()?;
        let input = if self.compatibility_mode == CpwCompatibilityMode::Qucs {
            permittivity.mapv(|value| Complex64::new(value.re, 0.0))
        } else {
            permittivity
        };
        let (quasi_impedance, quasi_effective, _, _, _) = self.quasi_static_for(&input)?;
        if self.compatibility_mode == CpwCompatibilityMode::Ads {
            return Ok((quasi_impedance, quasi_effective));
        }
        let geometry_log = (self.width / self.substrate_height).ln();
        let u = 0.015f64
            .mul_add(-geometry_log, 0.64)
            .mul_add(-geometry_log, 0.54);
        let v = 0.54f64
            .mul_add(-geometry_log, 0.86)
            .mul_add(-geometry_log, 0.43);
        let dispersion_factor = (u * (self.width / self.gap).ln() + v).exp();
        let impedance = Array1::from_shape_fn(self.frequency.points(), |point| {
            let substrate = input[point];
            let cutoff = SPEED_OF_LIGHT / (4.0 * self.substrate_height * (substrate - 1.0).sqrt());
            let quasi_root = quasi_effective[point].sqrt();
            let root = quasi_root
                + (substrate.sqrt() - quasi_root)
                    / (1.0
                        + dispersion_factor
                            * (Complex64::new(self.frequency.values_hz()[point], 0.0) / cutoff)
                                .powf(-1.8));
            quasi_impedance[point] * quasi_root / root
        });
        let effective = Array1::from_shape_fn(self.frequency.points(), |point| {
            let substrate = input[point];
            let cutoff = SPEED_OF_LIGHT / (4.0 * self.substrate_height * (substrate - 1.0).sqrt());
            let quasi_root = quasi_effective[point].sqrt();
            let root = quasi_root
                + (substrate.sqrt() - quasi_root)
                    / (1.0
                        + dispersion_factor
                            * (Complex64::new(self.frequency.values_hz()[point], 0.0) / cutoff)
                                .powf(-1.8));
            root.powi(2)
        });
        Ok((impedance, effective))
    }

    /// Calculate conductor and dielectric attenuation in nepers per meter.
    ///
    /// Conductor loss uses Wheeler's incremental-inductance rule; dielectric
    /// loss uses the frequency-dependent effective permittivity and tangent.
    ///
    /// # Errors
    ///
    /// Returns an error when dielectric dispersion or the quasi-static and
    /// frequency-dependent CPW calculations are invalid.
    pub fn attenuation(&self) -> Result<(Array1<f64>, Array1<f64>)> {
        let (permittivity, tangent) = self.dielectric_properties()?;
        let input = if self.compatibility_mode == CpwCompatibilityMode::Qucs {
            permittivity.mapv(|value| Complex64::new(value.re, 0.0))
        } else {
            permittivity.clone()
        };
        let (_, _, modulus, elliptic, complementary_elliptic) = self.quasi_static_for(&input)?;
        let (_, effective) = self.frequency_dependent_characteristics()?;
        let conductor = if let (Some(thickness), Some(resistivity)) = (
            self.thickness.filter(|value| *value > 0.0),
            self.resistivity,
        ) {
            let n = (1.0 - modulus) * 8.0 * std::f64::consts::PI / (thickness * (1.0 + modulus));
            let inner = self.width / 2.0;
            let outer = inner + self.gap;
            let geometry = (std::f64::consts::PI + (n * inner).ln()) / inner
                + (std::f64::consts::PI + (n * outer).ln()) / outer;
            let free_space_impedance = (FREE_SPACE_PERMEABILITY / FREE_SPACE_PERMITTIVITY).sqrt();
            Array1::from_shape_fn(self.frequency.points(), |point| {
                let surface_resistivity = (std::f64::consts::PI
                    * self.frequency.values_hz()[point]
                    * FREE_SPACE_PERMEABILITY
                    * resistivity)
                    .sqrt();
                surface_resistivity * effective[point].re.sqrt() * geometry
                    / (4.0
                        * free_space_impedance
                        * elliptic
                        * complementary_elliptic
                        * modulus.mul_add(-modulus, 1.0))
            })
        } else {
            Array1::zeros(self.frequency.points())
        };
        let dielectric = Array1::from_shape_fn(self.frequency.points(), |point| {
            let effective_real = effective[point].re;
            std::f64::consts::PI * permittivity[point].re / (permittivity[point].re - 1.0)
                * (effective_real - 1.0)
                / effective_real.sqrt()
                * tangent[point]
                * self.frequency.values_hz()[point]
                / SPEED_OF_LIGHT
        });
        Ok((conductor, dielectric))
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

impl Media for Cpw {
    fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    fn propagation_constant(&self) -> Result<Array1<Complex64>> {
        let (_, effective) = self.frequency_dependent_characteristics()?;
        let (conductor, dielectric) = self.attenuation()?;
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            Complex64::new(
                conductor[point] + dielectric[point],
                std::f64::consts::TAU
                    * self.frequency.values_hz()[point]
                    * effective[point].re.sqrt()
                    / SPEED_OF_LIGHT,
            )
        }))
    }

    fn characteristic_impedance(&self) -> Result<Array1<Complex64>> {
        if let Some(impedance) = &self.characteristic_impedance_override {
            Ok(impedance.clone())
        } else {
            Ok(self.frequency_dependent_characteristics()?.0)
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

impl fmt::Display for Cpw {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Coplanar Waveguide Media.  {}-{} {}. {} points\n W= {:.2e}m, S= {:.2e}m",
            self.frequency.start_scaled().unwrap_or_default(),
            self.frequency.stop_scaled().unwrap_or_default(),
            self.frequency.unit().symbol(),
            self.frequency.points(),
            self.width,
            self.gap,
        )
    }
}
