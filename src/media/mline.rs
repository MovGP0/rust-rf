use super::media::*;
use super::*;
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MicrostripQuasiStaticModel {
    #[default]
    HammerstadJensen,
    Wheeler,
    Schneider,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MicrostripDispersionModel {
    #[default]
    HammerstadJensen,
    None,
    Schneider,
    KirschningJansen,
    Yamashita,
    Kobayashi,
}

/// Microstrip transmission-line medium.
///
/// Origin: `skrf/media/mline.py::MLine`.
#[derive(Clone, Debug)]
pub struct MicrostripLine {
    pub frequency: Frequency,
    pub width: f64,
    pub substrate_height: f64,
    pub thickness: Option<f64>,
    pub relative_permittivity: f64,
    pub relative_permeability: f64,
    pub loss_tangent: f64,
    pub resistivity: Option<f64>,
    pub roughness: f64,
    pub dielectric_model: DielectricDispersionModel,
    pub quasi_static_model: MicrostripQuasiStaticModel,
    pub dispersion_model: MicrostripDispersionModel,
    pub compatibility_mode: CpwCompatibilityMode,
    pub port_z0: Option<Array1<Complex64>>,
    pub characteristic_impedance_override: Option<Array1<Complex64>>,
}

type MicrostripCharacteristics = (Array1<Complex64>, Array1<Complex64>, Array1<Complex64>);

impl MicrostripLine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frequency: Frequency,
        width: f64,
        substrate_height: f64,
        thickness: Option<f64>,
        relative_permittivity: f64,
        relative_permeability: f64,
        loss_tangent: f64,
        resistivity: Option<f64>,
        roughness: f64,
        dielectric_model: DielectricDispersionModel,
        quasi_static_model: MicrostripQuasiStaticModel,
        dispersion_model: MicrostripDispersionModel,
        compatibility_mode: CpwCompatibilityMode,
        port_z0: Option<Array1<Complex64>>,
        characteristic_impedance_override: Option<Array1<Complex64>>,
    ) -> Result<Self> {
        for (name, value) in [
            ("width", width),
            ("substrate height", substrate_height),
            ("relative permittivity", relative_permittivity),
            ("relative permeability", relative_permeability),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::Unsupported(format!(
                    "microstrip {name} must be positive and finite"
                )));
            }
        }
        if thickness.is_some_and(|value| !value.is_finite() || value < 0.0)
            || !loss_tangent.is_finite()
            || loss_tangent < 0.0
            || resistivity.is_some_and(|value| !value.is_finite() || value <= 0.0)
            || !roughness.is_finite()
            || roughness < 0.0
        {
            return Err(Error::Unsupported(
                "microstrip thickness/loss values are not physical".to_owned(),
            ));
        }
        if thickness.is_some_and(|value| value > 0.0) && resistivity.is_none() {
            return Err(Error::Unsupported(
                "microstrip conductor loss requires resistivity for non-zero thickness".to_owned(),
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
                    "microstrip {name} must match the frequency length"
                )));
            }
        }
        Ok(Self {
            frequency,
            width,
            substrate_height,
            thickness,
            relative_permittivity,
            relative_permeability,
            loss_tangent,
            resistivity,
            roughness,
            dielectric_model,
            quasi_static_model,
            dispersion_model,
            compatibility_mode,
            port_z0,
            characteristic_impedance_override,
        })
    }

    /// Resolves conductor resistivity through `skrf.data.materials`.
    pub fn set_resistivity_material(&mut self, material: &str) -> Result<()> {
        let properties = crate::data::MATERIALS
            .get(material.to_ascii_lowercase().as_str())
            .ok_or_else(|| Error::Unsupported(format!("unknown material `{material}`")))?;
        self.resistivity = Some(properties.resistivity_ohm_meter.ok_or_else(|| {
            Error::Unsupported(format!("material `{material}` does not define resistivity"))
        })?);
        Ok(())
    }

    fn dielectric_properties(&self) -> Result<(Array1<Complex64>, Array1<f64>)> {
        let cpw = Cpw::new(
            self.frequency.clone(),
            self.width,
            self.width,
            self.substrate_height,
            None,
            self.relative_permittivity,
            self.loss_tangent,
            None,
            self.dielectric_model,
            false,
            CpwCompatibilityMode::Native,
            None,
            None,
        )?;
        cpw.dielectric_properties()
    }

    fn hammerstad_impedance(normalized_width: Complex64) -> Complex64 {
        let free_space_impedance = (FREE_SPACE_PERMEABILITY / FREE_SPACE_PERMITTIVITY).sqrt();
        let factor = 6.0
            + (std::f64::consts::TAU - 6.0)
                * (-(Complex64::new(30.666, 0.0) / normalized_width).powf(0.7528)).exp();
        free_space_impedance / std::f64::consts::TAU
            * (factor / normalized_width
                + (Complex64::new(1.0, 0.0)
                    + (Complex64::new(2.0, 0.0) / normalized_width).powi(2))
                .sqrt())
            .ln()
    }

    fn hammerstad_effective_permittivity(
        normalized_width: Complex64,
        relative_permittivity: Complex64,
    ) -> Complex64 {
        let a = 1.0
            + ((normalized_width.powi(4) + (normalized_width / 52.0).powi(2))
                / (normalized_width.powi(4) + 0.432))
                .ln()
                / 49.0
            + (Complex64::new(1.0, 0.0) + (normalized_width / 18.1).powi(3)).ln() / 18.7;
        let b = 0.564 * ((relative_permittivity - 0.9) / (relative_permittivity + 3.0)).powf(0.053);
        (relative_permittivity + 1.0) / 2.0
            + (relative_permittivity - 1.0) / 2.0
                * (Complex64::new(1.0, 0.0) + 10.0 / normalized_width).powc(-a * b)
    }

    fn wheeler_quasi_static(&self, dielectric: &Array1<Complex64>) -> MicrostripCharacteristics {
        let free_space_impedance = (FREE_SPACE_PERMEABILITY / FREE_SPACE_PERMITTIVITY).sqrt();
        let ratio = self.width / self.substrate_height;
        let thickness_correction =
            self.thickness
                .filter(|thickness| *thickness > 0.0)
                .map_or(0.0, |thickness| {
                    thickness / std::f64::consts::PI
                        * (4.0 * std::f64::consts::E / (thickness / self.substrate_height).abs()
                            + (1.0 / (std::f64::consts::PI * (self.width / thickness + 1.1)))
                                .powi(2))
                        .ln()
                });
        let effective_width = dielectric.mapv(|permittivity| {
            self.width
                + (Complex64::new(1.0, 0.0) + Complex64::new(1.0, 0.0) / permittivity) / 2.0
                    * thickness_correction
        });
        let impedance = Array1::from_shape_fn(self.frequency.points(), |point| {
            let permittivity = dielectric[point];
            let width = effective_width[point];
            if ratio < 3.3 {
                let cp = (4.0 * self.substrate_height / width
                    + ((4.0 * self.substrate_height / width).powi(2) + 2.0).sqrt())
                .ln();
                let b = (permittivity - 1.0) / (permittivity + 1.0) / 2.0
                    * ((std::f64::consts::PI / 2.0).ln()
                        + (4.0 / std::f64::consts::PI).ln() / permittivity);
                (cp - b) * free_space_impedance
                    / (std::f64::consts::PI * (2.0 * (permittivity + 1.0)).sqrt())
            } else {
                let cp = Complex64::new(1.0 + (std::f64::consts::PI / 2.0).ln(), 0.0)
                    + (width / (2.0 * self.substrate_height) + 0.94).ln();
                let d = 1.0 / (2.0 * std::f64::consts::PI)
                    * (1.0 + (std::f64::consts::PI.powi(2) / 16.0).ln())
                    * (permittivity - 1.0)
                    / permittivity.powi(2);
                let x = 2.0 * 2.0_f64.ln() / std::f64::consts::PI
                    + width / (2.0 * self.substrate_height)
                    + (permittivity + 1.0) / (2.0 * std::f64::consts::PI * permittivity) * cp
                    + d;
                free_space_impedance / (2.0 * x * permittivity.sqrt())
            }
        });
        let effective = Array1::from_shape_fn(self.frequency.points(), |point| {
            let permittivity = dielectric[point];
            let width = effective_width[point];
            if ratio < 1.3 {
                let a = (8.0 * self.substrate_height / width).ln()
                    + (width / self.substrate_height).powi(2) / 32.0;
                let b = (permittivity - 1.0) / (permittivity + 1.0) / 2.0
                    * ((std::f64::consts::PI / 2.0).ln()
                        + (4.0 / std::f64::consts::PI).ln() / permittivity);
                (permittivity + 1.0) / 2.0 * (a / (a - b)).powi(2)
            } else {
                let normalized_width = width / self.substrate_height;
                let d = (permittivity - 1.0) / (2.0 * std::f64::consts::PI * permittivity)
                    * ((2.1349 * normalized_width + 4.0137).ln() - 0.5169 / permittivity);
                let e = normalized_width / 2.0
                    + (8.5397 * normalized_width + 16.0547).ln() / std::f64::consts::PI;
                permittivity * ((e - d) / e).powi(2)
            }
        });
        (impedance, effective, effective_width)
    }

    fn schneider_quasi_static(&self, dielectric: &Array1<Complex64>) -> MicrostripCharacteristics {
        let free_space_impedance = (FREE_SPACE_PERMEABILITY / FREE_SPACE_PERMITTIVITY).sqrt();
        let initial_ratio = self.width / self.substrate_height;
        let width_correction = self
            .thickness
            .filter(|thickness| *thickness > 0.0 && *thickness < self.width / 2.0)
            .map_or(0.0, |thickness| {
                let argument = if initial_ratio < 1.0 / (2.0 * std::f64::consts::PI) {
                    2.0 * std::f64::consts::PI * self.width / thickness
                } else {
                    self.substrate_height / thickness
                };
                let correction = thickness / std::f64::consts::PI * (1.0 + (2.0 * argument).ln());
                if thickness / correction >= 0.75 {
                    0.0
                } else {
                    correction
                }
            });
        let width = self.width + width_correction;
        let normalized_width = width / self.substrate_height;
        let effective = dielectric.mapv(|permittivity| {
            (permittivity + 1.0) / 2.0
                + (permittivity - 1.0) / (2.0 * (1.0 + 10.0 / normalized_width).sqrt())
        });
        let normalized_impedance = if normalized_width < 1.0 {
            (8.0 / normalized_width + normalized_width / 4.0).ln() / (2.0 * std::f64::consts::PI)
        } else {
            1.0 / (normalized_width + 2.42 - 0.44 / normalized_width
                + (1.0 - 1.0 / normalized_width).powi(6))
        };
        let impedance =
            effective.mapv(|value| free_space_impedance * normalized_impedance / value.sqrt());
        (
            impedance,
            effective,
            Array1::from_elem(self.frequency.points(), Complex64::new(width, 0.0)),
        )
    }

    fn kirschning_effective_permittivity(
        normalized_width: Complex64,
        normalized_frequency: f64,
        relative_permittivity: Complex64,
        quasi_effective: Complex64,
    ) -> Complex64 {
        let p1 = 0.27488
            + (0.6315 + 0.525 / (1.0 + 0.0157 * normalized_frequency).powi(20)) * normalized_width
            - 0.065683 * (-8.7513 * normalized_width).exp();
        let p2 = 0.33622 * (Complex64::new(1.0, 0.0) - (-0.03442 * relative_permittivity).exp());
        let p3 = 0.0363
            * (-4.6 * normalized_width).exp()
            * (1.0 - (-(normalized_frequency / 38.7).powf(4.97)).exp());
        let p4 = 1.0
            + 2.751
                * (Complex64::new(1.0, 0.0) - (-(relative_permittivity / 15.916).powi(8)).exp());
        let pf = p1 * p2 * ((0.1844 + p3 * p4) * normalized_frequency).powf(1.5763);
        relative_permittivity
            - (relative_permittivity - quasi_effective) / (Complex64::new(1.0, 0.0) + pf)
    }

    fn kirschning_impedance(
        normalized_width: Complex64,
        normalized_frequency: f64,
        relative_permittivity: Complex64,
        quasi_effective: Complex64,
        effective: Complex64,
        quasi_impedance: Complex64,
    ) -> Complex64 {
        let r1 = cap_real(0.03891 * relative_permittivity.powf(1.4), 20.0);
        let r2 = cap_real(0.2671 * normalized_width.powi(7), 20.0);
        let r3 = 4.766 * (-3.228 * normalized_width.powf(0.641)).exp();
        let r4 = 0.016 + (0.0514 * relative_permittivity).powf(4.524);
        let r5 = (normalized_frequency / 28.843).powi(12);
        let r6 = cap_real(22.20 * normalized_width.powf(1.92), 20.0);
        let r7 = 1.206 - 0.3144 * (-r1).exp() * (Complex64::new(1.0, 0.0) - (-r2).exp());
        let r8 = 1.0
            + 1.275
                * (Complex64::new(1.0, 0.0)
                    - (-0.004625
                        * r3
                        * relative_permittivity.powf(1.674)
                        * (normalized_frequency / 18.365).powf(2.745))
                    .exp());
        let r9 = 5.086 * r4 * r5 / (0.3838 + 0.386 * r4) * (-r6).exp() / (1.0 + 1.2992 * r5)
            * (relative_permittivity - 1.0).powi(6)
            / (1.0 + 10.0 * (relative_permittivity - 1.0).powi(6));
        let r10 = 0.00044 * relative_permittivity.powf(2.136) + 0.0184;
        let normalized_19 = (normalized_frequency / 19.47).powi(6);
        let r11 = normalized_19 / (1.0 + 0.0962 * normalized_19);
        let r12 = Complex64::new(1.0, 0.0) / (1.0 + 0.00245 * normalized_width.powi(2));
        let r13 = 0.9408 * effective.powc(r8) - 0.9603;
        let r14 = (0.9408 - r9) * quasi_effective.powc(r8) - 0.9603;
        let r15 = 0.707 * r10 * (normalized_frequency / 12.3).powf(1.097);
        let r16 = 1.0
            + 0.0503
                * relative_permittivity.powi(2)
                * r11
                * (Complex64::new(1.0, 0.0) - (-(normalized_width / 15.0).powi(6)).exp());
        let r17 = r7
            * (Complex64::new(1.0, 0.0)
                - 1.1241 * r12 / r16 * (-0.026 * normalized_frequency.powf(1.15656) - r15).exp());
        quasi_impedance * (r13 / r14).powc(r17)
    }

    pub fn quasi_static_characteristics(&self) -> Result<MicrostripCharacteristics> {
        let (dielectric, _) = self.dielectric_properties()?;
        let input = if self.compatibility_mode == CpwCompatibilityMode::Qucs {
            dielectric.mapv(|value| Complex64::new(value.re, 0.0))
        } else {
            dielectric
        };
        match self.quasi_static_model {
            MicrostripQuasiStaticModel::Wheeler => {
                return Ok(self.wheeler_quasi_static(&input));
            }
            MicrostripQuasiStaticModel::Schneider => {
                return Ok(self.schneider_quasi_static(&input));
            }
            MicrostripQuasiStaticModel::HammerstadJensen => {}
        }
        let normalized_width = self.width / self.substrate_height;
        let normalized_thickness = self.thickness.unwrap_or(0.0) / self.substrate_height;
        let width_correction = if normalized_thickness > 0.0 {
            normalized_thickness / std::f64::consts::PI
                * (1.0
                    + 4.0 * std::f64::consts::E / normalized_thickness
                        * (6.517 * normalized_width).sqrt().tanh().powi(2))
                .ln()
        } else {
            0.0
        };
        let homogeneous_width = normalized_width + width_correction;
        let homogeneous_impedance =
            Self::hammerstad_impedance(Complex64::new(homogeneous_width, 0.0));
        let effective_width = Array1::from_shape_fn(self.frequency.points(), |point| {
            let dielectric_correction = width_correction
                * (1.0
                    + Complex64::new(1.0, 0.0)
                        / (input[point] - Complex64::new(1.0, 0.0)).sqrt().cosh())
                / 2.0;
            Complex64::new(normalized_width, 0.0) + dielectric_correction
        });
        let effective = Array1::from_shape_fn(self.frequency.points(), |point| {
            Self::hammerstad_effective_permittivity(effective_width[point], input[point])
        });
        let impedance = Array1::from_shape_fn(self.frequency.points(), |point| {
            let dielectric_impedance = Self::hammerstad_impedance(effective_width[point]);
            dielectric_impedance / effective[point].sqrt()
        });
        let effective = Array1::from_shape_fn(self.frequency.points(), |point| {
            effective[point]
                * (homogeneous_impedance / Self::hammerstad_impedance(effective_width[point]))
                    .powi(2)
        });
        Ok((
            impedance,
            effective,
            effective_width * self.substrate_height,
        ))
    }

    pub fn frequency_dependent_characteristics(
        &self,
    ) -> Result<(Array1<Complex64>, Array1<Complex64>)> {
        let (quasi_impedance, quasi_effective, effective_width) =
            self.quasi_static_characteristics()?;
        if self.dispersion_model == MicrostripDispersionModel::None {
            return Ok((quasi_impedance, quasi_effective));
        }
        let (dielectric, _) = self.dielectric_properties()?;
        let input = if self.compatibility_mode == CpwCompatibilityMode::Qucs {
            dielectric.mapv(|value| Complex64::new(value.re, 0.0))
        } else {
            dielectric
        };
        let free_space_impedance = (FREE_SPACE_PERMEABILITY / FREE_SPACE_PERMITTIVITY).sqrt();
        let effective = Array1::from_shape_fn(self.frequency.points(), |point| {
            let frequency = self.frequency.values_hz()[point];
            let normalized_width = if self.compatibility_mode == CpwCompatibilityMode::Qucs {
                Complex64::new(self.width / self.substrate_height, 0.0)
            } else {
                effective_width[point] / self.substrate_height
            };
            match self.dispersion_model {
                MicrostripDispersionModel::None => quasi_effective[point],
                MicrostripDispersionModel::Schneider => {
                    let k = (quasi_effective[point] / input[point]).sqrt();
                    let normalized_frequency = 4.0 * self.substrate_height * frequency
                        / SPEED_OF_LIGHT
                        * (input[point] - 1.0).sqrt();
                    quasi_effective[point]
                        * ((Complex64::new(1.0, 0.0) + normalized_frequency.powi(2))
                            / (Complex64::new(1.0, 0.0) + k * normalized_frequency.powi(2)))
                        .powi(2)
                }
                MicrostripDispersionModel::HammerstadJensen => {
                    let factor = std::f64::consts::PI.powi(2) / 12.0 * (input[point] - 1.0)
                        / quasi_effective[point]
                        * (2.0 * std::f64::consts::PI * quasi_impedance[point]
                            / free_space_impedance)
                            .sqrt();
                    let normalized_frequency =
                        2.0 * FREE_SPACE_PERMEABILITY * self.substrate_height * frequency
                            / quasi_impedance[point];
                    input[point]
                        - (input[point] - quasi_effective[point])
                            / (Complex64::new(1.0, 0.0) + factor * normalized_frequency.powi(2))
                }
                MicrostripDispersionModel::KirschningJansen => {
                    Self::kirschning_effective_permittivity(
                        normalized_width,
                        frequency * self.substrate_height * 1.0e-6,
                        input[point],
                        quasi_effective[point],
                    )
                }
                MicrostripDispersionModel::Yamashita => {
                    let k = (input[point] / quasi_effective[point]).sqrt();
                    let fp = 4.0 * self.substrate_height * frequency / SPEED_OF_LIGHT
                        * (input[point] - 1.0).sqrt()
                        * (0.5 + (1.0 + 2.0 * (1.0 + normalized_width).log(10.0)).powi(2));
                    quasi_effective[point]
                        * ((Complex64::new(1.0, 0.0) + k * fp.powf(1.5) / 4.0)
                            / (Complex64::new(1.0, 0.0) + fp.powf(1.5) / 4.0))
                            .powi(2)
                }
                MicrostripDispersionModel::Kobayashi => {
                    let fk = SPEED_OF_LIGHT
                        * (input[point]
                            * ((quasi_effective[point] - 1.0)
                                / (input[point] - quasi_effective[point]))
                                .sqrt())
                        .atan()
                        / (2.0
                            * std::f64::consts::PI
                            * self.substrate_height
                            * (input[point] - quasi_effective[point]).sqrt());
                    let fh =
                        fk / (0.75 + (0.75 - 0.332 / input[point].powf(1.73)) * normalized_width);
                    let inverse_width = Complex64::new(1.0, 0.0)
                        / (Complex64::new(1.0, 0.0) + normalized_width.sqrt());
                    let n0 = 1.0 + inverse_width + 0.32 * inverse_width.powi(3);
                    let nc = if normalized_width.re < 0.7 {
                        1.0 + 1.4 / (1.0 + normalized_width)
                            * (0.15 - 0.235 * (-0.45 * frequency / fh).exp())
                    } else {
                        Complex64::new(1.0, 0.0)
                    };
                    let exponent = cap_real(n0 * nc, 2.32);
                    input[point]
                        - (input[point] - quasi_effective[point])
                            / (Complex64::new(1.0, 0.0)
                                + (Complex64::new(frequency, 0.0) / fh).powc(exponent))
                }
            }
        });
        let impedance = Array1::from_shape_fn(self.frequency.points(), |point| {
            match self.dispersion_model {
                MicrostripDispersionModel::None => quasi_impedance[point],
                MicrostripDispersionModel::Schneider => {
                    quasi_impedance[point] * (quasi_effective[point] / effective[point]).sqrt()
                }
                MicrostripDispersionModel::HammerstadJensen => {
                    quasi_impedance[point]
                        * (quasi_effective[point] / effective[point]).sqrt()
                        * (effective[point] - 1.0)
                        / (quasi_effective[point] - 1.0)
                }
                MicrostripDispersionModel::KirschningJansen => Self::kirschning_impedance(
                    if self.compatibility_mode == CpwCompatibilityMode::Qucs {
                        Complex64::new(self.width / self.substrate_height, 0.0)
                    } else {
                        effective_width[point] / self.substrate_height
                    },
                    self.frequency.values_hz()[point] * self.substrate_height * 1.0e-6,
                    input[point],
                    quasi_effective[point],
                    effective[point],
                    quasi_impedance[point],
                ),
                MicrostripDispersionModel::Yamashita | MicrostripDispersionModel::Kobayashi
                    if self.compatibility_mode == CpwCompatibilityMode::Qucs =>
                {
                    quasi_impedance[point]
                }
                MicrostripDispersionModel::Yamashita | MicrostripDispersionModel::Kobayashi => {
                    Self::kirschning_impedance(
                        effective_width[point] / self.substrate_height,
                        self.frequency.values_hz()[point] * self.substrate_height * 1.0e-6,
                        input[point],
                        quasi_effective[point],
                        effective[point],
                        quasi_impedance[point],
                    )
                }
            }
        });
        Ok((impedance, effective))
    }

    pub fn attenuation(&self) -> Result<(Array1<f64>, Array1<f64>)> {
        let (dielectric, tangent) = self.dielectric_properties()?;
        let (quasi_impedance, quasi_effective, _) = self.quasi_static_characteristics()?;
        let (impedance, effective) = self.frequency_dependent_characteristics()?;
        let (loss_impedance, loss_effective) =
            if self.compatibility_mode == CpwCompatibilityMode::Qucs {
                (&quasi_impedance, &quasi_effective)
            } else {
                (&impedance, &effective)
            };
        let conductor = if let (Some(thickness), Some(resistivity)) = (
            self.thickness.filter(|value| *value > 0.0),
            self.resistivity,
        ) {
            let _ = thickness;
            let free_space_impedance = (FREE_SPACE_PERMEABILITY / FREE_SPACE_PERMITTIVITY).sqrt();
            Array1::from_shape_fn(self.frequency.points(), |point| {
                let depth = (resistivity
                    / (std::f64::consts::PI
                        * self.frequency.values_hz()[point]
                        * self.relative_permeability
                        * FREE_SPACE_PERMEABILITY))
                    .sqrt();
                let surface = resistivity / depth;
                let current_distribution =
                    (-1.2 * (loss_impedance[point].re / free_space_impedance).powf(0.7)).exp();
                let roughness = 1.0
                    + 2.0 / std::f64::consts::PI * (1.4 * (self.roughness / depth).powi(2)).atan();
                surface / (loss_impedance[point].re * self.width) * current_distribution * roughness
            })
        } else {
            Array1::zeros(self.frequency.points())
        };
        let dielectric_loss = Array1::from_shape_fn(self.frequency.points(), |point| {
            let effective_real = loss_effective[point].re;
            std::f64::consts::PI * dielectric[point].re / (dielectric[point].re - 1.0)
                * (effective_real - 1.0)
                / effective_real.sqrt()
                * tangent[point]
                * self.frequency.values_hz()[point]
                / SPEED_OF_LIGHT
        });
        Ok((conductor, dielectric_loss))
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

fn cap_real(value: Complex64, maximum: f64) -> Complex64 {
    if value.re > maximum {
        Complex64::new(maximum, 0.0)
    } else {
        value
    }
}

impl Media for MicrostripLine {
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

impl fmt::Display for MicrostripLine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Microstrip Media.  {}-{} {}. {} points\n W= {:.2e}m, H= {:.2e}m",
            self.frequency.start_scaled().unwrap_or_default(),
            self.frequency.stop_scaled().unwrap_or_default(),
            self.frequency.unit().symbol(),
            self.frequency.points(),
            self.width,
            self.substrate_height,
        )
    }
}
