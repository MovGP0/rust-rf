use super::media::*;
use super::*;
/// A homogeneous plane-wave medium.
///
/// Origin: `skrf/media/freespace.py::Freespace`.
#[derive(Clone, Debug)]
pub struct Freespace {
    pub frequency: Frequency,
    pub relative_permittivity: Array1<Complex64>,
    pub relative_permeability: Array1<Complex64>,
    pub electric_loss_tangent: Option<Array1<f64>>,
    pub magnetic_loss_tangent: Option<Array1<f64>>,
    pub resistivity: Option<Array1<f64>>,
    pub port_z0: Option<Array1<Complex64>>,
    pub characteristic_impedance_override: Option<Array1<Complex64>>,
}

impl Freespace {
    /// Port of `skrf.media.Freespace.__init__`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frequency: Frequency,
        relative_permittivity: Array1<Complex64>,
        relative_permeability: Array1<Complex64>,
        electric_loss_tangent: Option<Array1<f64>>,
        magnetic_loss_tangent: Option<Array1<f64>>,
        resistivity: Option<Array1<f64>>,
        port_z0: Option<Array1<Complex64>>,
        characteristic_impedance_override: Option<Array1<Complex64>>,
    ) -> Result<Self> {
        let points = frequency.points();
        for (name, length) in [
            ("relative permittivity", relative_permittivity.len()),
            ("relative permeability", relative_permeability.len()),
        ] {
            if length != points {
                return Err(Error::IncompatibleShape(format!(
                    "freespace {name} has {length} values for {points} frequency points"
                )));
            }
        }
        for (name, values) in [
            ("electric loss tangent", electric_loss_tangent.as_ref()),
            ("magnetic loss tangent", magnetic_loss_tangent.as_ref()),
        ] {
            if values.is_some_and(|values| values.len() != points) {
                return Err(Error::IncompatibleShape(format!(
                    "freespace {name} must match the frequency length"
                )));
            }
        }
        for (name, values) in [
            ("resistivity", resistivity.as_ref().map(Array1::len)),
            ("port impedance", port_z0.as_ref().map(Array1::len)),
            (
                "characteristic-impedance override",
                characteristic_impedance_override.as_ref().map(Array1::len),
            ),
        ] {
            if values.is_some_and(|length| length != points) {
                return Err(Error::IncompatibleShape(format!(
                    "freespace {name} must match the frequency length"
                )));
            }
        }
        if resistivity.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        }) {
            return Err(Error::Unsupported(
                "freespace resistivity must contain positive finite values".to_owned(),
            ));
        }
        Ok(Self {
            frequency,
            relative_permittivity,
            relative_permeability,
            electric_loss_tangent,
            magnetic_loss_tangent,
            resistivity,
            port_z0,
            characteristic_impedance_override,
        })
    }

    pub fn from_scalars(
        frequency: Frequency,
        relative_permittivity: Complex64,
        relative_permeability: Complex64,
    ) -> Result<Self> {
        let points = frequency.points();
        Self::new(
            frequency,
            Array1::from_elem(points, relative_permittivity),
            Array1::from_elem(points, relative_permeability),
            None,
            None,
            None,
            None,
            None,
        )
    }

    pub fn vacuum(frequency: Frequency) -> Result<Self> {
        Self::from_scalars(
            frequency,
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        )
    }

    /// Resolves a conductor name or alias through `skrf.data.materials`.
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

    /// Complex dielectric permittivity in farads per meter.
    pub fn permittivity(&self) -> Array1<Complex64> {
        Array1::from_shape_fn(self.frequency.points(), |point| {
            let relative = self.electric_loss_tangent.as_ref().map_or(
                self.relative_permittivity[point],
                |loss_tangent| {
                    Complex64::new(
                        self.relative_permittivity[point].re,
                        -self.relative_permittivity[point].re * loss_tangent[point],
                    )
                },
            );
            relative * FREE_SPACE_PERMITTIVITY
        })
    }

    /// Complex magnetic permeability in henries per meter.
    pub fn permeability(&self) -> Array1<Complex64> {
        Array1::from_shape_fn(self.frequency.points(), |point| {
            let relative = self.magnetic_loss_tangent.as_ref().map_or(
                self.relative_permeability[point],
                |loss_tangent| {
                    Complex64::new(
                        self.relative_permeability[point].re,
                        -self.relative_permeability[point].re * loss_tangent[point],
                    )
                },
            );
            relative * FREE_SPACE_PERMEABILITY
        })
    }

    /// Permittivity with finite conductivity represented as dielectric loss.
    pub fn permittivity_with_resistivity(&self) -> Result<Array1<Complex64>> {
        let mut permittivity = self.permittivity();
        if let Some(resistivity) = &self.resistivity {
            let angular = self.frequency.angular();
            if angular.iter().any(|value| *value == 0.0) {
                return Err(Error::InvalidFrequency(
                    "freespace resistivity is undefined at zero frequency".to_owned(),
                ));
            }
            for point in 0..self.frequency.points() {
                permittivity[point] -=
                    Complex64::new(0.0, 1.0 / (resistivity[point] * angular[point]));
            }
        }
        Ok(permittivity)
    }

    /// Port of `skrf.media.Freespace.from_distributed_circuit`.
    pub fn from_distributed_circuit(circuit: &DistributedCircuit) -> Result<Self> {
        let angular = circuit.frequency.angular();
        if angular.iter().any(|value| *value == 0.0) {
            return Err(Error::InvalidFrequency(
                "conversion from a distributed circuit is undefined at zero frequency".to_owned(),
            ));
        }
        let impedance = circuit.distributed_impedance();
        let admittance = circuit.distributed_admittance();
        let points = circuit.frequency.points();
        let relative_permittivity = Array1::from_shape_fn(points, |point| {
            Complex64::new(0.0, -1.0) * admittance[point]
                / (angular[point] * FREE_SPACE_PERMITTIVITY)
        });
        let relative_permeability = Array1::from_shape_fn(points, |point| {
            Complex64::new(0.0, -1.0) * impedance[point]
                / (angular[point] * FREE_SPACE_PERMEABILITY)
        });
        Self::new(
            circuit.frequency.clone(),
            relative_permittivity,
            relative_permeability,
            None,
            None,
            None,
            circuit.port_z0.clone(),
            None,
        )
    }

    fn as_defined(&self) -> Result<DefinedGammaZ0> {
        DefinedGammaZ0::new(
            self.frequency.clone(),
            self.propagation_constant()?,
            self.characteristic_impedance()?,
            self.port_z0.clone(),
        )
    }

    /// Port of `skrf.media.Freespace.plot_ep` as backend-independent plot data.
    pub fn plot_permittivity(&self) -> crate::plotting::Plot {
        complex_material_plot(
            &self.frequency,
            "Relative permittivity",
            "ep_r",
            &self.relative_permittivity,
        )
    }

    /// Port of `skrf.media.Freespace.plot_mu` as backend-independent plot data.
    pub fn plot_permeability(&self) -> crate::plotting::Plot {
        complex_material_plot(
            &self.frequency,
            "Relative permeability",
            "mu_r",
            &self.relative_permeability,
        )
    }

    /// Port of `skrf.media.Freespace.plot_ep_mu` as backend-independent plot data.
    pub fn plot_permittivity_and_permeability(&self) -> crate::plotting::Plot {
        let mut plot = self.plot_permittivity();
        plot.title = "Relative permittivity and permeability".to_owned();
        plot.series.extend(self.plot_permeability().series);
        plot
    }
}

impl Media for Freespace {
    fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    fn propagation_constant(&self) -> Result<Array1<Complex64>> {
        let permittivity = self.permittivity_with_resistivity()?;
        let permeability = self.permeability();
        let angular = self.frequency.angular();
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            Complex64::new(0.0, angular[point]) * (permittivity[point] * permeability[point]).sqrt()
        }))
    }

    fn characteristic_impedance(&self) -> Result<Array1<Complex64>> {
        if let Some(impedance) = &self.characteristic_impedance_override {
            return Ok(impedance.clone());
        }
        let permittivity = self.permittivity_with_resistivity()?;
        if permittivity.iter().any(|value| value.norm_sqr() == 0.0) {
            return Err(Error::Unsupported(
                "freespace permittivity must be non-zero".to_owned(),
            ));
        }
        let permeability = self.permeability();
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            (permeability[point] / permittivity[point]).sqrt()
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

impl fmt::Display for Freespace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Freespace Media.  {}-{} {}.  {} points",
            self.frequency.start_scaled().unwrap_or_default(),
            self.frequency.stop_scaled().unwrap_or_default(),
            self.frequency.unit().symbol(),
            self.frequency.points(),
        )
    }
}

fn complex_material_plot(
    frequency: &Frequency,
    title: &str,
    label: &str,
    values: &Array1<Complex64>,
) -> crate::plotting::Plot {
    let x = frequency.values_hz().to_vec();
    crate::plotting::Plot {
        title: title.to_owned(),
        x_label: "Frequency (Hz)".to_owned(),
        y_label: "Relative material property".to_owned(),
        series: vec![
            crate::plotting::PlotSeries {
                label: format!("{label} real"),
                x: x.clone(),
                y: values.iter().map(|value| value.re).collect(),
            },
            crate::plotting::PlotSeries {
                label: format!("{label} imag"),
                x,
                y: values.iter().map(|value| value.im).collect(),
            },
        ],
    }
}
