use super::media::*;
use super::*;
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DielectricDispersionModel {
    #[default]
    FrequencyInvariant,
    DjordjevicSvensson {
        low_frequency_hz: f64,
        high_frequency_hz: f64,
        specification_frequency_hz: f64,
    },
}

#[derive(Clone, Debug)]
pub enum DefinedCharacteristicImpedance {
    Nominal(f64),
    Raw(Array1<Complex64>),
}

/// Transmission-line medium defined by attenuation, permittivity, loss tangent, and impedance.
///
/// Origin: `skrf/media/definedAEpTandZ0.py::DefinedAEpTandZ0`.
#[derive(Clone, Debug)]
pub struct DefinedAEpTandZ0 {
    pub frequency: Frequency,
    pub conductor_attenuation: Array1<f64>,
    pub attenuation_reference_frequency_hz: f64,
    pub relative_permittivity: Array1<f64>,
    pub loss_tangent: Array1<f64>,
    pub impedance: DefinedCharacteristicImpedance,
    pub port_z0: Option<Array1<Complex64>>,
    pub dispersion_model: DielectricDispersionModel,
}

impl DefinedAEpTandZ0 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frequency: Frequency,
        conductor_attenuation: Array1<f64>,
        attenuation_reference_frequency_hz: f64,
        relative_permittivity: Array1<f64>,
        loss_tangent: Array1<f64>,
        impedance: DefinedCharacteristicImpedance,
        port_z0: Option<Array1<Complex64>>,
        dispersion_model: DielectricDispersionModel,
    ) -> Result<Self> {
        let points = frequency.points();
        for (name, values) in [
            ("conductor attenuation", &conductor_attenuation),
            ("relative permittivity", &relative_permittivity),
            ("loss tangent", &loss_tangent),
        ] {
            if values.len() != points {
                return Err(Error::IncompatibleShape(format!(
                    "defined medium {name} has {} values for {points} frequency points",
                    values.len()
                )));
            }
        }
        if !attenuation_reference_frequency_hz.is_finite()
            || attenuation_reference_frequency_hz <= 0.0
        {
            return Err(Error::InvalidFrequency(
                "attenuation reference frequency must be positive and finite".to_owned(),
            ));
        }
        if conductor_attenuation
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(Error::Unsupported(
                "conductor attenuation must be finite and non-negative".to_owned(),
            ));
        }
        if relative_permittivity
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::Unsupported(
                "relative permittivity must be positive and finite".to_owned(),
            ));
        }
        if loss_tangent
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(Error::Unsupported(
                "loss tangent must be finite and non-negative".to_owned(),
            ));
        }
        match &impedance {
            DefinedCharacteristicImpedance::Nominal(value)
                if !value.is_finite() || *value <= 0.0 =>
            {
                return Err(Error::Unsupported(
                    "nominal impedance must be positive and finite".to_owned(),
                ));
            }
            DefinedCharacteristicImpedance::Raw(values)
                if values.len() != 1 && values.len() != points =>
            {
                return Err(Error::IncompatibleShape(
                    "raw characteristic impedance must contain one value or match the frequency length"
                        .to_owned(),
                ));
            }
            _ => {}
        }
        if port_z0
            .as_ref()
            .is_some_and(|values| values.len() != points)
        {
            return Err(Error::IncompatibleShape(
                "defined-medium port impedance must match the frequency length".to_owned(),
            ));
        }
        if let DielectricDispersionModel::DjordjevicSvensson {
            low_frequency_hz,
            high_frequency_hz,
            specification_frequency_hz,
        } = dispersion_model
        {
            if !low_frequency_hz.is_finite()
                || !high_frequency_hz.is_finite()
                || !specification_frequency_hz.is_finite()
                || low_frequency_hz <= 0.0
                || low_frequency_hz >= high_frequency_hz
                || specification_frequency_hz <= 0.0
            {
                return Err(Error::InvalidFrequency(
                    "Djordjevic-Svensson frequencies must satisfy 0 < low < high and specification > 0"
                        .to_owned(),
                ));
            }
        }
        Ok(Self {
            frequency,
            conductor_attenuation,
            attenuation_reference_frequency_hz,
            relative_permittivity,
            loss_tangent,
            impedance,
            port_z0,
            dispersion_model,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_scalars(
        frequency: Frequency,
        conductor_attenuation: f64,
        attenuation_reference_frequency_hz: f64,
        relative_permittivity: f64,
        loss_tangent: f64,
        nominal_impedance: f64,
        port_impedance: Option<Complex64>,
        dispersion_model: DielectricDispersionModel,
    ) -> Result<Self> {
        let points = frequency.points();
        Self::new(
            frequency,
            Array1::from_elem(points, conductor_attenuation),
            attenuation_reference_frequency_hz,
            Array1::from_elem(points, relative_permittivity),
            Array1::from_elem(points, loss_tangent),
            DefinedCharacteristicImpedance::Nominal(nominal_impedance),
            port_impedance.map(|value| Array1::from_elem(points, value)),
            dispersion_model,
        )
    }

    pub fn relative_permittivity_at_frequency(&self) -> Result<Array1<Complex64>> {
        match self.dispersion_model {
            DielectricDispersionModel::FrequencyInvariant => {
                Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
                    Complex64::new(
                        self.relative_permittivity[point],
                        -self.relative_permittivity[point] * self.loss_tangent[point],
                    )
                }))
            }
            DielectricDispersionModel::DjordjevicSvensson {
                low_frequency_hz,
                high_frequency_hz,
                specification_frequency_hz,
            } => {
                let k = ((Complex64::new(high_frequency_hz, specification_frequency_hz))
                    / Complex64::new(low_frequency_hz, specification_frequency_hz))
                .ln();
                if k.im == 0.0 {
                    return Err(Error::Unsupported(
                        "Djordjevic-Svensson logarithmic slope is singular".to_owned(),
                    ));
                }
                Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
                    let frequency = self.frequency.values_hz()[point];
                    let frequency_log = ((Complex64::new(high_frequency_hz, frequency))
                        / Complex64::new(low_frequency_hz, frequency))
                    .ln();
                    let dielectric_slope =
                        -self.loss_tangent[point] * self.relative_permittivity[point] / k.im;
                    let infinite_frequency_permittivity = self.relative_permittivity[point]
                        * (1.0 + self.loss_tangent[point] * k.re / k.im);
                    infinite_frequency_permittivity + dielectric_slope * frequency_log
                }))
            }
        }
    }

    pub fn frequency_dependent_loss_tangent(&self) -> Result<Array1<f64>> {
        let permittivity = self.relative_permittivity_at_frequency()?;
        if permittivity.iter().any(|value| value.re == 0.0) {
            return Err(Error::Unsupported(
                "frequency-dependent permittivity must have a non-zero real part".to_owned(),
            ));
        }
        Ok(permittivity.mapv(|value| -value.im / value.re))
    }

    pub fn conductor_attenuation_per_meter(&self) -> Result<Array1<f64>> {
        if self
            .frequency
            .values_hz()
            .iter()
            .any(|frequency| *frequency < 0.0)
        {
            return Err(Error::InvalidFrequency(
                "conductor attenuation requires non-negative frequencies".to_owned(),
            ));
        }
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            self.conductor_attenuation[point] * std::f64::consts::LN_10 / 20.0
                * (self.frequency.values_hz()[point] / self.attenuation_reference_frequency_hz)
                    .sqrt()
        }))
    }

    pub fn dielectric_attenuation_per_meter(&self) -> Result<Array1<f64>> {
        let permittivity = self.relative_permittivity_at_frequency()?;
        let loss_tangent = self.frequency_dependent_loss_tangent()?;
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            std::f64::consts::PI * permittivity[point].re.sqrt() * self.frequency.values_hz()[point]
                / SPEED_OF_LIGHT
                * loss_tangent[point]
        }))
    }

    pub fn phase_constant(&self) -> Result<Array1<f64>> {
        let permittivity = self.relative_permittivity_at_frequency()?;
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            std::f64::consts::TAU
                * self.frequency.values_hz()[point]
                * permittivity[point].re.sqrt()
                / SPEED_OF_LIGHT
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

impl fmt::Display for DefinedAEpTandZ0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let scaled = self.frequency.scaled();
        write!(
            formatter,
            "DefinedAEpTandZ0 Media. {}-{} {}, {} points",
            scaled.first().copied().unwrap_or_default(),
            scaled.last().copied().unwrap_or_default(),
            self.frequency.unit().symbol(),
            self.frequency.points()
        )
    }
}

impl Media for DefinedAEpTandZ0 {
    fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    fn propagation_constant(&self) -> Result<Array1<Complex64>> {
        let conductor = self.conductor_attenuation_per_meter()?;
        let dielectric = self.dielectric_attenuation_per_meter()?;
        let phase = self.phase_constant()?;
        Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
            Complex64::new(conductor[point] + dielectric[point], phase[point])
        }))
    }

    fn characteristic_impedance(&self) -> Result<Array1<Complex64>> {
        match &self.impedance {
            DefinedCharacteristicImpedance::Raw(values) if values.len() == 1 => {
                Ok(Array1::from_elem(self.frequency.points(), values[0]))
            }
            DefinedCharacteristicImpedance::Raw(values) => Ok(values.clone()),
            DefinedCharacteristicImpedance::Nominal(nominal) => {
                let conductor = self.conductor_attenuation_per_meter()?;
                let dielectric = self.dielectric_attenuation_per_meter()?;
                let angular = self.frequency.angular();
                Ok(Array1::from_shape_fn(self.frequency.points(), |point| {
                    let root_permittivity = self.relative_permittivity[point].sqrt();
                    let resistance = 2.0 * nominal * conductor[point];
                    let inductance = nominal * root_permittivity / SPEED_OF_LIGHT;
                    let conductance = 2.0 / nominal * dielectric[point];
                    let capacitance = root_permittivity / (SPEED_OF_LIGHT * nominal);
                    (Complex64::new(resistance, angular[point] * inductance)
                        / Complex64::new(conductance, angular[point] * capacitance))
                    .sqrt()
                }))
            }
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
