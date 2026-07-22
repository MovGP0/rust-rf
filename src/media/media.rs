//! Common transmission-media abstractions and a medium defined directly by
//! propagation constant and characteristic impedance.

use super::{
    Array1, Array2, Array3, Complex64, Error, Frequency, Network, Path, Result,
    SParameterDefinition, random_complex, random_gaussian_polar,
};
use num_traits::ToPrimitive;

/// A transmission medium capable of creating networks over a frequency axis.
///
/// Implementors provide the frequency, propagation constant $\gamma$, and
/// characteristic impedance `$Z_0$`. The default methods construct common
/// distributed and lumped components from those quantities.
pub trait Media {
    /// Returns the medium frequency axis.
    fn frequency(&self) -> &Frequency;

    /// Returns the complex propagation constant $\gamma=\alpha+j\beta$.
    ///
    /// # Errors
    ///
    /// Returns an error when the medium cannot evaluate its propagation constant.
    fn propagation_constant(&self) -> Result<Array1<Complex64>>;

    /// Returns the characteristic impedance `$Z_0$` at every frequency.
    ///
    /// # Errors
    ///
    /// Returns an error when the medium cannot evaluate its characteristic impedance.
    fn characteristic_impedance(&self) -> Result<Array1<Complex64>>;

    /// Returns the optional network port impedance used when generated networks are renormalized.
    fn port_impedance(&self) -> Option<&Array1<Complex64>> {
        None
    }

    /// Returns the number of frequency points.
    fn points(&self) -> usize {
        self.frequency().points()
    }

    /// Returns the attenuation constant $\alpha=\operatorname{Re}(\gamma)$ in Np/m.
    ///
    /// # Errors
    ///
    /// Returns an error when the propagation constant cannot be evaluated.
    fn attenuation_constant(&self) -> Result<Array1<f64>> {
        Ok(self.propagation_constant()?.mapv(|value| value.re))
    }

    /// Returns the phase constant $\beta=\operatorname{Im}(\gamma)$ in rad/m.
    ///
    /// # Errors
    ///
    /// Returns an error when the propagation constant cannot be evaluated.
    fn phase_constant(&self) -> Result<Array1<f64>> {
        Ok(self.propagation_constant()?.mapv(|value| value.im))
    }

    /// Returns the electrical length $\theta=\gamma d$ for a physical distance.
    ///
    /// When `degrees` is true the result is scaled by $180/\pi$.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite distance or unavailable propagation constant.
    fn electrical_length(&self, distance_meters: f64, degrees: bool) -> Result<Array1<Complex64>> {
        if !distance_meters.is_finite() {
            return Err(Error::Unsupported(
                "electrical-length distance must be finite".to_owned(),
            ));
        }
        let scale = if degrees {
            180.0 / std::f64::consts::PI
        } else {
            1.0
        };
        Ok(self
            .propagation_constant()?
            .mapv(|gamma| gamma * distance_meters * scale))
    }

    /// Converts an electrical angle to physical distance at every frequency.
    ///
    /// # Errors
    ///
    /// Returns an error when the phase constant is unavailable or contains zero.
    fn distance_from_electrical_length(&self, angle: f64, degrees: bool) -> Result<Array1<f64>> {
        let angle = if degrees { angle.to_radians() } else { angle };
        let beta = self.phase_constant()?;
        if beta.iter().any(|value| *value == 0.0) {
            return Err(Error::Unsupported(
                "electrical length requires non-zero phase constant".to_owned(),
            ));
        }
        Ok(beta.mapv(|value| angle / value))
    }

    /// Converts an electrical angle to distance at the center frequency.
    ///
    /// # Errors
    ///
    /// Returns an error when distances cannot be evaluated or the frequency axis is empty.
    fn center_distance_from_electrical_length(&self, angle: f64, degrees: bool) -> Result<f64> {
        let distances = self.distance_from_electrical_length(angle, degrees)?;
        distances.get(distances.len() / 2).copied().ok_or_else(|| {
            Error::InvalidFrequency(
                "electrical length requires a non-empty frequency axis".to_owned(),
            )
        })
    }

    /// Converts a length in [`LengthUnit`] to meters.
    ///
    /// # Errors
    ///
    /// Returns an error when the propagation constant or requested unit conversion is invalid.
    fn physical_length(&self, length: f64, unit: LengthUnit) -> Result<f64> {
        media_length_to_meters(
            self.frequency(),
            &self.propagation_constant()?,
            length,
            unit,
        )
    }

    /// Returns backend-independent plot data for the frequency axis.
    fn plot_frequency(&self) -> crate::plotting::Plot {
        let values = self.frequency().values_hz().to_vec();
        crate::plotting::Plot {
            title: "Media frequency".to_owned(),
            x_label: "Point".to_owned(),
            y_label: "Frequency (Hz)".to_owned(),
            series: vec![crate::plotting::PlotSeries {
                label: "frequency".to_owned(),
                x: (0..values.len())
                    .map(|point| point.to_f64().unwrap_or(f64::INFINITY))
                    .collect(),
                y: values,
            }],
        }
    }

    /// Writes frequency, $\gamma$, `$Z_0$`, and port impedance to a CSV file.
    ///
    /// # Errors
    ///
    /// Returns an error when medium values cannot be evaluated or the CSV cannot be written.
    fn write_csv(&self, path: impl AsRef<Path>) -> Result<()>
    where
        Self: Sized,
    {
        DefinedGammaZ0::new(
            self.frequency().clone(),
            self.propagation_constant()?,
            self.characteristic_impedance()?,
            self.port_impedance().cloned(),
        )?
        .write_csv(path)
    }

    /// Returns the complex phase velocity `$v_p=j\omega/\gamma$`.
    ///
    /// # Errors
    ///
    /// Returns an error when the propagation constant cannot be evaluated.
    fn phase_velocity(&self) -> Result<Array1<Complex64>> {
        let gamma = self.propagation_constant()?;
        Ok(Array1::from_shape_fn(self.frequency().points(), |point| {
            Complex64::new(0.0, self.frequency().angular()[point]) / gamma[point]
        }))
    }

    /// Returns group velocity `$v_g=d\omega/d\gamma$`.
    ///
    /// # Errors
    ///
    /// Returns an error when either required gradient cannot be evaluated.
    fn group_velocity(&self) -> Result<Array1<Complex64>> {
        let angular_gradient = self.frequency().angular_gradient()?;
        let gamma_gradient = complex_gradient(&self.propagation_constant()?)?;
        Ok(Array1::from_shape_fn(self.frequency().points(), |point| {
            angular_gradient[point] / gamma_gradient[point]
        }))
    }

    /// Creates a matched transmission line of the requested length.
    ///
    /// # Errors
    ///
    /// Returns an error when the implementation cannot construct the line.
    fn line(&self, length: f64, unit: LengthUnit) -> Result<Network>;

    /// Creates a zero-length through network.
    ///
    /// # Errors
    ///
    /// Returns an error when the implementation cannot construct the through network.
    fn thru(&self) -> Result<Network>;

    /// Creates a one-port load with the supplied reflection coefficient.
    ///
    /// # Errors
    ///
    /// Returns an error when the implementation cannot construct the load.
    fn load(&self, reflection_coefficient: Complex64) -> Result<Network>;

    /// Creates an ideal open circuit.
    ///
    /// # Errors
    ///
    /// Returns an error when the implementation cannot construct the open circuit.
    fn open(&self) -> Result<Network>;

    /// Creates an ideal short circuit.
    ///
    /// # Errors
    ///
    /// Returns an error when the implementation cannot construct the short circuit.
    fn short(&self) -> Result<Network>;

    /// Creates an ideal matched network with the requested number of ports.
    ///
    /// # Errors
    ///
    /// Returns an error for zero ports, incompatible impedance data, or network construction failure.
    fn match_network(
        &self,
        ports: usize,
        reference_impedance: Option<Array1<Complex64>>,
    ) -> Result<Network> {
        if ports == 0 {
            return Err(Error::Unsupported(
                "a matched network requires at least one port".to_owned(),
            ));
        }
        let points = self.frequency().points();
        let reference = match reference_impedance {
            Some(values) if values.len() == points => values,
            Some(values) => {
                return Err(Error::IncompatibleShape(format!(
                    "reference impedance has {} values for {points} frequency points",
                    values.len()
                )));
            }
            None => self.characteristic_impedance()?,
        };
        Network::new(
            self.frequency().clone(),
            Array3::zeros((points, ports, ports)),
            Array2::from_shape_fn((points, ports), |(point, _)| reference[point]),
        )
    }

    /// Creates an equal load on every port from frequency-dependent reflections.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible reflection data or matched-network construction failure.
    fn load_nports(
        &self,
        reflection_coefficient: &Array1<Complex64>,
        ports: usize,
        reference_impedance: Option<Array1<Complex64>>,
    ) -> Result<Network> {
        let points = self.frequency().points();
        if reflection_coefficient.len() != points {
            return Err(Error::IncompatibleShape(
                "load reflection coefficient must match the frequency length".to_owned(),
            ));
        }
        let mut network = self.match_network(ports, reference_impedance)?;
        for point in 0..points {
            for port in 0..ports {
                network.s[(point, port, port)] = reflection_coefficient[point];
            }
        }
        Ok(network)
    }

    /// Creates a two-port series impedance.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible impedance data or unsupported reference impedances.
    fn series_impedance(&self, impedance: &Array1<Complex64>) -> Result<Network> {
        let points = self.frequency().points();
        if impedance.len() != points {
            return Err(Error::IncompatibleShape(
                "series impedance must match the frequency length".to_owned(),
            ));
        }
        let mut network = self.match_network(2, None)?;
        for point in 0..points {
            let left = network.z0[(point, 0)];
            let right = network.z0[(point, 1)];
            if left.re <= 0.0 || right.re <= 0.0 {
                return Err(Error::Unsupported(
                    "power-wave lumped elements require positive real reference impedances"
                        .to_owned(),
                ));
            }
            let denominator = impedance[point] + left + right;
            if denominator.norm_sqr() == 0.0 {
                return Err(Error::Unsupported(
                    "series-element scattering denominator is zero".to_owned(),
                ));
            }
            let transmission = 2.0 * (left.re * right.re).sqrt() / denominator;
            network.s[(point, 0, 0)] = (impedance[point] - left.conj() + right) / denominator;
            network.s[(point, 1, 1)] = (impedance[point] + left - right.conj()) / denominator;
            network.s[(point, 0, 1)] = transmission;
            network.s[(point, 1, 0)] = transmission;
        }
        network.s_definition = SParameterDefinition::Power;
        Ok(network)
    }

    /// Creates a two-port series resistor.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid resistance data or series-network construction failure.
    fn resistor(&self, resistance: &Array1<f64>) -> Result<Network> {
        if resistance
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(Error::Unsupported(
                "resistance must be finite and non-negative".to_owned(),
            ));
        }
        self.series_impedance(&resistance.mapv(|value| Complex64::new(value, 0.0)))
    }

    /// Creates a two-port series capacitor with $Z=1/(j\omega C)$.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid capacitance data, zero frequency, or network construction failure.
    fn capacitor(&self, capacitance: &Array1<f64>) -> Result<Network> {
        let points = self.frequency().points();
        if capacitance.len() != points
            || capacitance
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::Unsupported(
                "capacitance must contain one positive finite value per frequency".to_owned(),
            ));
        }
        let angular = self.frequency().angular();
        if angular.iter().any(|value| *value == 0.0) {
            return Err(Error::InvalidFrequency(
                "capacitor scattering is singular at zero frequency".to_owned(),
            ));
        }
        self.series_impedance(&Array1::from_shape_fn(points, |point| {
            Complex64::new(0.0, -1.0 / (angular[point] * capacitance[point]))
        }))
    }

    /// Creates a two-port series inductor with $Z=j\omega L$.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid inductance data or series-network construction failure.
    fn inductor(&self, inductance: &Array1<f64>) -> Result<Network> {
        let points = self.frequency().points();
        if inductance.len() != points
            || inductance
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(Error::Unsupported(
                "inductance must contain one non-negative finite value per frequency".to_owned(),
            ));
        }
        let angular = self.frequency().angular();
        self.series_impedance(&Array1::from_shape_fn(points, |point| {
            Complex64::new(0.0, angular[point] * inductance[point])
        }))
    }

    /// Creates a power-wave impedance step between real positive impedances.
    ///
    /// # Errors
    ///
    /// Returns an error when the supplied impedances cannot form the requested mismatch.
    fn impedance_mismatch(
        &self,
        left_impedance: &Array1<f64>,
        right_impedance: &Array1<f64>,
    ) -> Result<Network> {
        self.impedance_mismatch_complex(
            &left_impedance.mapv(|value| Complex64::new(value, 0.0)),
            &right_impedance.mapv(|value| Complex64::new(value, 0.0)),
            SParameterDefinition::Power,
        )
    }

    /// Creates an impedance step using the selected scattering-wave definition.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible or invalid impedances or network construction failure.
    fn impedance_mismatch_complex(
        &self,
        left_impedance: &Array1<Complex64>,
        right_impedance: &Array1<Complex64>,
        definition: SParameterDefinition,
    ) -> Result<Network> {
        let points = self.frequency().points();
        if left_impedance.len() != points
            || right_impedance.len() != points
            || left_impedance
                .iter()
                .chain(right_impedance.iter())
                .any(|value| !value.re.is_finite() || !value.im.is_finite() || value.re <= 0.0)
        {
            return Err(Error::Unsupported(
                "mismatch impedances must contain finite values with positive real parts"
                    .to_owned(),
            ));
        }
        let mut scattering = Array3::zeros((points, 2, 2));
        let z0 = Array2::from_shape_fn((points, 2), |(point, port)| {
            if port == 0 {
                left_impedance[point]
            } else {
                right_impedance[point]
            }
        });
        for point in 0..points {
            let left = left_impedance[point];
            let right = right_impedance[point];
            let denominator = left + right;
            if denominator.norm_sqr() == 0.0 {
                return Err(Error::Unsupported(
                    "mismatch impedance sum must be non-zero".to_owned(),
                ));
            }
            let reflection = (right - left) / denominator;
            match definition {
                SParameterDefinition::Traveling => {
                    scattering[(point, 0, 0)] = reflection;
                    scattering[(point, 1, 1)] = -reflection;
                    scattering[(point, 1, 0)] =
                        (Complex64::new(1.0, 0.0) + reflection) * (left / right).sqrt();
                    scattering[(point, 0, 1)] =
                        (Complex64::new(1.0, 0.0) - reflection) * (right / left).sqrt();
                }
                SParameterDefinition::Pseudo => {
                    let normalization = (right / left).norm() * (left.re / right.re).sqrt();
                    scattering[(point, 0, 0)] = reflection;
                    scattering[(point, 1, 1)] = -reflection;
                    scattering[(point, 1, 0)] = 2.0 * right / (normalization * denominator);
                    scattering[(point, 0, 1)] = 2.0 * left * normalization / denominator;
                }
                SParameterDefinition::Power => {
                    let normalization = (left.re / right.re).sqrt();
                    scattering[(point, 0, 0)] = (right - left.conj()) / denominator;
                    scattering[(point, 1, 1)] = (left - right.conj()) / denominator;
                    scattering[(point, 1, 0)] = 2.0 * left.re / (normalization * denominator);
                    scattering[(point, 0, 1)] = 2.0 * right.re * normalization / denominator;
                }
            }
        }
        let mut network = Network::new(self.frequency().clone(), scattering, z0)?;
        network.s_definition = definition;
        Ok(network)
    }

    /// Creates an ideal lossless junction with the requested number of ports.
    ///
    /// # Errors
    ///
    /// Returns an error for fewer than two ports or matched-network construction failure.
    fn splitter(&self, ports: usize) -> Result<Network> {
        if ports < 2 {
            return Err(Error::Unsupported(
                "a splitter requires at least two ports".to_owned(),
            ));
        }
        let mut network = self.match_network(ports, None)?;
        for point in 0..self.frequency().points() {
            let inverse_sum = (0..ports)
                .map(|port| Complex64::new(1.0, 0.0) / network.z0[(point, port)])
                .sum::<Complex64>();
            for output in 0..ports {
                let output_z0 = network.z0[(point, output)];
                let other_inverse_sum = inverse_sum - Complex64::new(1.0, 0.0) / output_z0;
                let parallel = Complex64::new(1.0, 0.0) / other_inverse_sum;
                network.s[(point, output, output)] =
                    (parallel - output_z0.conj()) / (parallel + output_z0);
                for input in 0..ports {
                    if output != input {
                        let input_z0 = network.z0[(point, input)];
                        network.s[(point, output, input)] = 2.0
                            * (output_z0.re * input_z0.re).sqrt()
                            / (output_z0 * input_z0 * inverse_sum);
                    }
                }
            }
        }
        network.s_definition = SParameterDefinition::Power;
        Ok(network)
    }

    /// Creates an ideal three-port tee junction.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying splitter cannot be constructed.
    fn tee(&self) -> Result<Network> {
        self.splitter(3)
    }

    /// Creates a four-port balanced transmission line.
    ///
    /// # Errors
    ///
    /// Returns an error when medium properties, length conversion, or network construction fails.
    fn floating_line(&self, length: f64, unit: LengthUnit) -> Result<Network> {
        let gamma = self.propagation_constant()?;
        let distance = media_length_to_meters(self.frequency(), &gamma, length, unit)?;
        let points = self.frequency().points();
        let reference = self.characteristic_impedance()?;
        let mut scattering = Array3::zeros((points, 4, 4));
        for point in 0..points {
            let electrical = gamma[point] * distance;
            let exponential = electrical.exp();
            let exponential_squared = (2.0 * electrical).exp();
            let denominator = -1.0 + 9.0 * exponential_squared;
            let s11 = (1.0 + 3.0 * exponential_squared) / denominator;
            let s12 = 4.0 * exponential / denominator;
            let s13 = (-2.0 + 6.0 * exponential_squared) / denominator;
            let row = [s11, s12, s13, -s12];
            for output in 0..4 {
                for input in 0..4 {
                    scattering[(point, output, input)] = row[(input + 4 - output) % 4];
                }
            }
            scattering[(point, 1, 2)] = -s12;
            scattering[(point, 1, 3)] = s13;
            scattering[(point, 2, 1)] = -s12;
            scattering[(point, 2, 3)] = s12;
            scattering[(point, 3, 1)] = s13;
            scattering[(point, 3, 2)] = s12;
        }
        Network::new(
            self.frequency().clone(),
            scattering,
            Array2::from_shape_fn((points, 4), |(point, _)| reference[point]),
        )
    }

    /// Creates a one-port load behind a transmission-line delay.
    ///
    /// # Errors
    ///
    /// Returns an error when medium properties, length conversion, or load construction fails.
    fn delay_load(
        &self,
        reflection_coefficient: Complex64,
        length: f64,
        unit: LengthUnit,
    ) -> Result<Network> {
        let gamma = self.propagation_constant()?;
        let distance = media_length_to_meters(self.frequency(), &gamma, length, unit)?;
        let delayed = gamma.mapv(|value| reflection_coefficient * (-2.0 * value * distance).exp());
        self.load_nports(&delayed, 1, None)
    }

    /// Creates a delayed short circuit.
    ///
    /// # Errors
    ///
    /// Returns an error when the delayed load cannot be constructed.
    fn delay_short(&self, length: f64, unit: LengthUnit) -> Result<Network> {
        self.delay_load(Complex64::new(-1.0, 0.0), length, unit)
    }

    /// Creates a delayed open circuit.
    ///
    /// # Errors
    ///
    /// Returns an error when the delayed load cannot be constructed.
    fn delay_open(&self, length: f64, unit: LengthUnit) -> Result<Network> {
        self.delay_load(Complex64::new(1.0, 0.0), length, unit)
    }

    /// Connects a one-port network in shunt between two ports.
    ///
    /// # Errors
    ///
    /// Returns an error for an incompatible load, singular junction, or network construction failure.
    fn shunt_load(&self, load: &Network) -> Result<Network> {
        if load.ports() != 1 || load.frequency != *self.frequency() {
            return Err(Error::IncompatibleShape(
                "shunting a load requires an aligned one-port Network".to_owned(),
            ));
        }
        let mut network = self.match_network(2, None)?;
        for point in 0..self.frequency().points() {
            let reflection = load.s[(point, 0, 0)];
            let denominator = 3.0 + reflection;
            if denominator.norm_sqr() <= f64::EPSILON {
                return Err(Error::Unsupported(
                    "shunted load produced a singular junction".to_owned(),
                ));
            }
            let shunt_reflection = -(1.0 - reflection) / denominator;
            let transmission = 2.0 * (1.0 + reflection) / denominator;
            network.s[(point, 0, 0)] = shunt_reflection;
            network.s[(point, 1, 1)] = shunt_reflection;
            network.s[(point, 0, 1)] = transmission;
            network.s[(point, 1, 0)] = transmission;
        }
        Ok(network)
    }

    /// Creates a delayed load and connects it in shunt.
    ///
    /// # Errors
    ///
    /// Returns an error when delayed-load or shunt construction fails.
    fn shunt_delay_load(
        &self,
        reflection_coefficient: Complex64,
        length: f64,
        unit: LengthUnit,
    ) -> Result<Network> {
        self.shunt_load(&self.delay_load(reflection_coefficient, length, unit)?)
    }

    /// Creates a delayed open circuit connected in shunt.
    ///
    /// # Errors
    ///
    /// Returns an error when delayed-open or shunt construction fails.
    fn shunt_delay_open(&self, length: f64, unit: LengthUnit) -> Result<Network> {
        self.shunt_load(&self.delay_open(length, unit)?)
    }

    /// Creates a delayed short circuit connected in shunt.
    ///
    /// # Errors
    ///
    /// Returns an error when delayed-short or shunt construction fails.
    fn shunt_delay_short(&self, length: f64, unit: LengthUnit) -> Result<Network> {
        self.shunt_load(&self.delay_short(length, unit)?)
    }

    /// Creates a two-port shunt admittance.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible admittance data or network construction failure.
    fn shunt_admittance(&self, admittance: &Array1<Complex64>) -> Result<Network> {
        let points = self.frequency().points();
        if admittance.len() != points {
            return Err(Error::IncompatibleShape(
                "shunt admittance must match the frequency length".to_owned(),
            ));
        }
        let reference = self.characteristic_impedance()?;
        let mut scattering = Array3::zeros((points, 2, 2));
        for point in 0..points {
            let denominator = 2.0 + admittance[point] * reference[point];
            let reflection = -admittance[point] * reference[point] / denominator;
            let transmission = Complex64::new(2.0, 0.0) / denominator;
            scattering[(point, 0, 0)] = reflection;
            scattering[(point, 1, 1)] = reflection;
            scattering[(point, 0, 1)] = transmission;
            scattering[(point, 1, 0)] = transmission;
        }
        let mut network = Network::new(
            self.frequency().clone(),
            scattering,
            Array2::from_shape_fn((points, 2), |(point, _)| reference[point]),
        )?;
        network.s_definition = SParameterDefinition::Traveling;
        Ok(network)
    }

    /// Creates a shunt resistor.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid resistance data or shunt-network construction failure.
    fn shunt_resistor(&self, resistance: &Array1<f64>) -> Result<Network> {
        if resistance.len() != self.frequency().points()
            || resistance
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::Unsupported(
                "shunt resistance must contain positive finite values per frequency".to_owned(),
            ));
        }
        self.shunt_admittance(&resistance.mapv(|value| Complex64::new(1.0 / value, 0.0)))
    }

    /// Creates a shunt capacitor with $Y=j\omega C$.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid capacitance data or shunt-network construction failure.
    fn shunt_capacitor(&self, capacitance: &Array1<f64>) -> Result<Network> {
        if capacitance.len() != self.frequency().points()
            || capacitance
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(Error::Unsupported(
                "shunt capacitance must contain non-negative finite values per frequency"
                    .to_owned(),
            ));
        }
        let angular = self.frequency().angular();
        self.shunt_admittance(&Array1::from_shape_fn(self.frequency().points(), |point| {
            Complex64::new(0.0, angular[point] * capacitance[point])
        }))
    }

    /// Creates a shunt inductor with $Y=1/(j\omega L)$.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid inductance data, zero frequency, or shunt-network failure.
    fn shunt_inductor(&self, inductance: &Array1<f64>) -> Result<Network> {
        if inductance.len() != self.frequency().points()
            || inductance
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::Unsupported(
                "shunt inductance must contain positive finite values per frequency".to_owned(),
            ));
        }
        let angular = self.frequency().angular();
        if angular.iter().any(|value| *value == 0.0) {
            return Err(Error::InvalidFrequency(
                "shunt inductance is singular at zero frequency".to_owned(),
            ));
        }
        self.shunt_admittance(&Array1::from_shape_fn(self.frequency().points(), |point| {
            Complex64::new(0.0, -1.0 / (angular[point] * inductance[point]))
        }))
    }

    /// Creates a lossy series capacitor from its capacitance and quality factor.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid quality parameters or series-network construction failure.
    fn capacitor_with_q(
        &self,
        capacitance: &Array1<f64>,
        specification_frequency_hz: f64,
        quality_factor: f64,
    ) -> Result<Network> {
        if !specification_frequency_hz.is_finite()
            || specification_frequency_hz <= 0.0
            || !quality_factor.is_finite()
            || quality_factor <= 0.0
        {
            return Err(Error::Unsupported(
                "capacitor Q parameters must be positive and finite".to_owned(),
            ));
        }
        let angular = self.frequency().angular();
        self.series_impedance(&Array1::from_shape_fn(self.frequency().points(), |point| {
            Complex64::new(
                1.0 / (std::f64::consts::TAU
                    * specification_frequency_hz
                    * capacitance[point]
                    * quality_factor),
                -1.0 / (angular[point] * capacitance[point]),
            )
        }))
    }

    /// Creates a lossy series inductor from its inductance, quality factor, and DC resistance.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid quality parameters or series-network construction failure.
    fn inductor_with_q(
        &self,
        inductance: &Array1<f64>,
        specification_frequency_hz: f64,
        quality_factor: f64,
        dc_resistance: f64,
    ) -> Result<Network> {
        if !specification_frequency_hz.is_finite()
            || specification_frequency_hz <= 0.0
            || !quality_factor.is_finite()
            || quality_factor <= 0.0
            || !dc_resistance.is_finite()
            || dc_resistance < 0.0
        {
            return Err(Error::Unsupported(
                "inductor Q parameters must be finite with positive frequency and Q".to_owned(),
            ));
        }
        let angular = self.frequency().angular();
        self.series_impedance(&Array1::from_shape_fn(self.frequency().points(), |point| {
            let ac_resistance = angular[point] * inductance[point] / quality_factor;
            Complex64::new(
                dc_resistance.hypot(ac_resistance),
                angular[point] * inductance[point],
            )
        }))
    }

    /// Creates a matched attenuator, optionally followed by a line delay.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid transmission data or network and line construction failure.
    fn attenuator(
        &self,
        transmission: &Array1<f64>,
        decibels: bool,
        length: f64,
        unit: LengthUnit,
    ) -> Result<Network> {
        if transmission.len() != self.frequency().points()
            || transmission.iter().any(|value| !value.is_finite())
        {
            return Err(Error::IncompatibleShape(
                "attenuator transmission must contain one finite value per frequency".to_owned(),
            ));
        }
        let magnitude = transmission.mapv(|value| {
            if decibels {
                10.0_f64.powf(value / 20.0)
            } else {
                value
            }
        });
        if magnitude.iter().any(|value| *value < 0.0) {
            return Err(Error::Unsupported(
                "linear attenuator transmission must not be negative".to_owned(),
            ));
        }
        let mut attenuator = self.match_network(2, None)?;
        for point in 0..self.frequency().points() {
            let value = Complex64::new(magnitude[point], 0.0);
            attenuator.s[(point, 0, 1)] = value;
            attenuator.s[(point, 1, 0)] = value;
        }
        attenuator.cascade(&self.line(length, unit)?)
    }

    /// Creates a reciprocal lossless two-port with the supplied reflection coefficient.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible, non-finite, or over-unity reflection data.
    fn lossless_mismatch(&self, reflection: &Array1<Complex64>) -> Result<Network> {
        if reflection.len() != self.frequency().points()
            || reflection
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::IncompatibleShape(
                "lossless mismatch reflection must contain one finite value per frequency"
                    .to_owned(),
            ));
        }
        let mut network = self.match_network(2, None)?;
        for point in 0..self.frequency().points() {
            let reflection = reflection[point];
            if reflection.norm_sqr() > 1.0 + f64::EPSILON {
                return Err(Error::Unsupported(
                    "lossless mismatch reflection magnitude cannot exceed one".to_owned(),
                ));
            }
            let phase = reflection.arg();
            let transmission_phase = if phase <= 0.0 {
                phase + std::f64::consts::FRAC_PI_2
            } else {
                phase - std::f64::consts::FRAC_PI_2
            };
            let transmission = Complex64::from_polar(
                (1.0 - reflection.norm_sqr()).max(0.0).sqrt(),
                transmission_phase,
            );
            network.s[(point, 0, 0)] = reflection;
            network.s[(point, 1, 1)] = reflection;
            network.s[(point, 0, 1)] = transmission;
            network.s[(point, 1, 0)] = transmission;
        }
        Ok(network)
    }

    /// Creates a lossless mismatch from return loss in decibels.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid return-loss data or mismatch construction failure.
    fn lossless_mismatch_db(&self, return_loss_db: &Array1<f64>) -> Result<Network> {
        if return_loss_db.len() != self.frequency().points()
            || return_loss_db.iter().any(|value| !value.is_finite())
        {
            return Err(Error::IncompatibleShape(
                "return loss must contain one finite value per frequency".to_owned(),
            ));
        }
        self.lossless_mismatch(
            &return_loss_db.mapv(|value| Complex64::new(10.0_f64.powf(value / 20.0), 0.0)),
        )
    }

    /// Creates an ideal two-port isolator passing away from `source_port`.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid source port or through-network construction failure.
    fn isolator(&self, source_port: usize) -> Result<Network> {
        if source_port > 1 {
            return Err(Error::InvalidPort {
                port: source_port,
                ports: 2,
            });
        }
        let mut network = self.thru()?;
        let blocked_output = source_port;
        let blocked_input = 1 - source_port;
        for point in 0..self.frequency().points() {
            network.s[(point, blocked_output, blocked_input)] = Complex64::new(0.0, 0.0);
        }
        Ok(network)
    }

    /// Creates a random network with optional reciprocity, matching, and symmetry.
    ///
    /// # Errors
    ///
    /// Returns an error when the initial matched network cannot be constructed.
    fn random_network(
        &self,
        ports: usize,
        reciprocal: bool,
        matched: bool,
        symmetric: bool,
    ) -> Result<Network> {
        let mut network = self.match_network(ports, None)?;
        let values = random_complex(self.frequency().points() * ports, ports);
        for point in 0..self.frequency().points() {
            for output in 0..ports {
                for input in 0..ports {
                    network.s[(point, output, input)] = values[(point * ports + output, input)];
                }
            }
        }
        if reciprocal {
            for point in 0..self.frequency().points() {
                for output in 0..ports {
                    for input in 0..output {
                        network.s[(point, output, input)] = network.s[(point, input, output)];
                    }
                }
            }
        }
        if symmetric {
            for point in 0..self.frequency().points() {
                let diagonal = network.s[(point, 0, 0)];
                for port in 0..ports {
                    network.s[(point, port, port)] = diagonal;
                }
            }
        }
        if matched {
            for point in 0..self.frequency().points() {
                for port in 0..ports {
                    network.s[(point, port, port)] = Complex64::new(0.0, 0.0);
                }
            }
        }
        Ok(network)
    }

    /// Creates a network whose magnitude and phase are Gaussian random variables.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid distributions or matched-network construction failure.
    fn white_gaussian_polar(
        &self,
        phase_standard_deviation: f64,
        magnitude_standard_deviation: f64,
        ports: usize,
    ) -> Result<Network> {
        let mut network = self.match_network(ports, None)?;
        let values = random_gaussian_polar(
            self.frequency().points() * ports,
            ports,
            phase_standard_deviation,
            magnitude_standard_deviation,
        )?;
        for point in 0..self.frequency().points() {
            for output in 0..ports {
                for input in 0..ports {
                    network.s[(point, output, input)] = values[(point * ports + output, input)];
                }
            }
        }
        Ok(network)
    }

    /// Estimates distance from the unwrapped reflection-phase gradient.
    ///
    /// # Errors
    ///
    /// Returns an error for an incompatible network or unavailable gradients and propagation data.
    fn extract_distance(&self, network: &Network) -> Result<Array1<f64>> {
        if network.ports() != 1 || network.frequency != *self.frequency() {
            return Err(Error::IncompatibleShape(
                "distance extraction requires an aligned one-port Network".to_owned(),
            ));
        }
        let phase = unwrap_phase(
            &network
                .s
                .outer_iter()
                .map(|matrix| matrix[(0, 0)].arg())
                .collect::<Vec<_>>(),
        );
        let phase_gradient = real_gradient(&Array1::from_vec(phase))?;
        let beta = self.propagation_constant()?.mapv(|value| value.im);
        let beta_gradient = real_gradient(&beta)?;
        Ok(Array1::from_shape_fn(self.frequency().points(), |point| {
            -phase_gradient[point] / beta_gradient[point]
        }))
    }
}

pub(super) fn real_gradient(values: &Array1<f64>) -> Result<Array1<f64>> {
    if values.len() < 2 {
        return Err(Error::IncompatibleShape(
            "a gradient requires at least two samples".to_owned(),
        ));
    }
    let mut result = Array1::zeros(values.len());
    result[0] = values[1] - values[0];
    result[values.len() - 1] = values[values.len() - 1] - values[values.len() - 2];
    for index in 1..values.len() - 1 {
        result[index] = (values[index + 1] - values[index - 1]) / 2.0;
    }
    Ok(result)
}

pub(super) fn complex_gradient(values: &Array1<Complex64>) -> Result<Array1<Complex64>> {
    if values.len() < 2 {
        return Err(Error::IncompatibleShape(
            "a gradient requires at least two samples".to_owned(),
        ));
    }
    let mut result = Array1::zeros(values.len());
    result[0] = values[1] - values[0];
    result[values.len() - 1] = values[values.len() - 1] - values[values.len() - 2];
    for index in 1..values.len() - 1 {
        result[index] = (values[index + 1] - values[index - 1]) / 2.0;
    }
    Ok(result)
}

pub(super) fn unwrap_phase(values: &[f64]) -> Vec<f64> {
    let mut unwrapped = values.to_vec();
    for index in 1..unwrapped.len() {
        let mut difference = values[index] - values[index - 1];
        loop {
            if difference <= std::f64::consts::PI {
                break;
            }
            difference -= std::f64::consts::TAU;
        }
        loop {
            if difference >= -std::f64::consts::PI {
                break;
            }
            difference += std::f64::consts::TAU;
        }
        unwrapped[index] = unwrapped[index - 1] + difference;
    }
    unwrapped
}

pub(super) fn media_length_to_meters(
    frequency: &Frequency,
    gamma: &Array1<Complex64>,
    length: f64,
    unit: LengthUnit,
) -> Result<f64> {
    if !length.is_finite() {
        return Err(Error::Unsupported("line length must be finite".to_owned()));
    }
    match unit {
        LengthUnit::Meter => Ok(length),
        LengthUnit::Centimeter => Ok(length * 1.0e-2),
        LengthUnit::Millimeter => Ok(length * 1.0e-3),
        LengthUnit::Micrometer => Ok(length * 1.0e-6),
        LengthUnit::Inch => Ok(length * 0.0254),
        LengthUnit::Mil => Ok(length * 0.000_025_4),
        LengthUnit::Second
        | LengthUnit::Microsecond
        | LengthUnit::Nanosecond
        | LengthUnit::Picosecond => {
            let angular_gradient = frequency.angular_gradient()?;
            let gamma_gradient = complex_gradient(gamma)?;
            let velocity = angular_gradient
                .iter()
                .zip(gamma_gradient.iter())
                .map(|(angular, gamma)| -(*angular / *gamma).im)
                .sum::<f64>()
                / frequency.points().to_f64().unwrap_or(f64::INFINITY);
            let seconds = match unit {
                LengthUnit::Second => length,
                LengthUnit::Microsecond => length * 1.0e-6,
                LengthUnit::Nanosecond => length * 1.0e-9,
                LengthUnit::Picosecond => length * 1.0e-12,
                _ => unreachable!(),
            };
            Ok(seconds * velocity)
        }
        LengthUnit::Degree | LengthUnit::Radian => {
            if gamma.is_empty() || gamma[gamma.len() / 2].im == 0.0 {
                return Err(Error::Unsupported(
                    "electrical length requires a non-zero phase constant".to_owned(),
                ));
            }
            let radians = if unit == LengthUnit::Degree {
                length.to_radians()
            } else {
                length
            };
            Ok(radians / gamma[gamma.len() / 2].im)
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Unit used to specify a physical, temporal, or electrical line length.
pub enum LengthUnit {
    /// Meters.
    Meter,
    /// Centimeters.
    Centimeter,
    /// Millimeters.
    Millimeter,
    /// Micrometers.
    Micrometer,
    /// Inches.
    Inch,
    /// Thousandths of an inch.
    Mil,
    /// Seconds of propagation delay.
    Second,
    /// Microseconds of propagation delay.
    Microsecond,
    /// Nanoseconds of propagation delay.
    Nanosecond,
    /// Picoseconds of propagation delay.
    Picosecond,
    /// Degrees of electrical length at the center frequency.
    #[default]
    Degree,
    /// Radians of electrical length at the center frequency.
    Radian,
}

/// A medium whose propagation constant and characteristic impedance are supplied directly.
///
/// This is useful when measured, simulated, or analytically calculated values
/// of $\gamma$ and `$Z_0$` are already available.
#[derive(Clone, Debug, PartialEq)]
pub struct DefinedGammaZ0 {
    /// Frequencies at which the medium is defined.
    pub frequency: Frequency,
    /// Complex propagation constant $\gamma=\alpha+j\beta$.
    pub gamma: Array1<Complex64>,
    /// Complex characteristic impedance `$Z_0$`.
    pub z0: Array1<Complex64>,
    /// Optional impedance used to renormalize generated networks.
    pub port_z0: Option<Array1<Complex64>>,
}

impl DefinedGammaZ0 {
    /// Creates a medium from frequency-dependent $\gamma$ and `$Z_0$` arrays.
    ///
    /// Each array must contain one value per frequency point.
    ///
    /// # Errors
    ///
    /// Returns an error when any supplied array does not match the frequency length.
    pub fn new(
        frequency: Frequency,
        gamma: Array1<Complex64>,
        z0: Array1<Complex64>,
        port_z0: Option<Array1<Complex64>>,
    ) -> Result<Self> {
        let points = frequency.points();
        if gamma.len() != points || z0.len() != points {
            return Err(Error::IncompatibleShape(format!(
                "media frequency has {points} points, gamma has {}, and z0 has {}",
                gamma.len(),
                z0.len()
            )));
        }
        if port_z0
            .as_ref()
            .is_some_and(|values| values.len() != points)
        {
            return Err(Error::IncompatibleShape(
                "media port impedance must match the frequency length".to_owned(),
            ));
        }
        Ok(Self {
            frequency,
            gamma,
            z0,
            port_z0,
        })
    }

    /// Converts a line length from the requested unit to meters.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested length conversion is invalid.
    pub fn physical_length(&self, length: f64, unit: LengthUnit) -> Result<f64> {
        media_length_to_meters(&self.frequency, &self.gamma, length, unit)
    }

    fn network_reference_impedance(&self) -> &Array1<Complex64> {
        self.port_z0.as_ref().unwrap_or(&self.z0)
    }

    /// Writes frequency, `$Z_0$`, $\gamma$, and port impedance to CSV.
    ///
    /// # Errors
    ///
    /// Returns an error when the CSV file cannot be created or written.
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_path(path)
            .map_err(csv_error)?;
        writer
            .write_record([
                format!("f[{}]", self.frequency.unit().symbol()),
                "Re(z0)".to_owned(),
                "Im(z0)".to_owned(),
                "Re(gamma)".to_owned(),
                "Im(gamma)".to_owned(),
                "Re(z0_port)".to_owned(),
                "Im(z0_port)".to_owned(),
            ])
            .map_err(csv_error)?;
        let port_z0 = self.port_z0.as_ref().unwrap_or(&self.z0);
        let scaled_frequency = self.frequency.scaled();
        for point in 0..self.frequency.points() {
            writer
                .serialize((
                    scaled_frequency[point],
                    self.z0[point].re,
                    self.z0[point].im,
                    self.gamma[point].re,
                    self.gamma[point].im,
                    port_z0[point].re,
                    port_z0[point].im,
                ))
                .map_err(csv_error)?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Reads a medium written by [`write_csv`](Self::write_csv).
    ///
    /// # Errors
    ///
    /// Returns an error when the CSV cannot be read, parsed, or converted into a valid medium.
    pub fn from_csv(path: impl AsRef<Path>) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_path(path)
            .map_err(csv_error)?;
        let header = reader.headers().map_err(csv_error)?.clone();
        let frequency_header = header
            .get(0)
            .ok_or_else(|| Error::Parse("media CSV is missing its frequency header".to_owned()))?
            .trim_start_matches('#')
            .trim();
        let unit_name = frequency_header
            .strip_prefix("f[")
            .and_then(|value| value.strip_suffix(']'))
            .ok_or_else(|| {
                Error::Parse(format!(
                    "media CSV frequency header `{frequency_header}` does not contain f[unit]"
                ))
            })?;
        let unit = frequency_unit_from_symbol(unit_name)?;
        let mut frequency = Vec::new();
        let mut z0 = Vec::new();
        let mut gamma = Vec::new();
        let mut port_z0 = Vec::new();
        for (row_index, record) in reader.records().enumerate() {
            let record = record.map_err(csv_error)?;
            if record.len() != 7 {
                return Err(Error::Parse(format!(
                    "media CSV row {} has {} columns instead of 7",
                    row_index + 2,
                    record.len()
                )));
            }
            let values = record
                .iter()
                .map(|value| {
                    value.parse::<f64>().map_err(|error| {
                        Error::Parse(format!(
                            "invalid numeric value `{value}` in media CSV row {}: {error}",
                            row_index + 2
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            frequency.push(values[0]);
            z0.push(Complex64::new(values[1], values[2]));
            gamma.push(Complex64::new(values[3], values[4]));
            port_z0.push(Complex64::new(values[5], values[6]));
        }
        Self::new(
            Frequency::from_values(Array1::from_vec(frequency), unit)?,
            Array1::from_vec(gamma),
            Array1::from_vec(z0),
            Some(Array1::from_vec(port_z0)),
        )
    }
}

impl Media for DefinedGammaZ0 {
    /// Returns the stored frequency axis.
    fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    /// Returns the stored propagation constant.
    fn propagation_constant(&self) -> Result<Array1<Complex64>> {
        Ok(self.gamma.clone())
    }

    /// Returns the stored characteristic impedance.
    fn characteristic_impedance(&self) -> Result<Array1<Complex64>> {
        Ok(self.z0.clone())
    }

    /// Returns the optional network port impedance.
    fn port_impedance(&self) -> Option<&Array1<Complex64>> {
        self.port_z0.as_ref()
    }

    /// Creates a matched line with transmission $e^{-\gamma d}$.
    fn line(&self, length: f64, unit: LengthUnit) -> Result<Network> {
        let distance = self.physical_length(length, unit)?;
        let points = self.frequency.points();
        let mut scattering = Array3::zeros((points, 2, 2));
        for point in 0..points {
            let transmission = (-self.gamma[point] * distance).exp();
            scattering[(point, 0, 1)] = transmission;
            scattering[(point, 1, 0)] = transmission;
        }
        let characteristic_reference =
            Array2::from_shape_fn((points, 2), |(point, _)| self.z0[point]);
        let target_reference = self.port_z0.as_ref().map_or_else(
            || characteristic_reference.clone(),
            |port_z0| Array2::from_shape_fn((points, 2), |(point, _)| port_z0[point]),
        );
        if distance == 0.0 {
            return Network::new(self.frequency.clone(), scattering, target_reference);
        }
        let mut network =
            Network::new(self.frequency.clone(), scattering, characteristic_reference)?;
        network.s_definition = SParameterDefinition::Traveling;
        if target_reference == network.z0
            && target_reference.iter().all(|impedance| impedance.im == 0.0)
        {
            network.s_definition = SParameterDefinition::Power;
        } else {
            network.renormalize(target_reference, SParameterDefinition::Power)?;
        }
        Ok(network)
    }

    /// Creates a zero-length through network.
    fn thru(&self) -> Result<Network> {
        self.line(0.0, LengthUnit::Meter)
    }

    /// Creates a one-port load with the supplied reflection coefficient.
    fn load(&self, reflection_coefficient: Complex64) -> Result<Network> {
        let points = self.frequency.points();
        let scattering = Array3::from_elem((points, 1, 1), reflection_coefficient);
        let z0 = Array2::from_shape_fn((points, 1), |(point, _)| {
            self.network_reference_impedance()[point]
        });
        Network::new(self.frequency.clone(), scattering, z0)
    }

    /// Creates an ideal open circuit.
    fn open(&self) -> Result<Network> {
        self.load(Complex64::new(1.0, 0.0))
    }

    /// Creates an ideal short circuit for the configured reference impedance.
    fn short(&self) -> Result<Network> {
        let points = self.frequency.points();
        let reference = self.network_reference_impedance();
        let scattering = Array3::from_shape_fn((points, 1, 1), |(point, _, _)| {
            -reference[point].conj() / reference[point]
        });
        let z0 = Array2::from_shape_fn((points, 1), |(point, _)| reference[point]);
        Network::new(self.frequency.clone(), scattering, z0)
    }
}

fn csv_error(error: csv::Error) -> Error {
    Error::Io(std::io::Error::other(error))
}

fn frequency_unit_from_symbol(symbol: &str) -> Result<crate::FrequencyUnit> {
    match symbol.trim().to_ascii_lowercase().as_str() {
        "hz" => Ok(crate::FrequencyUnit::Hz),
        "khz" => Ok(crate::FrequencyUnit::KHz),
        "mhz" => Ok(crate::FrequencyUnit::MHz),
        "ghz" => Ok(crate::FrequencyUnit::GHz),
        "thz" => Ok(crate::FrequencyUnit::THz),
        _ => Err(Error::Parse(format!(
            "unsupported media CSV frequency unit `{symbol}`"
        ))),
    }
}
