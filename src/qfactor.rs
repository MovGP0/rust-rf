//! Resonator quality-factor extraction using the MAT 58 nonlinear models.
//!
//! [`QFactor`] fits one-port resonance data with NLQFIT6, NLQFIT7, or NLQFIT8
//! and derives loaded/unloaded Q, resonance frequency, bandwidth, fitted
//! networks, angular weights, and the resonance circle.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use num_traits::ToPrimitive;

use crate::{Error, Frequency, Network, Result};

const PARAMETER_COUNT: usize = 6;
const MAX_ITERATIONS: usize = 100;
const FIT_TOLERANCE: f64 = 1.0e-10;

/// Origin: `skrf/qfactor.py::OptimizedResult` for the NLQFIT6 model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OptimizedResult {
    /// Whether the optimizer met its convergence criterion.
    pub success: bool,
    /// Number of optimizer iterations performed.
    pub iterations: usize,
    /// Final sum-of-squares objective.
    pub objective: f64,
    /// Raw fitted model parameters.
    pub parameters: Vec<f64>,
    /// Real part of the detuned response.
    pub m1: f64,
    /// Imaginary part of the detuned response.
    pub m2: f64,
    /// Real resonance-circle displacement.
    pub m3: f64,
    /// Imaginary resonance-circle displacement.
    pub m4: f64,
    /// Fitted loaded quality factor $Q_{L}$.
    pub loaded_q: f64,
    /// Fitted resonant frequency in hertz.
    pub resonant_frequency_hz: f64,
    /// Root-mean-square complex residual error.
    pub rms_error: f64,
    /// Nonlinear model used for the fit.
    pub method: QFitMethod,
    /// Optional NLQFIT7/8 phase-delay slope in radians per hertz.
    pub phase_slope_radians_per_hz: Option<f64>,
    /// Optional NLQFIT8 complex leakage slope.
    pub leakage_slope: Option<Complex64>,
    /// Ratio used by angular reweighting, when enabled.
    pub weighting_ratio: Option<f64>,
    /// MAT 58 loop-plan string applied by the fit.
    pub loop_plan: String,
}

/// MAT 58 nonlinear resonator model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QFitMethod {
    /// Six-parameter resonance-circle model.
    #[default]
    Nlqfit6,
    /// Seven-parameter model including phase delay.
    Nlqfit7,
    /// Eight-parameter model including phase delay and leakage.
    Nlqfit8,
}

/// Origin: `skrf/qfactor.py::Qfactor.Q_circle`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QCircle {
    /// Diameter of the fitted resonance circle.
    pub diameter: f64,
    /// Complex response far from resonance.
    pub detuned: Complex64,
    /// Complex response at resonance.
    pub tuned: Complex64,
}

/// Origin: `skrf/qfactor.py::Qfactor`.
#[derive(Clone, Debug)]
pub struct QFactor {
    /// One-port network containing the measured resonance.
    pub network: Network,
    /// Measurement configuration used to interpret the circle.
    pub resonance_type: ResonanceType,
    initial_loaded_q: f64,
    initial_resonant_frequency_hz: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Resonator measurement configuration.
pub enum ResonanceType {
    /// Transmission-mode resonance.
    Transmission,
    /// Reflection-mode resonance using the primary correction.
    Reflection,
    /// Reflection-mode resonance using the alternate correction.
    ReflectionMethod2,
    /// Absorption-mode resonance.
    Absorption,
}

impl QFactor {
    /// Port of `skrf.qfactor.Qfactor.__init__` and its resonance seed selection.
    ///
    /// # Errors
    ///
    /// Returns an error if the network is not a valid one-port resonance data
    /// set or either supplied initial estimate is invalid.
    pub fn new(
        network: Network,
        resonance_type: ResonanceType,
        initial_loaded_q: Option<f64>,
        initial_resonant_frequency_hz: Option<f64>,
    ) -> Result<Self> {
        if network.ports() != 1 {
            return Err(Error::IncompatibleShape(
                "Q-factor fitting requires a one-port Network".to_owned(),
            ));
        }
        if network.frequency_points() < PARAMETER_COUNT {
            return Err(Error::IncompatibleShape(format!(
                "Q-factor fitting requires at least {PARAMETER_COUNT} frequency points"
            )));
        }
        let frequencies = network.frequency.values_hz();
        if !network.frequency.is_monotonic_increasing()
            || frequencies.iter().any(|frequency| *frequency <= 0.0)
        {
            return Err(Error::InvalidFrequency(
                "Q-factor fitting requires positive, monotonically increasing frequencies"
                    .to_owned(),
            ));
        }

        let resonant_frequency_hz = match initial_resonant_frequency_hz {
            Some(value) if value.is_finite() && value > 0.0 => value,
            Some(_) => {
                return Err(Error::InvalidFrequency(
                    "initial resonant frequency must be finite and positive".to_owned(),
                ));
            }
            None => {
                let extremum = network
                    .s
                    .outer_iter()
                    .enumerate()
                    .min_by(|(_, left), (_, right)| {
                        let left = left[(0, 0)].norm_sqr();
                        let right = right[(0, 0)].norm_sqr();
                        match resonance_type {
                            ResonanceType::Transmission => right.total_cmp(&left),
                            ResonanceType::Reflection
                            | ResonanceType::ReflectionMethod2
                            | ResonanceType::Absorption => left.total_cmp(&right),
                        }
                    })
                    .ok_or_else(|| {
                        Error::IncompatibleShape(
                            "Q-factor fitting requires at least two frequency points".to_owned(),
                        )
                    })?
                    .0;
                frequencies[extremum]
            }
        };

        let span = frequencies[frequencies.len() - 1] - frequencies[0];
        let loaded_q = match initial_loaded_q {
            Some(value) if value.is_finite() && value > 0.0 => value,
            Some(_) => {
                return Err(Error::Unsupported(
                    "initial loaded Q must be finite and positive".to_owned(),
                ));
            }
            None => 5.0 * resonant_frequency_hz / span,
        };

        Ok(Self {
            network,
            resonance_type,
            initial_loaded_q: loaded_q,
            initial_resonant_frequency_hz: resonant_frequency_hz,
        })
    }

    /// Fits the six-parameter MAT 58 resonator response used by NLQFIT6.
    ///
    /// See the [NPL MAT 58 report](https://eprintspublications.npl.co.uk/9304/).
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails or produces an invalid result.
    pub fn fit(&self) -> Result<OptimizedResult> {
        self.fit_method(QFitMethod::Nlqfit6)
    }

    /// Fits one of the NLQFIT6, NLQFIT7, or NLQFIT8 MAT 58 models.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails or produces an invalid result.
    pub fn fit_method(&self, method: QFitMethod) -> Result<OptimizedResult> {
        self.fit_with_loop_plan(method, "fwfwc")
    }

    /// Fits with the MAT 58 loop-plan language (`f`, `w`, and `c`).
    ///
    /// `f` performs one fit, `w` recalculates angular weights from the previous
    /// fit, and `c` repeats fitting until convergence.
    ///
    /// # Errors
    ///
    /// Returns an error if the loop plan is invalid or fitting fails.
    pub fn fit_with_loop_plan(
        &self,
        method: QFitMethod,
        loop_plan: &str,
    ) -> Result<OptimizedResult> {
        validate_loop_plan(loop_plan)?;
        let six_parameter = self.fit_six_parameter()?;
        let weighted = loop_plan.contains('w');
        let maximum_iterations = if loop_plan.contains('c') {
            MAX_ITERATIONS * 2
        } else {
            loop_plan
                .chars()
                .filter(|operation| *operation == 'f')
                .count()
        };
        if method == QFitMethod::Nlqfit6 && !weighted {
            let mut result = six_parameter;
            loop_plan.clone_into(&mut result.loop_plan);
            return Ok(result);
        }
        let mut result =
            self.fit_extended(method, &six_parameter, weighted, maximum_iterations.max(1))?;
        loop_plan.clone_into(&mut result.loop_plan);
        Ok(result)
    }

    fn fit_six_parameter(&self) -> Result<OptimizedResult> {
        let frequencies = self.network.frequency.values_hz().to_vec();
        let measured = self
            .network
            .s
            .outer_iter()
            .map(|matrix| matrix[(0, 0)])
            .collect::<Vec<_>>();
        let endpoint_count = (measured.len() / 10).max(1);
        let endpoint_sample_count = (2 * endpoint_count).to_f64().unwrap_or(f64::NAN);
        let detuned = measured
            .iter()
            .take(endpoint_count)
            .chain(measured.iter().rev().take(endpoint_count))
            .copied()
            .sum::<Complex64>()
            / endpoint_sample_count;
        let resonance_index = frequencies
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                (**left - self.initial_resonant_frequency_hz)
                    .abs()
                    .total_cmp(&(**right - self.initial_resonant_frequency_hz).abs())
            })
            .ok_or_else(|| {
                Error::IncompatibleShape(
                    "Q-factor fitting requires at least two frequency points".to_owned(),
                )
            })?
            .0;
        let resonant_delta = measured[resonance_index] - detuned;
        let mut parameters = [
            detuned.re,
            detuned.im,
            resonant_delta.re,
            resonant_delta.im,
            self.initial_loaded_q.ln(),
            self.initial_resonant_frequency_hz.ln(),
        ];
        let mut residuals = residual_vector(&frequencies, &measured, &parameters);
        let mut objective = squared_norm(&residuals);
        let mut damping = 1.0e-3;
        let mut iterations = 0;
        let mut success = false;

        for iteration in 1..=MAX_ITERATIONS {
            iterations = iteration;
            let jacobian = numerical_jacobian(&frequencies, &measured, &parameters);
            let mut normal = [[0.0; PARAMETER_COUNT]; PARAMETER_COUNT];
            let mut gradient = [0.0; PARAMETER_COUNT];
            for row in 0..residuals.len() {
                for column in 0..PARAMETER_COUNT {
                    gradient[column] =
                        jacobian[row][column].mul_add(residuals[row], gradient[column]);
                    for other in 0..PARAMETER_COUNT {
                        normal[column][other] = jacobian[row][column]
                            .mul_add(jacobian[row][other], normal[column][other]);
                    }
                }
            }
            for diagonal in 0..PARAMETER_COUNT {
                normal[diagonal][diagonal] += damping * normal[diagonal][diagonal].max(1.0);
                gradient[diagonal] = -gradient[diagonal];
            }
            let Some(step) = solve_six_by_six(normal, gradient) else {
                damping *= 10.0;
                continue;
            };
            let mut candidate = parameters;
            for index in 0..PARAMETER_COUNT {
                candidate[index] += step[index];
            }
            candidate[4] = candidate[4].clamp(-20.0, 30.0);
            candidate[5] =
                candidate[5].clamp(frequencies[0].ln(), frequencies[frequencies.len() - 1].ln());
            let candidate_residuals = residual_vector(&frequencies, &measured, &candidate);
            let candidate_objective = squared_norm(&candidate_residuals);

            if candidate_objective < objective {
                let relative_improvement =
                    (objective - candidate_objective) / objective.max(f64::EPSILON);
                parameters = candidate;
                residuals = candidate_residuals;
                objective = candidate_objective;
                damping = (damping * 0.3).max(1.0e-12);
                let relative_step = step
                    .iter()
                    .zip(parameters.iter())
                    .map(|(change, value)| change.abs() / value.abs().max(1.0))
                    .fold(0.0, f64::max);
                if relative_improvement < FIT_TOLERANCE || relative_step < FIT_TOLERANCE {
                    success = true;
                    break;
                }
            } else {
                damping = (damping * 10.0).min(1.0e12);
            }
        }

        Self::six_parameter_result(parameters, success, iterations, objective, measured.len())
    }

    fn six_parameter_result(
        parameters: [f64; PARAMETER_COUNT],
        success: bool,
        iterations: usize,
        objective: f64,
        measured_count: usize,
    ) -> Result<OptimizedResult> {
        let loaded_q = parameters[4].exp();
        let resonant_frequency_hz = parameters[5].exp();
        let measured_count = measured_count.to_f64().unwrap_or(f64::NAN);
        let rms_error = (objective / measured_count).sqrt();
        if !loaded_q.is_finite() || !resonant_frequency_hz.is_finite() || !rms_error.is_finite() {
            return Err(Error::Unsupported(
                "Q-factor optimization produced a non-finite solution".to_owned(),
            ));
        }

        Ok(OptimizedResult {
            success,
            iterations,
            objective,
            parameters: parameters.to_vec(),
            m1: parameters[0],
            m2: parameters[1],
            m3: parameters[2],
            m4: parameters[3],
            loaded_q,
            resonant_frequency_hz,
            rms_error,
            method: QFitMethod::Nlqfit6,
            phase_slope_radians_per_hz: None,
            leakage_slope: None,
            weighting_ratio: None,
            loop_plan: String::new(),
        })
    }

    fn fit_extended(
        &self,
        method: QFitMethod,
        initial: &OptimizedResult,
        weighted: bool,
        maximum_iterations: usize,
    ) -> Result<OptimizedResult> {
        let frequencies = self.network.frequency.values_hz().to_vec();
        let measured = self
            .network
            .s
            .outer_iter()
            .map(|matrix| matrix[(0, 0)])
            .collect::<Vec<_>>();
        let span = frequencies[frequencies.len() - 1] - frequencies[0];
        let parameter_count = match method {
            QFitMethod::Nlqfit6 => PARAMETER_COUNT,
            QFitMethod::Nlqfit7 => 7,
            QFitMethod::Nlqfit8 => 8,
        };
        let mut parameters = vec![
            initial.m1,
            initial.m2,
            initial.m3,
            initial.m4,
            initial.loaded_q.ln(),
            initial.resonant_frequency_hz.ln(),
        ];
        parameters.resize(parameter_count, 0.0);
        let mut damping = 1.0e-3;
        let mut iterations = 0;
        let mut success = false;
        let mut residuals =
            extended_residual_vector(&frequencies, &measured, &parameters, method, span, weighted);
        let mut objective = squared_norm(&residuals);

        for iteration in 1..=maximum_iterations {
            iterations = iteration;
            let jacobian = extended_numerical_jacobian(
                &frequencies,
                &measured,
                &parameters,
                method,
                span,
                weighted,
            );
            let (mut normal, mut gradient) =
                dense_normal_equations(&jacobian, &residuals, parameter_count);
            for diagonal in 0..parameter_count {
                normal[diagonal][diagonal] += damping * normal[diagonal][diagonal].abs().max(1.0);
                gradient[diagonal] = -gradient[diagonal];
            }
            let Some(step) = solve_dense_system(normal, gradient) else {
                damping *= 10.0;
                continue;
            };
            let mut candidate = parameters.clone();
            for index in 0..parameter_count {
                candidate[index] += step[index];
            }
            candidate[4] = candidate[4].clamp(-20.0, 30.0);
            candidate[5] =
                candidate[5].clamp(frequencies[0].ln(), frequencies[frequencies.len() - 1].ln());
            let candidate_residuals = extended_residual_vector(
                &frequencies,
                &measured,
                &candidate,
                method,
                span,
                weighted,
            );
            let candidate_objective = squared_norm(&candidate_residuals);
            if candidate_objective < objective {
                let relative_improvement =
                    (objective - candidate_objective) / objective.max(f64::EPSILON);
                parameters = candidate;
                residuals = candidate_residuals;
                objective = candidate_objective;
                damping = (damping * 0.3).max(1.0e-12);
                let relative_step = step
                    .iter()
                    .zip(&parameters)
                    .map(|(change, value)| change.abs() / value.abs().max(1.0))
                    .fold(0.0, f64::max);
                if relative_improvement < FIT_TOLERANCE || relative_step < FIT_TOLERANCE {
                    success = true;
                    break;
                }
            } else {
                damping = (damping * 10.0).min(1.0e12);
            }
        }

        let loaded_q = parameters[4].exp();
        let resonant_frequency_hz = parameters[5].exp();
        let (weighting_ratio, rms_error) = extended_fit_metrics(
            &frequencies,
            resonant_frequency_hz,
            loaded_q,
            objective,
            weighted,
        );
        Self::extended_result(
            method,
            &parameters,
            (success, iterations, objective),
            span,
            (weighting_ratio, rms_error),
        )
    }

    fn extended_result(
        method: QFitMethod,
        parameters: &[f64],
        optimizer: (bool, usize, f64),
        span: f64,
        metrics: (Option<f64>, f64),
    ) -> Result<OptimizedResult> {
        let (success, iterations, objective) = optimizer;
        let (weighting_ratio, rms_error) = metrics;
        let loaded_q = parameters[4].exp();
        let resonant_frequency_hz = parameters[5].exp();
        if !loaded_q.is_finite() || !resonant_frequency_hz.is_finite() || !rms_error.is_finite() {
            return Err(Error::Unsupported(
                "Q-factor optimization produced a non-finite solution".to_owned(),
            ));
        }
        Ok(OptimizedResult {
            success,
            iterations,
            objective,
            parameters: parameters.to_vec(),
            m1: parameters[0],
            m2: parameters[1],
            m3: parameters[2],
            m4: parameters[3],
            loaded_q,
            resonant_frequency_hz,
            rms_error,
            method,
            phase_slope_radians_per_hz: (method == QFitMethod::Nlqfit7)
                .then(|| parameters[6] / span),
            leakage_slope: (method == QFitMethod::Nlqfit8)
                .then(|| Complex64::new(parameters[6], parameters[7])),
            weighting_ratio,
            loop_plan: String::new(),
        })
    }

    /// Returns the diagonal MAT 58 angular weights.
    ///
    /// $$W_{i}=\frac{1}{\left[2Q_{L}(f_{i}-f_{L})/f_{L}\right]^2+1}.$$
    ///
    /// These weights reduce systematic error when frequency samples are evenly
    /// spaced rather than evenly distributed around the Q circle.
    #[must_use]
    pub fn angular_weights(
        frequencies_hz: &[f64],
        resonant_frequency_hz: f64,
        loaded_q: f64,
    ) -> Vec<f64> {
        frequencies_hz
            .iter()
            .map(|frequency| {
                let offset =
                    2.0 * loaded_q * (*frequency - resonant_frequency_hz) / resonant_frequency_hz;
                1.0 / (offset * offset + 1.0)
            })
            .collect()
    }

    /// Evaluates the fitted resonator response.
    ///
    /// For NLQFIT6,
    /// $$S=m_{1}+jm_{2}+\frac{m_{3}+jm_{4}}{1+jQ_{L}t},\qquad
    /// t=\frac{f}{f_{L}}-\frac{f_{L}}{f}.$$
    ///
    /// # Errors
    ///
    /// Returns an error if the optimized result or frequency axis is invalid.
    pub fn fitted_s(
        &self,
        result: &OptimizedResult,
        frequency: Option<&Frequency>,
    ) -> Result<Vec<Complex64>> {
        validate_result(result)?;
        let frequency = frequency.unwrap_or(&self.network.frequency);
        if frequency.values_hz().iter().any(|value| *value <= 0.0) {
            return Err(Error::InvalidFrequency(
                "the fitted response requires positive frequencies".to_owned(),
            ));
        }
        Ok(frequency
            .values_hz()
            .iter()
            .map(|frequency| model_response_for_result(*frequency, result))
            .collect())
    }

    /// Returns a one-port [`Network`] containing the fitted response.
    ///
    /// # Errors
    ///
    /// Returns an error if response evaluation or network construction fails.
    pub fn fitted_network(&self, result: &OptimizedResult) -> Result<Network> {
        let response = self.fitted_s(result, None)?;
        let s = Array3::from_shape_vec((response.len(), 1, 1), response)
            .map_err(|error| Error::IncompatibleShape(error.to_string()))?;
        let mut network = Network::new(self.network.frequency.clone(), s, self.network.z0.clone())?;
        network.name = self
            .network
            .name
            .as_ref()
            .map(|name| format!("{name}-fitted"));
        network.comments.clone_from(&self.network.comments);
        network.variables.clone_from(&self.network.variables);
        network.s_definition = self.network.s_definition;
        Ok(network)
    }

    /// Equivalent to `fitted_network`, sampled on another frequency axis.
    ///
    /// # Errors
    ///
    /// Returns an error if response evaluation or network construction fails.
    pub fn fitted_network_at(
        &self,
        result: &OptimizedResult,
        frequency: Frequency,
    ) -> Result<Network> {
        let response = self.fitted_s(result, Some(&frequency))?;
        let s = Array3::from_shape_vec((response.len(), 1, 1), response)
            .map_err(|error| Error::IncompatibleShape(error.to_string()))?;
        let reference = self.network.z0[(0, 0)];
        let points = frequency.points();
        Network::new(frequency, s, Array2::from_elem((points, 1), reference))
    }

    /// Returns the scaled Q-circle diameter and detuned/tuned points.
    ///
    /// # Errors
    ///
    /// Returns an error if the fitted result or requested scaling is invalid.
    pub fn q_circle(&self, result: &OptimizedResult, scaling: Option<f64>) -> Result<QCircle> {
        validate_result(result)?;
        let detuned = Complex64::new(result.m1, result.m2);
        let scaling = scaling.unwrap_or_else(|| 1.0 / detuned.norm());
        if !scaling.is_finite() || scaling <= 0.0 || detuned.norm_sqr() <= f64::EPSILON {
            return Err(Error::Unsupported(
                "Q-circle scaling requires a finite positive value and non-zero detuned response"
                    .to_owned(),
            ));
        }
        let tuned = detuned + Complex64::new(result.m3, result.m4);
        Ok(QCircle {
            diameter: (tuned - detuned).norm() * scaling,
            detuned: detuned * scaling,
            tuned: tuned * scaling,
        })
    }

    /// Estimates unloaded quality factor $Q_{0}$ from loaded $Q_{L}$ and coupling.
    ///
    /// # Errors
    ///
    /// Returns an error if circle scaling or coupling yields an invalid estimate.
    pub fn unloaded_q(&self, result: &OptimizedResult, scaling: Option<f64>) -> Result<f64> {
        let circle = match self.resonance_type {
            ResonanceType::Transmission | ResonanceType::ReflectionMethod2 if scaling.is_none() => {
                return Err(Error::Unsupported(
                    "this resonance type requires an explicit Q-circle scaling factor".to_owned(),
                ));
            }
            _ => self.q_circle(result, scaling)?,
        };
        let denominator = match self.resonance_type {
            ResonanceType::Transmission => 1.0 - circle.diameter,
            ResonanceType::Reflection => {
                let coupling = 2.0 / circle.diameter - 1.0;
                1.0 / (1.0 + 1.0 / coupling)
            }
            ResonanceType::ReflectionMethod2 => {
                let detuned = circle.detuned.norm();
                let tuned = circle.tuned.norm();
                let circle_numerator = tuned.mul_add(
                    -tuned,
                    circle.diameter.mul_add(circle.diameter, detuned * detuned),
                );
                let cosine = circle_numerator / (2.0 * detuned * circle.diameter);
                let touching_diameter =
                    detuned.mul_add(-detuned, 1.0) / detuned.mul_add(-cosine, 1.0);
                let coupling = touching_diameter / circle.diameter - 1.0;
                1.0 / (1.0 + 1.0 / coupling)
            }
            ResonanceType::Absorption => {
                let coupling = 1.0 / circle.diameter - 1.0;
                1.0 / (1.0 + 1.0 / coupling)
            }
        };
        if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
            return Err(Error::Unsupported(
                "Q-circle coupling produced a singular unloaded-Q estimate".to_owned(),
            ));
        }
        Ok(result.loaded_q / denominator)
    }

    /// Returns the half-power (3-dB) bandwidth.
    ///
    /// $$BW=\frac{f_{L}}{Q_{L}}.$$
    ///
    /// # Errors
    ///
    /// Returns an error if the optimized result is invalid.
    pub fn bandwidth_hz(result: &OptimizedResult) -> Result<f64> {
        validate_result(result)?;
        Ok(result.resonant_frequency_hz / result.loaded_q)
    }

    /// Resonant frequency expressed in the input network's display unit.
    ///
    /// Origin: `skrf.qfactor.Qfactor.f_L_scaled`.
    ///
    /// # Errors
    ///
    /// Returns an error if the optimized result is invalid.
    pub fn resonant_frequency_scaled(&self, result: &OptimizedResult) -> Result<f64> {
        validate_result(result)?;
        Ok(result.resonant_frequency_hz / self.network.frequency.unit().multiplier())
    }

    /// Three-decibel bandwidth expressed in the input network's display unit.
    ///
    /// Origin: `skrf.qfactor.Qfactor.BW_scaled`.
    ///
    /// # Errors
    ///
    /// Returns an error if the optimized result is invalid.
    pub fn bandwidth_scaled(&self, result: &OptimizedResult) -> Result<f64> {
        Ok(Self::bandwidth_hz(result)? / self.network.frequency.unit().multiplier())
    }
}

impl fmt::Display for QFactor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Q-factor of Network {}",
            self.network.name.as_deref().unwrap_or("<unnamed>")
        )
    }
}

fn dense_normal_equations(
    jacobian: &[Vec<f64>],
    residuals: &[f64],
    parameter_count: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut normal = vec![vec![0.0; parameter_count]; parameter_count];
    let mut gradient = vec![0.0; parameter_count];
    for row in 0..residuals.len() {
        for column in 0..parameter_count {
            gradient[column] = jacobian[row][column].mul_add(residuals[row], gradient[column]);
            for other in 0..parameter_count {
                normal[column][other] =
                    jacobian[row][column].mul_add(jacobian[row][other], normal[column][other]);
            }
        }
    }
    (normal, gradient)
}

fn extended_fit_metrics(
    frequencies: &[f64],
    resonant_frequency_hz: f64,
    loaded_q: f64,
    objective: f64,
    weighted: bool,
) -> (Option<f64>, f64) {
    let weights = if weighted {
        QFactor::angular_weights(frequencies, resonant_frequency_hz, loaded_q)
    } else {
        vec![1.0; frequencies.len()]
    };
    let weighting_ratio = weighted.then(|| {
        let maximum = weights.iter().copied().reduce(f64::max).unwrap_or(1.0);
        let minimum = weights.iter().copied().reduce(f64::min).unwrap_or(1.0);
        maximum / minimum
    });
    let rms_error = (objective / weights.iter().sum::<f64>()).sqrt();
    (weighting_ratio, rms_error)
}

fn validate_loop_plan(loop_plan: &str) -> Result<()> {
    if loop_plan.is_empty() {
        return Err(Error::Unsupported(
            "Q-factor loop plan must not be empty".to_owned(),
        ));
    }
    if loop_plan
        .chars()
        .any(|operation| !matches!(operation, 'f' | 'w' | 'c'))
    {
        return Err(Error::Unsupported(
            "Q-factor loop plan may contain only 'f', 'w', and 'c'".to_owned(),
        ));
    }
    if loop_plan.starts_with('w') || loop_plan.ends_with('w') {
        return Err(Error::Unsupported(
            "Q-factor loop plan must not start or end with weight calculation".to_owned(),
        ));
    }
    Ok(())
}

fn result_parameters(result: &OptimizedResult) -> [f64; PARAMETER_COUNT] {
    [
        result.m1,
        result.m2,
        result.m3,
        result.m4,
        result.loaded_q.ln(),
        result.resonant_frequency_hz.ln(),
    ]
}

fn validate_result(result: &OptimizedResult) -> Result<()> {
    if [
        result.m1,
        result.m2,
        result.m3,
        result.m4,
        result.loaded_q,
        result.resonant_frequency_hz,
    ]
    .iter()
    .all(|value| value.is_finite())
        && result.loaded_q > 0.0
        && result.resonant_frequency_hz > 0.0
    {
        Ok(())
    } else {
        Err(Error::Unsupported(
            "the optimized Q-factor result is incomplete or invalid".to_owned(),
        ))
    }
}

fn model_response(frequency_hz: f64, parameters: [f64; PARAMETER_COUNT]) -> Complex64 {
    let detuned = Complex64::new(parameters[0], parameters[1]);
    let resonant_delta = Complex64::new(parameters[2], parameters[3]);
    let loaded_q = parameters[4].exp();
    let resonant_frequency_hz = parameters[5].exp();
    let fractional_offset =
        frequency_hz / resonant_frequency_hz - resonant_frequency_hz / frequency_hz;
    detuned + resonant_delta / Complex64::new(1.0, loaded_q * fractional_offset)
}

fn model_response_for_result(frequency_hz: f64, result: &OptimizedResult) -> Complex64 {
    let base = model_response(frequency_hz, result_parameters(result));
    match result.method {
        QFitMethod::Nlqfit6 => base,
        QFitMethod::Nlqfit7 => {
            let phase = result.phase_slope_radians_per_hz.unwrap_or(0.0)
                * (frequency_hz - result.resonant_frequency_hz);
            base * Complex64::from_polar(1.0, phase)
        }
        QFitMethod::Nlqfit8 => {
            let offset =
                2.0 * (frequency_hz - result.resonant_frequency_hz) / result.resonant_frequency_hz;
            base + result.leakage_slope.unwrap_or_default() * offset
        }
    }
}

fn extended_model_response(
    frequency_hz: f64,
    parameters: &[f64],
    method: QFitMethod,
    span_hz: f64,
) -> Complex64 {
    let base = model_response(
        frequency_hz,
        [
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
            parameters[5],
        ],
    );
    let resonant_frequency_hz = parameters[5].exp();
    match method {
        QFitMethod::Nlqfit6 => base,
        QFitMethod::Nlqfit7 => {
            let normalized_offset = (frequency_hz - resonant_frequency_hz) / span_hz;
            base * Complex64::from_polar(1.0, parameters[6] * normalized_offset)
        }
        QFitMethod::Nlqfit8 => {
            let offset = 2.0 * (frequency_hz - resonant_frequency_hz) / resonant_frequency_hz;
            base + Complex64::new(parameters[6], parameters[7]) * offset
        }
    }
}

fn extended_residual_vector(
    frequencies_hz: &[f64],
    measured: &[Complex64],
    parameters: &[f64],
    method: QFitMethod,
    span_hz: f64,
    weighted: bool,
) -> Vec<f64> {
    let loaded_q = parameters[4].exp();
    let resonant_frequency_hz = parameters[5].exp();
    let weights = if weighted {
        QFactor::angular_weights(frequencies_hz, resonant_frequency_hz, loaded_q)
    } else {
        vec![1.0; frequencies_hz.len()]
    };
    let mut residuals = Vec::with_capacity(measured.len() * 2);
    for ((frequency, measured), weight) in frequencies_hz.iter().zip(measured).zip(weights) {
        let residual = (extended_model_response(*frequency, parameters, method, span_hz)
            - measured)
            * weight.sqrt();
        residuals.push(residual.re);
        residuals.push(residual.im);
    }
    residuals
}

fn extended_numerical_jacobian(
    frequencies_hz: &[f64],
    measured: &[Complex64],
    parameters: &[f64],
    method: QFitMethod,
    span_hz: f64,
    weighted: bool,
) -> Vec<Vec<f64>> {
    let mut jacobian = vec![vec![0.0; parameters.len()]; measured.len() * 2];
    for parameter in 0..parameters.len() {
        let step = f64::EPSILON.sqrt() * parameters[parameter].abs().max(1.0);
        let mut lower = parameters.to_vec();
        let mut upper = parameters.to_vec();
        lower[parameter] -= step;
        upper[parameter] += step;
        let lower_residuals =
            extended_residual_vector(frequencies_hz, measured, &lower, method, span_hz, weighted);
        let upper_residuals =
            extended_residual_vector(frequencies_hz, measured, &upper, method, span_hz, weighted);
        for row in 0..jacobian.len() {
            jacobian[row][parameter] = (upper_residuals[row] - lower_residuals[row]) / (2.0 * step);
        }
    }
    jacobian
}

fn residual_vector(
    frequencies_hz: &[f64],
    measured: &[Complex64],
    parameters: &[f64; PARAMETER_COUNT],
) -> Vec<f64> {
    let mut residuals = Vec::with_capacity(measured.len() * 2);
    for (frequency, measured) in frequencies_hz.iter().zip(measured.iter()) {
        let residual = model_response(*frequency, *parameters) - measured;
        residuals.push(residual.re);
        residuals.push(residual.im);
    }
    residuals
}

fn numerical_jacobian(
    frequencies_hz: &[f64],
    measured: &[Complex64],
    parameters: &[f64; PARAMETER_COUNT],
) -> Vec<[f64; PARAMETER_COUNT]> {
    let mut jacobian = vec![[0.0; PARAMETER_COUNT]; measured.len() * 2];
    for parameter in 0..PARAMETER_COUNT {
        let step = f64::EPSILON.sqrt() * parameters[parameter].abs().max(1.0);
        let mut lower = *parameters;
        let mut upper = *parameters;
        lower[parameter] -= step;
        upper[parameter] += step;
        let lower_residuals = residual_vector(frequencies_hz, measured, &lower);
        let upper_residuals = residual_vector(frequencies_hz, measured, &upper);
        for row in 0..jacobian.len() {
            jacobian[row][parameter] = (upper_residuals[row] - lower_residuals[row]) / (2.0 * step);
        }
    }
    jacobian
}

fn squared_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum()
}

fn solve_six_by_six(
    mut matrix: [[f64; PARAMETER_COUNT]; PARAMETER_COUNT],
    mut right: [f64; PARAMETER_COUNT],
) -> Option<[f64; PARAMETER_COUNT]> {
    for pivot in 0..PARAMETER_COUNT {
        let best = (pivot..PARAMETER_COUNT).max_by(|left, right_index| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right_index][pivot].abs())
        })?;
        if matrix[best][pivot].abs() <= f64::EPSILON {
            return None;
        }
        matrix.swap(pivot, best);
        right.swap(pivot, best);
        let pivot_row = matrix[pivot];
        for row in pivot + 1..PARAMETER_COUNT {
            let multiplier = matrix[row][pivot] / matrix[pivot][pivot];
            for (value, pivot_value) in matrix[row][pivot..]
                .iter_mut()
                .zip(pivot_row[pivot..].iter())
            {
                *value -= multiplier * pivot_value;
            }
            right[row] = multiplier.mul_add(-right[pivot], right[row]);
        }
    }

    let mut solution = [0.0; PARAMETER_COUNT];
    for row in (0..PARAMETER_COUNT).rev() {
        let mut value = right[row];
        for column in row + 1..PARAMETER_COUNT {
            value = matrix[row][column].mul_add(-solution[column], value);
        }
        solution[row] = value / matrix[row][row];
    }
    Some(solution)
}

fn solve_dense_system(mut matrix: Vec<Vec<f64>>, mut right: Vec<f64>) -> Option<Vec<f64>> {
    let size = right.len();
    if matrix.len() != size || matrix.iter().any(|row| row.len() != size) {
        return None;
    }
    for pivot in 0..size {
        let best = (pivot..size).max_by(|left, right_index| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right_index][pivot].abs())
        })?;
        if matrix[best][pivot].abs() <= f64::EPSILON {
            return None;
        }
        matrix.swap(pivot, best);
        right.swap(pivot, best);
        let pivot_row = matrix[pivot].clone();
        for row in pivot + 1..size {
            let multiplier = matrix[row][pivot] / matrix[pivot][pivot];
            for (value, pivot_value) in matrix[row][pivot..].iter_mut().zip(&pivot_row[pivot..]) {
                *value -= multiplier * pivot_value;
            }
            right[row] = multiplier.mul_add(-right[pivot], right[row]);
        }
    }
    let mut solution = vec![0.0; size];
    for row in (0..size).rev() {
        let mut value = right[row];
        for (column, solution_value) in solution.iter().enumerate().skip(row + 1) {
            value -= matrix[row][column] * solution_value;
        }
        solution[row] = value / matrix[row][row];
    }
    Some(solution)
}
use std::fmt;
