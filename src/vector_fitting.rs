use std::fs::File;
use std::io::Write;
use std::path::Path;

use ndarray::{Array1, Array2, Array3};
use ndarray_npy::{NpzReader, NpzWriter};
use num_complex::Complex64;

use crate::math::left_solve;
use crate::plotting::{Component, Plot, PlotSeries};
use crate::{Error, Network, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct StateSpaceModel {
    pub a: Array2<f64>,
    pub b: Array2<f64>,
    pub c: Array2<f64>,
    pub d: Array2<f64>,
    pub e: Array2<f64>,
}

/// Origin: `skrf/vectorFitting.py::VectorFitting`.
#[derive(Clone, Debug)]
pub struct VectorFitting {
    pub network: Network,
    pub poles: Array1<Complex64>,
    pub residues: Array2<Complex64>,
    pub proportional_coefficients: Array1<f64>,
    pub constant_coefficients: Array1<f64>,
}

impl VectorFitting {
    pub fn new(network: Network) -> Self {
        let responses = network.ports() * network.ports();
        Self {
            network,
            poles: Array1::zeros(0),
            residues: Array2::zeros((responses, 0)),
            proportional_coefficients: Array1::zeros(responses),
            constant_coefficients: Array1::zeros(responses),
        }
    }

    /// Fits residues for stable, linearly spaced initial poles.
    ///
    /// This is the residue-identification stage of the upstream vector fitting
    /// algorithm. Complex poles store only the positive-imaginary member of a
    /// conjugate pair, matching scikit-rf's pole representation.
    pub fn vector_fit(&mut self, real_poles: usize, complex_poles: usize) -> Result<()> {
        if real_poles + complex_poles == 0 {
            return Err(Error::Unsupported(
                "vector fitting requires at least one pole".to_owned(),
            ));
        }
        let frequencies = self.network.frequency.values_hz().clone();
        if frequencies.len() < 2 || !self.network.frequency.is_monotonic_increasing() {
            return Err(Error::InvalidFrequency(
                "vector fitting requires at least two increasing frequency samples".to_owned(),
            ));
        }
        let normalization = frequencies.iter().sum::<f64>() / frequencies.len() as f64;
        if !normalization.is_finite() || normalization <= 0.0 {
            return Err(Error::InvalidFrequency(
                "vector fitting requires a positive frequency scale".to_owned(),
            ));
        }
        let minimum = if frequencies[0] == 0.0 {
            frequencies
                .iter()
                .copied()
                .find(|frequency| *frequency > 0.0)
                .ok_or_else(|| {
                    Error::InvalidFrequency(
                        "vector fitting cannot initialize poles from an all-zero axis".to_owned(),
                    )
                })?
                / 1000.0
        } else {
            frequencies[0]
        };
        let maximum = frequencies[frequencies.len() - 1];
        let mut normalized_poles = Vec::with_capacity(real_poles + complex_poles);
        for frequency in linear_space(minimum / normalization, maximum / normalization, real_poles)
        {
            normalized_poles.push(Complex64::new(-std::f64::consts::TAU * frequency, 0.0));
        }
        for frequency in linear_space(
            minimum / normalization,
            maximum / normalization,
            complex_poles,
        ) {
            let omega = std::f64::consts::TAU * frequency;
            normalized_poles.push(Complex64::new(-0.01 * omega, omega));
        }

        self.fit_normalized_poles(&frequencies, normalization, &normalized_poles)
    }

    /// Fits model coefficients using caller-supplied poles in radians per second.
    ///
    /// This is the typed Rust counterpart of upstream `init_pole_spacing="custom"`.
    pub fn fit_with_poles(&mut self, poles: &Array1<Complex64>) -> Result<()> {
        if poles.is_empty()
            || poles
                .iter()
                .any(|pole| !pole.re.is_finite() || !pole.im.is_finite() || pole.re >= 0.0)
        {
            return Err(Error::Unsupported(
                "custom poles must be a non-empty set of finite stable poles".to_owned(),
            ));
        }
        let frequencies = self.network.frequency.values_hz().clone();
        if frequencies.len() < 2 || !self.network.frequency.is_monotonic_increasing() {
            return Err(Error::InvalidFrequency(
                "vector fitting requires at least two increasing frequency samples".to_owned(),
            ));
        }
        let normalization = frequencies.iter().sum::<f64>() / frequencies.len() as f64;
        if !normalization.is_finite() || normalization <= 0.0 {
            return Err(Error::InvalidFrequency(
                "vector fitting requires a positive frequency scale".to_owned(),
            ));
        }
        let normalized_poles = poles
            .iter()
            .map(|pole| *pole / normalization)
            .collect::<Vec<_>>();
        self.fit_normalized_poles(&frequencies, normalization, &normalized_poles)
    }

    /// Iteratively relocates poles before the final residue identification.
    ///
    /// This implements the shared-denominator Sanathanan-Koerner stage used by
    /// upstream `VectorFitting._pole_relocation`.
    pub fn vector_fit_relocating(
        &mut self,
        real_poles: usize,
        complex_poles: usize,
        maximum_iterations: usize,
        tolerance: f64,
    ) -> Result<()> {
        if real_poles + complex_poles == 0
            || maximum_iterations == 0
            || !tolerance.is_finite()
            || tolerance < 0.0
        {
            return Err(Error::Unsupported(
                "pole relocation requires poles, iterations, and non-negative finite tolerance"
                    .to_owned(),
            ));
        }
        let frequencies = self.network.frequency.values_hz().clone();
        if frequencies.len() < 2 || !self.network.frequency.is_monotonic_increasing() {
            return Err(Error::InvalidFrequency(
                "vector fitting requires at least two increasing frequency samples".to_owned(),
            ));
        }
        let normalization = frequencies.iter().sum::<f64>() / frequencies.len() as f64;
        if !normalization.is_finite() || normalization <= 0.0 {
            return Err(Error::InvalidFrequency(
                "vector fitting requires a positive frequency scale".to_owned(),
            ));
        }
        let minimum = frequencies
            .iter()
            .copied()
            .find(|frequency| *frequency > 0.0)
            .ok_or_else(|| {
                Error::InvalidFrequency(
                    "vector fitting cannot initialize poles from an all-zero axis".to_owned(),
                )
            })?;
        let maximum = frequencies[frequencies.len() - 1];
        let mut poles = linear_space(minimum / normalization, maximum / normalization, real_poles)
            .into_iter()
            .map(|frequency| Complex64::new(-std::f64::consts::TAU * frequency, 0.0))
            .chain(
                linear_space(
                    minimum / normalization,
                    maximum / normalization,
                    complex_poles,
                )
                .into_iter()
                .map(|frequency| {
                    let omega = std::f64::consts::TAU * frequency;
                    Complex64::new(-0.01 * omega, omega)
                }),
            )
            .collect::<Vec<_>>();
        for _ in 0..maximum_iterations {
            let next = relocate_poles(&self.network, &frequencies, normalization, &poles)?;
            let change = pole_set_change(&poles, &next);
            poles = next;
            if change <= tolerance {
                break;
            }
        }
        self.fit_normalized_poles(&frequencies, normalization, &poles)
    }

    fn fit_normalized_poles(
        &mut self,
        frequencies: &Array1<f64>,
        normalization: f64,
        normalized_poles: &[Complex64],
    ) -> Result<()> {
        let columns = coefficient_count(normalized_poles);
        if 2 * frequencies.len() < columns {
            return Err(Error::IncompatibleShape(format!(
                "{} complex samples cannot identify {columns} real model coefficients",
                frequencies.len()
            )));
        }
        let design = design_matrix(frequencies, normalization, normalized_poles);
        let responses = self.network.ports() * self.network.ports();
        let mut residues = Array2::zeros((responses, normalized_poles.len()));
        let mut constants = Array1::zeros(responses);
        let mut proportionals = Array1::zeros(responses);
        for output in 0..self.network.ports() {
            for input in 0..self.network.ports() {
                let response = output * self.network.ports() + input;
                let mut right = Vec::with_capacity(2 * frequencies.len());
                for point in 0..frequencies.len() {
                    let value = self.network.s[(point, output, input)];
                    right.push(value.re);
                    right.push(value.im);
                }
                let solution = solve_least_squares(&design, &right)?;
                let mut column = 0;
                for (pole_index, pole) in normalized_poles.iter().enumerate() {
                    if pole.im == 0.0 {
                        residues[(response, pole_index)] =
                            Complex64::new(solution[column] * normalization, 0.0);
                        column += 1;
                    } else {
                        residues[(response, pole_index)] = Complex64::new(
                            solution[column] * normalization,
                            solution[column + 1] * normalization,
                        );
                        column += 2;
                    }
                }
                constants[response] = solution[column];
                proportionals[response] = solution[column + 1] / normalization;
            }
        }
        self.poles = Array1::from_vec(
            normalized_poles
                .iter()
                .map(|pole| *pole * normalization)
                .collect(),
        );
        self.residues = residues;
        self.constant_coefficients = constants;
        self.proportional_coefficients = proportionals;
        self.enforce_dc_sample()?;
        Ok(())
    }

    /// Baseline automatic fit using the upstream default initial model order.
    pub fn auto_fit(&mut self) -> Result<()> {
        self.vector_fit(3, 3)
    }

    /// Adaptive-order fit which retains the lowest-error model encountered.
    pub fn auto_fit_with_tolerance(
        &mut self,
        maximum_model_order: usize,
        target_rms_error: f64,
    ) -> Result<()> {
        if maximum_model_order == 0 || !target_rms_error.is_finite() || target_rms_error < 0.0 {
            return Err(Error::Unsupported(
                "automatic fitting requires positive order and non-negative finite tolerance"
                    .to_owned(),
            ));
        }
        let mut best = None;
        for order in 1..=maximum_model_order {
            let real_poles = order % 2;
            let complex_poles = order / 2;
            let mut candidate = Self::new(self.network.clone());
            candidate.vector_fit(real_poles, complex_poles)?;
            let error = candidate.rms_error()?;
            if best
                .as_ref()
                .is_none_or(|(best_error, _): &(f64, Self)| error < *best_error)
            {
                best = Some((error, candidate));
            }
            if error <= target_rms_error {
                break;
            }
        }
        let (_, best) = best.ok_or_else(|| {
            Error::Unsupported("automatic fitting did not evaluate a model order".to_owned())
        })?;
        *self = best;
        Ok(())
    }

    /// Port of `VectorFitting.get_model_order`.
    pub fn model_order(poles: &Array1<Complex64>) -> usize {
        poles
            .iter()
            .map(|pole| if pole.im == 0.0 { 1 } else { 2 })
            .sum()
    }

    /// Port of `VectorFitting.get_spurious` using trapezoidal energy norms.
    pub fn spurious_poles(
        poles: &Array1<Complex64>,
        residues: &Array2<Complex64>,
        frequency_samples: usize,
        gamma: f64,
    ) -> Result<Vec<bool>> {
        if residues.ncols() != poles.len() {
            return Err(Error::IncompatibleShape(format!(
                "{} poles require residue columns, got {:?}",
                poles.len(),
                residues.dim()
            )));
        }
        if frequency_samples < 2 || !gamma.is_finite() || gamma < 0.0 {
            return Err(Error::Unsupported(
                "spurious-pole classification requires at least two samples and non-negative finite gamma"
                    .to_owned(),
            ));
        }
        let complex_indexes = poles
            .iter()
            .enumerate()
            .filter_map(|(index, pole)| (pole.im > 0.0).then_some(index))
            .collect::<Vec<_>>();
        let mut spurious = vec![false; poles.len()];
        if complex_indexes.is_empty() {
            return Ok(spurious);
        }
        let minimum = complex_indexes
            .iter()
            .map(|index| poles[*index].im)
            .fold(f64::INFINITY, f64::min)
            / 3.0;
        let maximum = complex_indexes
            .iter()
            .map(|index| poles[*index].im)
            .fold(f64::NEG_INFINITY, f64::max)
            * 3.0;
        let omega = Array1::linspace(minimum, maximum, frequency_samples);
        let mut norms = Array2::<f64>::zeros((residues.nrows(), complex_indexes.len()));
        for response in 0..residues.nrows() {
            for (candidate, pole_index) in complex_indexes.iter().copied().enumerate() {
                let pole = poles[pole_index];
                let residue = residues[(response, pole_index)];
                let mut integral = 0.0;
                for sample in 0..frequency_samples - 1 {
                    let evaluate = |angular_frequency: f64| {
                        let s = Complex64::new(0.0, angular_frequency);
                        let value = residue / (s - pole) + residue.conj() / (s - pole.conj());
                        value.norm_sqr()
                    };
                    let left = evaluate(omega[sample]);
                    let right = evaluate(omega[sample + 1]);
                    integral += (omega[sample + 1] - omega[sample]) * (left + right) / 2.0;
                }
                norms[(response, candidate)] = integral.sqrt();
            }
        }
        let mean = norms.iter().sum::<f64>() / norms.len() as f64;
        if mean > 0.0 {
            for (candidate, pole_index) in complex_indexes.iter().copied().enumerate() {
                spurious[pole_index] = (0..residues.nrows())
                    .all(|response| norms[(response, candidate)] / mean < gamma);
            }
        }
        Ok(spurious)
    }

    /// Port of `VectorFitting._get_ABCDE`.
    pub fn state_space(&self) -> Result<StateSpaceModel> {
        self.validate_model_state()?;
        let ports = self.network.ports();
        let order = Self::model_order(&self.poles);
        let dimension = order * ports;
        let mut a = Array2::eye(dimension);
        let mut b = Array2::zeros((dimension, ports));
        let mut state = 0;
        for input in 0..ports {
            for pole in &self.poles {
                if pole.im == 0.0 {
                    a[(state, state)] = pole.re;
                    b[(state, input)] = 1.0;
                    state += 1;
                } else {
                    a[(state, state)] = pole.re;
                    a[(state, state + 1)] = pole.im;
                    a[(state + 1, state)] = -pole.im;
                    a[(state + 1, state + 1)] = pole.re;
                    b[(state, input)] = 2.0;
                    state += 2;
                }
            }
        }
        let mut c = Array2::zeros((ports, dimension));
        let mut d = Array2::zeros((ports, ports));
        let mut e = Array2::zeros((ports, ports));
        for output in 0..ports {
            for input in 0..ports {
                let response = output * ports + input;
                let mut residue_state = input * order;
                for (pole_index, pole) in self.poles.iter().enumerate() {
                    let residue = self.residues[(response, pole_index)];
                    c[(output, residue_state)] = residue.re;
                    residue_state += 1;
                    if pole.im != 0.0 {
                        c[(output, residue_state)] = residue.im;
                        residue_state += 1;
                    }
                }
                d[(output, input)] = self.constant_coefficients[response];
                e[(output, input)] = self.proportional_coefficients[response];
            }
        }
        Ok(StateSpaceModel { a, b, c, d, e })
    }

    /// Port of `VectorFitting._get_s_from_ABCDE`.
    pub fn response_from_state_space(
        frequencies_hz: &Array1<f64>,
        model: &StateSpaceModel,
    ) -> Result<Array3<Complex64>> {
        let dimension = model.a.nrows();
        let ports = model.d.nrows();
        if dimension == 0
            || model.a.ncols() != dimension
            || model.b.dim() != (dimension, ports)
            || model.c.dim() != (ports, dimension)
            || model.d.dim() != (ports, ports)
            || model.e.dim() != (ports, ports)
        {
            return Err(Error::IncompatibleShape(
                "state-space matrices have incompatible dimensions".to_owned(),
            ));
        }
        if frequencies_hz
            .iter()
            .any(|frequency| !frequency.is_finite())
        {
            return Err(Error::InvalidFrequency(
                "state-space evaluation frequencies must be finite".to_owned(),
            ));
        }
        let points = frequencies_hz.len();
        let mut system = Array3::<Complex64>::zeros((points, dimension, dimension));
        let mut identity = Array3::<Complex64>::zeros((points, dimension, dimension));
        for point in 0..points {
            let s = Complex64::new(0.0, std::f64::consts::TAU * frequencies_hz[point]);
            for row in 0..dimension {
                for column in 0..dimension {
                    system[(point, row, column)] = Complex64::new(-model.a[(row, column)], 0.0);
                }
                system[(point, row, row)] += s;
                identity[(point, row, row)] = Complex64::new(1.0, 0.0);
            }
        }
        let inverse = left_solve(&system, &identity)?;
        Ok(Array3::from_shape_fn(
            (points, ports, ports),
            |(point, output, input)| {
                let mut value = Complex64::new(
                    model.d[(output, input)],
                    std::f64::consts::TAU * frequencies_hz[point] * model.e[(output, input)],
                );
                for row in 0..dimension {
                    for column in 0..dimension {
                        value += model.c[(output, row)]
                            * inverse[(point, row, column)]
                            * model.b[(column, input)];
                    }
                }
                value
            },
        ))
    }

    /// Port of `skrf.vectorFitting.VectorFitting.get_model_response`.
    pub fn model_response(
        &self,
        i: usize,
        j: usize,
        frequencies_hz: &Array1<f64>,
    ) -> Result<Array1<Complex64>> {
        let ports = self.network.ports();
        if i >= ports {
            return Err(Error::InvalidPort { port: i, ports });
        }
        if j >= ports {
            return Err(Error::InvalidPort { port: j, ports });
        }
        let responses = ports * ports;
        if self.poles.is_empty()
            || self.residues.dim() != (responses, self.poles.len())
            || self.constant_coefficients.len() != responses
            || self.proportional_coefficients.len() != responses
        {
            return Err(Error::Unsupported(
                "the vector model has not been fitted".to_owned(),
            ));
        }
        if frequencies_hz
            .iter()
            .any(|frequency| !frequency.is_finite())
        {
            return Err(Error::InvalidFrequency(
                "model evaluation frequencies must be finite".to_owned(),
            ));
        }
        let response = i * ports + j;
        Ok(frequencies_hz.mapv(|frequency| {
            let s = Complex64::new(0.0, std::f64::consts::TAU * frequency);
            let mut value =
                self.constant_coefficients[response] + self.proportional_coefficients[response] * s;
            for (pole_index, pole) in self.poles.iter().enumerate() {
                let residue = self.residues[(response, pole_index)];
                value += residue / (s - pole);
                if pole.im != 0.0 {
                    value += residue.conj() / (s - pole.conj());
                }
            }
            value
        }))
    }

    pub fn rms_error(&self) -> Result<f64> {
        let mut squared_error = 0.0;
        let mut samples = 0;
        for output in 0..self.network.ports() {
            for input in 0..self.network.ports() {
                let model =
                    self.model_response(output, input, self.network.frequency.values_hz())?;
                for point in 0..self.network.frequency_points() {
                    squared_error +=
                        (model[point] - self.network.s[(point, output, input)]).norm_sqr();
                    samples += 1;
                }
            }
        }
        Ok((squared_error / samples as f64).sqrt())
    }

    /// Backend-neutral data for upstream `VectorFitting.plot` wrappers.
    pub fn model_plot(
        &self,
        component: Component,
        ports: Option<(usize, usize)>,
        frequencies_hz: Option<&Array1<f64>>,
    ) -> Result<Plot> {
        self.validate_model_state()?;
        let frequencies = frequencies_hz
            .unwrap_or_else(|| self.network.frequency.values_hz())
            .clone();
        let pairs = match ports {
            Some((output, input)) => {
                self.validate_ports(output, input)?;
                vec![(output, input)]
            }
            None => (0..self.network.ports())
                .flat_map(|output| (0..self.network.ports()).map(move |input| (output, input)))
                .collect(),
        };
        let use_samples = frequencies == *self.network.frequency.values_hz();
        let mut series = Vec::with_capacity(pairs.len() * if use_samples { 2 } else { 1 });
        for (output, input) in pairs {
            if use_samples {
                series.push(PlotSeries {
                    label: format!("S{}{} samples", output + 1, input + 1),
                    x: frequencies.to_vec(),
                    y: (0..frequencies.len())
                        .map(|point| {
                            project_component(self.network.s[(point, output, input)], component)
                        })
                        .collect(),
                });
            }
            series.push(PlotSeries {
                label: format!("S{}{} fit", output + 1, input + 1),
                x: frequencies.to_vec(),
                y: self
                    .model_response(output, input, &frequencies)?
                    .iter()
                    .map(|value| project_component(*value, component))
                    .collect(),
            });
        }
        Ok(Plot {
            title: "Vector-fitted model".to_owned(),
            x_label: "Frequency (Hz)".to_owned(),
            y_label: vector_component_label(component).to_owned(),
            series,
        })
    }

    /// Singular values of the fitted scattering matrix over frequency.
    pub fn singular_value_plot(&self, frequencies_hz: Option<&Array1<f64>>) -> Result<Plot> {
        self.validate_model_state()?;
        let frequencies = frequencies_hz
            .unwrap_or_else(|| self.network.frequency.values_hz())
            .clone();
        let scattering = Self::response_from_state_space(&frequencies, &self.state_space()?)?;
        Ok(Plot {
            title: "Vector-fitted singular values".to_owned(),
            x_label: "Frequency (Hz)".to_owned(),
            y_label: "Singular value".to_owned(),
            series: vec![PlotSeries {
                label: "maximum singular value".to_owned(),
                x: frequencies.to_vec(),
                y: (0..frequencies.len())
                    .map(|point| largest_singular_value(&scattering, point))
                    .collect(),
            }],
        })
    }

    /// Sampled passivity violation bands for scattering models.
    pub fn passivity_bands(
        &self,
        frequency_samples: usize,
        maximum_frequency_hz: Option<f64>,
    ) -> Result<Vec<(f64, f64)>> {
        self.validate_model_state()?;
        if frequency_samples < 2 {
            return Err(Error::Unsupported(
                "passivity testing requires at least two frequency samples".to_owned(),
            ));
        }
        let maximum = maximum_frequency_hz
            .unwrap_or_else(|| self.network.frequency.stop().unwrap_or(0.0) * 1.2);
        if !maximum.is_finite() || maximum <= 0.0 {
            return Err(Error::InvalidFrequency(
                "passivity testing requires a positive maximum frequency".to_owned(),
            ));
        }
        let frequencies = Array1::linspace(0.0, maximum, frequency_samples);
        let scattering = Self::response_from_state_space(&frequencies, &self.state_space()?)?;
        let violations = (0..frequency_samples)
            .map(|point| largest_singular_value(&scattering, point) > 1.0 + 1.0e-9)
            .collect::<Vec<_>>();
        let mut bands = Vec::new();
        let mut start = None;
        for (point, violation) in violations.iter().copied().enumerate() {
            if violation && start.is_none() {
                start = Some(frequencies[point]);
            }
            if !violation {
                if let Some(begin) = start.take() {
                    bands.push((begin, frequencies[point - 1]));
                }
            } else if point + 1 == frequency_samples {
                if let Some(begin) = start.take() {
                    bands.push((begin, frequencies[point]));
                }
            }
        }
        Ok(bands)
    }

    pub fn is_passive(&self) -> Result<bool> {
        Ok(self.passivity_bands(200, None)?.is_empty())
    }

    /// Enforces sampled passivity by uniformly scaling the fitted transfer matrix.
    pub fn enforce_passivity(
        &mut self,
        frequency_samples: usize,
        maximum_frequency_hz: Option<f64>,
    ) -> Result<()> {
        self.validate_model_state()?;
        if frequency_samples < 2 {
            return Err(Error::Unsupported(
                "passivity enforcement requires at least two frequency samples".to_owned(),
            ));
        }
        let maximum = maximum_frequency_hz
            .unwrap_or_else(|| self.network.frequency.stop().unwrap_or(0.0) * 1.2);
        if !maximum.is_finite() || maximum <= 0.0 {
            return Err(Error::InvalidFrequency(
                "passivity enforcement requires a positive maximum frequency".to_owned(),
            ));
        }
        let frequencies = Array1::linspace(0.0, maximum, frequency_samples);
        let scattering = Self::response_from_state_space(&frequencies, &self.state_space()?)?;
        let maximum_singular = (0..frequency_samples)
            .map(|point| largest_singular_value(&scattering, point))
            .fold(0.0_f64, f64::max);
        if maximum_singular > 1.0 {
            let scale = (1.0 - 1.0e-9) / maximum_singular;
            self.residues.mapv_inplace(|value| value * scale);
            self.constant_coefficients
                .mapv_inplace(|value| value * scale);
            self.proportional_coefficients
                .mapv_inplace(|value| value * scale);
        }
        Ok(())
    }

    /// NumPy-compatible NPZ persistence for fitted model coefficients.
    pub fn write_npz(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate_model_state()?;
        let file = File::create(path)?;
        let mut archive = NpzWriter::new(file);
        archive.add_array("poles", &self.poles).map_err(npy_error)?;
        archive
            .add_array("residues", &self.residues)
            .map_err(npy_error)?;
        archive
            .add_array("constants", &self.constant_coefficients)
            .map_err(npy_error)?;
        archive
            .add_array("proportionals", &self.proportional_coefficients)
            .map_err(npy_error)?;
        archive.finish().map_err(npy_error)?;
        Ok(())
    }

    pub fn read_npz(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::open(path)?;
        let mut archive = NpzReader::new(file).map_err(npy_error)?;
        self.poles = archive.by_name("poles.npy").map_err(npy_error)?;
        self.residues = archive.by_name("residues.npy").map_err(npy_error)?;
        self.constant_coefficients = archive.by_name("constants.npy").map_err(npy_error)?;
        self.proportional_coefficients = archive.by_name("proportionals.npy").map_err(npy_error)?;
        self.validate_model_state()
    }

    /// Writes an S-parameter state-space equivalent subcircuit.
    ///
    /// Origin: `VectorFitting.write_spice_subcircuit_s`.
    pub fn write_spice_subcircuit(
        &self,
        path: impl AsRef<Path>,
        model_name: &str,
        create_reference_pins: bool,
    ) -> Result<()> {
        self.validate_model_state()?;
        if model_name.is_empty()
            || !model_name
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            return Err(Error::Unsupported(
                "SPICE model names may contain only ASCII letters, digits, and underscores"
                    .to_owned(),
            ));
        }
        let mut file = File::create(path)?;
        writeln!(file, "* EQUIVALENT CIRCUIT FOR VECTOR FITTED S-MATRIX")?;
        writeln!(file, "* Created using rust-rf vector_fitting.rs")?;
        writeln!(file, "*")?;
        let pins = (0..self.network.ports())
            .flat_map(|port| {
                if create_reference_pins {
                    vec![format!("p{}", port + 1), format!("p{}_ref", port + 1)]
                } else {
                    vec![format!("p{}", port + 1)]
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(file, ".SUBCKT {model_name} {pins}")?;
        let build_proportional = self
            .proportional_coefficients
            .iter()
            .any(|coefficient| *coefficient != 0.0);

        for output in 0..self.network.ports() {
            let port = output + 1;
            let output_reference = spice_reference_node(output, create_reference_pins);
            let z0 = self.network.z0[(0, output)].re;
            if !z0.is_finite() || z0 <= 0.0 {
                return Err(Error::Unsupported(format!(
                    "SPICE synthesis requires positive real reference impedance at port {port}"
                )));
            }
            let voltage_wave_gain = 0.5 / z0.sqrt();
            let current_wave_gain = 0.5 * z0.sqrt();
            let reflected_wave_gain = 2.0 / z0.sqrt();
            writeln!(file, "*")?;
            writeln!(file, "* Port network for port {port}")?;
            writeln!(file, "V{port} p{port} s{port} 0")?;
            writeln!(file, "R{port} s{port} {output_reference} {z0}")?;

            for input in 0..self.network.ports() {
                let input_port = input + 1;
                let input_reference = spice_reference_node(input, create_reference_pins);
                let input_z0 = self.network.z0[(0, input)].re;
                if !input_z0.is_finite() || input_z0 <= 0.0 {
                    return Err(Error::Unsupported(format!(
                        "SPICE synthesis requires positive real reference impedance at port {input_port}"
                    )));
                }
                let response = output * self.network.ports() + input;
                let constant = self.constant_coefficients[response];
                if constant != 0.0 {
                    let voltage_gain = reflected_wave_gain * constant * 0.5 / input_z0.sqrt();
                    let current_gain = reflected_wave_gain * constant * 0.5 * input_z0.sqrt();
                    writeln!(
                        file,
                        "Gd{port}_{input_port} {output_reference} s{port} p{input_port} {input_reference} {voltage_gain}"
                    )?;
                    writeln!(
                        file,
                        "Fd{port}_{input_port} {output_reference} s{port} V{input_port} {current_gain}"
                    )?;
                }
                let proportional = self.proportional_coefficients[response];
                if build_proportional && proportional != 0.0 {
                    let gain = reflected_wave_gain * proportional;
                    writeln!(
                        file,
                        "Ge{port}_{input_port} {output_reference} s{port} e{input_port} 0 {gain}"
                    )?;
                }
                for (pole_index, pole) in self.poles.iter().enumerate() {
                    let state = pole_index + 1;
                    let residue = self.residues[(response, pole_index)];
                    let real_gain = reflected_wave_gain * residue.re;
                    if pole.im == 0.0 {
                        writeln!(
                            file,
                            "Gr{state}_{port}_{input_port} {output_reference} s{port} x{state}_a{input_port} 0 {real_gain}"
                        )?;
                    } else {
                        let imaginary_gain = reflected_wave_gain * residue.im;
                        writeln!(
                            file,
                            "Gr{state}_re_{port}_{input_port} {output_reference} s{port} x{state}_re_a{input_port} 0 {real_gain}"
                        )?;
                        writeln!(
                            file,
                            "Gr{state}_im_{port}_{input_port} {output_reference} s{port} x{state}_im_a{input_port} 0 {imaginary_gain}"
                        )?;
                    }
                }
            }

            writeln!(file, "*")?;
            writeln!(file, "* State networks driven by port {port}")?;
            for (pole_index, pole) in self.poles.iter().enumerate() {
                let state = pole_index + 1;
                if pole.im == 0.0 {
                    writeln!(file, "Cx{state}_a{port} x{state}_a{port} 0 1.0")?;
                    writeln!(
                        file,
                        "Gx{state}_a{port} 0 x{state}_a{port} p{port} {output_reference} {voltage_wave_gain}"
                    )?;
                    writeln!(
                        file,
                        "Fx{state}_a{port} 0 x{state}_a{port} V{port} {current_wave_gain}"
                    )?;
                    writeln!(
                        file,
                        "Rp{state}_a{port} 0 x{state}_a{port} {}",
                        -1.0 / pole.re
                    )?;
                } else {
                    writeln!(file, "Cx{state}_re_a{port} x{state}_re_a{port} 0 1.0")?;
                    writeln!(
                        file,
                        "Gx{state}_re_a{port} 0 x{state}_re_a{port} p{port} {output_reference} {}",
                        2.0 * voltage_wave_gain
                    )?;
                    writeln!(
                        file,
                        "Fx{state}_re_a{port} 0 x{state}_re_a{port} V{port} {}",
                        2.0 * current_wave_gain
                    )?;
                    writeln!(
                        file,
                        "Rp{state}_re_re_a{port} 0 x{state}_re_a{port} {}",
                        -1.0 / pole.re
                    )?;
                    writeln!(
                        file,
                        "Gp{state}_re_im_a{port} 0 x{state}_re_a{port} x{state}_im_a{port} 0 {}",
                        pole.im
                    )?;
                    writeln!(file, "Cx{state}_im_a{port} x{state}_im_a{port} 0 1.0")?;
                    writeln!(
                        file,
                        "Gp{state}_im_re_a{port} 0 x{state}_im_a{port} x{state}_re_a{port} 0 {}",
                        -pole.im
                    )?;
                    writeln!(
                        file,
                        "Rp{state}_im_im_a{port} 0 x{state}_im_a{port} {}",
                        -1.0 / pole.re
                    )?;
                }
            }
            if build_proportional {
                writeln!(file, "*")?;
                writeln!(file, "* Network with derivative of input a_{port}")?;
                writeln!(file, "Le{port} e{port} 0 1.0")?;
                writeln!(
                    file,
                    "Ge{port} 0 e{port} p{port} {output_reference} {voltage_wave_gain}"
                )?;
                writeln!(file, "Fe{port} 0 e{port} V{port} {current_wave_gain}")?;
            }
        }
        writeln!(file, ".ENDS {model_name}")?;
        Ok(())
    }

    fn validate_model_state(&self) -> Result<()> {
        let responses = self.network.ports() * self.network.ports();
        if self.poles.is_empty()
            || self.residues.dim() != (responses, self.poles.len())
            || self.constant_coefficients.len() != responses
            || self.proportional_coefficients.len() != responses
        {
            return Err(Error::Unsupported(
                "the vector model has not been fitted".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_ports(&self, output: usize, input: usize) -> Result<()> {
        if output >= self.network.ports() {
            return Err(Error::InvalidPort {
                port: output,
                ports: self.network.ports(),
            });
        }
        if input >= self.network.ports() {
            return Err(Error::InvalidPort {
                port: input,
                ports: self.network.ports(),
            });
        }
        Ok(())
    }

    fn enforce_dc_sample(&mut self) -> Result<()> {
        if self.network.frequency.values_hz()[0] != 0.0 {
            return Ok(());
        }
        let zero = ndarray::array![0.0];
        for output in 0..self.network.ports() {
            for input in 0..self.network.ports() {
                let response = output * self.network.ports() + input;
                let fitted = self.model_response(output, input, &zero)?[0];
                self.constant_coefficients[response] +=
                    self.network.s[(0, output, input)].re - fitted.re;
            }
        }
        Ok(())
    }
}

fn spice_reference_node(port: usize, create_reference_pins: bool) -> String {
    if create_reference_pins {
        format!("p{}_ref", port + 1)
    } else {
        "0".to_owned()
    }
}

fn project_component(value: Complex64, component: Component) -> f64 {
    match component {
        Component::Decibels => 20.0 * value.norm().log10(),
        Component::Decibels10 => 10.0 * value.norm().log10(),
        Component::Magnitude => value.norm(),
        Component::PhaseDegrees => value.arg().to_degrees(),
        Component::Real => value.re,
        Component::Imaginary => value.im,
        Component::Vswr => (1.0 + value.norm()) / (1.0 - value.norm()),
    }
}

fn vector_component_label(component: Component) -> &'static str {
    match component {
        Component::Decibels => "Magnitude (dB)",
        Component::Decibels10 => "Magnitude (dB10)",
        Component::Magnitude => "Magnitude",
        Component::PhaseDegrees => "Phase (degrees)",
        Component::Real => "Real",
        Component::Imaginary => "Imaginary",
        Component::Vswr => "VSWR",
    }
}

fn largest_singular_value(scattering: &Array3<Complex64>, point: usize) -> f64 {
    let ports = scattering.dim().1;
    let mut vector = vec![Complex64::new(1.0 / (ports as f64).sqrt(), 0.0); ports];
    let mut singular = 0.0;
    for _ in 0..32 {
        let transformed = (0..ports)
            .map(|row| {
                (0..ports)
                    .map(|column| scattering[(point, row, column)] * vector[column])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        let adjoint = (0..ports)
            .map(|column| {
                (0..ports)
                    .map(|row| scattering[(point, row, column)].conj() * transformed[row])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        let norm = adjoint
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        for (value, next) in vector.iter_mut().zip(adjoint) {
            *value = next / norm;
        }
        singular = transformed
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
    }
    singular
}

fn npy_error(error: impl std::fmt::Display) -> Error {
    Error::Io(std::io::Error::other(error.to_string()))
}

fn relocate_poles(
    network: &Network,
    frequencies_hz: &Array1<f64>,
    normalization: f64,
    poles: &[Complex64],
) -> Result<Vec<Complex64>> {
    let basis_count = coefficient_count(poles) - 2;
    let numerator_count = basis_count + 2;
    let responses = network.ports() * network.ports();
    let denominator_offset = responses * numerator_count;
    let columns = denominator_offset + basis_count + 1;
    let mut design = Vec::with_capacity(2 * responses * frequencies_hz.len() + 1);
    let mut right = Vec::with_capacity(design.capacity());
    let mut basis_sums = vec![0.0; basis_count];
    for point in 0..frequencies_hz.len() {
        let s = Complex64::new(
            0.0,
            std::f64::consts::TAU * frequencies_hz[point] / normalization,
        );
        let basis = rational_basis(s, poles);
        for (sum, value) in basis_sums.iter_mut().zip(&basis) {
            *sum += value.re;
        }
        for output in 0..network.ports() {
            for input in 0..network.ports() {
                let response = output * network.ports() + input;
                let measured = network.s[(point, output, input)];
                let numerator_offset = response * numerator_count;
                let mut complex_row = vec![Complex64::new(0.0, 0.0); columns];
                for (basis_index, value) in basis.iter().copied().enumerate() {
                    complex_row[numerator_offset + basis_index] = value;
                    complex_row[denominator_offset + basis_index] = -measured * value;
                }
                complex_row[numerator_offset + basis_count] = Complex64::new(1.0, 0.0);
                complex_row[numerator_offset + basis_count + 1] = s;
                complex_row[columns - 1] = -measured;
                design.push(complex_row.iter().map(|value| value.re).collect());
                right.push(0.0);
                design.push(complex_row.iter().map(|value| value.im).collect());
                right.push(0.0);
            }
        }
    }
    let mut normalization_row = vec![0.0; columns];
    for (basis_index, sum) in basis_sums.into_iter().enumerate() {
        normalization_row[denominator_offset + basis_index] = sum;
    }
    normalization_row[columns - 1] = frequencies_hz.len() as f64;
    design.push(normalization_row);
    right.push(frequencies_hz.len() as f64);
    let solution = solve_least_squares(&design, &right)?;
    let denominator = &solution[denominator_offset..denominator_offset + basis_count];
    let constant = solution[columns - 1];
    let divisor = if constant.abs() < 1.0e-8 {
        if constant.is_sign_negative() {
            -1.0e-8
        } else {
            1.0e-8
        }
    } else {
        constant
    };
    let mut state = faer::Mat::<Complex64>::zeros(basis_count, basis_count);
    let mut row = 0;
    for pole in poles {
        if pole.im == 0.0 {
            state[(row, row)] = *pole;
            for column in 0..basis_count {
                state[(row, column)] -= denominator[column] / divisor;
            }
            row += 1;
        } else {
            state[(row, row)] = Complex64::new(pole.re, 0.0);
            state[(row, row + 1)] = Complex64::new(pole.im, 0.0);
            state[(row + 1, row)] = Complex64::new(-pole.im, 0.0);
            state[(row + 1, row + 1)] = Complex64::new(pole.re, 0.0);
            for column in 0..basis_count {
                state[(row, column)] -= 2.0 * denominator[column] / divisor;
            }
            row += 2;
        }
    }
    let decomposition = state.eigen().map_err(|error| {
        Error::Unsupported(format!("pole eigendecomposition failed: {error:?}"))
    })?;
    let eigenvalues = decomposition.S().column_vector();
    let imaginary_tolerance = 1.0e-10;
    let mut relocated = (0..basis_count)
        .filter_map(|index| {
            let value = eigenvalues[index];
            (value.im >= -imaginary_tolerance).then(|| {
                Complex64::new(
                    -value.re.abs(),
                    if value.im.abs() <= imaginary_tolerance {
                        0.0
                    } else {
                        value.im.abs()
                    },
                )
            })
        })
        .collect::<Vec<_>>();
    relocated.sort_by(|left, right| {
        left.im
            .total_cmp(&right.im)
            .then_with(|| left.re.total_cmp(&right.re))
    });
    if coefficient_count(&relocated) - 2 != basis_count {
        return Err(Error::Unsupported(
            "pole relocation did not preserve model order".to_owned(),
        ));
    }
    Ok(relocated)
}

fn rational_basis(s: Complex64, poles: &[Complex64]) -> Vec<Complex64> {
    let mut basis = Vec::with_capacity(coefficient_count(poles) - 2);
    for pole in poles {
        if pole.im == 0.0 {
            basis.push(Complex64::new(1.0, 0.0) / (s - pole));
        } else {
            let positive = Complex64::new(1.0, 0.0) / (s - pole);
            let negative = Complex64::new(1.0, 0.0) / (s - pole.conj());
            basis.push(positive + negative);
            basis.push(Complex64::new(0.0, 1.0) * (positive - negative));
        }
    }
    basis
}

fn pole_set_change(previous: &[Complex64], current: &[Complex64]) -> f64 {
    if previous.len() != current.len() {
        return f64::INFINITY;
    }
    previous
        .iter()
        .zip(current)
        .map(|(left, right)| (*left - *right).norm() / left.norm().max(1.0))
        .fold(0.0, f64::max)
}

fn coefficient_count(poles: &[Complex64]) -> usize {
    poles
        .iter()
        .map(|pole| if pole.im == 0.0 { 1 } else { 2 })
        .sum::<usize>()
        + 2
}

fn design_matrix(
    frequencies_hz: &Array1<f64>,
    normalization: f64,
    poles: &[Complex64],
) -> Vec<Vec<f64>> {
    let columns = coefficient_count(poles);
    let mut design = vec![vec![0.0; columns]; 2 * frequencies_hz.len()];
    for (point, frequency) in frequencies_hz.iter().enumerate() {
        let s = Complex64::new(0.0, std::f64::consts::TAU * *frequency / normalization);
        let mut column = 0;
        for pole in poles {
            if pole.im == 0.0 {
                let basis = Complex64::new(1.0, 0.0) / (s - pole);
                design[2 * point][column] = basis.re;
                design[2 * point + 1][column] = basis.im;
                column += 1;
            } else {
                let positive = Complex64::new(1.0, 0.0) / (s - pole);
                let negative = Complex64::new(1.0, 0.0) / (s - pole.conj());
                let real_residue_basis = positive + negative;
                let imaginary_residue_basis = Complex64::new(0.0, 1.0) * (positive - negative);
                design[2 * point][column] = real_residue_basis.re;
                design[2 * point + 1][column] = real_residue_basis.im;
                design[2 * point][column + 1] = imaginary_residue_basis.re;
                design[2 * point + 1][column + 1] = imaginary_residue_basis.im;
                column += 2;
            }
        }
        design[2 * point][column] = 1.0;
        design[2 * point + 1][column] = 0.0;
        design[2 * point][column + 1] = 0.0;
        design[2 * point + 1][column + 1] = s.im;
    }
    design
}

fn solve_least_squares(design: &[Vec<f64>], right: &[f64]) -> Result<Vec<f64>> {
    let columns = design[0].len();
    let mut normal = vec![vec![0.0; columns]; columns];
    let mut projected = vec![0.0; columns];
    for (row, value) in design.iter().zip(right.iter()) {
        for column in 0..columns {
            projected[column] += row[column] * value;
            for other in 0..columns {
                normal[column][other] += row[column] * row[other];
            }
        }
    }
    let diagonal_scale = (0..columns)
        .map(|index| normal[index][index])
        .fold(0.0, f64::max)
        .max(1.0);
    for (index, row) in normal.iter_mut().enumerate() {
        row[index] += diagonal_scale * 1.0e-14;
    }
    solve_linear_system(normal, projected).ok_or_else(|| {
        Error::Unsupported("vector fitting least-squares system is singular".to_owned())
    })
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut right: Vec<f64>) -> Option<Vec<f64>> {
    let dimension = right.len();
    for pivot in 0..dimension {
        let best = (pivot..dimension).max_by(|left, right_index| {
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
        for row in pivot + 1..dimension {
            let multiplier = matrix[row][pivot] / pivot_row[pivot];
            for (value, pivot_value) in matrix[row][pivot..]
                .iter_mut()
                .zip(pivot_row[pivot..].iter())
            {
                *value -= multiplier * pivot_value;
            }
            right[row] -= multiplier * right[pivot];
        }
    }
    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let tail = matrix[row][row + 1..]
            .iter()
            .zip(solution[row + 1..].iter())
            .map(|(coefficient, value)| coefficient * value)
            .sum::<f64>();
        solution[row] = (right[row] - tail) / matrix[row][row];
    }
    Some(solution)
}

fn linear_space(start: f64, stop: f64, points: usize) -> Vec<f64> {
    match points {
        0 => Vec::new(),
        1 => vec![start],
        _ => (0..points)
            .map(|index| {
                if index + 1 == points {
                    stop
                } else {
                    start + (stop - start) * index as f64 / (points - 1) as f64
                }
            })
            .collect(),
    }
}
