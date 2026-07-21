use std::sync::{Mutex, OnceLock};

use ndarray::{Array1, Array2, Array3, ArrayD, Axis, IxDyn};
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use realfft::RealFftPlanner;
use rustfft::FftPlanner;

use crate::constants::{
    LOG_OF_NEGATIVE, MINIMUM_EIGENVALUE, MINIMUM_EIGENVALUE_RATIO, NUMERICAL_INFINITY,
};
use crate::{Error, Result};

static RANDOM_GENERATOR: OnceLock<Mutex<StdRng>> = OnceLock::new();

/// Port of `skrf.mathFunctions.complex_2_magnitude`.
pub fn complex_magnitude(value: Complex64) -> f64 {
    value.norm()
}

/// Port of `skrf.mathFunctions.complex_2_db`.
pub fn complex_to_db(value: Complex64) -> f64 {
    magnitude_to_db(value.norm(), true)
}

/// Port of `skrf.mathFunctions.complex_2_db10`.
pub fn complex_to_db10(value: Complex64) -> f64 {
    magnitude_to_db10(value.norm(), true)
}

/// Port of `skrf.mathFunctions.complex_2_radian`.
pub fn complex_angle_radians(value: Complex64) -> f64 {
    value.arg()
}

/// Port of `skrf.mathFunctions.complex_2_degree`.
pub fn complex_angle_degrees(value: Complex64) -> f64 {
    radians_to_degrees(value.arg())
}

/// Port of `skrf.mathFunctions.complex_2_quadrature`.
pub fn complex_quadrature(value: Complex64) -> (f64, f64) {
    let magnitude = value.norm();
    (magnitude, value.arg() * magnitude)
}

/// Port of `skrf.mathFunctions.complex_2_reim`.
pub fn complex_real_imaginary(value: Complex64) -> (f64, f64) {
    (value.re, value.im)
}

/// Port of `skrf.mathFunctions.complex_components`.
pub fn complex_components(value: Complex64) -> (f64, f64, f64, f64, f64) {
    let (magnitude, arc_length) = complex_quadrature(value);
    (
        value.re,
        value.im,
        complex_angle_degrees(value),
        magnitude,
        arc_length,
    )
}

/// Port of `skrf.mathFunctions.magnitude_2_db`.
pub fn magnitude_to_db(value: f64, replace_nan: bool) -> f64 {
    replace_logarithm_nan(20.0 * value.log10(), replace_nan)
}

/// Port of `skrf.mathFunctions.mag_2_db10`.
pub fn magnitude_to_db10(value: f64, replace_nan: bool) -> f64 {
    replace_logarithm_nan(10.0 * value.log10(), replace_nan)
}

/// Port of `skrf.mathFunctions.db_2_magnitude`.
pub fn db_to_magnitude(value: f64) -> f64 {
    10.0_f64.powf(value / 20.0)
}

/// Port of `skrf.mathFunctions.db10_2_mag` for complex dB values.
pub fn db10_to_complex_magnitude(value: Complex64) -> Complex64 {
    (value * (std::f64::consts::LN_10 / 10.0)).exp()
}

/// Port of `skrf.mathFunctions.db10_2_mag` for real dB values.
pub fn db10_to_magnitude(value: f64) -> f64 {
    10.0_f64.powf(value / 10.0)
}

/// Port of `skrf.mathFunctions.magdeg_2_reim`.
pub fn magnitude_degrees_to_complex(magnitude: f64, degrees: f64) -> Complex64 {
    Complex64::from_polar(magnitude, degrees_to_radians(degrees))
}

/// Port of `skrf.mathFunctions.dbdeg_2_reim`.
pub fn db_degrees_to_complex(db: f64, degrees: f64) -> Complex64 {
    magnitude_degrees_to_complex(db_to_magnitude(db), degrees)
}

/// Port of `skrf.mathFunctions.db_2_np`.
pub fn db_to_nepers(db: f64) -> f64 {
    std::f64::consts::LN_10 / 20.0 * db
}

/// Port of `skrf.mathFunctions.np_2_db`.
pub fn nepers_to_db(nepers: f64) -> f64 {
    20.0 / std::f64::consts::LN_10 * nepers
}

/// Port of `skrf.mathFunctions.radian_2_degree`.
pub fn radians_to_degrees(radians: f64) -> f64 {
    radians * 180.0 / std::f64::consts::PI
}

/// Port of `skrf.mathFunctions.degree_2_radian`.
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

/// Port of `skrf.mathFunctions.feet_2_meter`.
pub fn feet_to_meters(feet: f64) -> f64 {
    0.3048 * feet
}

/// Port of `skrf.mathFunctions.meter_2_feet`.
pub fn meters_to_feet(meters: f64) -> f64 {
    3.28084 * meters
}

/// Port of `skrf.mathFunctions.db_per_100feet_2_db_per_100meter`.
pub fn db_per_100_feet_to_db_per_100_meters(value: f64) -> f64 {
    value * 100.0 / feet_to_meters(100.0)
}

/// Port of `skrf.mathFunctions.inf_to_num`.
pub fn infinity_to_number(value: f64) -> f64 {
    if value == f64::INFINITY {
        NUMERICAL_INFINITY
    } else if value == f64::NEG_INFINITY {
        -NUMERICAL_INFINITY
    } else {
        value
    }
}

pub fn infinities_to_numbers(values: &Array1<f64>) -> Array1<f64> {
    values.mapv(infinity_to_number)
}

/// Port of `skrf.mathFunctions.cross_ratio`.
pub fn cross_ratio(a: Complex64, b: Complex64, c: Complex64, d: Complex64) -> Complex64 {
    (a - b) * (c - d) / ((a - d) * (c - b))
}

/// Port of `skrf.mathFunctions.unwrap_rad` for a one-dimensional phase trace.
pub fn unwrap_radians(phase: &Array1<f64>) -> Array1<f64> {
    if phase.is_empty() {
        return phase.clone();
    }
    let mut result = phase.clone();
    let mut offset = 0.0;
    for index in 1..phase.len() {
        let difference = phase[index] - phase[index - 1];
        if difference > std::f64::consts::PI {
            offset -= 2.0 * std::f64::consts::PI;
        } else if difference < -std::f64::consts::PI {
            offset += 2.0 * std::f64::consts::PI;
        }
        result[index] += offset;
    }
    result
}

/// Port of `skrf.mathFunctions.sqrt_known_sign` for scalar complex values.
pub fn sqrt_known_sign(value_squared: Complex64, approximation: Complex64) -> Complex64 {
    let root = value_squared.sqrt();
    if root.arg().signum() == approximation.arg().signum() {
        root
    } else {
        root.conj()
    }
}

/// Port of `skrf.mathFunctions.find_correct_sign` for scalar complex values.
pub fn find_correct_sign(
    first: Complex64,
    second: Complex64,
    approximation: Complex64,
) -> Complex64 {
    if first.arg().signum() == approximation.arg().signum() {
        first
    } else {
        second
    }
}

/// Port of `skrf.mathFunctions.find_closest` for scalar complex values.
pub fn find_closest(first: Complex64, second: Complex64, approximation: Complex64) -> Complex64 {
    if (first - approximation).norm() < (second - approximation).norm() {
        first
    } else {
        second
    }
}

/// Port of `skrf.mathFunctions.sqrt_phase_unwrap`.
pub fn sqrt_phase_unwrap(values: &Array1<Complex64>) -> Array1<Complex64> {
    let phase = unwrap_radians(&values.mapv(|value| value.arg()));
    Array1::from_shape_fn(values.len(), |index| {
        Complex64::from_polar(values[index].norm().sqrt(), phase[index] / 2.0)
    })
}

/// Port of `skrf.mathFunctions.dirac_delta`.
pub fn dirac_delta(value: f64) -> f64 {
    if value == 0.0 { 1.0 } else { 0.0 }
}

/// Port of `skrf.mathFunctions.neuman`.
pub fn neumann_number(value: f64) -> f64 {
    2.0 - dirac_delta(value)
}

/// Null-space basis computed from the full singular-value decomposition.
///
/// Origin: `skrf.mathFunctions.null`.
pub fn null_space(matrix: &Array2<Complex64>, epsilon: f64) -> Result<Array2<Complex64>> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(Error::Unsupported(
            "null-space epsilon must be finite and non-negative".to_owned(),
        ));
    }
    let faer_matrix =
        faer::Mat::<Complex64>::from_fn(matrix.nrows(), matrix.ncols(), |row, column| {
            matrix[(row, column)]
        });
    let decomposition = faer_matrix
        .svd()
        .map_err(|error| Error::Unsupported(format!("SVD failed: {error:?}")))?;
    let singular = decomposition.S().column_vector();
    let indices = (0..singular.nrows())
        .filter(|index| singular[*index].re.abs() <= epsilon)
        .collect::<Vec<_>>();
    let vectors = decomposition.V();
    Ok(Array2::from_shape_fn(
        (matrix.ncols(), indices.len()),
        |(row, column)| vectors[(row, indices[column])],
    ))
}

/// Applies a scalar real function independently to the real and imaginary parts.
///
/// Origin: `skrf.mathFunctions.complexify` for the common first-argument form.
pub fn complexify(value: Complex64, function: impl Fn(f64) -> f64) -> Complex64 {
    Complex64::new(function(value.re), function(value.im))
}

/// Port of `skrf.mathFunctions.complex2Scalar`.
pub fn complex_to_scalar(values: &[Complex64]) -> Array1<f64> {
    Array1::from_iter(values.iter().flat_map(|value| [value.re, value.im]))
}

/// Port of `skrf.mathFunctions.scalar2Complex`.
pub fn scalar_to_complex(values: &[f64]) -> Result<Array1<Complex64>> {
    if values.len() % 2 != 0 {
        return Err(Error::IncompatibleShape(
            "serialized complex data must contain real/imaginary pairs".to_owned(),
        ));
    }
    Ok(Array1::from_iter(
        values
            .chunks_exact(2)
            .map(|pair| Complex64::new(pair[0], pair[1])),
    ))
}

/// Port of `skrf.mathFunctions.flatten_c_mat` using NumPy's default Fortran order.
pub fn flatten_complex_matrix(matrix: &Array2<Complex64>) -> Array1<f64> {
    complex_to_scalar(
        &(0..matrix.ncols())
            .flat_map(|column| (0..matrix.nrows()).map(move |row| matrix[(row, column)]))
            .collect::<Vec<_>>(),
    )
}

pub fn is_square(matrix: &Array2<Complex64>) -> bool {
    matrix.nrows() == matrix.ncols()
}

/// Port of `skrf.mathFunctions.get_Hermitian_transpose`.
pub fn hermitian_transpose(matrix: &Array2<Complex64>) -> Array2<Complex64> {
    Array2::from_shape_fn((matrix.ncols(), matrix.nrows()), |(row, column)| {
        matrix[(column, row)].conj()
    })
}

/// Port of `skrf.mathFunctions.is_symmetric`.
pub fn is_symmetric(matrix: &Array2<Complex64>, tolerance: f64) -> bool {
    is_square(matrix)
        && (0..matrix.nrows()).all(|row| {
            (0..matrix.ncols())
                .all(|column| (matrix[(row, column)] - matrix[(column, row)]).norm() <= tolerance)
        })
}

/// Port of `skrf.mathFunctions.is_Hermitian`.
pub fn is_hermitian(matrix: &Array2<Complex64>, tolerance: f64) -> bool {
    is_square(matrix)
        && (0..matrix.nrows()).all(|row| {
            (0..matrix.ncols()).all(|column| {
                (matrix[(row, column)] - matrix[(column, row)].conj()).norm() <= tolerance
            })
        })
}

/// Port of `skrf.mathFunctions.is_unitary`.
pub fn is_unitary(matrix: &Array2<Complex64>, tolerance: f64) -> bool {
    if !is_square(matrix) {
        return false;
    }
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            let product = (0..matrix.nrows())
                .map(|inner| matrix[(inner, row)].conj() * matrix[(inner, column)])
                .sum::<Complex64>();
            let expected = if row == column { 1.0 } else { 0.0 };
            if (product - expected).norm() > tolerance {
                return false;
            }
        }
    }
    true
}

/// Port of `skrf.mathFunctions.is_positive_definite`.
pub fn is_positive_definite(matrix: &Array2<Complex64>, tolerance: f64) -> bool {
    if !is_hermitian(matrix, tolerance) {
        return false;
    }
    let size = matrix.nrows();
    let mut lower = Array2::<Complex64>::zeros((size, size));
    for row in 0..size {
        for column in 0..=row {
            let correction = (0..column)
                .map(|inner| lower[(row, inner)] * lower[(column, inner)].conj())
                .sum::<Complex64>();
            let residual = matrix[(row, column)] - correction;
            if row == column {
                if residual.im.abs() > tolerance || residual.re <= tolerance {
                    return false;
                }
                lower[(row, column)] = Complex64::new(residual.re.sqrt(), 0.0);
            } else {
                lower[(row, column)] = residual / lower[(column, column)];
            }
        }
    }
    true
}

/// Port of `skrf.mathFunctions.is_positive_semidefinite` using an LDL-H
/// factorization, avoiding a dependency on a general eigensolver.
pub fn is_positive_semidefinite(matrix: &Array2<Complex64>, tolerance: f64) -> bool {
    if !is_hermitian(matrix, tolerance) {
        return false;
    }
    let size = matrix.nrows();
    let mut lower = Array2::<Complex64>::zeros((size, size));
    let mut diagonal = vec![0.0; size];
    for row in 0..size {
        lower[(row, row)] = Complex64::new(1.0, 0.0);
    }
    for column in 0..size {
        let diagonal_residual = matrix[(column, column)]
            - (0..column)
                .map(|inner| {
                    lower[(column, inner)] * lower[(column, inner)].conj() * diagonal[inner]
                })
                .sum::<Complex64>();
        if diagonal_residual.im.abs() > tolerance || diagonal_residual.re < -tolerance {
            return false;
        }
        diagonal[column] = diagonal_residual.re.max(0.0);
        for row in column + 1..size {
            let residual = matrix[(row, column)]
                - (0..column)
                    .map(|inner| {
                        lower[(row, inner)] * lower[(column, inner)].conj() * diagonal[inner]
                    })
                    .sum::<Complex64>();
            if diagonal[column] <= tolerance {
                if residual.norm() > tolerance {
                    return false;
                }
            } else {
                lower[(row, column)] = residual / diagonal[column];
            }
        }
    }
    true
}

/// Port of `skrf.mathFunctions.rsolve` for batches of square matrices.
pub fn right_solve(
    coefficients: &Array3<Complex64>,
    right_hand_side: &Array3<Complex64>,
) -> Result<Array3<Complex64>> {
    let (batches, rows, columns) = coefficients.dim();
    if rows != columns || right_hand_side.dim() != (batches, rows, columns) {
        return Err(Error::IncompatibleShape(format!(
            "right solve requires matching batches of square matrices, got {:?} and {:?}",
            coefficients.dim(),
            right_hand_side.dim()
        )));
    }

    let mut result = Array3::zeros((batches, rows, columns));
    for batch in 0..batches {
        let transposed_coefficients = Array2::from_shape_fn((rows, columns), |(row, column)| {
            coefficients[(batch, column, row)]
        });
        for result_row in 0..rows {
            let right =
                Array1::from_shape_fn(rows, |index| right_hand_side[(batch, result_row, index)]);
            let solution = solve_linear_system(&transposed_coefficients, &right)?;
            for column in 0..columns {
                result[(batch, result_row, column)] = solution[column];
            }
        }
    }
    Ok(result)
}

/// Solves `A @ X = B` for batches of square matrices. This is the direct
/// counterpart to scikit-rf's NumPy `solve` calls used by parameter conversion.
pub fn left_solve(
    coefficients: &Array3<Complex64>,
    right_hand_side: &Array3<Complex64>,
) -> Result<Array3<Complex64>> {
    let (batches, rows, columns) = coefficients.dim();
    if rows != columns || right_hand_side.dim() != (batches, rows, columns) {
        return Err(Error::IncompatibleShape(format!(
            "left solve requires matching batches of square matrices, got {:?} and {:?}",
            coefficients.dim(),
            right_hand_side.dim()
        )));
    }

    let mut result = Array3::zeros((batches, rows, columns));
    for batch in 0..batches {
        let coefficient_matrix = Array2::from_shape_fn((rows, columns), |index| {
            coefficients[(batch, index.0, index.1)]
        });
        for result_column in 0..columns {
            let right =
                Array1::from_shape_fn(rows, |row| right_hand_side[(batch, row, result_column)]);
            let solution = solve_linear_system(&coefficient_matrix, &right)?;
            for row in 0..rows {
                result[(batch, row, result_column)] = solution[row];
            }
        }
    }
    Ok(result)
}

/// Rust-native equivalent of `set_rand_rng`, using a deterministic seed.
pub fn set_random_seed(seed: u64) {
    let generator = RANDOM_GENERATOR.get_or_init(|| Mutex::new(StdRng::seed_from_u64(seed)));
    *generator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = StdRng::seed_from_u64(seed);
}

/// Port of `skrf.mathFunctions.rand_c` for a two-dimensional result.
pub fn random_complex(rows: usize, columns: usize) -> Array2<Complex64> {
    let generator = RANDOM_GENERATOR.get_or_init(|| Mutex::new(StdRng::seed_from_u64(0)));
    let mut generator = generator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Array2::from_shape_fn((rows, columns), |_| {
        Complex64::new(
            1.0 - 2.0 * generator.random::<f64>(),
            1.0 - 2.0 * generator.random::<f64>(),
        )
    })
}

/// Random polar samples used by `skrf.media.Media.white_gaussian_polar`.
pub fn random_gaussian_polar(
    rows: usize,
    columns: usize,
    phase_standard_deviation: f64,
    magnitude_standard_deviation: f64,
) -> Result<Array2<Complex64>> {
    if !phase_standard_deviation.is_finite()
        || phase_standard_deviation < 0.0
        || !magnitude_standard_deviation.is_finite()
        || magnitude_standard_deviation < 0.0
    {
        return Err(Error::Unsupported(
            "Gaussian polar deviations must be finite and non-negative".to_owned(),
        ));
    }
    let phase = (phase_standard_deviation > 0.0)
        .then(|| Normal::new(0.0, phase_standard_deviation))
        .transpose()
        .map_err(|error| Error::Unsupported(format!("invalid phase distribution: {error}")))?;
    let magnitude = (magnitude_standard_deviation > 0.0)
        .then(|| Normal::new(0.0, magnitude_standard_deviation))
        .transpose()
        .map_err(|error| Error::Unsupported(format!("invalid magnitude distribution: {error}")))?;
    let generator = RANDOM_GENERATOR.get_or_init(|| Mutex::new(StdRng::seed_from_u64(0)));
    let mut generator = generator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Ok(Array2::from_shape_fn((rows, columns), |_| {
        Complex64::from_polar(
            magnitude
                .as_ref()
                .map_or(0.0, |distribution| distribution.sample(&mut *generator)),
            phase
                .as_ref()
                .map_or(0.0, |distribution| distribution.sample(&mut *generator)),
        )
    }))
}

/// Draws independent zero-mean Gaussian values with per-element deviations.
///
/// This supports `skrf.networkSet.NetworkSet.add_polar_noise` while sharing
/// the deterministic generator configured through [`set_random_seed`].
pub fn random_normal_like(standard_deviations: &Array3<f64>) -> Result<Array3<f64>> {
    if standard_deviations
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(Error::Unsupported(
            "normal deviations must be finite and non-negative".to_owned(),
        ));
    }
    let generator = RANDOM_GENERATOR.get_or_init(|| Mutex::new(StdRng::seed_from_u64(0)));
    let mut generator = generator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let values = standard_deviations
        .iter()
        .map(|deviation| {
            if *deviation == 0.0 {
                Ok(0.0)
            } else {
                Normal::new(0.0, *deviation)
                    .map(|distribution| distribution.sample(&mut *generator))
                    .map_err(|error| {
                        Error::Unsupported(format!("invalid normal distribution: {error}"))
                    })
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Array3::from_shape_vec(standard_deviations.raw_dim(), values)
        .map_err(|error| Error::IncompatibleShape(error.to_string()))
}

/// Converts a one-sided complex spectrum into a centered real time signal.
///
/// Origin: `skrf.mathFunctions.psd2TimeDomain`.
pub fn psd_to_time_domain(
    frequency: &Array1<f64>,
    spectrum: &Array1<Complex64>,
    window: crate::time::Window,
) -> Result<(Array1<f64>, Array1<f64>)> {
    if frequency.len() != spectrum.len() || frequency.len() < 2 {
        return Err(Error::IncompatibleShape(
            "PSD conversion requires matching frequency/spectrum arrays with at least two points"
                .to_owned(),
        ));
    }
    if frequency
        .windows(2)
        .into_iter()
        .any(|pair| pair[1] <= pair[0])
    {
        return Err(Error::InvalidFrequency(
            "PSD conversion requires increasing frequencies".to_owned(),
        ));
    }
    let window = crate::time::window_samples(&window, frequency.len())?;
    let windowed = Array1::from_shape_fn(spectrum.len(), |index| spectrum[index] * window[index]);
    let mut complete = windowed
        .iter()
        .skip(1)
        .rev()
        .map(|value| value.conj())
        .chain(windowed.iter().copied())
        .collect::<Vec<_>>();
    complete = ifft_shift_complex(&complete);
    let mut planner = FftPlanner::new();
    planner
        .plan_fft_inverse(complete.len())
        .process(&mut complete);
    let scale = complete.len() as f64;
    complete.iter_mut().for_each(|value| *value /= scale);
    complete = ifft_shift_complex(&complete);
    let period = 1.0 / (frequency[1] - frequency[0]).abs();
    let count = complete.len();
    let time = Array1::from_shape_fn(count, |index| {
        -period / 2.0 + period * index as f64 / (count - 1) as f64
    });
    Ok((
        time,
        Array1::from_iter(complete.into_iter().map(|value| value.re)),
    ))
}

/// Floater-Hormann barycentric rational interpolator.
///
/// Origin: `skrf.mathFunctions.rational_interp` for axis zero and complex values.
#[derive(Clone, Debug)]
pub struct RationalInterpolator {
    x: Array1<f64>,
    y: Array1<Complex64>,
    weights: Array1<f64>,
    epsilon: f64,
}

impl RationalInterpolator {
    pub fn new(
        x: &Array1<f64>,
        y: &Array1<Complex64>,
        degree: usize,
        epsilon: f64,
        assume_sorted: bool,
    ) -> Result<Self> {
        if x.len() != y.len() || x.len() <= degree || x.len() < 2 {
            return Err(Error::IncompatibleShape(
                "rational interpolation requires matching arrays and more points than its degree"
                    .to_owned(),
            ));
        }
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(Error::Unsupported(
                "rational interpolation epsilon must be finite and non-negative".to_owned(),
            ));
        }
        let mut pairs = x.iter().copied().zip(y.iter().copied()).collect::<Vec<_>>();
        if !assume_sorted {
            pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
        }
        if pairs.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
            return Err(Error::InvalidFrequency(
                "rational interpolation coordinates must be strictly increasing".to_owned(),
            ));
        }
        let x = Array1::from_iter(pairs.iter().map(|pair| pair.0));
        let y = Array1::from_iter(pairs.iter().map(|pair| pair.1));
        let count = x.len();
        let scale = (x[count / 2] - x[count / 2 - 1]).powi(degree as i32);
        let mut weights = Array1::zeros(count);
        for k in 0..count {
            for i in k.saturating_sub(degree)..(k + 1).min(count - degree) {
                let mut product = scale;
                for j in i..(i + degree + 1).min(count) {
                    if j != k {
                        product /= x[k] - x[j];
                    }
                }
                weights[k] += if i % 2 == 0 { product } else { -product };
            }
        }
        Ok(Self {
            x,
            y,
            weights,
            epsilon,
        })
    }

    pub fn evaluate(&self, targets: &Array1<f64>) -> Array1<Complex64> {
        Array1::from_shape_fn(targets.len(), |target_index| {
            let target = targets[target_index];
            if let Some(index) = self
                .x
                .iter()
                .position(|value| (*value - target).abs() < self.epsilon)
            {
                return self.y[index];
            }
            let mut numerator = Complex64::new(0.0, 0.0);
            let mut denominator = 0.0;
            for index in 0..self.x.len() {
                let weight = self.weights[index] / (target - self.x[index]);
                numerator += self.y[index] * weight;
                denominator += weight;
            }
            numerator / denominator
        })
    }
}

/// Axis-zero Floater-Hormann interpolation for arbitrary trailing dimensions.
///
/// This is the full array-shape counterpart of `skrf.mathFunctions.rational_interp`.
#[derive(Clone, Debug)]
pub struct RationalInterpolatorAxis0 {
    x: Array1<f64>,
    y: ArrayD<Complex64>,
    weights: Array1<f64>,
    epsilon: f64,
}

impl RationalInterpolatorAxis0 {
    pub fn new(
        x: &Array1<f64>,
        y: &ArrayD<Complex64>,
        degree: usize,
        epsilon: f64,
        assume_sorted: bool,
    ) -> Result<Self> {
        if y.ndim() == 0 || y.shape()[0] != x.len() || x.len() <= degree || x.len() < 2 {
            return Err(Error::IncompatibleShape(
                "axis-zero interpolation requires y.shape[0] == x.len() and enough points"
                    .to_owned(),
            ));
        }
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(Error::Unsupported(
                "rational interpolation epsilon must be finite and non-negative".to_owned(),
            ));
        }
        let mut order = (0..x.len()).collect::<Vec<_>>();
        if !assume_sorted {
            order.sort_by(|left, right| x[*left].total_cmp(&x[*right]));
        }
        let x = Array1::from_iter(order.iter().map(|index| x[*index]));
        if x.windows(2).into_iter().any(|pair| pair[0] >= pair[1]) {
            return Err(Error::InvalidFrequency(
                "rational interpolation coordinates must be strictly increasing".to_owned(),
            ));
        }
        let mut shape = y.shape().to_vec();
        shape[0] = x.len();
        let mut sorted_y = ArrayD::zeros(IxDyn(&shape));
        for (target, source) in order.iter().copied().enumerate() {
            sorted_y
                .index_axis_mut(Axis(0), target)
                .assign(&y.index_axis(Axis(0), source));
        }
        let weights = floater_hormann_weights(&x, degree);
        Ok(Self {
            x,
            y: sorted_y,
            weights,
            epsilon,
        })
    }

    pub fn evaluate(&self, targets: &Array1<f64>) -> ArrayD<Complex64> {
        let mut shape = self.y.shape().to_vec();
        shape[0] = targets.len();
        let mut output = ArrayD::zeros(IxDyn(&shape));
        for (target_index, target) in targets.iter().copied().enumerate() {
            let mut destination = output.index_axis_mut(Axis(0), target_index);
            if let Some(source) = self
                .x
                .iter()
                .position(|value| (*value - target).abs() < self.epsilon)
            {
                destination.assign(&self.y.index_axis(Axis(0), source));
                continue;
            }
            let mut denominator = 0.0;
            for source in 0..self.x.len() {
                let weight = self.weights[source] / (target - self.x[source]);
                denominator += weight;
                destination.zip_mut_with(&self.y.index_axis(Axis(0), source), |output, value| {
                    *output += *value * weight;
                });
            }
            destination.mapv_inplace(|value| value / denominator);
        }
        output
    }
}

fn floater_hormann_weights(x: &Array1<f64>, degree: usize) -> Array1<f64> {
    let count = x.len();
    let scale = (x[count / 2] - x[count / 2 - 1]).powi(degree as i32);
    let mut weights = Array1::zeros(count);
    for k in 0..count {
        for i in k.saturating_sub(degree)..(k + 1).min(count - degree) {
            let mut product = scale;
            for j in i..(i + degree + 1).min(count) {
                if j != k {
                    product /= x[k] - x[j];
                }
            }
            weights[k] += if i % 2 == 0 { product } else { -product };
        }
    }
    weights
}

/// Centered complex inverse FFT matching NumPy's shift convention.
///
/// Origin: `skrf.mathFunctions.ifft`.
pub fn inverse_fft_centered(values: &Array1<Complex64>) -> Array1<Complex64> {
    if values.is_empty() {
        return Array1::zeros(0);
    }
    let mut shifted = ifft_shift_complex(&values.to_vec());
    let mut planner = FftPlanner::new();
    planner
        .plan_fft_inverse(shifted.len())
        .process(&mut shifted);
    let scale = shifted.len() as f64;
    shifted.iter_mut().for_each(|value| *value /= scale);
    Array1::from(fft_shift_complex(&shifted))
}

/// Centered real inverse FFT matching NumPy's `irfft` normalization.
///
/// Origin: `skrf.mathFunctions.irfft`.
pub fn inverse_real_fft_centered(
    values: &Array1<Complex64>,
    output_length: Option<usize>,
) -> Result<Array1<f64>> {
    let length = output_length.unwrap_or_else(|| values.len().saturating_sub(1) * 2);
    if length == 0 {
        return Ok(Array1::zeros(0));
    }
    let required = length / 2 + 1;
    let mut spectrum = vec![Complex64::new(0.0, 0.0); required];
    for (target, source) in spectrum.iter_mut().zip(values.iter()) {
        *target = *source;
    }
    let mut output = vec![0.0; length];
    let mut planner = RealFftPlanner::<f64>::new();
    planner
        .plan_fft_inverse(length)
        .process(&mut spectrum, &mut output)
        .map_err(|error| Error::Unsupported(format!("inverse real FFT failed: {error}")))?;
    output.iter_mut().for_each(|value| *value /= length as f64);
    let split = length.div_ceil(2);
    Ok(Array1::from_iter(
        output[split..].iter().chain(&output[..split]).copied(),
    ))
}

/// Centered inverse FFT along axis zero for arbitrary trailing dimensions.
pub fn inverse_fft_centered_axis0(values: &ArrayD<Complex64>) -> Result<ArrayD<Complex64>> {
    if values.ndim() == 0 {
        return Err(Error::IncompatibleShape(
            "axis-zero FFT requires at least one dimension".to_owned(),
        ));
    }
    let length = values.shape()[0];
    if length == 0 {
        return Ok(values.clone());
    }
    let mut output = values.clone();
    let mut planner = FftPlanner::new();
    let transform = planner.plan_fft_inverse(length);
    for mut lane in output.lanes_mut(Axis(0)) {
        let mut transformed = ifft_shift_complex(&lane.iter().copied().collect::<Vec<_>>());
        transform.process(&mut transformed);
        transformed
            .iter_mut()
            .for_each(|value| *value /= length as f64);
        let transformed = fft_shift_complex(&transformed);
        for (destination, source) in lane.iter_mut().zip(transformed) {
            *destination = source;
        }
    }
    Ok(output)
}

/// Centered real inverse FFT along axis zero for arbitrary trailing dimensions.
pub fn inverse_real_fft_centered_axis0(
    values: &ArrayD<Complex64>,
    output_length: Option<usize>,
) -> Result<ArrayD<f64>> {
    if values.ndim() == 0 {
        return Err(Error::IncompatibleShape(
            "axis-zero inverse real FFT requires at least one dimension".to_owned(),
        ));
    }
    let length = output_length.unwrap_or_else(|| values.shape()[0].saturating_sub(1) * 2);
    let mut shape = values.shape().to_vec();
    shape[0] = length;
    let mut output = ArrayD::zeros(IxDyn(&shape));
    if length == 0 {
        return Ok(output);
    }
    let required = length / 2 + 1;
    let mut planner = RealFftPlanner::<f64>::new();
    let transform = planner.plan_fft_inverse(length);
    for (input, mut destination) in values
        .lanes(Axis(0))
        .into_iter()
        .zip(output.lanes_mut(Axis(0)))
    {
        let mut spectrum = vec![Complex64::new(0.0, 0.0); required];
        for (target, source) in spectrum.iter_mut().zip(input.iter()) {
            *target = *source;
        }
        let mut transformed = vec![0.0; length];
        transform
            .process(&mut spectrum, &mut transformed)
            .map_err(|error| Error::Unsupported(format!("inverse real FFT failed: {error}")))?;
        transformed
            .iter_mut()
            .for_each(|value| *value /= length as f64);
        let split = length.div_ceil(2);
        for (target, source) in destination
            .iter_mut()
            .zip(transformed[split..].iter().chain(&transformed[..split]))
        {
            *target = *source;
        }
    }
    Ok(output)
}

/// Nudges small eigenvalues to avoid singular matrix equations.
///
/// Origin: `skrf.mathFunctions.nudge_eig`.
pub fn nudge_eigenvalues(
    matrices: &Array3<Complex64>,
    condition: Option<f64>,
    minimum: Option<f64>,
) -> Result<Array3<Complex64>> {
    let (batches, rows, columns) = matrices.dim();
    if rows != columns {
        return Err(Error::IncompatibleShape(format!(
            "eigenvalue nudging requires square matrices, got {rows}x{columns}"
        )));
    }
    let condition = condition.unwrap_or(MINIMUM_EIGENVALUE_RATIO);
    let minimum = minimum.unwrap_or(MINIMUM_EIGENVALUE);
    if !condition.is_finite() || condition < 0.0 || !minimum.is_finite() || minimum < 0.0 {
        return Err(Error::Unsupported(
            "eigenvalue thresholds must be finite and non-negative".to_owned(),
        ));
    }
    let mut output = matrices.clone();
    for batch in 0..batches {
        let matrix = faer::Mat::<Complex64>::from_fn(rows, columns, |row, column| {
            matrices[(batch, row, column)]
        });
        let decomposition = matrix
            .eigen()
            .map_err(|error| Error::Unsupported(format!("eigendecomposition failed: {error:?}")))?;
        let eigenvalues = decomposition.S().column_vector();
        let maximum = (0..rows)
            .map(|index| eigenvalues[index].norm())
            .fold(0.0_f64, f64::max);
        let threshold = (condition * maximum).max(minimum);
        if (0..rows).all(|index| eigenvalues[index].norm() >= threshold) {
            continue;
        }
        let eigenvectors = decomposition.U();
        let left = Array3::from_shape_fn((1, rows, columns), |(_, row, column)| {
            eigenvectors[(row, column)]
        });
        let scaled = Array3::from_shape_fn((1, rows, columns), |(_, row, column)| {
            let eigenvalue = if eigenvalues[column].norm() < threshold {
                Complex64::new(threshold, 0.0)
            } else {
                eigenvalues[column]
            };
            eigenvectors[(row, column)] * eigenvalue
        });
        let reconstructed = right_solve(&left, &scaled)?;
        for row in 0..rows {
            for column in 0..columns {
                output[(batch, row, column)] = reconstructed[(0, row, column)];
            }
        }
    }
    Ok(output)
}

fn fft_shift_complex(values: &[Complex64]) -> Vec<Complex64> {
    let split = values.len().div_ceil(2);
    values[split..]
        .iter()
        .chain(&values[..split])
        .copied()
        .collect()
}

fn ifft_shift_complex(values: &[Complex64]) -> Vec<Complex64> {
    let split = values.len() / 2;
    values[split..]
        .iter()
        .chain(&values[..split])
        .copied()
        .collect()
}

fn replace_logarithm_nan(value: f64, replace_nan: bool) -> f64 {
    if replace_nan && value.is_nan() {
        LOG_OF_NEGATIVE
    } else {
        value
    }
}

fn solve_linear_system(
    coefficients: &Array2<Complex64>,
    right_hand_side: &Array1<Complex64>,
) -> Result<Array1<Complex64>> {
    let size = coefficients.nrows();
    let mut augmented = Array2::zeros((size, size + 1));
    for row in 0..size {
        for column in 0..size {
            augmented[(row, column)] = coefficients[(row, column)];
        }
        augmented[(row, size)] = right_hand_side[row];
    }

    for pivot_column in 0..size {
        let pivot_row = (pivot_column..size)
            .max_by(|left, right| {
                augmented[(*left, pivot_column)]
                    .norm_sqr()
                    .total_cmp(&augmented[(*right, pivot_column)].norm_sqr())
            })
            .ok_or_else(|| Error::Unsupported("cannot solve an empty matrix".to_owned()))?;
        if augmented[(pivot_row, pivot_column)].norm_sqr() <= f64::EPSILON {
            return Err(Error::Unsupported(
                "cannot solve a singular matrix".to_owned(),
            ));
        }
        if pivot_row != pivot_column {
            for column in pivot_column..=size {
                augmented.swap((pivot_row, column), (pivot_column, column));
            }
        }

        let pivot = augmented[(pivot_column, pivot_column)];
        for column in pivot_column..=size {
            augmented[(pivot_column, column)] /= pivot;
        }

        for row in 0..size {
            if row == pivot_column {
                continue;
            }
            let factor = augmented[(row, pivot_column)];
            for column in pivot_column..=size {
                let pivot_value = augmented[(pivot_column, column)];
                augmented[(row, column)] -= factor * pivot_value;
            }
        }
    }

    Ok(Array1::from_shape_fn(size, |row| augmented[(row, size)]))
}

/// Numerical seam for the functions originating in `skrf/mathFunctions.py`.
pub trait Interpolator {
    fn interpolate_real(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        target: &Array1<f64>,
    ) -> Result<Array1<f64>>;

    fn interpolate_complex(
        &self,
        x: &Array1<f64>,
        y: &Array1<Complex64>,
        target: &Array1<f64>,
    ) -> Result<Array1<Complex64>>;
}

pub trait SpecialFunctions {
    fn bessel_i(&self, order: f64, value: f64) -> Result<f64>;

    fn complete_elliptic_integral_first_kind(&self, parameter: f64) -> Result<f64>;

    fn bessel_j_zero(&self, order: usize, index: usize, derivative: bool) -> Result<f64>;
}

/// Return the one-based `index`th positive zero of integer-order `J_n` or its derivative.
///
/// This is the Rust numerical replacement for SciPy's `jn_zeros` and `jnp_zeros`, used by
/// `skrf/media/circularWaveguide.py`.
pub fn bessel_j_zero(order: usize, index: usize, derivative: bool) -> Result<f64> {
    if index == 0 {
        return Err(Error::Unsupported(
            "Bessel-zero indices are one-based".to_owned(),
        ));
    }
    let order = i32::try_from(order)
        .map_err(|_| Error::Unsupported("Bessel order is too large".to_owned()))?;
    let evaluate = |value: f64| {
        if derivative {
            if order == 0 {
                -libm::j1(value)
            } else {
                (libm::jn(order - 1, value) - libm::jn(order + 1, value)) / 2.0
            }
        } else {
            libm::jn(order, value)
        }
    };

    let step = std::f64::consts::PI / 32.0;
    let maximum = (index as f64 + f64::from(order) / 2.0 + 3.0) * std::f64::consts::PI;
    let mut left = f64::EPSILON.sqrt();
    let mut left_value = evaluate(left);
    let mut roots_found = 0;
    while left < maximum {
        let right = (left + step).min(maximum);
        let right_value = evaluate(right);
        if left_value.is_finite()
            && right_value.is_finite()
            && left_value.signum() != right_value.signum()
        {
            let mut lower = left;
            let mut upper = right;
            let mut lower_value = left_value;
            for _ in 0..80 {
                let middle = (lower + upper) / 2.0;
                let middle_value = evaluate(middle);
                if middle_value == 0.0 {
                    lower = middle;
                    upper = middle;
                    break;
                }
                if lower_value.signum() == middle_value.signum() {
                    lower = middle;
                    lower_value = middle_value;
                } else {
                    upper = middle;
                }
            }
            roots_found += 1;
            if roots_found == index {
                return Ok((lower + upper) / 2.0);
            }
        }
        left = right;
        left_value = right_value;
    }
    Err(Error::Unsupported(format!(
        "could not bracket Bessel zero {index} for order {order}"
    )))
}

/// Complete elliptic integral of the first kind `K(m)` for a real parameter.
///
/// Uses the arithmetic-geometric mean identity and replaces `scipy.special.ellipk`.
pub fn complete_elliptic_integral_first_kind(parameter: f64) -> Result<f64> {
    if !parameter.is_finite() || !(0.0..1.0).contains(&parameter) {
        return Err(Error::Unsupported(
            "the elliptic-integral parameter must satisfy 0 <= m < 1".to_owned(),
        ));
    }
    let mut arithmetic = 1.0;
    let mut geometric = (1.0 - parameter).sqrt();
    for _ in 0..64 {
        let next_arithmetic = (arithmetic + geometric) / 2.0;
        let next_geometric = (arithmetic * geometric).sqrt();
        if (next_arithmetic - next_geometric).abs() <= f64::EPSILON * next_arithmetic {
            return Ok(std::f64::consts::PI / (2.0 * next_arithmetic));
        }
        arithmetic = next_arithmetic;
        geometric = next_geometric;
    }
    Ok(std::f64::consts::PI / (2.0 * arithmetic))
}

/// Modified Bessel function `I_1(x)` evaluated by its entire power series.
pub fn modified_bessel_i1(value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(Error::Unsupported(
            "modified Bessel input must be finite".to_owned(),
        ));
    }
    let half = value / 2.0;
    let mut term = half;
    let mut sum = term;
    for index in 1..=256 {
        term *= half * half / (index as f64 * (index + 1) as f64);
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs().max(1.0) {
            return Ok(sum);
        }
    }
    Ok(sum)
}
