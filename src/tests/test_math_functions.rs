use approx::assert_relative_eq;
use ndarray::{Array1, Array2, Array3, array};
use num_complex::Complex64;
use rust_rf::constants::{LOG_OF_NEGATIVE, NUMERICAL_INFINITY};
use rust_rf::math::{
    RationalInterpolator, RationalInterpolatorAxis0, bessel_j_zero,
    complete_elliptic_integral_first_kind, complex_angle_degrees, complex_components,
    complex_magnitude, complex_quadrature, complex_real_imaginary, complex_to_db10,
    complex_to_scalar, complexify, db_degrees_to_complex, db_per_100_feet_to_db_per_100_meters,
    db_to_nepers, db10_to_complex_magnitude, dirac_delta, feet_to_meters, find_closest,
    find_correct_sign, flatten_complex_matrix, hermitian_transpose, infinities_to_numbers,
    infinity_to_number, inverse_fft_centered, inverse_fft_centered_axis0,
    inverse_real_fft_centered, inverse_real_fft_centered_axis0, is_hermitian, is_positive_definite,
    is_positive_semidefinite, is_symmetric, is_unitary, magnitude_degrees_to_complex,
    magnitude_to_db, magnitude_to_db10, meters_to_feet, modified_bessel_i1, nepers_to_db,
    neumann_number, nudge_eigenvalues, null_space, psd_to_time_domain, radians_to_degrees,
    random_complex, random_gaussian_polar, right_solve, scalar_to_complex, set_random_seed,
    sqrt_known_sign, sqrt_phase_unwrap, unwrap_radians,
};
use rust_rf::time::Window;

const TOLERANCE: f64 = 1.0e-6;

#[test]
fn converts_complex_components() {
    assert_relative_eq!(complex_magnitude(Complex64::new(3.0, 4.0)), 5.0);
    assert_relative_eq!(complex_to_db10(Complex64::new(6.0, 8.0)), 10.0);
    assert_relative_eq!(complex_angle_degrees(Complex64::new(0.0, 1.0)), 90.0);
    assert_eq!(complex_real_imaginary(Complex64::new(1.0, 2.0)), (1.0, 2.0));

    let quadrature = complex_quadrature(Complex64::new(0.0, 2.0));
    assert_relative_eq!(quadrature.0, 2.0);
    assert_relative_eq!(quadrature.1, std::f64::consts::PI);

    let components = complex_components(Complex64::new(0.0, 2.0));
    assert_relative_eq!(components.0, 0.0);
    assert_relative_eq!(components.1, 2.0);
    assert_relative_eq!(components.2, 90.0);
    assert_relative_eq!(components.3, 2.0);
    assert_relative_eq!(components.4, std::f64::consts::PI);
}

#[test]
fn converts_logarithmic_magnitudes() {
    assert_relative_eq!(magnitude_to_db(10.0, true), 20.0);
    assert_relative_eq!(magnitude_to_db10(10.0, true), 10.0);
    assert_eq!(magnitude_to_db(0.0, true), f64::NEG_INFINITY);
    assert_eq!(magnitude_to_db(-1.0, true), LOG_OF_NEGATIVE);
    assert!(magnitude_to_db(-1.0, false).is_nan());

    let complex = db10_to_complex_magnitude(Complex64::new(3.0, 4.0));
    let expected = (Complex64::new(3.0, 4.0) * (std::f64::consts::LN_10 / 10.0)).exp();
    assert_complex_close(complex, expected);
}

#[test]
fn converts_magnitude_and_phase_to_complex() {
    assert_complex_close(
        magnitude_degrees_to_complex(1.0, 90.0),
        Complex64::new(0.0, 1.0),
    );
    assert_complex_close(db_degrees_to_complex(20.0, 90.0), Complex64::new(0.0, 10.0));
}

#[test]
fn converts_physical_and_logarithmic_units() {
    assert_relative_eq!(nepers_to_db(1.0), 20.0 / std::f64::consts::LN_10);
    assert_relative_eq!(db_to_nepers(1.0), std::f64::consts::LN_10 / 20.0);
    assert_relative_eq!(radians_to_degrees(std::f64::consts::PI), 180.0);
    assert_relative_eq!(feet_to_meters(0.01), 0.003048);
    assert_relative_eq!(feet_to_meters(1.0), 0.3048);
    assert_relative_eq!(meters_to_feet(0.01), 0.0328084);
    assert_relative_eq!(meters_to_feet(1.0), 3.28084);
    assert_relative_eq!(
        db_per_100_feet_to_db_per_100_meters(2.5),
        8.2020997375,
        epsilon = 1.0e-8
    );
}

#[test]
fn replaces_infinities_with_skrf_sentinel() {
    assert_eq!(infinity_to_number(f64::INFINITY), NUMERICAL_INFINITY);
    assert_eq!(infinity_to_number(f64::NEG_INFINITY), -NUMERICAL_INFINITY);
    assert_eq!(
        infinities_to_numbers(&array![0.0, f64::INFINITY, 0.0, f64::NEG_INFINITY]),
        array![0.0, NUMERICAL_INFINITY, 0.0, -NUMERICAL_INFINITY]
    );
}

#[test]
fn right_solve_satisfies_x_times_a_equals_b() {
    let coefficients = Array3::from_shape_vec(
        (1, 2, 2),
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(3.0, -1.0),
        ],
    )
    .expect("shape should be valid");
    let right_hand_side = Array3::from_shape_vec(
        (1, 2, 2),
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(-1.0, 1.0),
            Complex64::new(2.0, 2.0),
        ],
    )
    .expect("shape should be valid");

    let solution = right_solve(&coefficients, &right_hand_side)
        .expect("coefficient matrix should be invertible");
    for row in 0..2 {
        for column in 0..2 {
            let actual = (0..2)
                .map(|inner| solution[(0, row, inner)] * coefficients[(0, inner, column)])
                .sum();
            assert_complex_close(actual, right_hand_side[(0, row, column)]);
        }
    }
}

#[test]
fn finds_integer_order_bessel_zeros() {
    assert_relative_eq!(
        bessel_j_zero(0, 1, false).expect("J0 root should be found"),
        2.404_825_557_695_773,
        epsilon = 1.0e-12
    );
    assert_relative_eq!(
        bessel_j_zero(1, 1, true).expect("J1 derivative root should be found"),
        1.841_183_781_340_659_3,
        epsilon = 1.0e-12
    );
    assert!(bessel_j_zero(0, 0, false).is_err());
}

#[test]
fn calculates_complete_elliptic_integrals() {
    assert_relative_eq!(
        complete_elliptic_integral_first_kind(0.0).expect("K(0) should be defined"),
        std::f64::consts::PI / 2.0,
        epsilon = 1.0e-15
    );
    assert_relative_eq!(
        complete_elliptic_integral_first_kind(0.5).expect("K(0.5) should be defined"),
        1.854_074_677_301_371_9,
        epsilon = 1.0e-14
    );
    assert!(complete_elliptic_integral_first_kind(1.0).is_err());
}

#[test]
fn calculates_modified_bessel_i1() {
    assert_relative_eq!(
        modified_bessel_i1(0.0).expect("I1(0) should be defined"),
        0.0,
        epsilon = 1.0e-15
    );
    assert_relative_eq!(
        modified_bessel_i1(1.0).expect("I1(1) should be defined"),
        0.565_159_103_992_485,
        epsilon = 1.0e-14
    );
}

#[test]
fn seeded_complex_random_values_are_repeatable() {
    set_random_seed(42);
    let first_seed_42 = random_complex(2, 2);
    set_random_seed(43);
    let first_seed_43 = random_complex(2, 2);
    set_random_seed(42);
    let second_seed_42 = random_complex(2, 2);
    set_random_seed(43);
    let second_seed_43 = random_complex(2, 2);

    assert_eq!(first_seed_42, second_seed_42);
    assert_eq!(first_seed_43, second_seed_43);
    assert_ne!(first_seed_42, first_seed_43);
    assert!(
        first_seed_42
            .iter()
            .all(|value| { (-1.0..=1.0).contains(&value.re) && (-1.0..=1.0).contains(&value.im) })
    );
}

#[test]
fn seeded_gaussian_polar_values_are_repeatable() {
    set_random_seed(123);
    let first =
        random_gaussian_polar(2, 3, 0.1, 0.2).expect("Gaussian polar samples should be generated");
    set_random_seed(123);
    let second =
        random_gaussian_polar(2, 3, 0.1, 0.2).expect("Gaussian polar samples should be generated");
    assert_eq!(first, second);
    assert!(first.iter().any(|value| value.norm() > 0.0));
    assert_eq!(
        random_gaussian_polar(1, 1, 0.0, 0.0).expect("zero-deviation samples should be generated")
            [(0, 0)],
        Complex64::new(0.0, 0.0)
    );
    assert!(random_gaussian_polar(1, 1, -1.0, 0.1).is_err());
}

#[test]
fn unwraps_phase_and_selects_complex_roots() {
    let unwrapped = unwrap_radians(&array![
        0.0,
        1.5 * std::f64::consts::PI,
        1.75 * std::f64::consts::PI
    ]);
    assert_relative_eq!(unwrapped[1], -0.5 * std::f64::consts::PI);
    assert_relative_eq!(unwrapped[2], -0.25 * std::f64::consts::PI);

    let squared = Complex64::new(3.0, 4.0);
    assert_complex_close(
        sqrt_known_sign(squared, Complex64::new(2.0, 1.0)),
        Complex64::new(2.0, 1.0),
    );
    assert_complex_close(
        sqrt_known_sign(squared, Complex64::new(2.0, -1.0)),
        Complex64::new(2.0, -1.0),
    );
    assert_eq!(
        find_correct_sign(
            Complex64::new(1.0, 1.0),
            Complex64::new(-1.0, -1.0),
            Complex64::new(2.0, 1.0)
        ),
        Complex64::new(1.0, 1.0)
    );
    assert_eq!(
        find_closest(
            Complex64::new(1.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(3.0, 0.0)
        ),
        Complex64::new(4.0, 0.0)
    );

    let values = array![
        Complex64::from_polar(4.0, 170.0_f64.to_radians()),
        Complex64::from_polar(4.0, -170.0_f64.to_radians())
    ];
    let roots = sqrt_phase_unwrap(&values);
    assert_relative_eq!(roots[0].arg().to_degrees(), 85.0, epsilon = TOLERANCE);
    assert_relative_eq!(roots[1].arg().to_degrees(), 95.0, epsilon = TOLERANCE);
    assert_eq!(dirac_delta(0.0), 1.0);
    assert_eq!(dirac_delta(1.0), 0.0);
    assert_eq!(neumann_number(0.0), 1.0);
    assert_eq!(neumann_number(1.0), 2.0);
}

#[test]
fn serializes_complex_matrices_in_fortran_order() {
    let values = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let scalars = complex_to_scalar(&values);
    assert_eq!(scalars, array![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        scalar_to_complex(scalars.as_slice().expect("array should be contiguous"))
            .expect("pairs should deserialize"),
        array![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]
    );
    assert!(scalar_to_complex(&[1.0]).is_err());

    let matrix = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0),
        ],
    )
    .expect("shape should be valid");
    assert_eq!(
        flatten_complex_matrix(&matrix),
        array![1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 4.0, 4.0]
    );
}

#[test]
fn evaluates_complex_matrix_predicates() {
    let swap = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    assert!(is_unitary(&swap, 1.0e-12));
    assert!(is_symmetric(&swap, 1.0e-12));

    let hermitian = array![
        [Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)],
        [Complex64::new(0.0, -1.0), Complex64::new(2.0, 0.0)]
    ];
    assert!(is_hermitian(&hermitian, 1.0e-12));
    assert!(is_positive_definite(&hermitian, 1.0e-12));
    assert_eq!(hermitian_transpose(&hermitian), hermitian);

    let semidefinite = array![
        [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]
    ];
    assert!(is_positive_semidefinite(&semidefinite, 1.0e-12));
    assert!(!is_positive_definite(&semidefinite, 1.0e-12));

    let indefinite = array![
        [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        [Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)]
    ];
    assert!(!is_positive_semidefinite(&indefinite, 1.0e-12));
}

#[test]
fn calculates_null_spaces_and_complexified_functions() {
    let matrix = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    let basis = null_space(&matrix, 1.0e-15).expect("null space should be computed");
    assert_eq!(basis.dim(), (2, 1));
    assert_relative_eq!(basis[(0, 0)].norm(), 0.0, epsilon = 1.0e-12);
    assert_relative_eq!(basis[(1, 0)].norm(), 1.0, epsilon = 1.0e-12);

    assert_eq!(
        complexify(Complex64::new(2.0, -3.0), |value| value * value),
        Complex64::new(4.0, 9.0)
    );
}

#[test]
fn performs_rational_interpolation() {
    let x = array![3.0, 0.0, 2.0, 1.0, 4.0];
    let y = x.mapv(|value| Complex64::new(value * value + 2.0 * value + 1.0, -value));
    let interpolator =
        RationalInterpolator::new(&x, &y, 2, 1.0e-12, false).expect("interpolator should build");
    let target = array![0.0, 0.5, 2.0, 3.5];
    let values = interpolator.evaluate(&target);
    for (actual, target) in values.iter().zip(target) {
        assert_relative_eq!(
            actual.re,
            target * target + 2.0 * target + 1.0,
            epsilon = 1.0e-10
        );
        assert_relative_eq!(actual.im, -target, epsilon = 1.0e-10);
    }
}

#[test]
fn interpolates_multidimensional_values_along_axis_zero() {
    let x = array![2.0, 0.0, 1.0];
    let y = array![
        [Complex64::new(4.0, 0.0), Complex64::new(5.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)]
    ]
    .into_dyn();
    let interpolator = RationalInterpolatorAxis0::new(&x, &y, 2, 1.0e-12, false)
        .expect("multidimensional interpolator should build");
    let values = interpolator.evaluate(&array![0.5, 1.5]);
    assert_eq!(values.shape(), &[2, 2]);
    assert_relative_eq!(values[[0, 0]].re, 0.25, epsilon = 1.0e-12);
    assert_relative_eq!(values[[0, 1]].re, 2.0, epsilon = 1.0e-12);
    assert_relative_eq!(values[[1, 0]].re, 2.25, epsilon = 1.0e-12);
    assert_relative_eq!(values[[1, 1]].re, 4.0, epsilon = 1.0e-12);
}

#[test]
fn converts_spectra_to_centered_time_domain() {
    let spectrum = Array1::from_elem(5, Complex64::new(1.0, 0.0));
    let transformed = inverse_fft_centered(&spectrum);
    assert_eq!(transformed.len(), 5);
    assert_relative_eq!(transformed[2].re, 1.0, epsilon = 1.0e-12);
    assert!(
        transformed
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != 2)
            .all(|(_, value)| value.norm() < 1.0e-12)
    );

    let real = inverse_real_fft_centered(
        &array![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0)
        ],
        Some(4),
    )
    .expect("real inverse FFT should succeed");
    assert_eq!(real.len(), 4);
    assert!(real.iter().all(|value| value.is_finite()));

    let (time, signal) = psd_to_time_domain(
        &array![1.0, 2.0, 3.0],
        &array![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0)
        ],
        Window::Hamming,
    )
    .expect("PSD should transform");
    assert_eq!(time.len(), 5);
    assert_eq!(signal.len(), 5);
    assert_relative_eq!(time[0], -0.5, epsilon = 1.0e-12);
    assert_relative_eq!(time[4], 0.5, epsilon = 1.0e-12);
}

#[test]
fn transforms_multidimensional_spectra_along_axis_zero() {
    let spectrum = Array2::from_shape_fn((5, 2), |(_, column)| {
        Complex64::new((column + 1) as f64, 0.0)
    })
    .into_dyn();
    let transformed = inverse_fft_centered_axis0(&spectrum).expect("axis-zero FFT should succeed");
    assert_eq!(transformed.shape(), &[5, 2]);
    assert_relative_eq!(transformed[[2, 0]].re, 1.0, epsilon = 1.0e-12);
    assert_relative_eq!(transformed[[2, 1]].re, 2.0, epsilon = 1.0e-12);
    for row in [0, 1, 3, 4] {
        assert!(transformed[[row, 0]].norm() < 1.0e-12);
        assert!(transformed[[row, 1]].norm() < 1.0e-12);
    }

    let real_spectrum = Array2::from_shape_fn((3, 2), |(_, column)| {
        Complex64::new((column + 1) as f64, 0.0)
    })
    .into_dyn();
    let real = inverse_real_fft_centered_axis0(&real_spectrum, Some(4))
        .expect("axis-zero real FFT should succeed");
    assert_eq!(real.shape(), &[4, 2]);
    assert_relative_eq!(real[[2, 0]], 1.0, epsilon = 1.0e-12);
    assert_relative_eq!(real[[2, 1]], 2.0, epsilon = 1.0e-12);
}

#[test]
fn nudges_small_eigenvalues() {
    let zero = Array3::zeros((3, 2, 2));
    let nudged = nudge_eigenvalues(&zero, None, None).expect("zero matrices should be nudged");
    for batch in 0..3 {
        assert_relative_eq!(nudged[(batch, 0, 0)].re, 1.0e-12, epsilon = 1.0e-18);
        assert_relative_eq!(nudged[(batch, 1, 1)].re, 1.0e-12, epsilon = 1.0e-18);
        assert!(nudged[(batch, 0, 1)].norm() < 1.0e-18);
        assert!(nudged[(batch, 1, 0)].norm() < 1.0e-18);
    }

    let identity = Array3::from_shape_fn((1, 4, 4), |(_, row, column)| {
        Complex64::new(if row == column { 1.0 } else { 0.0 }, 0.0)
    });
    assert_eq!(
        nudge_eigenvalues(&identity, None, None).expect("identity should remain stable"),
        identity
    );
    let custom = nudge_eigenvalues(&Array3::zeros((1, 2, 2)), None, Some(1.0e-10))
        .expect("custom minimum should apply");
    assert_relative_eq!(custom[(0, 0, 0)].re, 1.0e-10, epsilon = 1.0e-16);
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(actual.re, expected.re, epsilon = TOLERANCE);
    assert_relative_eq!(actual.im, expected.im, epsilon = TOLERANCE);
}
