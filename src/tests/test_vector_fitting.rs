//! Integration tests for vector fitting, model persistence, passivity, and
//! state-space export.

use approx::assert_relative_eq;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rust_rf::data::DATA;
use rust_rf::plotting::Component;
use rust_rf::vector_fitting::VectorFitting;
use rust_rf::{Frequency, Network};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
/// Fits a real pole with residue, constant, and proportional terms and verifies
/// both direct and state-space responses.
fn fits_real_pole_residue_constant_and_proportional_terms() {
    let pole = Complex64::new(-std::f64::consts::TAU * 1.0e9, 0.0);
    let residue = Complex64::new(3.0e9, 0.0);
    let constant = 0.2;
    let proportional = 2.0e-12;
    let network = model_network(101, |frequency| {
        let s = Complex64::new(0.0, std::f64::consts::TAU * frequency);
        constant + proportional * s + residue / (s - pole)
    })
    .expect("analytic model network should be valid");
    let sampled = network.frequency.values_hz().clone();
    let expected = network.s.clone();
    let mut fitting = VectorFitting::new(network);

    fitting
        .vector_fit(1, 0)
        .expect("real-pole fit should succeed");

    assert_relative_eq!(fitting.poles[0].re, pole.re, max_relative = 1.0e-12);
    assert_relative_eq!(
        fitting.residues[(0, 0)].re,
        residue.re,
        max_relative = 1.0e-9
    );
    assert_relative_eq!(
        fitting.constant_coefficients[0],
        constant,
        epsilon = 1.0e-10
    );
    assert_relative_eq!(
        fitting.proportional_coefficients[0],
        proportional,
        max_relative = 1.0e-8
    );
    let response = fitting
        .model_response(0, 0, &sampled)
        .expect("model should be evaluated");
    for (point, actual) in response.iter().enumerate() {
        assert_relative_eq!(actual.re, expected[(point, 0, 0)].re, epsilon = 1.0e-9);
        assert_relative_eq!(actual.im, expected[(point, 0, 0)].im, epsilon = 1.0e-9);
    }
    assert_eq!(VectorFitting::model_order(&fitting.poles), 1);
    let state_space = fitting
        .state_space()
        .expect("state-space model should be assembled");
    assert_eq!(state_space.a.dim(), (1, 1));
    assert_relative_eq!(state_space.a[(0, 0)], pole.re, max_relative = 1.0e-12);
    let state_response = VectorFitting::response_from_state_space(&sampled, &state_space)
        .expect("state-space response should be evaluated");
    for point in 0..sampled.len() {
        assert_relative_eq!(
            state_response[(point, 0, 0)].re,
            response[point].re,
            epsilon = 1.0e-9
        );
        assert_relative_eq!(
            state_response[(point, 0, 0)].im,
            response[point].im,
            epsilon = 1.0e-9
        );
    }
    assert!(fitting.rms_error().expect("RMS error should be available") < 1.0e-9);
}

#[test]
/// Fits a complex-conjugate pole pair and verifies its state-space realization.
fn fits_complex_conjugate_pole_pair() {
    let omega = std::f64::consts::TAU * 1.0e9;
    let pole = Complex64::new(-0.01 * omega, omega);
    let residue = Complex64::new(2.0e9, 1.0e9);
    let network = model_network(101, |frequency| {
        let s = Complex64::new(0.0, std::f64::consts::TAU * frequency);
        0.1 + residue / (s - pole) + residue.conj() / (s - pole.conj())
    })
    .expect("analytic model network should be valid");
    let sampled = network.frequency.values_hz().clone();
    let mut fitting = VectorFitting::new(network);

    fitting
        .vector_fit(0, 1)
        .expect("complex-pole fit should succeed");

    assert_relative_eq!(fitting.poles[0].re, pole.re, max_relative = 1.0e-12);
    assert_relative_eq!(fitting.poles[0].im, pole.im, max_relative = 1.0e-12);
    assert_relative_eq!(
        fitting.residues[(0, 0)].re,
        residue.re,
        max_relative = 1.0e-8
    );
    assert_relative_eq!(
        fitting.residues[(0, 0)].im,
        residue.im,
        max_relative = 1.0e-8
    );
    assert_eq!(VectorFitting::model_order(&fitting.poles), 2);
    let state_space = fitting
        .state_space()
        .expect("complex state-space model should be assembled");
    assert_eq!(state_space.a.dim(), (2, 2));
    assert_relative_eq!(state_space.a[(0, 1)], pole.im, max_relative = 1.0e-12);
    assert_relative_eq!(state_space.a[(1, 0)], -pole.im, max_relative = 1.0e-12);
    assert_relative_eq!(state_space.b[(0, 0)], 2.0);
    assert_relative_eq!(state_space.b[(1, 0)], 0.0);
    let direct = fitting
        .model_response(0, 0, &sampled)
        .expect("direct model should evaluate");
    let state_response = VectorFitting::response_from_state_space(&sampled, &state_space)
        .expect("state-space response should evaluate");
    for point in 0..sampled.len() {
        assert_relative_eq!(
            state_response[(point, 0, 0)].re,
            direct[point].re,
            epsilon = 1.0e-8
        );
        assert_relative_eq!(
            state_response[(point, 0, 0)].im,
            direct[point].im,
            epsilon = 1.0e-8
        );
    }
    assert!(fitting.rms_error().expect("RMS error should be available") < 1.0e-8);
}

#[test]
/// Rejects invalid model state and exercises automatic model fitting.
fn validates_model_state_and_auto_fit() {
    let network = model_network(101, |_| Complex64::new(0.25, 0.0))
        .expect("analytic model network should be valid");
    let mut fitting = VectorFitting::new(network);
    assert!(
        fitting
            .model_response(0, 0, &ndarray::array![1.0e9])
            .is_err()
    );
    assert!(fitting.vector_fit(0, 0).is_err());
    fitting.auto_fit().expect("automatic fit should succeed");
    assert_eq!(fitting.poles.len(), 6);
    assert!(fitting.rms_error().expect("RMS error should be available") < 1.0e-6);
    assert!(
        fitting
            .model_response(1, 0, &ndarray::array![1.0e9])
            .is_err()
    );
}

#[test]
/// Classifies complex poles with negligible residue energy as spurious.
fn classifies_low_energy_complex_poles_as_spurious() {
    let poles = ndarray::array![
        Complex64::new(-1.0e8, 1.0e9),
        Complex64::new(-2.0e8, 2.0e9),
        Complex64::new(-3.0e8, 0.0)
    ];
    let residues = Array2::from_shape_vec(
        (2, 3),
        vec![
            Complex64::new(1.0e9, 0.5e9),
            Complex64::new(1.0, 0.5),
            Complex64::new(1.0e9, 0.0),
            Complex64::new(0.8e9, 0.2e9),
            Complex64::new(0.8, 0.2),
            Complex64::new(1.0e9, 0.0),
        ],
    )
    .expect("shape should be valid");
    let spurious = VectorFitting::spurious_poles(&poles, &residues, 101, 0.03)
        .expect("classification should succeed");
    assert_eq!(spurious, vec![false, true, false]);
    assert!(VectorFitting::spurious_poles(&poles, &Array2::zeros((1, 2)), 101, 0.03).is_err());
}

#[test]
/// Writes fitted parameters to a `NumPy` archive and restores them unchanged.
fn writes_and_reads_numpy_model_archives() {
    let network = model_network(101, |frequency| {
        let s = Complex64::new(0.0, std::f64::consts::TAU * frequency);
        0.2 + Complex64::new(3.0e9, 0.0) / (s - Complex64::new(-std::f64::consts::TAU * 1.0e9, 0.0))
    })
    .expect("analytic model network should be valid");
    let mut fitting = VectorFitting::new(network.clone());
    fitting.vector_fit(1, 0).expect("model should fit");
    let directory = std::env::temp_dir().join(format!(
        "rust-rf-vector-fitting-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should follow the Unix epoch")
            .as_nanos()
    ));
    fs::create_dir(&directory).expect("temporary directory should be created");
    let archive = directory.join("model.npz");

    fitting
        .write_npz(&archive)
        .expect("model archive should be written");
    let mut restored = VectorFitting::new(network);
    restored
        .read_npz(&archive)
        .expect("model archive should be read");

    assert_eq!(restored.poles, fitting.poles);
    assert_eq!(restored.residues, fitting.residues);
    assert_eq!(
        restored.constant_coefficients,
        fitting.constant_coefficients
    );
    assert_eq!(
        restored.proportional_coefficients,
        fitting.proportional_coefficients
    );
    fs::remove_dir_all(directory).expect("temporary directory should be removed");
}

#[test]
/// Detects sampled passivity violations and enforces a passive response.
fn detects_and_enforces_sampled_passivity() {
    let network = model_network(21, |_| Complex64::new(1.2, 0.0))
        .expect("analytic model network should be valid");
    let mut fitting = VectorFitting::new(network);
    fitting.poles = ndarray::array![Complex64::new(-1.0e9, 0.0)];
    fitting.residues = ndarray::array![[Complex64::new(0.0, 0.0)]];
    fitting.constant_coefficients = ndarray::array![1.2];
    fitting.proportional_coefficients = ndarray::array![0.0];

    assert!(!fitting.is_passive().expect("passivity should be tested"));
    assert_eq!(
        fitting
            .passivity_bands(21, Some(10.0e9))
            .expect("violation bands should be available"),
        vec![(0.0, 10.0e9)]
    );
    fitting
        .enforce_passivity(201, Some(10.0e9))
        .expect("passivity should be enforced");
    assert!(fitting.is_passive().expect("passivity should be tested"));
    assert!(fitting.constant_coefficients[0] < 1.0);
}

#[test]
/// Retains the smallest model that satisfies the adaptive-fit tolerance.
fn adaptive_fit_retains_the_smallest_sufficient_model() {
    let network = model_network(101, |frequency| {
        let s = Complex64::new(0.0, std::f64::consts::TAU * frequency);
        0.25 + Complex64::new(3.0e9, 0.0)
            / (s - Complex64::new(-std::f64::consts::TAU * 1.0e9, 0.0))
    })
    .expect("analytic model network should be valid");
    let mut fitting = VectorFitting::new(network);

    fitting
        .auto_fit_with_tolerance(5, 1.0e-8)
        .expect("adaptive fit should succeed");

    assert_eq!(VectorFitting::model_order(&fitting.poles), 1);
    assert!(fitting.rms_error().expect("RMS error should be available") < 1.0e-8);
}

#[test]
/// Fits caller-supplied poles and builds response and singular-value plot data.
fn fits_with_caller_supplied_poles_and_builds_plot_data() {
    let pole = Complex64::new(-std::f64::consts::TAU * 1.0e9, 0.0);
    let network = model_network(101, |frequency| {
        let s = Complex64::new(0.0, std::f64::consts::TAU * frequency);
        0.2 + Complex64::new(3.0e9, 0.0) / (s - pole)
    })
    .expect("analytic model network should be valid");
    let mut fitting = VectorFitting::new(network);

    fitting
        .fit_with_poles(&ndarray::array![pole])
        .expect("custom poles should be accepted");

    assert_relative_eq!(fitting.poles[0].re, pole.re, max_relative = 1.0e-12);
    assert!(fitting.rms_error().expect("RMS error should be available") < 1.0e-9);
    let plot = fitting
        .model_plot(Component::Magnitude, Some((0, 0)), None)
        .expect("plot data should be built");
    assert_eq!(plot.series.len(), 2);
    assert_eq!(plot.series[0].x.len(), 101);
    let singular = fitting
        .singular_value_plot(None)
        .expect("singular values should be plotted");
    assert_eq!(singular.series.len(), 1);
    assert_eq!(singular.series[0].y.len(), 101);
    assert!(
        fitting
            .fit_with_poles(&ndarray::array![Complex64::new(1.0, 0.0)])
            .is_err()
    );
}

#[test]
/// Writes the fitted state-space model as a SPICE subcircuit.
fn writes_state_space_spice_subcircuit() {
    let network = model_network(21, |_| Complex64::new(0.2, 0.0))
        .expect("analytic model network should be valid");
    let mut fitting = VectorFitting::new(network);
    fitting.poles = ndarray::array![Complex64::new(-1.0e9, 0.0), Complex64::new(-2.0e9, 3.0e9)];
    fitting.residues = ndarray::array![[Complex64::new(1.0e8, 0.0), Complex64::new(2.0e8, 3.0e8)]];
    fitting.constant_coefficients = ndarray::array![0.2];
    fitting.proportional_coefficients = ndarray::array![1.0e-12];
    let directory = std::env::temp_dir().join(format!(
        "rust-rf-vector-spice-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should follow the Unix epoch")
            .as_nanos()
    ));
    fs::create_dir(&directory).expect("temporary directory should be created");
    let path = directory.join("model.sp");

    fitting
        .write_spice_subcircuit(&path, "fitted_model", true)
        .expect("SPICE subcircuit should be written");

    let netlist = fs::read_to_string(&path).expect("SPICE subcircuit should be readable");
    assert!(netlist.contains(".SUBCKT fitted_model p1 p1_ref"));
    assert!(netlist.contains("R1 s1 p1_ref 50"));
    assert!(netlist.contains("Cx1_a1"));
    assert!(netlist.contains("Cx2_re_a1"));
    assert!(netlist.contains("Cx2_im_a1"));
    assert!(netlist.contains("Le1 e1 0 1.0"));
    assert!(netlist.ends_with(".ENDS fitted_model\n"));
    fs::remove_dir_all(directory).expect("temporary directory should be removed");
}

#[test]
/// Relocates initial poles toward the measured response and improves fit error.
fn relocates_initial_poles_to_the_measured_response() {
    let pole = Complex64::new(-std::f64::consts::TAU * 3.25e9, 0.0);
    let network = model_network(201, |frequency| {
        let s = Complex64::new(0.0, std::f64::consts::TAU * frequency);
        0.15 + Complex64::new(4.0e9, 0.0) / (s - pole)
    })
    .expect("analytic model network should be valid");
    let mut fixed = VectorFitting::new(network.clone());
    fixed
        .vector_fit(1, 0)
        .expect("fixed-pole fit should succeed");
    let fixed_error = fixed.rms_error().expect("RMS error should be available");
    let mut relocated = VectorFitting::new(network);

    relocated
        .vector_fit_relocating(1, 0, 12, 1.0e-10)
        .expect("pole relocation should succeed");

    assert_relative_eq!(relocated.poles[0].re, pole.re, max_relative = 1.0e-6);
    assert!(
        relocated
            .rms_error()
            .expect("RMS error should be available")
            < 1.0e-8
    );
    assert!(
        relocated
            .rms_error()
            .expect("RMS error should be available")
            < fixed_error
    );
}

#[test]
/// Preserves the real measured response at the DC sample.
fn preserves_the_measured_dc_sample() {
    let pole = Complex64::new(-1.0e9, 0.0);
    let frequency = Frequency::from_hz(ndarray::Array1::linspace(0.0, 10.0e9, 101))
        .expect("frequency should be valid");
    let s = Array3::from_shape_fn((101, 1, 1), |(point, _, _)| {
        let laplace = Complex64::new(0.0, std::f64::consts::TAU * frequency.values_hz()[point]);
        0.25 + Complex64::new(2.0e9, 0.0) / (laplace - pole)
    });
    let network = Network::new(
        frequency,
        s,
        Array2::from_elem((101, 1), Complex64::new(50.0, 0.0)),
    )
    .expect("network should be valid");
    let expected_dc = network.s[(0, 0, 0)].re;
    let mut fitting = VectorFitting::new(network);

    fitting.vector_fit(1, 0).expect("model should fit");

    let actual = fitting
        .model_response(0, 0, &ndarray::array![0.0])
        .expect("DC response should evaluate")[0];
    assert_relative_eq!(actual.re, expected_dc, epsilon = 1.0e-12);
}

#[test]
/// Fits the upstream ring-slot example within the expected RMS error.
fn fits_the_upstream_ring_slot_example() {
    let network = DATA.ring_slot().expect("ring-slot fixture should parse");
    let mut fitting = VectorFitting::new(network);

    fitting
        .vector_fit_relocating(2, 0, 20, 1.0e-6)
        .expect("ring-slot model should fit");

    assert!(fitting.rms_error().expect("RMS error should be available") < 0.02);
}

#[test]
/// Fits the upstream 190 GHz measured network within the expected RMS error.
fn fits_the_upstream_190_ghz_measurement() {
    let network = Network::read_touchstone(vector_fixture("190ghz_tx_measured.s2p"))
        .expect("190 GHz fixture should parse");
    let mut fitting = VectorFitting::new(network);

    fitting
        .vector_fit_relocating(4, 4, 20, 1.0e-6)
        .expect("measured model should fit");

    assert!(fitting.rms_error().expect("RMS error should be available") < 0.02);
}

/// Builds a one-port network by sampling an analytic model.
fn model_network(points: usize, model: impl Fn(f64) -> Complex64) -> rust_rf::Result<Network> {
    let frequency = Frequency::from_hz(ndarray::Array1::linspace(1.0e9, 10.0e9, points))?;
    let s = Array3::from_shape_fn((points, 1, 1), |(point, _, _)| {
        model(frequency.values_hz()[point])
    });
    Network::new(
        frequency,
        s,
        Array2::from_elem((points, 1), Complex64::new(50.0, 0.0)),
    )
}

/// Locates a vector-fitting fixture in the Rust integration-test data.
fn vector_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("vector_fitting")
        .join(name)
}
