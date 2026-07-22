//! MAT 58 Q-factor fitting and derived-quantity regressions.

use approx::assert_relative_eq;
use ndarray::{Array2, Array3, array};
use num_complex::Complex64;
use rust_rf::constants::SPEED_OF_LIGHT;
use rust_rf::qfactor::{OptimizedResult, QFactor, QFitMethod, ResonanceType};
use rust_rf::{Error, Frequency, FrequencyUnit, Network};

/// Checks recovery of a synthetic six-parameter resonator response.
#[test]
fn fits_six_parameter_resonator_response() -> rust_rf::Result<()> {
    let expected_q = 250.0;
    let expected_frequency = 1.0e9;
    let detuned = Complex64::new(0.1, -0.02);
    let resonant_delta = Complex64::new(0.7, 0.1);
    let network = synthetic_resonator(
        0.95e9,
        1.05e9,
        201,
        expected_frequency,
        expected_q,
        detuned,
        resonant_delta,
    )?;
    let qfactor = QFactor::new(
        network.clone(),
        ResonanceType::Transmission,
        Some(200.0),
        Some(1.001e9),
    )
    .expect("Q-factor construction should succeed");

    let result = qfactor.fit().expect("six-parameter fit should succeed");

    assert!(result.success);
    assert_eq!(result.loop_plan, "fwfwc");
    assert!(result.weighting_ratio.is_some());
    assert_relative_eq!(result.loaded_q, expected_q, max_relative = 1.0e-6);
    assert_relative_eq!(
        result.resonant_frequency_hz,
        expected_frequency,
        max_relative = 1.0e-8
    );
    assert_relative_eq!(result.m1, detuned.re, epsilon = 1.0e-8);
    assert_relative_eq!(result.m2, detuned.im, epsilon = 1.0e-8);
    assert_relative_eq!(result.m3, resonant_delta.re, epsilon = 1.0e-8);
    assert_relative_eq!(result.m4, resonant_delta.im, epsilon = 1.0e-8);
    assert!(result.rms_error < 1.0e-9);

    let fitted = qfactor
        .fitted_network(&result)
        .expect("fitted network should be generated");
    assert_eq!(fitted.frequency, network.frequency);
    for (actual, expected) in fitted.s.iter().zip(network.s.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-8);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-8);
    }
    Ok(())
}

/// Checks Q-circle geometry, unloaded Q, angular weights, and 3-dB bandwidth.
#[test]
fn calculates_circle_unloaded_q_weights_and_bandwidth() -> rust_rf::Result<()> {
    let network = synthetic_resonator(
        0.9e9,
        1.1e9,
        11,
        1.0e9,
        100.0,
        Complex64::new(0.5, 0.0),
        Complex64::new(-0.2, 0.0),
    )?;
    let result = OptimizedResult {
        success: true,
        m1: 0.5,
        m2: 0.0,
        m3: -0.2,
        m4: 0.0,
        loaded_q: 100.0,
        resonant_frequency_hz: 1.0e9,
        ..OptimizedResult::default()
    };
    let qfactor = QFactor::new(network, ResonanceType::Reflection, None, None)
        .expect("Q-factor construction should succeed");

    let circle = qfactor
        .q_circle(&result, None)
        .expect("Q-circle calculation should succeed");
    assert_relative_eq!(circle.diameter, 0.4, epsilon = 1.0e-12);
    assert_relative_eq!(circle.detuned.re, 1.0, epsilon = 1.0e-12);
    assert_relative_eq!(circle.tuned.re, 0.6, epsilon = 1.0e-12);
    assert_relative_eq!(
        qfactor
            .unloaded_q(&result, None)
            .expect("unloaded Q should be calculated"),
        125.0,
        epsilon = 1.0e-12
    );
    assert_relative_eq!(
        QFactor::bandwidth_hz(&result).expect("bandwidth should be calculated"),
        10.0e6,
        epsilon = 1.0e-12
    );
    assert_relative_eq!(
        qfactor
            .resonant_frequency_scaled(&result)
            .expect("scaled frequency should be calculated"),
        1.0e9,
        epsilon = 1.0e-12
    );
    assert_relative_eq!(
        qfactor
            .bandwidth_scaled(&result)
            .expect("scaled bandwidth should be calculated"),
        10.0e6,
        epsilon = 1.0e-12
    );
    let weights = QFactor::angular_weights(&[0.99e9, 1.0e9, 1.01e9], 1.0e9, 100.0);
    assert_relative_eq!(weights[0], 0.2, epsilon = 1.0e-12);
    assert_relative_eq!(weights[1], 1.0, epsilon = 1.0e-12);
    assert_relative_eq!(weights[2], 0.2, epsilon = 1.0e-12);
    Ok(())
}

/// Checks input-network validation and resonance-specific scaling requirements.
#[test]
fn validates_network_shape_and_required_scaling() {
    let frequency =
        Frequency::from_values(array![0.9, 0.94, 0.98, 1.02, 1.06, 1.1], FrequencyUnit::GHz)
            .expect("frequency should be valid");
    let two_port = Network::new(
        frequency.clone(),
        Array3::zeros((6, 2, 2)),
        Array2::from_elem((6, 2), Complex64::new(50.0, 0.0)),
    )
    .expect("network should be valid");
    assert!(QFactor::new(two_port, ResonanceType::Reflection, None, None).is_err());

    let one_port = Network::new(
        frequency,
        Array3::zeros((6, 1, 1)),
        Array2::from_elem((6, 1), Complex64::new(50.0, 0.0)),
    )
    .expect("network should be valid");
    let result = OptimizedResult {
        m1: 0.5,
        m3: 0.1,
        loaded_q: 100.0,
        resonant_frequency_hz: 1.0e9,
        ..OptimizedResult::default()
    };
    let transmission = QFactor::new(one_port, ResonanceType::Transmission, None, None)
        .expect("Q-factor construction should succeed");
    assert!(transmission.unloaded_q(&result, None).is_err());
    for invalid in ["", "wfc", "fcw", "fxc"] {
        assert!(
            transmission
                .fit_with_loop_plan(QFitMethod::Nlqfit6, invalid)
                .is_err()
        );
    }
    let unweighted = transmission
        .fit_with_loop_plan(QFitMethod::Nlqfit6, "f")
        .expect("one unweighted fit should be accepted");
    assert_eq!(unweighted.loop_plan, "f");
    assert!(unweighted.weighting_ratio.is_none());
}

/// Checks NLQFIT7 recovery of transmission-line phase delay.
#[test]
fn fits_seven_parameter_phase_delay_model() -> rust_rf::Result<()> {
    let expected_q = 420.0;
    let expected_frequency = 1.0e9;
    let phase_slope = 4.0e-9;
    let mut network = synthetic_resonator(
        0.97e9,
        1.03e9,
        241,
        expected_frequency,
        expected_q,
        Complex64::new(0.2, -0.05),
        Complex64::new(0.55, 0.08),
    )?;
    for point in 0..network.frequency_points() {
        let offset = network.frequency.values_hz()[point] - expected_frequency;
        network.s[(point, 0, 0)] *= Complex64::from_polar(1.0, phase_slope * offset);
    }
    let qfactor = QFactor::new(
        network,
        ResonanceType::Reflection,
        Some(350.0),
        Some(0.999e9),
    )
    .expect("Q-factor construction should succeed");
    let result = qfactor
        .fit_method(QFitMethod::Nlqfit7)
        .expect("seven-parameter fit should succeed");

    assert!(result.success);
    assert_eq!(result.method, QFitMethod::Nlqfit7);
    assert_relative_eq!(result.loaded_q, expected_q, max_relative = 2.0e-4);
    assert_relative_eq!(
        result.resonant_frequency_hz,
        expected_frequency,
        max_relative = 1.0e-6
    );
    assert_relative_eq!(
        result
            .phase_slope_radians_per_hz
            .expect("phase slope should be reported"),
        phase_slope,
        max_relative = 2.0e-3
    );
    assert!(result.weighting_ratio.is_some());
    Ok(())
}

/// Checks NLQFIT8 recovery of frequency-dependent leakage.
#[test]
fn fits_eight_parameter_frequency_dependent_leakage_model() -> rust_rf::Result<()> {
    let expected_q = 310.0;
    let expected_frequency = 2.0e9;
    let leakage = Complex64::new(0.035, -0.018);
    let mut network = synthetic_resonator(
        1.94e9,
        2.06e9,
        241,
        expected_frequency,
        expected_q,
        Complex64::new(0.08, 0.02),
        Complex64::new(0.62, -0.04),
    )?;
    for point in 0..network.frequency_points() {
        let frequency = network.frequency.values_hz()[point];
        network.s[(point, 0, 0)] +=
            leakage * (2.0 * (frequency - expected_frequency) / expected_frequency);
    }
    let qfactor = QFactor::new(
        network,
        ResonanceType::Transmission,
        Some(280.0),
        Some(2.001e9),
    )
    .expect("Q-factor construction should succeed");
    let result = qfactor
        .fit_method(QFitMethod::Nlqfit8)
        .expect("eight-parameter fit should succeed");

    assert!(result.success);
    assert_eq!(result.method, QFitMethod::Nlqfit8);
    assert_relative_eq!(result.loaded_q, expected_q, max_relative = 2.0e-4);
    assert_relative_eq!(
        result.resonant_frequency_hz,
        expected_frequency,
        max_relative = 1.0e-6
    );
    let fitted_leakage = result
        .leakage_slope
        .expect("leakage slope should be reported");
    assert_relative_eq!(fitted_leakage.re, leakage.re, max_relative = 2.0e-3);
    assert_relative_eq!(fitted_leakage.im, leakage.im, max_relative = 2.0e-3);
    Ok(())
}

/// Compares all fit methods with the published NPL MAT 58 datasets.
#[test]
fn matches_npl_mat58_reference_datasets() -> rust_rf::Result<()> {
    let figure6b = fixture_network("Figure6b.txt")?;
    let transmission = QFactor::new(figure6b, ResonanceType::Transmission, None, None)
        .expect("Figure 6b should construct");
    let result6 = transmission.fit().expect("Figure 6b NLQFIT6 should fit");
    assert_relative_eq!(
        result6.resonant_frequency_hz,
        3.987_848e9,
        max_relative = 2.0e-6
    );
    assert_relative_eq!(result6.loaded_q, 7_454.0, max_relative = 5.0e-3);

    // Upstream NPL MAT 58 dataset: Figure27.
    let absorption_network = fixture_network("Figure27.txt")?;
    let absorption = QFactor::new(absorption_network, ResonanceType::Absorption, None, None)
        .expect("Figure 27 should construct");
    let result27 = absorption.fit().expect("Figure 27 NLQFIT6 should fit");
    assert_relative_eq!(
        result27.resonant_frequency_hz,
        6.072_255_67e9,
        max_relative = 2.0e-6
    );
    assert_relative_eq!(result27.loaded_q, 56_019.85, max_relative = 1.0e-2);

    let table6c27 = fixture_network("Table6c27.txt")?;
    let reflection = QFactor::new(table6c27, ResonanceType::Reflection, None, None)
        .expect("Table 6c27 should construct");
    // Upstream result7: Table 6c27 evaluated with NLQFIT7.
    let reflection_fit_result = reflection
        .fit_method(QFitMethod::Nlqfit7)
        .expect("Table 6c27 NLQFIT7 should fit");
    assert_relative_eq!(
        reflection_fit_result.resonant_frequency_hz,
        3.652_938e9,
        max_relative = 2.0e-6
    );
    assert_relative_eq!(reflection_fit_result.loaded_q, 708.0, max_relative = 2.0e-2);

    // Upstream NPL MAT 58 dataset: Figure23.
    let mut leakage_network = fixture_network("Figure23.txt")?;
    for point in 0..leakage_network.frequency_points() {
        let frequency = leakage_network.frequency.values_hz()[point];
        leakage_network.s[(point, 0, 0)] *= Complex64::from_polar(
            1.0,
            -std::f64::consts::TAU * frequency * 1.2 / SPEED_OF_LIGHT,
        );
    }
    let peak = (0..leakage_network.frequency_points())
        .max_by(|left, right| {
            leakage_network.s[(*left, 0, 0)]
                .norm_sqr()
                .total_cmp(&leakage_network.s[(*right, 0, 0)].norm_sqr())
        })
        .expect("Figure 23 should contain samples");
    let seed_frequency = leakage_network.frequency.values_hz()[peak];
    let seed_q = 5.0 * seed_frequency
        / (leakage_network.frequency.values_hz()[leakage_network.frequency_points() - 1]
            - leakage_network.frequency.values_hz()[0]);
    let leakage = QFactor::new(
        leakage_network,
        ResonanceType::Transmission,
        Some(seed_q),
        Some(seed_frequency),
    )
    .expect("Figure 23 should construct");
    let result8 = leakage
        .fit_method(QFitMethod::Nlqfit8)
        .expect("Figure 23 NLQFIT8 should fit");
    assert_relative_eq!(
        result8.resonant_frequency_hz,
        9.760_155_71e9,
        max_relative = 2.0e-6
    );
    assert_relative_eq!(result8.loaded_q, 4_760.04, max_relative = 2.0e-2);
    Ok(())
}

fn synthetic_resonator(
    start_hz: f64,
    stop_hz: f64,
    points: usize,
    resonant_frequency_hz: f64,
    loaded_q: f64,
    detuned: Complex64,
    resonant_delta: Complex64,
) -> rust_rf::Result<Network> {
    let frequency = Frequency::from_hz(ndarray::Array1::linspace(start_hz, stop_hz, points))?;
    let s = Array3::from_shape_fn((points, 1, 1), |(point, _, _)| {
        let frequency_hz = frequency.values_hz()[point];
        let offset = frequency_hz / resonant_frequency_hz - resonant_frequency_hz / frequency_hz;
        detuned + resonant_delta / Complex64::new(1.0, loaded_q * offset)
    });
    let z0 = Array2::from_elem((points, 1), Complex64::new(50.0, 0.0));
    Network::new(frequency, s, z0)
}

fn fixture_network(name: &str) -> rust_rf::Result<Network> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("qfactor")
        .join(name);
    let mut frequency_ghz = Vec::new();
    let mut scattering = Vec::new();
    for line in fs::read_to_string(path)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        let values = line
            .split_whitespace()
            .map(|value| {
                value.parse::<f64>().map_err(|error| {
                    Error::Parse(format!("invalid Q-factor fixture value {value:?}: {error}"))
                })
            })
            .collect::<rust_rf::Result<Vec<_>>>()?;
        if values.len() < 3 {
            return Err(Error::Parse(
                "Q-factor fixture rows need frequency, real, and imaginary values".to_owned(),
            ));
        }
        frequency_ghz.push(values[0]);
        scattering.push(Complex64::new(values[1], values[2]));
    }
    let frequency =
        Frequency::from_values(ndarray::Array1::from_vec(frequency_ghz), FrequencyUnit::GHz)?;
    let points = frequency.points();
    let scattering = Array3::from_shape_vec((points, 1, 1), scattering).map_err(|error| {
        Error::IncompatibleShape(format!(
            "invalid Q-factor fixture scattering shape: {error}"
        ))
    })?;
    Network::new(
        frequency,
        scattering,
        Array2::from_elem((points, 1), Complex64::new(50.0, 0.0)),
    )
}
use std::fs;
use std::path::PathBuf;
