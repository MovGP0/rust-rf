use std::collections::BTreeMap;

use approx::assert_relative_eq;
use ndarray::{Array2, Array3, array};
use num_complex::Complex64;
use rust_rf::calibration::{
    Calibration, EightTerm, EnhancedResponse, ErrorNetworkResult, Lmr16, Lrm, Lrrm, LrrmMatchFit,
    Mrc, MultiportSolt, NistMultilineTrl, Normalization, OnePort, Sddl, SddlWeikle, SixteenTerm,
    Trl, TugMultilineTrl, TwelveTerm, TwoPortOnePath, UnknownThru, align_measured_ideals,
    compute_switch_terms, convert_8term_2_12term, convert_12term_2_8term, convert_pnacoefs_2_skrf,
    convert_skrfcoefs_2_pna, error_dict_2_network, ideal_coefs_12term, terminate, terminate_nport,
    two_port_error_vector_2_ts, unterminate,
};
use rust_rf::{Frequency, Network, NetworkSet};

#[test]
fn solves_and_applies_three_term_one_port_calibration() {
    let frequency =
        Frequency::from_hz(array![1.0e9, 2.0e9, 3.0e9]).expect("frequency should be valid");
    let directivity = Complex64::new(0.02, 0.01);
    let tracking = Complex64::new(0.8, 0.1);
    let source_match = Complex64::new(0.05, -0.02);
    let standards = [
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];
    let ideals = standards
        .iter()
        .map(|standard| constant_one_port(frequency.clone(), *standard))
        .collect::<Vec<_>>();
    let measured = standards
        .iter()
        .map(|standard| {
            constant_one_port(
                frequency.clone(),
                embed_value(*standard, directivity, tracking, source_match),
            )
        })
        .collect::<Vec<_>>();
    let mut calibration = OnePort::new(measured, ideals).expect("standards should be valid");
    let raw_dut = one_port(
        frequency.clone(),
        &[
            Complex64::new(0.2, 0.1),
            Complex64::new(-0.3, 0.2),
            Complex64::new(0.1, -0.4),
        ]
        .map(|ideal| embed_value(ideal, directivity, tracking, source_match)),
    );
    assert!(calibration.apply(&raw_dut).is_err());

    calibration.run().expect("calibration should solve");

    for value in &calibration.coefficients()["directivity"] {
        assert_complex_close(*value, directivity);
    }
    for value in &calibration.coefficients()["reflection tracking"] {
        assert_complex_close(*value, tracking);
    }
    for value in &calibration.coefficients()["source match"] {
        assert_complex_close(*value, source_match);
    }
    let corrected = calibration
        .apply(&raw_dut)
        .expect("DUT correction should succeed");
    let expected = [
        Complex64::new(0.2, 0.1),
        Complex64::new(-0.3, 0.2),
        Complex64::new(0.1, -0.4),
    ];
    for (point, expected) in expected.iter().enumerate() {
        assert_complex_close(corrected.s[(point, 0, 0)], *expected);
    }
    let embedded = calibration
        .embed(&corrected)
        .expect("DUT embedding should succeed");
    for (actual, expected) in embedded.s.iter().zip(raw_dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }

    assert_eq!(calibration.standards(), 4);
    assert_eq!(
        calibration.frequency().expect("frequency should exist"),
        &frequency
    );
    let coefficient_networks = calibration
        .coefficient_networks()
        .expect("coefficient networks should be available");
    assert_eq!(coefficient_networks.len(), 3);
    assert_eq!(
        coefficient_networks["directivity"].name.as_deref(),
        Some("directivity")
    );
    assert_complex_close(
        coefficient_networks["directivity"].s[(0, 0, 0)],
        directivity,
    );

    let calibrated_standards = calibration
        .calibrated_standards()
        .expect("standards should calibrate");
    for (calibrated, ideal) in calibrated_standards.iter().zip(calibration.ideals()) {
        for (actual, expected) in calibrated.s.iter().zip(ideal.s.iter()) {
            assert_complex_close(*actual, *expected);
        }
    }
    let residuals = calibration
        .residual_networks()
        .expect("residuals should be available");
    for residual in residuals {
        for value in residual.s {
            assert_complex_close(value, Complex64::new(0.0, 0.0));
        }
    }

    let mut set = NetworkSet::new(vec![raw_dut.clone(), raw_dut], Some("dut-set".to_owned()))
        .expect("network set should be valid");
    set.set_parameter("temperature", vec![20.0, 30.0])
        .expect("parameter should be valid");
    let calibrated_set = calibration
        .apply_network_set(&set)
        .expect("network set calibration should succeed");
    assert_eq!(calibrated_set.name.as_deref(), Some("dut-set"));
    assert_eq!(calibrated_set.parameters["temperature"], vec![20.0, 30.0]);
    for network in calibrated_set.networks {
        for (point, expected) in expected.iter().enumerate() {
            assert_complex_close(network.s[(point, 0, 0)], *expected);
        }
    }
}

#[test]
fn terminates_and_unterminates_two_port_switch_terms() {
    let frequency =
        Frequency::from_hz(array![1.0e9, 2.0e9, 3.0e9]).expect("frequency should be valid");
    let ideal = complex_two_port(
        frequency,
        [
            [Complex64::new(0.12, 0.03), Complex64::new(0.62, -0.08)],
            [Complex64::new(0.71, 0.05), Complex64::new(-0.09, 0.02)],
        ],
    );
    let forward = array![
        Complex64::new(0.04, 0.01),
        Complex64::new(0.03, -0.01),
        Complex64::new(0.02, 0.015),
    ];
    let reverse = array![
        Complex64::new(-0.02, 0.01),
        Complex64::new(-0.015, 0.005),
        Complex64::new(-0.01, -0.005),
    ];

    let measured = terminate(&ideal, &forward, &reverse).expect("switch-term termination");
    let corrected = unterminate(&measured, &forward, &reverse).expect("switch-term untermination");

    for (actual, expected) in corrected.s.iter().zip(ideal.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn terminates_nport_with_the_upstream_two_port_gamma_order() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency should be valid");
    let network = complex_two_port(
        frequency.clone(),
        [
            [Complex64::new(0.12, 0.03), Complex64::new(0.71, -0.04)],
            [Complex64::new(0.68, 0.06), Complex64::new(-0.09, 0.02)],
        ],
    );
    let forward = ndarray::Array1::from_vec(vec![
        Complex64::new(0.04, 0.01),
        Complex64::new(0.03, -0.02),
    ]);
    let reverse = ndarray::Array1::from_vec(vec![
        Complex64::new(-0.02, 0.01),
        Complex64::new(0.01, 0.03),
    ]);
    let expected = terminate(&network, &forward, &reverse).expect("termination should succeed");
    let reverse_network = one_port(frequency.clone(), reverse.as_slice().expect("contiguous"));
    let forward_network = one_port(frequency, forward.as_slice().expect("contiguous"));
    let actual = terminate_nport(&network, &[reverse_network, forward_network])
        .expect("N-port termination should succeed");
    for (actual, expected) in actual.s.iter().zip(expected.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn computes_switch_terms_from_reciprocal_device_measurements() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency should be valid");
    let forward = ndarray::Array1::from_vec(vec![
        Complex64::new(0.04, 0.01),
        Complex64::new(0.03, -0.02),
    ]);
    let reverse = ndarray::Array1::from_vec(vec![
        Complex64::new(-0.02, 0.01),
        Complex64::new(0.01, 0.03),
    ]);
    let reciprocal_devices = [
        [
            [Complex64::new(0.10, 0.02), Complex64::new(0.80, -0.03)],
            [Complex64::new(0.80, -0.03), Complex64::new(-0.05, 0.01)],
        ],
        [
            [Complex64::new(-0.23, 0.05), Complex64::new(0.55, 0.07)],
            [Complex64::new(0.55, 0.07), Complex64::new(0.18, -0.04)],
        ],
        [
            [Complex64::new(0.31, -0.09), Complex64::new(0.37, 0.11)],
            [Complex64::new(0.37, 0.11), Complex64::new(-0.27, 0.08)],
        ],
        [
            [Complex64::new(-0.08, 0.13), Complex64::new(0.66, -0.12)],
            [Complex64::new(0.66, -0.12), Complex64::new(0.22, 0.06)],
        ],
    ];
    let measurements = reciprocal_devices
        .into_iter()
        .map(|values| {
            terminate(
                &complex_two_port(frequency.clone(), values),
                &forward,
                &reverse,
            )
            .expect("termination should succeed")
        })
        .collect::<Vec<_>>();

    let (computed_forward, computed_reverse) =
        compute_switch_terms(&measurements).expect("switch terms should solve");
    for point in 0..frequency.points() {
        assert_complex_close(computed_forward.s[(point, 0, 0)], forward[point]);
        assert_complex_close(computed_reverse.s[(point, 0, 0)], reverse[point]);
    }
}

#[test]
fn converts_between_twelve_and_eight_term_coefficients() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency should be valid");
    let ideal = ideal_coefs_12term(&frequency);
    let ideal_eight =
        convert_12term_2_8term(&ideal, true).expect("ideal coefficients should convert");
    let ideal_round_trip =
        convert_8term_2_12term(&ideal_eight).expect("ideal coefficients should round trip");
    for (name, expected) in &ideal {
        let actual = &ideal_round_trip[name];
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected);
        }
    }

    let points = frequency.points();
    let eight = BTreeMap::from([
        (
            "forward directivity".to_owned(),
            constant_array(points, 0.02),
        ),
        (
            "forward source match".to_owned(),
            constant_array(points, 0.04),
        ),
        (
            "forward reflection tracking".to_owned(),
            constant_array(points, 0.82),
        ),
        (
            "forward isolation".to_owned(),
            constant_array(points, 0.001),
        ),
        (
            "reverse directivity".to_owned(),
            constant_array(points, -0.01),
        ),
        (
            "reverse source match".to_owned(),
            constant_array(points, 0.03),
        ),
        (
            "reverse reflection tracking".to_owned(),
            constant_array(points, 0.91),
        ),
        (
            "reverse isolation".to_owned(),
            constant_array(points, -0.002),
        ),
        (
            "forward switch term".to_owned(),
            constant_array(points, 0.06),
        ),
        (
            "reverse switch term".to_owned(),
            constant_array(points, -0.04),
        ),
        ("k".to_owned(), constant_array(points, 1.13)),
    ]);
    let twelve = convert_8term_2_12term(&eight).expect("eight-term coefficients should convert");
    let round_trip =
        convert_12term_2_8term(&twelve, false).expect("twelve-term coefficients should convert");
    for name in ["forward switch term", "reverse switch term", "k"] {
        for (actual, expected) in round_trip[name].iter().zip(eight[name].iter()) {
            assert_complex_close(*actual, *expected);
        }
    }
}

#[test]
fn converts_pna_coefficient_names_without_losing_values() {
    let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency should be valid");
    let coefficients = ideal_coefs_12term(&frequency);
    let pna =
        convert_skrfcoefs_2_pna(&coefficients, (1, 2)).expect("scikit-rf names should convert");
    assert!(pna.contains_key("Directivity(1,1)"));
    assert!(pna.contains_key("LoadMatch(2,1)"));
    assert!(pna.contains_key("TransmissionTracking(1,2)"));
    let round_trip = convert_pnacoefs_2_skrf(&pna).expect("PNA names should convert");
    assert_eq!(round_trip, coefficients);

    let three_term = BTreeMap::from([
        ("directivity".to_owned(), constant_array(1, 0.01)),
        ("source match".to_owned(), constant_array(1, 0.02)),
        ("reflection tracking".to_owned(), constant_array(1, 0.9)),
    ]);
    let one_port_pna =
        convert_skrfcoefs_2_pna(&three_term, (3, 4)).expect("one-port names should convert");
    assert!(one_port_pna.contains_key("Directivity(3,3)"));
    assert_eq!(
        convert_pnacoefs_2_skrf(&one_port_pna).expect("one-port names should round trip"),
        three_term
    );
}

#[test]
fn aligns_named_standards_and_builds_error_networks() {
    let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency should be valid");
    let mut ideal_open = constant_one_port(frequency.clone(), Complex64::new(1.0, 0.0));
    ideal_open.name = Some("open".to_owned());
    let mut ideal_short = constant_one_port(frequency.clone(), Complex64::new(-1.0, 0.0));
    ideal_short.name = Some("short".to_owned());
    let mut measured_short = ideal_short.clone();
    measured_short.name = Some("measured-short-01".to_owned());
    let mut measured_open = ideal_open.clone();
    measured_open.name = Some("measured-open-01".to_owned());
    let (measured, ideals) = align_measured_ideals(
        &[measured_short.clone(), measured_open.clone()],
        &[ideal_open.clone(), ideal_short.clone()],
    );
    assert_eq!(measured, vec![measured_short, measured_open]);
    assert_eq!(ideals, vec![ideal_short, ideal_open]);

    let coefficients = BTreeMap::from([
        ("directivity".to_owned(), constant_array(1, 0.01)),
        ("source match".to_owned(), constant_array(1, 0.02)),
        ("reflection tracking".to_owned(), constant_array(1, 0.81)),
    ]);
    let ErrorNetworkResult::One(error_network) =
        error_dict_2_network(&coefficients, &frequency, true)
            .expect("three-term error network should build")
    else {
        panic!("three-term coefficients should produce one network");
    };
    assert_complex_close(error_network.s[(0, 0, 0)], Complex64::new(0.01, 0.0));
    assert_complex_close(error_network.s[(0, 1, 1)], Complex64::new(0.02, 0.0));
    assert_complex_close(error_network.s[(0, 0, 1)], Complex64::new(0.9, 0.0));
    assert_complex_close(error_network.s[(0, 1, 0)], Complex64::new(0.9, 0.0));
}

#[test]
fn builds_two_port_error_t_matrices() {
    let coefficients = BTreeMap::from([
        ("det_X".to_owned(), constant_array(2, 2.0)),
        ("det_Y".to_owned(), constant_array(2, 3.0)),
        ("e00".to_owned(), constant_array(2, 4.0)),
        ("e11".to_owned(), constant_array(2, 5.0)),
        ("e22".to_owned(), constant_array(2, 6.0)),
        ("e33".to_owned(), constant_array(2, 7.0)),
        ("k".to_owned(), constant_array(2, 8.0)),
    ]);
    let (t1, t2, t3, t4) =
        two_port_error_vector_2_ts(&coefficients).expect("T matrices should build");
    assert_complex_close(t1[(0, 0, 0)], Complex64::new(-2.0, 0.0));
    assert_complex_close(t1[(0, 1, 1)], Complex64::new(-24.0, 0.0));
    assert_complex_close(t2[(0, 0, 0)], Complex64::new(4.0, 0.0));
    assert_complex_close(t2[(0, 1, 1)], Complex64::new(56.0, 0.0));
    assert_complex_close(t3[(0, 0, 0)], Complex64::new(-5.0, 0.0));
    assert_complex_close(t3[(0, 1, 1)], Complex64::new(-48.0, 0.0));
    assert_complex_close(t4[(0, 0, 0)], Complex64::new(1.0, 0.0));
    assert_complex_close(t4[(0, 1, 1)], Complex64::new(8.0, 0.0));
}

#[test]
fn validates_one_port_standard_count_and_shape() {
    let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency should be valid");
    let standards = vec![
        constant_one_port(frequency.clone(), Complex64::new(-1.0, 0.0)),
        constant_one_port(frequency, Complex64::new(1.0, 0.0)),
    ];
    assert!(OnePort::new(standards.clone(), standards).is_err());
}

#[test]
fn applies_simple_thru_normalization() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency should be valid");
    let first = one_port(
        frequency.clone(),
        &[Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)],
    );
    let second = one_port(
        frequency.clone(),
        &[Complex64::new(4.0, 0.0), Complex64::new(8.0, 0.0)],
    );
    let mut calibration =
        Normalization::new(vec![first, second]).expect("normalization should be valid");
    calibration
        .run()
        .expect("normalization run should validate");
    let measured = one_port(
        frequency,
        &[Complex64::new(6.0, 0.0), Complex64::new(18.0, 0.0)],
    );

    let corrected = calibration
        .apply(&measured)
        .expect("normalization should apply");
    assert_complex_close(corrected.s[(0, 0, 0)], Complex64::new(2.0, 0.0));
    assert_complex_close(corrected.s[(1, 0, 0)], Complex64::new(3.0, 0.0));
    let restored = calibration
        .embed(&corrected)
        .expect("normalization embedding should apply");
    for (actual, expected) in restored.s.iter().zip(measured.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn solves_applies_and_embeds_twelve_term_two_port_calibration() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let reflects = [-1.0, 0.0, 1.0]
        .into_iter()
        .map(|reflection| two_port(frequency.clone(), [[reflection, 0.0], [0.0, reflection]]))
        .collect::<Vec<_>>();
    let thru = two_port(frequency.clone(), [[0.0, 1.0], [1.0, 0.0]]);
    let mut ideals = reflects;
    ideals.push(thru);
    let coefficients = BTreeMap::from([
        ("forward directivity".to_owned(), constant_array(2, 0.01)),
        ("forward source match".to_owned(), constant_array(2, 0.03)),
        (
            "forward reflection tracking".to_owned(),
            constant_array(2, 0.8),
        ),
        (
            "forward transmission tracking".to_owned(),
            constant_array(2, 0.7),
        ),
        ("forward load match".to_owned(), constant_array(2, 0.05)),
        ("forward isolation".to_owned(), constant_array(2, 0.0)),
        ("reverse directivity".to_owned(), constant_array(2, 0.02)),
        ("reverse source match".to_owned(), constant_array(2, 0.04)),
        (
            "reverse reflection tracking".to_owned(),
            constant_array(2, 0.9),
        ),
        (
            "reverse transmission tracking".to_owned(),
            constant_array(2, 0.75),
        ),
        ("reverse load match".to_owned(), constant_array(2, 0.06)),
        ("reverse isolation".to_owned(), constant_array(2, 0.0)),
    ]);
    let seed = TwelveTerm {
        measured: ideals.clone(),
        ideals: ideals.clone(),
        coefficients,
    };
    let measured = ideals
        .iter()
        .map(|ideal| seed.embed(ideal))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let mut calibration = TwelveTerm::new(measured, ideals).expect("calibration standards");

    calibration
        .run()
        .expect("twelve-term calibration should solve");

    let dut = two_port(frequency, [[0.2, 0.1], [0.6, -0.1]]);
    let raw = seed.embed(&dut).expect("DUT should embed");
    let corrected = calibration.apply(&raw).expect("DUT should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
    let restored = calibration.embed(&corrected).expect("DUT should re-embed");
    for (actual, expected) in restored.s.iter().zip(raw.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn corrects_one_path_measurements_in_both_orientations() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let ideals = vec![
        two_port(frequency.clone(), [[-1.0, 0.0], [0.0, -1.0]]),
        two_port(frequency.clone(), [[0.0, 0.0], [0.0, 0.0]]),
        two_port(frequency.clone(), [[1.0, 0.0], [0.0, 1.0]]),
        two_port(frequency.clone(), [[0.0, 1.0], [1.0, 0.0]]),
    ];
    let coefficients = BTreeMap::from([
        ("forward directivity".to_owned(), constant_array(2, 0.01)),
        ("forward source match".to_owned(), constant_array(2, 0.03)),
        (
            "forward reflection tracking".to_owned(),
            constant_array(2, 0.8),
        ),
        (
            "forward transmission tracking".to_owned(),
            constant_array(2, 0.7),
        ),
        ("forward load match".to_owned(), constant_array(2, 0.05)),
        ("forward isolation".to_owned(), constant_array(2, 0.0)),
        ("reverse directivity".to_owned(), constant_array(2, 0.01)),
        ("reverse source match".to_owned(), constant_array(2, 0.03)),
        (
            "reverse reflection tracking".to_owned(),
            constant_array(2, 0.8),
        ),
        (
            "reverse transmission tracking".to_owned(),
            constant_array(2, 0.7),
        ),
        ("reverse load match".to_owned(), constant_array(2, 0.05)),
        ("reverse isolation".to_owned(), constant_array(2, 0.0)),
    ]);
    let seed = TwelveTerm {
        measured: ideals.clone(),
        ideals: ideals.clone(),
        coefficients,
    };
    let measured = ideals
        .iter()
        .map(|ideal| seed.embed(ideal))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let mut calibration =
        TwoPortOnePath::new(measured.clone(), ideals.clone(), 0).expect("one-path standards");
    let mut enhanced =
        EnhancedResponse::new(measured, ideals, 0).expect("enhanced-response standards");

    calibration
        .run()
        .expect("one-path calibration should solve");
    enhanced
        .run()
        .expect("enhanced-response calibration should solve");

    let dut = two_port(frequency, [[0.2, 0.1], [0.6, -0.1]]);
    let forward = seed.embed(&dut).expect("forward DUT should embed");
    let reverse = seed
        .embed(&dut.flipped().expect("DUT should flip"))
        .expect("reverse DUT should embed");
    let corrected = calibration
        .apply_pair(&forward, &reverse)
        .expect("both orientations should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
    let partial = enhanced
        .apply(&forward)
        .expect("single orientation should partially calibrate");
    for point in 0..partial.frequency_points() {
        assert_relative_eq!(
            partial.s[(point, 0, 0)].re,
            dut.s[(point, 0, 0)].re,
            epsilon = 5.0e-3
        );
        assert_relative_eq!(
            partial.s[(point, 1, 0)].re,
            dut.s[(point, 1, 0)].re,
            epsilon = 5.0e-3
        );
        assert_complex_close(partial.s[(point, 0, 1)], Complex64::new(0.0, 0.0));
        assert_complex_close(partial.s[(point, 1, 1)], Complex64::new(0.0, 0.0));
    }
}

#[test]
fn calibrates_three_port_solt_from_common_port_thrus() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let nport = |values: [[f64; 3]; 3]| {
        Network::new(
            frequency.clone(),
            Array3::from_shape_fn((2, 3, 3), |(_, row, column)| {
                Complex64::new(values[row][column], 0.0)
            }),
            Array2::from_elem((2, 3), Complex64::new(50.0, 0.0)),
        )
        .expect("three-port network")
    };
    let ideals = vec![
        nport([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        nport([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        nport([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
        nport([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        nport([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ];
    let measured = ideals.clone();
    let mut calibration = MultiportSolt::new(measured, ideals, vec![[0, 1], [0, 2]], None)
        .expect("multiport SOLT standards");

    calibration.run().expect("multiport SOLT should solve");

    let dut = nport([[0.2, 0.1, 0.05], [0.6, -0.1, 0.12], [0.2, 0.3, 0.15]]);
    let raw = dut.clone();
    let corrected = calibration.apply(&raw).expect("multiport DUT correction");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
    let restored = calibration
        .embed(&corrected)
        .expect("multiport DUT embedding");
    for (actual, expected) in restored.s.iter().zip(raw.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn solves_applies_and_embeds_eight_term_two_port_calibration() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let ideals = vec![
        two_port(frequency.clone(), [[-1.0, 0.0], [0.0, -1.0]]),
        two_port(frequency.clone(), [[0.0, 0.0], [0.0, 0.0]]),
        two_port(frequency.clone(), [[0.0, 1.0], [1.0, 0.0]]),
        two_port(frequency.clone(), [[0.2, 0.7], [0.4, -0.1]]),
        two_port(frequency.clone(), [[0.4, -0.3], [0.6, 0.2]]),
    ];
    let coefficients = BTreeMap::from([
        ("forward directivity".to_owned(), constant_array(2, 0.01)),
        ("forward source match".to_owned(), constant_array(2, 0.03)),
        (
            "forward reflection tracking".to_owned(),
            constant_array(2, 0.8),
        ),
        ("forward isolation".to_owned(), constant_array(2, 0.0)),
        ("forward switch term".to_owned(), constant_array(2, 0.0)),
        ("reverse directivity".to_owned(), constant_array(2, 0.02)),
        ("reverse source match".to_owned(), constant_array(2, 0.04)),
        (
            "reverse reflection tracking".to_owned(),
            constant_array(2, 0.9),
        ),
        ("reverse isolation".to_owned(), constant_array(2, 0.0)),
        ("reverse switch term".to_owned(), constant_array(2, 0.0)),
        ("k".to_owned(), constant_array(2, 1.1)),
    ]);
    let seed = EightTerm {
        measured: ideals.clone(),
        ideals: ideals.clone(),
        coefficients,
    };
    let measured = ideals
        .iter()
        .map(|ideal| seed.embed(ideal))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let mut calibration = EightTerm::new(measured, ideals).expect("calibration standards");

    calibration
        .run()
        .expect("eight-term calibration should solve");

    let dut = two_port(frequency, [[0.2, 0.1], [0.6, -0.1]]);
    let raw = seed.embed(&dut).expect("DUT should embed");
    let corrected = calibration.apply(&raw).expect("DUT should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
    let restored = calibration.embed(&corrected).expect("DUT should re-embed");
    for (actual, expected) in restored.s.iter().zip(raw.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn solves_sddl_with_partially_known_delay_shorts() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let actuals = vec![
        constant_one_port(frequency.clone(), Complex64::new(-1.0, 0.0)),
        constant_one_port(frequency.clone(), Complex64::from_polar(1.0, -2.4)),
        constant_one_port(frequency.clone(), Complex64::from_polar(1.0, -1.2)),
        constant_one_port(frequency.clone(), Complex64::new(0.2, 0.2)),
    ];
    let ideals = vec![
        actuals[0].clone(),
        constant_one_port(frequency.clone(), Complex64::from_polar(1.0, -2.0)),
        constant_one_port(frequency.clone(), Complex64::from_polar(1.0, -0.8)),
        actuals[3].clone(),
    ];
    let seed = OnePort {
        measured: actuals.clone(),
        ideals: actuals.clone(),
        coefficients: BTreeMap::from([
            ("directivity".to_owned(), constant_array(2, 0.02)),
            ("reflection tracking".to_owned(), constant_array(2, 0.8)),
            ("source match".to_owned(), constant_array(2, 0.04)),
        ]),
    };
    let measured = actuals
        .iter()
        .map(|actual| seed.embed(actual))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let mut calibration = Sddl::new(measured, ideals).expect("SDDL standards");

    calibration.run().expect("SDDL calibration should solve");

    let dut = constant_one_port(frequency, Complex64::new(0.3, -0.15));
    let raw = seed.embed(&dut).expect("DUT should embed");
    let corrected = calibration.apply(&raw).expect("DUT should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn solves_weikle_short_delay_delay_load_calibration() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let actuals = [
        Complex64::new(-1.0, 0.0),
        Complex64::from_polar(1.0, -2.0),
        Complex64::from_polar(1.0, -0.8),
        Complex64::new(0.2, 0.2),
    ];
    let ideals = [
        Complex64::new(-1.0, 0.0),
        Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_2),
        Complex64::from_polar(1.0, -std::f64::consts::PI),
        Complex64::new(0.2, 0.2),
    ]
    .map(|value| constant_one_port(frequency.clone(), value))
    .to_vec();
    let seed = OnePort {
        measured: ideals.clone(),
        ideals: ideals.clone(),
        coefficients: BTreeMap::from([
            ("directivity".to_owned(), constant_array(2, 0.02)),
            ("reflection tracking".to_owned(), constant_array(2, 0.8)),
            ("source match".to_owned(), constant_array(2, 0.05)),
        ]),
    };
    let measured = actuals
        .into_iter()
        .map(|value| seed.embed(&constant_one_port(frequency.clone(), value)))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let mut calibration =
        SddlWeikle::new(measured, ideals).expect("SDDLWeikle standards should be valid");

    calibration
        .run()
        .expect("SDDLWeikle calibration should solve");

    let dut = constant_one_port(frequency, Complex64::new(0.3, -0.15));
    let raw = seed.embed(&dut).expect("DUT should embed");
    let corrected = calibration.apply(&raw).expect("DUT should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn determines_trl_line_and_reflect_before_eight_term_solution() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let thru = two_port(frequency.clone(), [[0.0, 1.0], [1.0, 0.0]]);
    let reflect = two_port(frequency.clone(), [[-0.9, 0.0], [0.0, -0.9]]);
    let line = two_port(frequency.clone(), [[0.0, -0.7], [-0.7, 0.0]]);
    let actuals = vec![thru.clone(), reflect, line];
    let seed = EightTerm {
        measured: actuals.clone(),
        ideals: actuals.clone(),
        coefficients: BTreeMap::from([
            ("forward directivity".to_owned(), constant_array(2, 0.01)),
            ("forward source match".to_owned(), constant_array(2, 0.03)),
            (
                "forward reflection tracking".to_owned(),
                constant_array(2, 0.8),
            ),
            ("forward isolation".to_owned(), constant_array(2, 0.0)),
            ("forward switch term".to_owned(), constant_array(2, 0.0)),
            ("reverse directivity".to_owned(), constant_array(2, 0.02)),
            ("reverse source match".to_owned(), constant_array(2, 0.04)),
            (
                "reverse reflection tracking".to_owned(),
                constant_array(2, 0.9),
            ),
            ("reverse isolation".to_owned(), constant_array(2, 0.0)),
            ("reverse switch term".to_owned(), constant_array(2, 0.0)),
            ("k".to_owned(), constant_array(2, 1.1)),
        ]),
    };
    let measured = actuals
        .iter()
        .map(|actual| seed.embed(actual))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let ideals = vec![
        thru,
        two_port(frequency.clone(), [[-1.0, 0.0], [0.0, -1.0]]),
        two_port(frequency.clone(), [[0.0, -0.8], [-0.8, 0.0]]),
    ];
    let mut calibration = Trl::single_reflect(measured, ideals).expect("TRL standards");

    calibration.run().expect("TRL calibration should solve");

    let dut = two_port(frequency, [[0.2, 0.1], [0.6, -0.1]]);
    let raw = seed.embed(&dut).expect("DUT should embed");
    let corrected = calibration.apply(&raw).expect("DUT should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn solves_nist_and_tug_multiline_trl_calibrations() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let line = |length: f64| {
        Network::new(
            frequency.clone(),
            Array3::from_shape_fn((2, 2, 2), |(point, row, column)| {
                if row == column {
                    Complex64::new(0.0, 0.0)
                } else {
                    let angular = 2.0 * std::f64::consts::PI * frequency.values_hz()[point];
                    Complex64::new(0.0, -angular * length / 299_792_458.0).exp()
                }
            }),
            Array2::from_elem((2, 2), Complex64::new(50.0, 0.0)),
        )
        .expect("line")
    };
    let lengths = vec![0.0, 0.03, 0.06];
    let actuals = vec![
        line(lengths[0]),
        two_port(frequency.clone(), [[-0.9, 0.0], [0.0, -0.9]]),
        line(lengths[1]),
        line(lengths[2]),
    ];
    let seed = eight_term_seed(actuals.clone());
    let measured = actuals
        .iter()
        .map(|ideal| seed.embed(ideal))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let mut nist = NistMultilineTrl::new(
        measured.clone(),
        vec![Complex64::new(-1.0, 0.0)],
        lengths.clone(),
        Complex64::new(1.0, 0.0),
    )
    .expect("NIST multiline standards");
    let mut tug = TugMultilineTrl::new(
        measured,
        vec![Complex64::new(-1.0, 0.0)],
        lengths,
        Complex64::new(1.0, 0.0),
    )
    .expect("TUG multiline standards");

    nist.run().expect("NIST multiline TRL should solve");
    tug.run().expect("TUG multiline TRL should solve");

    let dut = two_port(frequency, [[0.2, 0.1], [0.6, -0.1]]);
    let raw = seed.embed(&dut).expect("DUT should embed");
    for corrected in [
        nist.apply(&raw).expect("NIST DUT correction"),
        tug.apply(&raw).expect("TUG DUT correction"),
    ] {
        for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
            assert_complex_close(*actual, *expected);
        }
    }
    for propagation in [
        nist.propagation_constant
            .expect("NIST propagation constant"),
        tug.propagation_constant.expect("TUG propagation constant"),
    ] {
        for (point, value) in propagation.iter().enumerate() {
            let expected = 2.0 * std::f64::consts::PI * (point + 1) as f64 * 1.0e9 / 299_792_458.0;
            assert_relative_eq!(value.im, expected, epsilon = 1.0e-8);
        }
    }
}

#[test]
fn solves_applies_and_embeds_sixteen_term_calibration() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let ideals = vec![
        two_port(frequency.clone(), [[-1.0, 0.0], [0.0, -1.0]]),
        two_port(frequency.clone(), [[0.0, 0.0], [0.0, 0.0]]),
        two_port(frequency.clone(), [[0.0, 1.0], [1.0, 0.0]]),
        two_port(frequency.clone(), [[0.2, 0.7], [0.4, -0.1]]),
        two_port(frequency.clone(), [[0.4, -0.3], [0.6, 0.2]]),
        two_port(frequency.clone(), [[-0.2, 0.5], [-0.4, 0.3]]),
        two_port(frequency.clone(), [[0.1, -0.8], [0.2, 0.5]]),
        two_port(frequency.clone(), [[0.7, 0.2], [-0.6, -0.4]]),
    ];
    let coefficients = BTreeMap::from([
        ("forward directivity".to_owned(), constant_array(2, 0.01)),
        ("reverse directivity".to_owned(), constant_array(2, 0.02)),
        ("forward source match".to_owned(), constant_array(2, 0.03)),
        ("reverse source match".to_owned(), constant_array(2, 0.04)),
        (
            "forward reflection tracking".to_owned(),
            constant_array(2, 0.8),
        ),
        (
            "reverse reflection tracking".to_owned(),
            constant_array(2, 0.9),
        ),
        ("k".to_owned(), constant_array(2, 1.1)),
        ("forward isolation".to_owned(), constant_array(2, 0.001)),
        ("reverse isolation".to_owned(), constant_array(2, 0.002)),
        (
            "forward port 1 isolation".to_owned(),
            constant_array(2, 0.003),
        ),
        (
            "reverse port 1 isolation".to_owned(),
            constant_array(2, 0.004),
        ),
        (
            "forward port 2 isolation".to_owned(),
            constant_array(2, 0.005),
        ),
        (
            "reverse port 2 isolation".to_owned(),
            constant_array(2, 0.006),
        ),
        (
            "forward port isolation".to_owned(),
            constant_array(2, 0.007),
        ),
        (
            "reverse port isolation".to_owned(),
            constant_array(2, 0.008),
        ),
        ("forward switch term".to_owned(), constant_array(2, 0.0)),
        ("reverse switch term".to_owned(), constant_array(2, 0.0)),
    ]);
    let seed = SixteenTerm {
        measured: ideals.clone(),
        ideals: ideals.clone(),
        coefficients,
    };
    let measured = ideals
        .iter()
        .map(|ideal| seed.embed(ideal))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let mut calibration = SixteenTerm::new(measured, ideals).expect("calibration standards");

    calibration
        .run()
        .expect("sixteen-term calibration should solve");

    let dut = two_port(frequency, [[0.2, 0.1], [0.6, -0.1]]);
    let raw = seed.embed(&dut).expect("DUT should embed");
    let corrected = calibration.apply(&raw).expect("DUT should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
    let restored = calibration.embed(&corrected).expect("DUT should re-embed");
    for (actual, expected) in restored.s.iter().zip(raw.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn solves_unknown_reciprocal_thru_from_reflect_standards() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let mut ideals = [-1.0, 0.0, 1.0]
        .into_iter()
        .map(|reflection| two_port(frequency.clone(), [[reflection, 0.0], [0.0, reflection]]))
        .collect::<Vec<_>>();
    ideals.push(two_port(frequency.clone(), [[0.1, -0.7], [-0.7, -0.05]]));
    let seed = EightTerm {
        measured: ideals.clone(),
        ideals: ideals.clone(),
        coefficients: BTreeMap::from([
            ("forward directivity".to_owned(), constant_array(2, 0.01)),
            ("forward source match".to_owned(), constant_array(2, 0.03)),
            (
                "forward reflection tracking".to_owned(),
                constant_array(2, 0.8),
            ),
            ("forward isolation".to_owned(), constant_array(2, 0.0)),
            ("forward switch term".to_owned(), constant_array(2, 0.0)),
            ("reverse directivity".to_owned(), constant_array(2, 0.02)),
            ("reverse source match".to_owned(), constant_array(2, 0.04)),
            (
                "reverse reflection tracking".to_owned(),
                constant_array(2, 0.9),
            ),
            ("reverse isolation".to_owned(), constant_array(2, 0.0)),
            ("reverse switch term".to_owned(), constant_array(2, 0.0)),
            ("k".to_owned(), constant_array(2, 1.1)),
        ]),
    };
    let measured = ideals
        .iter()
        .map(|ideal| seed.embed(ideal))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let mut approximations = ideals.clone();
    approximations[3] = two_port(frequency.clone(), [[0.0, -1.0], [-1.0, 0.0]]);
    let mut calibration =
        UnknownThru::new(measured, approximations).expect("unknown-thru standards");

    calibration
        .run()
        .expect("unknown-thru calibration should solve");

    let dut = two_port(frequency, [[0.2, 0.1], [0.6, -0.1]]);
    let raw = seed.embed(&dut).expect("DUT should embed");
    let corrected = calibration.apply(&raw).expect("DUT should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn solves_misreflection_residual_calibration() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let actuals = vec![
        two_port(frequency.clone(), [[-1.0, 0.0], [0.0, -1.0]]),
        complex_two_port(
            frequency.clone(),
            [
                [Complex64::from_polar(1.0, -2.2), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, -1.1)],
            ],
        ),
        complex_two_port(
            frequency.clone(),
            [
                [Complex64::from_polar(1.0, -1.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, -2.0)],
            ],
        ),
        two_port(frequency.clone(), [[0.2, 0.0], [0.0, 0.2]]),
        two_port(frequency.clone(), [[0.1, -0.7], [-0.7, -0.05]]),
    ];
    let seed = eight_term_seed(actuals.clone());
    let measured = actuals
        .iter()
        .map(|ideal| seed.embed(ideal))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let approximations = vec![
        two_port(frequency.clone(), [[-1.0, 0.0], [0.0, -1.0]]),
        two_port(frequency.clone(), [[0.0, -1.0], [-1.0, 0.0]]),
        two_port(frequency.clone(), [[0.0, -1.0], [-1.0, 0.0]]),
        two_port(frequency.clone(), [[0.2, 0.0], [0.0, 0.2]]),
        two_port(frequency.clone(), [[0.0, -1.0], [-1.0, 0.0]]),
    ];
    let mut calibration = Mrc::new(measured, approximations).expect("MRC standards");

    calibration.run().expect("MRC calibration should solve");

    let dut = two_port(frequency, [[0.2, 0.1], [0.6, -0.1]]);
    let raw = seed.embed(&dut).expect("DUT should embed");
    let corrected = calibration.apply(&raw).expect("DUT should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn solves_line_reflect_match_calibration() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let line = two_port(frequency.clone(), [[0.0, -0.7], [-0.7, 0.0]]);
    let reflect = two_port(frequency.clone(), [[-0.9, 0.0], [0.0, -0.9]]);
    let matched = two_port(frequency.clone(), [[0.0, 0.0], [0.0, 0.0]]);
    let actuals = vec![line.clone(), reflect, matched.clone()];
    let seed = EightTerm {
        measured: actuals.clone(),
        ideals: actuals.clone(),
        coefficients: BTreeMap::from([
            ("forward directivity".to_owned(), constant_array(2, 0.01)),
            ("forward source match".to_owned(), constant_array(2, 0.03)),
            (
                "forward reflection tracking".to_owned(),
                constant_array(2, 0.8),
            ),
            ("forward isolation".to_owned(), constant_array(2, 0.0)),
            ("forward switch term".to_owned(), constant_array(2, 0.0)),
            ("reverse directivity".to_owned(), constant_array(2, 0.02)),
            ("reverse source match".to_owned(), constant_array(2, 0.04)),
            (
                "reverse reflection tracking".to_owned(),
                constant_array(2, 0.9),
            ),
            ("reverse isolation".to_owned(), constant_array(2, 0.0)),
            ("reverse switch term".to_owned(), constant_array(2, 0.0)),
            ("k".to_owned(), constant_array(2, 1.1)),
        ]),
    };
    let measured = actuals
        .iter()
        .map(|actual| seed.embed(actual))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let ideals = vec![
        line,
        two_port(frequency.clone(), [[-1.0, 0.0], [0.0, -1.0]]),
        matched,
    ];
    let mut calibration = Lrm::new(measured, ideals).expect("LRM standards");

    calibration.run().expect("LRM calibration should solve");

    let dut = two_port(frequency, [[0.2, 0.1], [0.6, -0.1]]);
    let raw = seed.embed(&dut).expect("DUT should embed");
    let corrected = calibration.apply(&raw).expect("DUT should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn solves_lmr16_from_known_reflect() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let thru = two_port(frequency.clone(), [[0.0, 0.8], [0.8, 0.0]]);
    let match_match = two_port(frequency.clone(), [[0.0, 0.0], [0.0, 0.0]]);
    let reflect_reflect = two_port(frequency.clone(), [[-1.0, 0.0], [0.0, -1.0]]);
    let reflect_match = two_port(frequency.clone(), [[-1.0, 0.0], [0.0, 0.0]]);
    let match_reflect = two_port(frequency.clone(), [[0.0, 0.0], [0.0, -1.0]]);
    let actuals = vec![
        thru,
        match_match,
        reflect_reflect,
        reflect_match,
        match_reflect,
    ];
    let seed = SixteenTerm {
        measured: actuals.clone(),
        ideals: actuals.clone(),
        coefficients: sixteen_term_coefficients(2),
    };
    let measured = actuals
        .iter()
        .map(|ideal| seed.embed(ideal))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let ideal_reflect = constant_one_port(frequency.clone(), Complex64::new(-1.0, 0.0));
    let mut calibration = Lmr16::new(measured, ideal_reflect, true, None).expect("LMR16 standards");

    calibration.run().expect("LMR16 calibration should solve");

    let dut = two_port(frequency, [[0.2, 0.1], [0.6, -0.1]]);
    let raw = seed.embed(&dut).expect("DUT should embed");
    let corrected = calibration.apply(&raw).expect("DUT should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn solves_line_reflect_reflect_match_per_frequency() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let line = two_port(frequency.clone(), [[0.0, 0.8], [0.8, 0.0]]);
    let ideal_reflect1 = two_port(frequency.clone(), [[-1.0, 0.0], [0.0, -1.0]]);
    let ideal_reflect2 = two_port(frequency.clone(), [[1.0, 0.0], [0.0, 1.0]]);
    let angular = frequency
        .values_hz()
        .iter()
        .map(|value| 2.0 * std::f64::consts::PI * value)
        .collect::<Vec<_>>();
    let reflect1_values = angular
        .iter()
        .map(|value| {
            let resistance = 50.0 * (1.0 - 0.95) / (1.0 + 0.95);
            let impedance = Complex64::new(resistance, value * 5.0e-12);
            (impedance - 50.0) / (impedance + 50.0)
        })
        .collect::<Vec<_>>();
    let reflect2_values = angular
        .iter()
        .map(|value| {
            let impedance = Complex64::new(0.0, -1.0 / (value * 5.0e-15));
            (impedance - 50.0) / (impedance + 50.0)
        })
        .collect::<Vec<_>>();
    let match_values = angular
        .iter()
        .map(|value| {
            let impedance = Complex64::new(50.0, value * 20.0e-12);
            (impedance - 50.0) / (impedance + 50.0)
        })
        .collect::<Vec<_>>();
    let diagonal = |port0: &[Complex64], port1: &[Complex64]| {
        Network::new(
            frequency.clone(),
            Array3::from_shape_fn((2, 2, 2), |(point, row, column)| {
                if row != column {
                    Complex64::new(0.0, 0.0)
                } else if row == 0 {
                    port0[point]
                } else {
                    port1[point]
                }
            }),
            Array2::from_elem((2, 2), Complex64::new(50.0, 0.0)),
        )
        .expect("reflect standard")
    };
    let reflect1 = diagonal(&reflect1_values, &reflect1_values);
    let reflect2 = diagonal(&reflect2_values, &reflect2_values);
    let matched = diagonal(&match_values, &reflect2_values);
    let ideal_match = Network::new(
        frequency.clone(),
        Array3::zeros((2, 2, 2)),
        Array2::from_elem((2, 2), Complex64::new(50.0, 0.0)),
    )
    .expect("ideal match");
    let actuals = vec![line.clone(), reflect1.clone(), reflect2.clone(), matched];
    let seed = EightTerm {
        measured: actuals.clone(),
        ideals: actuals.clone(),
        coefficients: BTreeMap::from([
            ("forward directivity".to_owned(), constant_array(2, 0.01)),
            ("forward source match".to_owned(), constant_array(2, 0.03)),
            (
                "forward reflection tracking".to_owned(),
                constant_array(2, 0.8),
            ),
            ("forward isolation".to_owned(), constant_array(2, 0.0)),
            ("forward switch term".to_owned(), constant_array(2, 0.0)),
            ("reverse directivity".to_owned(), constant_array(2, 0.02)),
            ("reverse source match".to_owned(), constant_array(2, 0.04)),
            (
                "reverse reflection tracking".to_owned(),
                constant_array(2, 0.9),
            ),
            ("reverse isolation".to_owned(), constant_array(2, 0.0)),
            ("reverse switch term".to_owned(), constant_array(2, 0.0)),
            ("k".to_owned(), constant_array(2, 1.1)),
        ]),
    };
    let measured = actuals
        .iter()
        .map(|actual| seed.embed(actual))
        .collect::<rust_rf::Result<Vec<_>>>()
        .expect("standards should embed");
    let ideals = vec![line, ideal_reflect1, ideal_reflect2, ideal_match];
    let mut calibration = Lrrm::new(
        measured.clone(),
        ideals.clone(),
        50.0,
        LrrmMatchFit::PerFrequency,
    )
    .expect("LRRM standards");

    calibration.run().expect("LRRM calibration should solve");

    for value in calibration
        .solved_inductance
        .as_ref()
        .expect("inductance should be solved")
    {
        assert_relative_eq!(*value, 20.0e-12, max_relative = 1.0e-3);
    }

    let dut = two_port(frequency, [[0.2, 0.1], [0.6, -0.1]]);
    let raw = seed.embed(&dut).expect("DUT should embed");
    let corrected = calibration.apply(&raw).expect("DUT should calibrate");
    for (actual, expected) in corrected.s.iter().zip(dut.s.iter()) {
        assert_complex_close(*actual, *expected);
    }

    for fit in [
        LrrmMatchFit::Inductance,
        LrrmMatchFit::InductanceCapacitance,
    ] {
        let mut fitted =
            Lrrm::new(measured.clone(), ideals.clone(), 50.0, fit).expect("fitted LRRM standards");
        fitted.run().expect("fitted LRRM should solve");
        let fitted_corrected = fitted.apply(&raw).expect("fitted LRRM should calibrate");
        for (actual, expected) in fitted_corrected.s.iter().zip(dut.s.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-7);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-7);
        }
        for value in fitted
            .solved_inductance
            .as_ref()
            .expect("fitted inductance")
        {
            assert_relative_eq!(*value, 20.0e-12, max_relative = 1.0e-3);
        }
    }
}

fn sixteen_term_coefficients(points: usize) -> BTreeMap<String, ndarray::Array1<Complex64>> {
    BTreeMap::from([
        (
            "forward directivity".to_owned(),
            constant_array(points, 0.01),
        ),
        (
            "reverse directivity".to_owned(),
            constant_array(points, 0.02),
        ),
        (
            "forward source match".to_owned(),
            constant_array(points, 0.03),
        ),
        (
            "reverse source match".to_owned(),
            constant_array(points, 0.04),
        ),
        (
            "forward reflection tracking".to_owned(),
            constant_array(points, 0.8),
        ),
        (
            "reverse reflection tracking".to_owned(),
            constant_array(points, 0.9),
        ),
        ("k".to_owned(), constant_array(points, 1.1)),
        (
            "forward isolation".to_owned(),
            constant_array(points, 0.001),
        ),
        (
            "reverse isolation".to_owned(),
            constant_array(points, 0.002),
        ),
        (
            "forward port 1 isolation".to_owned(),
            constant_array(points, 0.003),
        ),
        (
            "reverse port 1 isolation".to_owned(),
            constant_array(points, 0.004),
        ),
        (
            "forward port 2 isolation".to_owned(),
            constant_array(points, 0.005),
        ),
        (
            "reverse port 2 isolation".to_owned(),
            constant_array(points, 0.006),
        ),
        (
            "forward port isolation".to_owned(),
            constant_array(points, 0.007),
        ),
        (
            "reverse port isolation".to_owned(),
            constant_array(points, 0.008),
        ),
        (
            "forward switch term".to_owned(),
            constant_array(points, 0.0),
        ),
        (
            "reverse switch term".to_owned(),
            constant_array(points, 0.0),
        ),
    ])
}

fn eight_term_seed(ideals: Vec<Network>) -> EightTerm {
    let points = ideals[0].frequency_points();
    EightTerm {
        measured: ideals.clone(),
        ideals,
        coefficients: BTreeMap::from([
            (
                "forward directivity".to_owned(),
                constant_array(points, 0.01),
            ),
            (
                "forward source match".to_owned(),
                constant_array(points, 0.03),
            ),
            (
                "forward reflection tracking".to_owned(),
                constant_array(points, 0.8),
            ),
            ("forward isolation".to_owned(), constant_array(points, 0.0)),
            (
                "forward switch term".to_owned(),
                constant_array(points, 0.0),
            ),
            (
                "reverse directivity".to_owned(),
                constant_array(points, 0.02),
            ),
            (
                "reverse source match".to_owned(),
                constant_array(points, 0.04),
            ),
            (
                "reverse reflection tracking".to_owned(),
                constant_array(points, 0.9),
            ),
            ("reverse isolation".to_owned(), constant_array(points, 0.0)),
            (
                "reverse switch term".to_owned(),
                constant_array(points, 0.0),
            ),
            ("k".to_owned(), constant_array(points, 1.1)),
        ]),
    }
}

fn embed_value(
    ideal: Complex64,
    directivity: Complex64,
    tracking: Complex64,
    source_match: Complex64,
) -> Complex64 {
    let a = tracking - directivity * source_match;
    (directivity + a * ideal) / (Complex64::new(1.0, 0.0) - source_match * ideal)
}

fn constant_one_port(frequency: Frequency, value: Complex64) -> Network {
    let points = frequency.points();
    one_port(frequency, &vec![value; points])
}

fn one_port(frequency: Frequency, values: &[Complex64]) -> Network {
    let points = frequency.points();
    let s = Array3::from_shape_fn((points, 1, 1), |(point, _, _)| values[point]);
    Network::new(
        frequency,
        s,
        Array2::from_elem((points, 1), Complex64::new(50.0, 0.0)),
    )
    .expect("network should be valid")
}

fn two_port(frequency: Frequency, values: [[f64; 2]; 2]) -> Network {
    let points = frequency.points();
    Network::new(
        frequency,
        Array3::from_shape_fn((points, 2, 2), |(_, output, input)| {
            Complex64::new(values[output][input], 0.0)
        }),
        Array2::from_elem((points, 2), Complex64::new(50.0, 0.0)),
    )
    .expect("network should be valid")
}

fn complex_two_port(frequency: Frequency, values: [[Complex64; 2]; 2]) -> Network {
    let points = frequency.points();
    Network::new(
        frequency,
        Array3::from_shape_fn((points, 2, 2), |(_, output, input)| values[output][input]),
        Array2::from_elem((points, 2), Complex64::new(50.0, 0.0)),
    )
    .expect("complex network should be valid")
}

fn constant_array(points: usize, value: f64) -> ndarray::Array1<Complex64> {
    ndarray::Array1::from_elem(points, Complex64::new(value, 0.0))
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-10);
    assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-10);
}
