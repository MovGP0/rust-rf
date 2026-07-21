use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use approx::assert_relative_eq;
use ndarray::{Array2, Array3, array};
use num_complex::Complex64;
use rust_rf::network::{
    abcd_to_s, active_s, active_vswr, active_y, active_z, average, cascade_list, concatenate_ports,
    flip_ports, g_to_s, h_to_s, h_to_z, one_ports_to_nport, overlap, parallel_connect, passivity,
    reciprocity, s_to_abcd, s_to_g, s_to_h, s_to_t, s_to_y, s_to_z, scattering_standard_deviation,
    stitch, t_to_s, two_port_measurements_to_nport, two_port_reflect, two_port_to_nport, y_to_s,
    z_to_h, z_to_s,
};
use rust_rf::{
    Frequency, FrequencyUnit, InterpolationMode, Network, SParameterDefinition, SweepType,
};

const TOLERANCE: f64 = 1.0e-10;

#[test]
fn validates_network_shapes() {
    let frequency = Frequency::new(1.0, 2.0, 2, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    assert!(
        Network::new(
            frequency.clone(),
            Array3::zeros((1, 2, 2)),
            Array2::zeros((1, 2))
        )
        .is_err()
    );
    assert!(Network::new(frequency, Array3::zeros((2, 2, 1)), Array2::zeros((2, 2))).is_err());
}

#[test]
fn linearly_interpolates_s_parameters_and_impedance() {
    let frequency = Frequency::from_hz(array![1.0, 3.0]).expect("frequency should be valid");
    let s = Array3::from_shape_fn((2, 1, 1), |(point, _, _)| {
        Complex64::new(2.0 * point as f64, -2.0 * point as f64)
    });
    let z0 = Array2::from_shape_fn((2, 1), |(point, _)| {
        Complex64::new(50.0 + 10.0 * point as f64, 0.0)
    });
    let network = Network::new(frequency, s, z0).expect("network should be valid");
    let target = Frequency::from_hz(array![1.0, 2.0, 3.0]).expect("target should be valid");

    let interpolated = network
        .interpolate(&target)
        .expect("interpolation should succeed");
    assert_complex_close(interpolated.s[(1, 0, 0)], Complex64::new(1.0, -1.0));
    assert_complex_close(interpolated.z0[(1, 0)], Complex64::new(55.0, 0.0));
    let polar = network
        .interpolate_with_mode(&target, InterpolationMode::PolarLinear)
        .expect("polar interpolation should succeed");
    assert_relative_eq!(
        polar.s[(1, 0, 0)].norm(),
        2.0_f64.sqrt(),
        epsilon = TOLERANCE
    );
    let rational = network
        .interpolate_with_mode(&target, InterpolationMode::Rational { degree: 1 })
        .expect("rational interpolation should succeed");
    assert_complex_close(rational.s[(1, 0, 0)], Complex64::new(1.0, -1.0));
    let cubic = network
        .interpolate_with_mode(&target, InterpolationMode::Cubic)
        .expect("cubic interpolation should succeed");
    assert_complex_close(cubic.s[(1, 0, 0)], Complex64::new(1.0, -1.0));
}

#[test]
fn cascades_and_inverts_two_port_networks() {
    let first = matched_two_port(Complex64::new(0.8, 0.1));
    let second = matched_two_port(Complex64::new(0.6, -0.2));
    let cascaded = first.cascade(&second).expect("cascade should succeed");
    assert_complex_close(
        cascaded.s[(0, 1, 0)],
        first.s[(0, 1, 0)] * second.s[(0, 1, 0)],
    );
    assert_complex_close(
        cascaded.s[(0, 0, 1)],
        first.s[(0, 0, 1)] * second.s[(0, 0, 1)],
    );

    let identity = first
        .cascade(&first.inverse().expect("inverse should exist"))
        .expect("inverse cascade should succeed");
    assert_complex_close(identity.s[(0, 0, 0)], Complex64::new(0.0, 0.0));
    assert_complex_close(identity.s[(0, 1, 1)], Complex64::new(0.0, 0.0));
    assert_complex_close(identity.s[(0, 1, 0)], Complex64::new(1.0, 0.0));
    assert_complex_close(identity.s[(0, 0, 1)], Complex64::new(1.0, 0.0));
}

#[test]
fn performs_elementwise_arithmetic_power_and_deembedding() {
    let left = matched_two_port(Complex64::new(0.8, 0.1));
    let right = matched_two_port(Complex64::new(0.5, -0.2));

    assert_complex_close(
        left.add_elementwise(&right)
            .expect("addition should succeed")
            .s[(0, 1, 0)],
        left.s[(0, 1, 0)] + right.s[(0, 1, 0)],
    );
    assert_complex_close(
        left.subtract_elementwise(&right)
            .expect("subtraction should succeed")
            .s[(0, 1, 0)],
        left.s[(0, 1, 0)] - right.s[(0, 1, 0)],
    );
    assert_complex_close(
        left.multiply_elementwise(&right)
            .expect("multiplication should succeed")
            .s[(0, 1, 0)],
        left.s[(0, 1, 0)] * right.s[(0, 1, 0)],
    );
    assert_complex_close(
        left.divide_elementwise(&right)
            .expect("division should succeed")
            .s[(0, 1, 0)],
        left.s[(0, 1, 0)] / right.s[(0, 1, 0)],
    );
    assert_complex_close(
        left.elementwise_power(2.0).expect("power should succeed").s[(0, 1, 0)],
        left.s[(0, 1, 0)].powf(2.0),
    );

    let dut = matched_two_port(Complex64::new(0.7, 0.05));
    let measured = left
        .cascade(&dut)
        .and_then(|network| network.cascade(&right))
        .expect("fixture cascade should succeed");
    let restored = measured
        .deembed(&left, Some(&right))
        .expect("deembedding should succeed");
    assert_parameter_matrices_close(&restored.s, &dut.s);
}

#[test]
fn writes_touchstone_that_can_be_read_back() {
    let network = matched_two_port(Complex64::new(0.75, -0.25));
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should follow the Unix epoch")
        .as_nanos();
    let temporary_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".temp");
    let temporary_directory =
        temporary_root.join(format!("network-roundtrip-{}-{unique}", std::process::id()));
    fs::create_dir_all(&temporary_directory).expect("temporary directory should be created");
    let path = temporary_directory.join("network.s2p");
    network
        .write_touchstone(&path)
        .expect("Touchstone write should succeed");
    let restored = Network::read_touchstone(&path).expect("Touchstone read should succeed");

    assert_eq!(restored.frequency, network.frequency);
    for (actual, expected) in restored.s.iter().zip(network.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
    fs::remove_file(path).expect("temporary Touchstone file should be removed");
    fs::remove_dir(temporary_directory).expect("empty temporary directory should be removed");
    let _ = fs::remove_dir(temporary_root);
}

#[test]
fn round_trips_impedance_and_admittance_for_all_wave_definitions() {
    let scattering = representative_scattering();
    let reference = Array2::from_shape_vec(
        (1, 2),
        vec![Complex64::new(50.0, 5.0), Complex64::new(75.0, -3.0)],
    )
    .expect("shape should be valid");

    for definition in [
        SParameterDefinition::Power,
        SParameterDefinition::Pseudo,
        SParameterDefinition::Traveling,
    ] {
        let impedance =
            s_to_z(&scattering, &reference, definition).expect("S to Z conversion should succeed");
        let restored =
            z_to_s(&impedance, &reference, definition).expect("Z to S conversion should succeed");
        assert_parameter_matrices_close(&restored, &scattering);

        let admittance =
            s_to_y(&scattering, &reference, definition).expect("S to Y conversion should succeed");
        let restored =
            y_to_s(&admittance, &reference, definition).expect("Y to S conversion should succeed");
        assert_parameter_matrices_close(&restored, &scattering);
        let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency");
        assert_parameter_matrices_close(
            &Network::from_impedance(frequency.clone(), &impedance, reference.clone(), definition)
                .expect("network from impedance")
                .s,
            &scattering,
        );
        assert_parameter_matrices_close(
            &Network::from_admittance(frequency, &admittance, reference.clone(), definition)
                .expect("network from admittance")
                .s,
            &scattering,
        );
    }
}

#[test]
fn matches_one_port_impedance_formula_for_real_reference() {
    let scattering = Array3::from_shape_vec((1, 1, 1), vec![Complex64::new(0.2, 0.1)])
        .expect("shape should be valid");
    let reference = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
    let expected = reference[(0, 0)] * (Complex64::new(1.0, 0.0) + scattering[(0, 0, 0)])
        / (Complex64::new(1.0, 0.0) - scattering[(0, 0, 0)]);

    for definition in [
        SParameterDefinition::Power,
        SParameterDefinition::Pseudo,
        SParameterDefinition::Traveling,
    ] {
        let impedance =
            s_to_z(&scattering, &reference, definition).expect("conversion should succeed");
        assert_complex_close(impedance[(0, 0, 0)], expected);
    }
}

#[test]
fn renormalizes_and_restores_network_reference_impedance() {
    let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency should be valid");
    let original_reference = Array2::from_elem((1, 2), Complex64::new(50.0, 0.0));
    let mut network = Network::new(
        frequency,
        representative_scattering(),
        original_reference.clone(),
    )
    .expect("network should be valid");
    let original_scattering = network.s.clone();
    let new_reference = Array2::from_shape_vec(
        (1, 2),
        vec![Complex64::new(25.0, 2.0), Complex64::new(75.0, -4.0)],
    )
    .expect("shape should be valid");

    network
        .renormalize(new_reference, SParameterDefinition::Power)
        .expect("renormalization should succeed");
    network
        .renormalize(original_reference, SParameterDefinition::Power)
        .expect("reverse renormalization should succeed");

    assert_parameter_matrices_close(&network.s, &original_scattering);
}

#[test]
fn round_trips_scattering_transfer_parameters() {
    let scattering = representative_scattering();
    let transfer = s_to_t(&scattering).expect("S to T conversion should succeed");
    let s11 = scattering[(0, 0, 0)];
    let s12 = scattering[(0, 0, 1)];
    let s21 = scattering[(0, 1, 0)];
    let s22 = scattering[(0, 1, 1)];
    assert_complex_close(transfer[(0, 0, 0)], s12 - s11 * s22 / s21);
    assert_complex_close(transfer[(0, 0, 1)], s11 / s21);
    assert_complex_close(transfer[(0, 1, 0)], -s22 / s21);
    assert_complex_close(transfer[(0, 1, 1)], Complex64::new(1.0, 0.0) / s21);
    let restored = t_to_s(&transfer).expect("T to S conversion should succeed");
    assert_parameter_matrices_close(&restored, &scattering);
    assert!(s_to_t(&Array3::zeros((1, 1, 1))).is_err());
}

#[test]
fn round_trips_abcd_parameters() {
    let scattering = representative_scattering();
    let reference = Array2::from_elem((1, 2), Complex64::new(50.0, 0.0));
    let abcd = s_to_abcd(&scattering, &reference).expect("S to ABCD conversion should succeed");
    let restored = abcd_to_s(&abcd, &reference).expect("ABCD to S conversion should succeed");
    assert_parameter_matrices_close(&restored, &scattering);

    let unequal = Array2::from_shape_vec(
        (1, 2),
        vec![Complex64::new(50.0, 0.0), Complex64::new(75.0, 0.0)],
    )
    .expect("shape should be valid");
    assert!(s_to_abcd(&scattering, &unequal).is_err());
}

#[test]
fn connects_arbitrary_n_port_networks() {
    let splitter = ideal_three_way();
    let extension = matched_two_port(Complex64::new(1.0, 0.0));
    let connected = splitter
        .connect(1, &extension, 0)
        .expect("N-port connection should succeed");
    assert_eq!(connected.ports(), 3);
    for output in 0..3 {
        for input in 0..3 {
            let expected = if output == input {
                -1.0 / 3.0
            } else {
                2.0 / 3.0
            };
            assert_complex_close(
                connected.s[(0, output, input)],
                Complex64::new(expected, 0.0),
            );
        }
    }

    let left = matched_two_port_with_reference(50.0);
    let right = matched_two_port_with_reference(75.0);
    let mismatch = left
        .connect(1, &right, 0)
        .expect("unequal-reference connection should succeed");
    assert_complex_close(mismatch.s[(0, 0, 0)], Complex64::new(0.2, 0.0));
    assert_complex_close(mismatch.s[(0, 1, 1)], Complex64::new(-0.2, 0.0));
    assert_relative_eq!(
        mismatch.s[(0, 1, 0)].norm_sqr() + mismatch.s[(0, 0, 0)].norm_sqr(),
        1.0,
        epsilon = TOLERANCE
    );
}

#[test]
fn inner_connects_ports_of_one_network() {
    let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency should be valid");
    let mut s = Array3::zeros((1, 4, 4));
    s[(0, 0, 1)] = Complex64::new(1.0, 0.0);
    s[(0, 1, 0)] = Complex64::new(1.0, 0.0);
    s[(0, 2, 3)] = Complex64::new(1.0, 0.0);
    s[(0, 3, 2)] = Complex64::new(1.0, 0.0);
    let network = Network::new(
        frequency,
        s,
        Array2::from_elem((1, 4), Complex64::new(50.0, 0.0)),
    )
    .expect("network should be valid");
    let connected = network
        .inner_connect(1, 2)
        .expect("inner connection should succeed");
    assert_eq!(connected.ports(), 2);
    assert_complex_close(connected.s[(0, 0, 0)], Complex64::new(0.0, 0.0));
    assert_complex_close(connected.s[(0, 1, 1)], Complex64::new(0.0, 0.0));
    assert_complex_close(connected.s[(0, 1, 0)], Complex64::new(1.0, 0.0));
    assert_complex_close(connected.s[(0, 0, 1)], Complex64::new(1.0, 0.0));
}

#[test]
fn round_trips_hybrid_and_inverse_hybrid_parameters() {
    let scattering = representative_scattering();
    let reference = Array2::from_shape_vec(
        (1, 2),
        vec![Complex64::new(50.0, 5.0), Complex64::new(75.0, -3.0)],
    )
    .expect("shape should be valid");

    let impedance = s_to_z(&scattering, &reference, SParameterDefinition::Power)
        .expect("S to Z conversion should succeed");
    let hybrid = z_to_h(&impedance).expect("Z to H conversion should succeed");
    let restored_impedance = h_to_z(&hybrid).expect("H to Z conversion should succeed");
    assert_parameter_matrices_close(&restored_impedance, &impedance);

    let hybrid = s_to_h(&scattering, &reference, SParameterDefinition::Power)
        .expect("S to H conversion should succeed");
    let restored = h_to_s(&hybrid, &reference, SParameterDefinition::Power)
        .expect("H to S conversion should succeed");
    assert_parameter_matrices_close(&restored, &scattering);

    let inverse_hybrid = s_to_g(&scattering, &reference, SParameterDefinition::Power)
        .expect("S to G conversion should succeed");
    let restored = g_to_s(&inverse_hybrid, &reference, SParameterDefinition::Power)
        .expect("G to S conversion should succeed");
    assert_parameter_matrices_close(&restored, &scattering);
}

#[test]
fn calculates_passivity_and_reciprocity_metrics() {
    let mut scattering = Array3::zeros((1, 2, 2));
    scattering[(0, 0, 0)] = Complex64::new(0.3, 0.0);
    scattering[(0, 0, 1)] = Complex64::new(0.4, 0.0);
    scattering[(0, 1, 0)] = Complex64::new(0.0, 0.0);
    scattering[(0, 1, 1)] = Complex64::new(0.5, 0.0);

    let metric = passivity(&scattering).expect("passivity should be defined");
    assert_complex_close(metric[(0, 0, 0)], Complex64::new(0.3, 0.0));
    assert_complex_close(metric[(0, 0, 1)], Complex64::new(0.12_f64.sqrt(), 0.0));
    assert_complex_close(metric[(0, 1, 0)], Complex64::new(0.12_f64.sqrt(), 0.0));
    assert_complex_close(metric[(0, 1, 1)], Complex64::new(0.41_f64.sqrt(), 0.0));

    let metric = reciprocity(&scattering).expect("reciprocity should be defined");
    assert_relative_eq!(metric[(0, 0, 1)], 0.4, epsilon = TOLERANCE);
    assert_relative_eq!(metric[(0, 1, 0)], 0.4, epsilon = TOLERANCE);
    assert!(passivity(&Array3::zeros((1, 1, 1))).is_err());
    assert!(reciprocity(&Array3::zeros((1, 1, 1))).is_err());
}

#[test]
fn flips_even_port_scattering_matrices() {
    let scattering = Array3::from_shape_fn((1, 4, 4), |(_, row, column)| {
        Complex64::new((10 * row + column) as f64, 0.0)
    });
    let flipped = flip_ports(&scattering).expect("port flipping should succeed");
    assert_complex_close(flipped[(0, 0, 0)], scattering[(0, 2, 2)]);
    assert_complex_close(flipped[(0, 1, 3)], scattering[(0, 3, 1)]);
    assert!(flip_ports(&Array3::zeros((1, 3, 3))).is_err());
}

#[test]
fn calculates_active_network_parameters() {
    let scattering = Array3::from_shape_vec(
        (1, 2, 2),
        vec![
            Complex64::new(0.1, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.3, 0.0),
            Complex64::new(0.4, 0.0),
        ],
    )
    .expect("shape should be valid");
    let excitation = array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let reference = Array2::from_shape_vec(
        (1, 2),
        vec![Complex64::new(50.0, 0.0), Complex64::new(75.0, 0.0)],
    )
    .expect("shape should be valid");

    let active = active_s(&scattering, &excitation).expect("active S should succeed");
    assert_complex_close(active[(0, 0)], Complex64::new(0.5, 0.0));
    assert_complex_close(active[(0, 1)], Complex64::new(0.55, 0.0));

    let impedance =
        active_z(&scattering, &reference, &excitation).expect("active Z should succeed");
    assert_complex_close(impedance[(0, 0)], Complex64::new(150.0, 0.0));
    assert_complex_close(impedance[(0, 1)], Complex64::new(75.0 * 1.55 / 0.45, 0.0));
    let admittance =
        active_y(&scattering, &reference, &excitation).expect("active Y should succeed");
    assert_complex_close(admittance[(0, 0)], Complex64::new(1.0 / 150.0, 0.0));
    assert_complex_close(
        admittance[(0, 1)],
        Complex64::new(0.45 / (75.0 * 1.55), 0.0),
    );
    let vswr = active_vswr(&scattering, &excitation).expect("active VSWR should succeed");
    assert_relative_eq!(vswr[(0, 0)], 3.0, epsilon = TOLERANCE);
    assert_relative_eq!(vswr[(0, 1)], 1.55 / 0.45, epsilon = TOLERANCE);
}

#[test]
fn exposes_derived_parameter_component_and_time_properties() {
    let network = matched_two_port(Complex64::new(0.8, 0.1));
    assert_eq!(network.impedance().expect("impedance").dim(), (1, 2, 2));
    assert_eq!(network.admittance().expect("admittance").dim(), (1, 2, 2));
    assert_eq!(network.hybrid().expect("hybrid").dim(), (1, 2, 2));
    assert_eq!(
        network.inverse_hybrid().expect("inverse hybrid").dim(),
        (1, 2, 2)
    );
    assert_eq!(
        network
            .scattering_transfer()
            .expect("scattering transfer")
            .dim(),
        (1, 2, 2)
    );
    assert_eq!(network.abcd().expect("ABCD").dim(), (1, 2, 2));
    assert_eq!(
        network.s_magnitude()[(0, 1, 0)],
        network.s[(0, 1, 0)].norm()
    );
    assert_relative_eq!(
        network.s_phase_degrees()[(0, 1, 0)],
        network.s[(0, 1, 0)].arg().to_degrees(),
        epsilon = TOLERANCE
    );
    assert_eq!(network.s_real()[(0, 1, 0)], 0.8);
    assert_eq!(network.s_imaginary()[(0, 1, 0)], 0.1);
    assert_eq!(network.s_time().expect("time response").dim(), (1, 2, 2));
    assert!(network.is_passive(1.0e-12).expect("passivity"));
    assert!(network.is_reciprocal(1.0).expect("reciprocity"));
    assert!(network.is_symmetric(1.0e-12).expect("symmetry"));
    assert_eq!(
        network
            .scattering_error(&network)
            .expect("scattering error"),
        array![0.0]
    );
}

#[test]
fn round_trips_equal_impedance_mixed_mode_conversion() {
    let first = matched_two_port(Complex64::new(0.8, 0.1));
    let mut four_port = first
        .connect(1, &first, 0)
        .expect("networks should connect");
    // The connection above produces a two-port, so construct a deterministic four-port matrix.
    four_port.s = Array3::from_shape_fn((1, 4, 4), |(_, output, input)| {
        Complex64::new((output * 4 + input) as f64 / 100.0, 0.01 * input as f64)
    });
    four_port.z0 = Array2::from_elem((1, 4), Complex64::new(50.0, 0.0));
    four_port.port_modes = vec![rust_rf::network::PortMode::SingleEnded; 4];

    let mixed = four_port
        .single_ended_to_mixed_mode(2)
        .expect("mixed-mode conversion should succeed");
    assert_eq!(mixed.z0[(0, 0)], Complex64::new(100.0, 0.0));
    assert_eq!(mixed.z0[(0, 2)], Complex64::new(25.0, 0.0));
    let restored = mixed
        .mixed_mode_to_single_ended(2)
        .expect("single-ended conversion should succeed");
    assert_parameter_matrices_close(&restored.s, &four_port.s);
}

#[test]
fn calculates_group_delay_stability_subnetworks_crops_and_port_delay() {
    let frequency =
        Frequency::from_hz(array![1.0e9, 2.0e9, 3.0e9]).expect("frequency should be valid");
    let delay = 0.1e-9;
    let s = Array3::from_shape_fn((3, 2, 2), |(point, output, input)| {
        if output != input {
            Complex64::from_polar(
                0.5,
                -std::f64::consts::TAU * frequency.values_hz()[point] * delay,
            )
        } else {
            Complex64::new(0.1, 0.0)
        }
    });
    let network = Network::new(
        frequency,
        s,
        Array2::from_elem((3, 2), Complex64::new(50.0, 0.0)),
    )
    .expect("network should be valid");

    assert_relative_eq!(
        network.group_delay().expect("group delay")[(1, 1, 0)],
        delay,
        epsilon = 1.0e-12
    );
    assert_eq!(network.stability_factor().expect("stability").len(), 3);
    assert_eq!(
        network
            .maximum_stable_gain()
            .expect("maximum stable gain")
            .len(),
        3
    );
    assert_eq!(network.maximum_gain().expect("maximum gain").len(), 3);
    assert_eq!(network.unilateral_gain().expect("unilateral gain").len(), 3);
    let windowed = network
        .windowed(&rust_rf::time::Window::Hann, true)
        .expect("windowed network");
    assert_eq!(windowed.frequency, network.frequency);
    assert_eq!(
        network
            .impulse_response(1, 0, Some(&rust_rf::time::Window::Hann))
            .expect("impulse response")
            .1
            .len(),
        3
    );
    assert_eq!(
        network
            .step_response(1, 0, None)
            .expect("step response")
            .1
            .len(),
        3
    );
    assert_eq!(network.subnetwork(&[1]).expect("subnetwork").ports(), 1);
    assert_eq!(
        network
            .cropped(1.5e9, 2.5e9)
            .expect("cropped network")
            .frequency_points(),
        1
    );
    let delayed = network.delayed_port(0, 90.0).expect("port delay");
    assert_complex_close(
        delayed.s[(0, 1, 0)],
        network.s[(0, 1, 0)] * Complex64::new(0.0, -1.0),
    );
    assert_complex_close(
        network.rotated(90.0).expect("network rotation").s[(0, 1, 0)],
        network.s[(0, 1, 0)] * Complex64::new(0.0, -1.0),
    );
    assert_parameter_matrices_close(
        &network
            .with_added_polar_noise(0.0, 0.0, false)
            .expect("zero additive noise")
            .s,
        &network.s,
    );
    assert_parameter_matrices_close(
        &network
            .with_multiplicative_noise(0.0, 0.0)
            .expect("zero multiplicative noise")
            .s,
        &network.s,
    );
    assert_complex_close(
        network.nudged(1.0e-12).expect("nudge").s[(0, 0, 0)],
        network.s[(0, 0, 0)] + 1.0e-12,
    );
    assert_eq!(
        network
            .stability_circle(0, 181)
            .expect("stability circle")
            .dim(),
        (3, 181)
    );
    assert_eq!(
        network
            .gain_circle(1, -3.0, 181)
            .expect("gain circle")
            .dim(),
        (3, 181)
    );
    let mut noisy = network.clone();
    noisy
        .set_noise_parameters(
            network.frequency.clone(),
            array![1.0, 1.1, 1.2],
            array![
                Complex64::new(0.1, 0.0),
                Complex64::new(0.1, 0.0),
                Complex64::new(0.1, 0.0)
            ],
            array![5.0, 5.0, 5.0],
        )
        .expect("noise parameters");
    assert_eq!(
        noisy.minimum_noise_factor().expect("minimum noise").len(),
        3
    );
    assert_eq!(
        noisy
            .optimal_noise_impedance()
            .expect("optimal impedance")
            .len(),
        3
    );
    assert_eq!(
        noisy
            .noise_figure_circle(2.0, 181)
            .expect("noise circle")
            .dim(),
        (3, 181)
    );
}

#[test]
fn combines_overlaps_stitches_averages_and_assembles_reflections() {
    let line = matched_two_port(Complex64::new(0.8, 0.0));
    let cascaded = cascade_list(&[line.clone(), line.clone()]).expect("cascade list");
    assert_complex_close(cascaded.s[(0, 1, 0)], Complex64::new(0.64, 0.0));
    assert_eq!(
        concatenate_ports(&[line.clone(), line.clone()])
            .expect("port concatenation")
            .ports(),
        4
    );
    let parallel =
        parallel_connect(&[line.clone(), line.clone()], &[0, 0]).expect("parallel connection");
    assert_eq!(parallel.ports(), 2);
    assert!(parallel.s.iter().all(|value| value.is_finite()));

    let mut shifted = line.clone();
    shifted.s.mapv_inplace(|value| value + 0.2);
    let mean = average(&[line.clone(), shifted.clone()]).expect("average");
    assert_complex_close(
        mean.s[(0, 1, 0)],
        (line.s[(0, 1, 0)] + shifted.s[(0, 1, 0)]) / 2.0,
    );
    assert_relative_eq!(
        scattering_standard_deviation(&[line, shifted]).expect("standard deviation")[(0, 1, 0)],
        0.1,
        epsilon = TOLERANCE
    );

    let first_frequency = Frequency::from_hz(array![1.0, 2.0, 3.0]).expect("frequency");
    let second_frequency = Frequency::from_hz(array![2.0, 3.0, 4.0]).expect("frequency");
    let build = |frequency: Frequency| {
        let points = frequency.points();
        Network::new(
            frequency,
            Array3::from_elem((points, 1, 1), Complex64::new(0.25, 0.0)),
            Array2::from_elem((points, 1), Complex64::new(50.0, 0.0)),
        )
        .expect("one-port")
    };
    let first = build(first_frequency);
    let second = build(second_frequency);
    let (left, right) = overlap(&first, &second).expect("frequency overlap");
    assert_eq!(left.frequency.values_hz(), right.frequency.values_hz());
    assert_eq!(left.frequency_points(), 2);
    let third = build(Frequency::from_hz(array![4.0, 5.0]).expect("frequency"));
    assert_eq!(
        stitch(&first, &third)
            .expect("frequency stitch")
            .frequency_points(),
        5
    );
    let reflect = two_port_reflect(&left, Some(&right)).expect("two-port reflect");
    assert_eq!(reflect.ports(), 2);
    assert_eq!(reflect.s[(0, 0, 1)], Complex64::new(0.0, 0.0));
    assert_eq!(
        one_ports_to_nport(&[left, right])
            .expect("one-port assembly")
            .ports(),
        2
    );
    assert_eq!(
        two_port_to_nport(&reflect, 1, 3, 4)
            .expect("two-port embedding")
            .ports(),
        4
    );
    assert_eq!(
        two_port_measurements_to_nport(&[(0, 2, reflect)], 3)
            .expect("two-port measurement reconstruction")
            .ports(),
        3
    );
}

fn matched_two_port(transmission: Complex64) -> Network {
    let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency should be valid");
    let mut s = Array3::zeros((1, 2, 2));
    s[(0, 1, 0)] = transmission;
    s[(0, 0, 1)] = transmission;
    let z0 = Array2::from_elem((1, 2), Complex64::new(50.0, 0.0));
    Network::new(frequency, s, z0).expect("network should be valid")
}

fn matched_two_port_with_reference(reference: f64) -> Network {
    let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency should be valid");
    let mut s = Array3::zeros((1, 2, 2));
    s[(0, 1, 0)] = Complex64::new(1.0, 0.0);
    s[(0, 0, 1)] = Complex64::new(1.0, 0.0);
    Network::new(
        frequency,
        s,
        Array2::from_elem((1, 2), Complex64::new(reference, 0.0)),
    )
    .expect("network should be valid")
}

fn ideal_three_way() -> Network {
    let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency should be valid");
    let s = Array3::from_shape_fn((1, 3, 3), |(_, output, input)| {
        Complex64::new(
            if output == input {
                -1.0 / 3.0
            } else {
                2.0 / 3.0
            },
            0.0,
        )
    });
    Network::new(
        frequency,
        s,
        Array2::from_elem((1, 3), Complex64::new(50.0, 0.0)),
    )
    .expect("network should be valid")
}

fn representative_scattering() -> Array3<Complex64> {
    Array3::from_shape_vec(
        (1, 2, 2),
        vec![
            Complex64::new(0.1, 0.02),
            Complex64::new(0.2, -0.03),
            Complex64::new(0.7, 0.05),
            Complex64::new(0.05, -0.01),
        ],
    )
    .expect("shape should be valid")
}

fn assert_parameter_matrices_close(actual: &Array3<Complex64>, expected: &Array3<Complex64>) {
    assert_eq!(actual.dim(), expected.dim());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(actual.re, expected.re, epsilon = TOLERANCE);
    assert_relative_eq!(actual.im, expected.im, epsilon = TOLERANCE);
}
