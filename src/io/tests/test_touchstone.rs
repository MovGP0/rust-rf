use std::fs::File;
use std::path::PathBuf;

use num_complex::Complex64;
use rust_rf::Network;
use rust_rf::io::{Touchstone, TouchstoneFormat, TouchstoneParameter};

const TOLERANCE: f64 = 1.0e-12;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn reads_real_imaginary_touchstone_data() {
    let touchstone =
        Touchstone::from_path(fixture("simple_touchstone.s2p")).expect("fixture should parse");

    assert_eq!(touchstone.rank, 2);
    assert_eq!(touchstone.format, TouchstoneFormat::RealImaginary);
    assert_eq!(
        touchstone.frequencies_hz().as_slice(),
        Some(&[1.0e9, 1.1e9][..])
    );
    assert_complex_close(touchstone.resistance, Complex64::new(50.0, 50.0));
    let s = touchstone.s_parameters();
    assert_complex_close(s[(0, 0, 0)], Complex64::new(1.0, 2.0));
    assert_complex_close(s[(0, 1, 0)], Complex64::new(3.0, 4.0));
    assert_complex_close(s[(0, 0, 1)], Complex64::new(5.0, 6.0));
    assert_complex_close(s[(0, 1, 1)], Complex64::new(7.0, 8.0));
    assert_complex_close(s[(1, 0, 0)], Complex64::new(9.0, 10.0));
    assert_eq!(
        touchstone.comments_after_option_line[0],
        "freq ReS11 ImS11 ReS21 ImS21 ReS12 ImS12 ReS22 ImS22"
    );
}

#[test]
fn retains_the_first_of_multiple_option_lines() {
    let touchstone =
        Touchstone::from_path(fixture("double_option_line.s2p")).expect("fixture should parse");
    assert_complex_close(touchstone.resistance, Complex64::new(10.0, 10.0));
}

#[test]
fn reads_from_a_stream() {
    let file = File::open(fixture("simple_touchstone.s2p")).expect("fixture should open");
    let touchstone = Touchstone::from_reader(file, 2).expect("stream should parse");
    assert_eq!(
        touchstone.frequencies_hz().as_slice(),
        Some(&[1.0e9, 1.1e9][..])
    );
    assert_complex_close(
        touchstone.s_parameters()[(1, 1, 1)],
        Complex64::new(15.0, 16.0),
    );
}

#[test]
fn converts_s_parameter_data_formats() {
    let touchstone =
        Touchstone::from_path(fixture("simple_touchstone.s2p")).expect("fixture should parse");
    let real_imaginary = touchstone.s_parameter_data(TouchstoneFormat::RealImaginary);
    assert_eq!(real_imaginary["S11R"].as_slice(), Some(&[1.0, 9.0][..]));
    assert_eq!(real_imaginary["S21I"].as_slice(), Some(&[4.0, 12.0][..]));

    let decibel = touchstone.s_parameter_data(TouchstoneFormat::DecibelAngle);
    assert!((decibel["S11DB"][0] - 20.0 * 5.0_f64.sqrt().log10()).abs() <= TOLERANCE);
    assert!((decibel["S11A"][0] - Complex64::new(1.0, 2.0).arg().to_degrees()).abs() <= TOLERANCE);
}

#[test]
fn exposes_touchstone_comments_variables_format_names_and_gamma_z0() {
    let text = b"! Created with skrf\n! width = 2.54 mm\n! operator note\n# GHz S RI R 50\n1 0 0\n";
    let touchstone = Touchstone::from_reader(&text[..], 1).expect("metadata fixture should parse");
    assert_eq!(
        touchstone.comments_excluding(&["Created with skrf"]),
        "width = 2.54 mm\noperator note"
    );
    assert_eq!(
        touchstone.comment_variables()["width"],
        ("2.54".to_owned(), "mm".to_owned())
    );
    assert_eq!(touchstone.format_description(None), "GHz S RI R 50+0i");
    assert_eq!(
        touchstone.format_description(Some(TouchstoneFormat::DecibelAngle)),
        "Hz S DB R 50+0i"
    );
    assert_eq!(
        touchstone.s_parameter_names(TouchstoneFormat::RealImaginary),
        vec!["S11I", "S11R", "frequency"]
    );
    assert_eq!(touchstone.gamma_z0(), (None, None));

    let hfss = Touchstone::from_path(fixture("ansys_modal_data.s2p"))
        .expect("HFSS modal fixture should parse");
    let (gamma, z0) = hfss.gamma_z0();
    assert!(gamma.is_some());
    assert!(z0.is_some());
}

#[test]
fn decodes_windows_1252_touchstone_comments() {
    let bytes = b"! 50 \x80 fixture\n# GHz S RI R 50\n1 0 0\n";
    let touchstone =
        Touchstone::from_reader(&bytes[..], 1).expect("Windows-1252 Touchstone should parse");
    assert_eq!(touchstone.comments, vec!["50 € fixture"]);
}

#[test]
fn reads_magnitude_angle_and_decibel_angle_data() {
    let magnitude_angle = b"# MHz S MA R 50\n1 2 90\n";
    let touchstone = Touchstone::from_reader(&magnitude_angle[..], 1)
        .expect("magnitude-angle data should parse");
    assert_eq!(touchstone.frequencies_hz()[0], 1.0e6);
    assert_complex_close(
        touchstone.s_parameters()[(0, 0, 0)],
        Complex64::new(0.0, 2.0),
    );

    let decibel_angle = b"# Hz S DB R 50\n10 20 -90\n";
    let touchstone =
        Touchstone::from_reader(&decibel_angle[..], 1).expect("decibel-angle data should parse");
    assert_complex_close(
        touchstone.s_parameters()[(0, 0, 0)],
        Complex64::new(0.0, -10.0),
    );
}

#[test]
fn constructs_network_from_touchstone() {
    let network =
        Network::read_touchstone(fixture("simple_touchstone.s2p")).expect("network should parse");
    assert_eq!(network.ports(), 2);
    assert_eq!(network.frequency_points(), 2);
    assert_eq!(network.z0.dim(), (2, 2));
    assert_complex_close(network.z0[(0, 0)], Complex64::new(50.0, 50.0));
    assert_eq!(network.name.as_deref(), Some("simple_touchstone"));
}

#[test]
fn reads_touchstone_two_keywords_ports_and_references() {
    let touchstone =
        Touchstone::from_path(fixture("ansys.ts")).expect("Touchstone 2 fixture should parse");
    assert_eq!(touchstone.version, "2.0");
    assert_eq!(touchstone.rank, 3);
    assert_eq!(
        touchstone.port_names,
        vec![
            "U29_B6_1024G_EAS3QB_A_DBI.37.FD_0-1",
            "U29_B6_1024G_EAS3QB_A_DBI.38.GND",
            "U40_178BGA.E10.FD_0-1"
        ]
    );
    assert_eq!(
        touchstone.reference_impedances,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(50.0, 0.0),
            Complex64::new(50.0, 0.0)
        ]
    );
    let network = touchstone.network().expect("network should be constructed");
    assert_eq!(network.port_names, touchstone.port_names);
    assert_complex_close(network.z0[(0, 0)], Complex64::new(1.0, 0.0));
    assert_complex_close(network.z0[(0, 2)], Complex64::new(50.0, 0.0));
}

#[test]
fn expands_touchstone_two_triangular_matrix_data() {
    let lower = b"[Version] 2.0\n# Hz S RI R 50\n[Number of Ports] 2\n[Number of Frequencies] 1\n[Matrix Format] Lower\n[Network Data]\n1 1 0 2 0 3 0\n[End]\n";
    let touchstone =
        Touchstone::from_reader(&lower[..], 0).expect("lower-matrix Touchstone data should parse");
    let s = touchstone.s_parameters();
    assert_complex_close(s[(0, 0, 0)], Complex64::new(1.0, 0.0));
    assert_complex_close(s[(0, 1, 0)], Complex64::new(2.0, 0.0));
    assert_complex_close(s[(0, 0, 1)], Complex64::new(2.0, 0.0));
    assert_complex_close(s[(0, 1, 1)], Complex64::new(3.0, 0.0));
}

#[test]
fn reads_hfss_per_frequency_impedance_and_propagation_data() {
    let touchstone = Touchstone::from_path(fixture("ansys_modal_data.s2p"))
        .expect("HFSS modal fixture should parse");
    let impedance = touchstone
        .port_impedances
        .as_ref()
        .expect("HFSS impedance should be present");
    assert_complex_close(impedance[(0, 0)], Complex64::new(51.0, 1.0));
    assert_complex_close(impedance[(0, 1)], Complex64::new(52.0, 2.0));
    assert_complex_close(impedance[(1, 0)], Complex64::new(61.0, 11.0));
    assert_complex_close(impedance[(1, 1)], Complex64::new(62.0, 12.0));
    let gamma = touchstone
        .propagation_constants
        .as_ref()
        .expect("HFSS propagation constants should be present");
    assert_complex_close(gamma[(0, 0)], Complex64::new(0.00653730315138823, 0.0));
    assert_complex_close(gamma[(0, 1)], Complex64::new(0.00654089320037521, 0.0));

    let network = touchstone.network().expect("network should be constructed");
    assert_complex_close(network.z0[(1, 0)], Complex64::new(61.0, 11.0));
}

#[test]
fn reads_touchstone_noise_data_and_two_port_order() {
    let touchstone =
        Touchstone::from_path(fixture("noise.ts")).expect("Touchstone noise fixture should parse");
    let noise = touchstone
        .noise
        .as_ref()
        .expect("noise data should be present");
    assert_eq!(noise.dim(), (2, 5));
    assert_eq!(noise[(0, 0)], 4.0e9);
    assert_eq!(noise[(0, 1)], 0.7);
    assert_eq!(noise[(1, 4)], 20.0);
    assert_complex_close(
        touchstone.s_parameters()[(0, 1, 0)],
        Complex64::from_polar(3.57, 157.0_f64.to_radians()),
    );
    assert_complex_close(
        touchstone.s_parameters()[(0, 0, 1)],
        Complex64::from_polar(0.04, 76.0_f64.to_radians()),
    );
    assert_eq!(
        touchstone.reference_impedances,
        vec![Complex64::new(50.0, 0.0), Complex64::new(25.0, 0.0)]
    );
    let network = touchstone.network().expect("network should retain noise");
    let network_noise = network.noise.expect("typed noise should be attached");
    assert_eq!(network_noise.frequency.values_hz()[0], 4.0e9);
    assert_eq!(network_noise.minimum_noise_figure_db[0], 0.7);
    assert_eq!(network_noise.equivalent_noise_resistance[1], 20.0);
}

#[test]
fn converts_impedance_and_admittance_parameter_files_to_scattering() {
    let impedance = b"[Version] 2.0\n# Hz Z RI R 50\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 50 0\n[End]\n";
    let touchstone =
        Touchstone::from_reader(&impedance[..], 0).expect("impedance-parameter data should parse");
    assert_eq!(touchstone.parameter, TouchstoneParameter::Impedance);
    assert_complex_close(
        touchstone.s_parameters()[(0, 0, 0)],
        Complex64::new(0.0, 0.0),
    );

    let admittance = b"[Version] 2.0\n# Hz Y RI R 50\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0.02 0\n[End]\n";
    let touchstone = Touchstone::from_reader(&admittance[..], 0)
        .expect("admittance-parameter data should parse");
    assert_eq!(touchstone.parameter, TouchstoneParameter::Admittance);
    assert_complex_close(
        touchstone.s_parameters()[(0, 0, 0)],
        Complex64::new(0.0, 0.0),
    );
}

#[test]
fn converts_hybrid_and_inverse_hybrid_parameter_files_to_scattering() {
    let impedance = b"[Version] 2.0\n# Hz Z RI R 50\n[Number of Ports] 2\n[Number of Frequencies] 1\n[Network Data]\n1 75 0 20 0 30 0 80 0\n[End]\n";
    let expected = Touchstone::from_reader(&impedance[..], 0)
        .expect("comparison impedance-parameter data should parse");
    let hybrid = b"[Version] 2.0\n# Hz H RI R 50\n[Number of Ports] 2\n[Number of Frequencies] 1\n[Network Data]\n1 67.5 0 0.25 0 -0.375 0 0.0125 0\n[End]\n";
    let touchstone =
        Touchstone::from_reader(&hybrid[..], 0).expect("hybrid-parameter data should parse");
    assert_eq!(touchstone.parameter, TouchstoneParameter::Hybrid);
    for row in 0..2 {
        for column in 0..2 {
            assert_complex_close(
                touchstone.s_parameters()[(0, row, column)],
                expected.s_parameters()[(0, row, column)],
            );
        }
    }

    let admittance = b"[Version] 2.0\n# Hz Y RI R 50\n[Number of Ports] 2\n[Number of Frequencies] 1\n[Network Data]\n1 0.03 0 -0.008 0 -0.012 0 0.025 0\n[End]\n";
    let expected = Touchstone::from_reader(&admittance[..], 0)
        .expect("comparison admittance-parameter data should parse");
    let inverse_hybrid = b"[Version] 2.0\n# Hz G RI R 50\n[Number of Ports] 2\n[Number of Frequencies] 1\n[Network Data]\n1 0.02616 0 -0.32 0 0.48 0 40 0\n[End]\n";
    let touchstone = Touchstone::from_reader(&inverse_hybrid[..], 0)
        .expect("inverse-hybrid-parameter data should parse");
    assert_eq!(touchstone.parameter, TouchstoneParameter::InverseHybrid);
    for row in 0..2 {
        for column in 0..2 {
            assert_complex_close(
                touchstone.s_parameters()[(0, row, column)],
                expected.s_parameters()[(0, row, column)],
            );
        }
    }
}

#[test]
fn applies_touchstone_one_normalization_before_parameter_conversion() {
    let normalized_impedance = b"# Hz Z RI R 50\n1 1 0\n";
    let touchstone = Touchstone::from_reader(&normalized_impedance[..], 1)
        .expect("normalized impedance data should parse");
    assert_complex_close(
        touchstone.s_parameters()[(0, 0, 0)],
        Complex64::new(0.0, 0.0),
    );
}

#[test]
fn rejects_hybrid_parameters_for_non_two_port_files() {
    let hybrid = b"[Version] 2.0\n# Hz H RI R 50\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 1 0\n[End]\n";
    let error = Touchstone::from_reader(&hybrid[..], 0)
        .expect_err("one-port hybrid data should be rejected");
    assert!(error.to_string().contains("2x2"));
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert!((actual.re - expected.re).abs() <= TOLERANCE);
    assert!((actual.im - expected.im).abs() <= TOLERANCE);
}
