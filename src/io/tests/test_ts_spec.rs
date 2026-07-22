//! Regressions based on the official Touchstone specification examples.

use std::path::PathBuf;

use approx::assert_relative_eq;
use rust_rf::io::{Touchstone, TouchstoneFormat, TouchstoneParameter, touchstone_string};
use rust_rf::{Network, PortMode};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ts-spec")
        .join(name)
}

/// Parses every official example fixture and verifies its network rank.
#[test]
fn parses_every_official_touchstone_example_fixture() {
    let examples = [
        ("ex_1.ts", 4),
        ("ex_2.ts", 1),
        ("ex_3.ts", 2),
        ("ex_4.ts", 4),
        ("ex_5.ts", 4),
        ("ex_6.ts", 4),
        ("ex_7.ts", 1),
        ("ex_8.s1p", 1),
        ("ex_9.s1p", 1),
        ("ex_10.ts", 1),
        ("ex_11.s2p", 2),
        ("ex_12.ts", 2),
        ("ex_12_g.ts", 2),
        ("ex_13.s2p", 2),
        ("ex_14.s4p", 4),
        ("ex_16.ts", 6),
        ("ex_17.ts", 2),
        ("ex_18.s2p", 2),
    ];
    for (name, ports) in examples {
        let network = Network::read_touchstone(fixture(name))
            .unwrap_or_else(|error| panic!("{name} should parse: {error}"));
        assert_eq!(network.ports(), ports, "{name}");
        if name != "ex_1.ts" {
            assert!(network.frequency_points() > 0, "{name}");
        }
    }
}

/// Verifies reference impedances, representative matrix values, and noise data.
#[test]
fn reads_reference_impedances_matrix_values_and_noise() {
    let example_3 = Network::read_touchstone(fixture("ex_3.ts")).expect("example 3");
    assert_eq!(example_3.frequency.values_hz().to_vec(), vec![1e9, 2e9]);
    assert_relative_eq!(example_3.s[(0, 0, 0)].re, 111.0, epsilon = 1e-14);
    assert_relative_eq!(example_3.s[(1, 1, 1)].re, 222.0, epsilon = 1e-14);

    let example_4 = Network::read_touchstone(fixture("ex_4.ts")).expect("example 4");
    let references = example_4.z0.row(0).mapv(|value| value.re).to_vec();
    assert_eq!(references, vec![50.0, 75.0, 0.01, 0.01]);

    let example_17 = Network::read_touchstone(fixture("ex_17.ts")).expect("example 17");
    let noise = example_17.noise.expect("example 17 contains noise data");
    assert_eq!(noise.frequency.values_hz().to_vec(), vec![4e9, 18e9]);
    assert_eq!(noise.minimum_noise_figure_db.to_vec(), vec![0.7, 2.7]);
    assert_eq!(noise.equivalent_noise_resistance.to_vec(), vec![19.0, 20.0]);
}

/// Normalizes mixed-mode port ordering and its reference impedances.
#[test]
fn normalizes_touchstone_mixed_mode_order() {
    let network = Network::read_touchstone(fixture("ex_16.ts")).expect("example 16");
    assert_eq!(
        network.port_modes,
        vec![
            PortMode::SingleEnded,
            PortMode::Differential,
            PortMode::Common,
            PortMode::SingleEnded,
            PortMode::Differential,
            PortMode::Common,
        ]
    );
    let references = network.z0.row(0).mapv(|value| value.re).to_vec();
    let expected = [50.0, 150.0, 37.5, 50.0, 0.02, 0.005];
    for (actual, expected) in references.iter().zip(expected) {
        assert_relative_eq!(actual, &expected, epsilon = 1e-14);
    }
}

/// Round-trips every supported network-parameter and numeric-value format.
#[test]
fn round_trips_all_touchstone_parameter_and_value_formats() {
    let network = Network::read_touchstone(fixture("ex_12.ts")).expect("example 12");
    for parameter in [
        TouchstoneParameter::Scattering,
        TouchstoneParameter::Impedance,
        TouchstoneParameter::Admittance,
        TouchstoneParameter::Hybrid,
        TouchstoneParameter::InverseHybrid,
    ] {
        for format in [
            TouchstoneFormat::RealImaginary,
            TouchstoneFormat::MagnitudeAngle,
            TouchstoneFormat::DecibelAngle,
        ] {
            let text =
                touchstone_string(&network, parameter, format).expect("Touchstone should render");
            let restored = Touchstone::from_reader(text.as_bytes(), 2)
                .expect("rendered Touchstone should parse")
                .network()
                .expect("parsed Touchstone should build a Network");
            for (actual, expected) in restored.s.iter().zip(&network.s) {
                assert_relative_eq!(actual.re, expected.re, epsilon = 1e-10);
                assert_relative_eq!(actual.im, expected.im, epsilon = 1e-10);
            }
        }
    }
}
