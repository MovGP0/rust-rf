#![allow(unused_imports)]

use approx::assert_relative_eq;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use rust_rf::constants::{FREE_SPACE_PERMEABILITY, FREE_SPACE_PERMITTIVITY, SPEED_OF_LIGHT};
use rust_rf::math::db_to_nepers;
use rust_rf::math::set_random_seed;
use rust_rf::media::{
    AttenuationUnit, CircularWaveguide, Coaxial, Cpw, CpwCompatibilityMode, DefinedAEpTandZ0,
    DefinedCharacteristicImpedance, DefinedGammaZ0, DielectricDispersionModel, DistributedCircuit,
    Freespace, LengthUnit, Media, MicrostripDispersionModel, MicrostripLine,
    MicrostripQuasiStaticModel, RectangularWaveguide, WaveguideMode,
};
use rust_rf::{Frequency, FrequencyUnit, Network, SweepType};

#[test]
fn calculates_hammerstad_jensen_microstrip_characteristics() {
    let frequency = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let microstrip = MicrostripLine::new(
        frequency.clone(),
        3.0e-3,
        1.55e-3,
        Some(35.0e-6),
        4.413,
        1.0,
        0.0182,
        Some(1.7e-8),
        0.15e-6,
        DielectricDispersionModel::FrequencyInvariant,
        MicrostripQuasiStaticModel::HammerstadJensen,
        MicrostripDispersionModel::HammerstadJensen,
        CpwCompatibilityMode::Qucs,
        Some(Array1::from_elem(1, Complex64::new(50.0, 0.0))),
        None,
    )
    .expect("microstrip should be valid");
    let impedance = microstrip
        .characteristic_impedance()
        .expect("microstrip impedance should be defined");
    let (_, effective) = microstrip
        .frequency_dependent_characteristics()
        .expect("microstrip effective permittivity should be defined");
    assert!(((impedance[0].re - 49.142) / 49.142).abs() < 0.01);
    assert!(((effective[0].re - 3.324) / 3.324).abs() < 0.01);
    assert!(
        microstrip
            .line(25.0, LengthUnit::Millimeter)
            .expect("microstrip line should be constructed")
            .s[(0, 1, 0)]
            .norm()
            < 1.0
    );

    let zero_thickness = MicrostripLine::new(
        frequency,
        3.0e-3,
        1.55e-3,
        Some(0.0),
        4.413,
        1.0,
        0.0,
        None,
        0.0,
        DielectricDispersionModel::FrequencyInvariant,
        MicrostripQuasiStaticModel::HammerstadJensen,
        MicrostripDispersionModel::None,
        CpwCompatibilityMode::Qucs,
        None,
        None,
    )
    .expect("zero-thickness microstrip should be valid");
    let (_, _, effective_width) = zero_thickness
        .quasi_static_characteristics()
        .expect("zero-thickness quasi-static values should be defined");
    assert_relative_eq!(effective_width[0].re, 3.0e-3, epsilon = 1.0e-16);
    assert_eq!(
        zero_thickness
            .attenuation()
            .expect("zero-thickness attenuation should be defined")
            .0[0],
        0.0
    );
}

#[test]
fn matches_upstream_microstrip_quasi_static_models() {
    let cases = [
        (
            MicrostripQuasiStaticModel::Wheeler,
            50.25326993664797,
            3.2908553711645645,
        ),
        (
            MicrostripQuasiStaticModel::Schneider,
            49.91669071856352,
            3.2497653483627915,
        ),
        (
            MicrostripQuasiStaticModel::HammerstadJensen,
            50.250686925360405,
            3.174965337163942,
        ),
    ];
    for (model, expected_impedance, expected_effective) in cases {
        let line = reference_microstrip(model, MicrostripDispersionModel::None);
        let (impedance, effective, _) = line
            .quasi_static_characteristics()
            .expect("quasi-static analysis should succeed");
        assert_relative_eq!(impedance[0].re, expected_impedance, max_relative = 5.0e-8);
        assert_relative_eq!(effective[0].re, expected_effective, max_relative = 5.0e-8);
    }
}

#[test]
fn matches_upstream_microstrip_dispersion_models() {
    let cases = [
        (
            MicrostripDispersionModel::Schneider,
            [50.2417205841579, 49.460917331488425, 44.13716658488521],
            [3.176098672663985, 3.2771676602888085, 4.11541934109831],
        ),
        (
            MicrostripDispersionModel::HammerstadJensen,
            [50.32134063158393, 54.956763626732084, 64.00728762637556],
            [3.1796190181353805, 3.496027915825514, 4.178006987788865],
        ),
        (
            MicrostripDispersionModel::KirschningJansen,
            [50.23096812314026, 52.70273556614812, 80.14258050526186],
            [3.1884772169290407, 3.44507466274657, 4.1286772455509295],
        ),
        (
            MicrostripDispersionModel::Yamashita,
            [50.250686925360405; 3],
            [3.1895925742278344, 3.497212129448975, 4.1340488409114915],
        ),
        (
            MicrostripDispersionModel::Kobayashi,
            [50.250686925360405; 3],
            [3.188493392139607, 3.4516442451970297, 4.108645706642019],
        ),
    ];
    for (model, expected_impedance, expected_effective) in cases {
        let line = reference_microstrip(MicrostripQuasiStaticModel::HammerstadJensen, model);
        let (impedance, effective) = line
            .frequency_dependent_characteristics()
            .expect("dispersion analysis should succeed");
        for point in 0..3 {
            assert_relative_eq!(
                impedance[point].re,
                expected_impedance[point],
                max_relative = 5.0e-8
            );
            assert_relative_eq!(
                effective[point].re,
                expected_effective[point],
                max_relative = 5.0e-8
            );
        }
    }
}

#[test]
fn matches_mline_qucs_fixtures() {
    let fixtures = [
        (
            "mline,hammerstad,hammerstad.s2p",
            MicrostripQuasiStaticModel::HammerstadJensen,
            MicrostripDispersionModel::HammerstadJensen,
        ),
        (
            "mline,hammerstad,kirschning.s2p",
            MicrostripQuasiStaticModel::HammerstadJensen,
            MicrostripDispersionModel::KirschningJansen,
        ),
        (
            "mline,hammerstad,kobayashi.s2p",
            MicrostripQuasiStaticModel::HammerstadJensen,
            MicrostripDispersionModel::Kobayashi,
        ),
        (
            "mline,hammerstad,yamashita.s2p",
            MicrostripQuasiStaticModel::HammerstadJensen,
            MicrostripDispersionModel::Yamashita,
        ),
        (
            "mline,wheeler,schneider.s2p",
            MicrostripQuasiStaticModel::Wheeler,
            MicrostripDispersionModel::Schneider,
        ),
        (
            "mline,schneider,schneider.s2p",
            MicrostripQuasiStaticModel::Schneider,
            MicrostripDispersionModel::Schneider,
        ),
    ];
    for (name, quasi_static_model, dispersion_model) in fixtures {
        let reference = read_mline_fixture("qucs", name);
        let mut line = reference_mline(
            reference.frequency.clone(),
            DielectricDispersionModel::FrequencyInvariant,
            quasi_static_model,
            dispersion_model,
            CpwCompatibilityMode::Qucs,
        );
        let actual = line
            .line(25.0e-3, LengthUnit::Meter)
            .expect("microstrip line should be constructed");
        assert_network_ratio_within(&actual, &reference, 0.1, 1.0, 0.1, 1.0, name);
        line.set_resistivity_material("cu")
            .expect("copper should resolve");
        assert!(line.to_string().contains("Microstrip Media"));
    }
}

#[test]
fn matches_mline_ads_fixtures() {
    let fixtures = [
        (
            "mlin,freqencyinvariant,kirschning.s2p",
            DielectricDispersionModel::FrequencyInvariant,
            MicrostripDispersionModel::KirschningJansen,
        ),
        (
            "mlin,djordjevicsvensson,kirschning.s2p",
            djordjevic_svensson(),
            MicrostripDispersionModel::KirschningJansen,
        ),
        (
            "mlin,freqencyinvariant,kobayashi.s2p",
            DielectricDispersionModel::FrequencyInvariant,
            MicrostripDispersionModel::Kobayashi,
        ),
        (
            "mlin,djordjevicsvensson,kobayashi.s2p",
            djordjevic_svensson(),
            MicrostripDispersionModel::Kobayashi,
        ),
        (
            "mlin,freqencyinvariant,yamashita.s2p",
            DielectricDispersionModel::FrequencyInvariant,
            MicrostripDispersionModel::Yamashita,
        ),
        (
            "mlin,djordjevicsvensson,yamashita.s2p",
            djordjevic_svensson(),
            MicrostripDispersionModel::Yamashita,
        ),
    ];
    for (name, dielectric_model, dispersion_model) in fixtures {
        let reference = read_mline_fixture("ads", name);
        let line = reference_mline(
            reference.frequency.clone(),
            dielectric_model,
            MicrostripQuasiStaticModel::HammerstadJensen,
            dispersion_model,
            CpwCompatibilityMode::Native,
        );
        let actual = line
            .line(25.0e-3, LengthUnit::Meter)
            .expect("microstrip line should be constructed");
        assert_network_ratio_within(&actual, &reference, 1.0, 10.0, 0.1, 1.0, name);
    }
}

fn djordjevic_svensson() -> DielectricDispersionModel {
    DielectricDispersionModel::DjordjevicSvensson {
        low_frequency_hz: 1.0e3,
        high_frequency_hz: 1.0e12,
        specification_frequency_hz: 1.0e9,
    }
}

fn read_mline_fixture(simulator: &str, name: &str) -> Network {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/media/mline")
        .join(simulator)
        .join(name);
    Network::read_touchstone(path).expect("microstrip fixture should load")
}

fn reference_mline(
    frequency: Frequency,
    dielectric_model: DielectricDispersionModel,
    quasi_static_model: MicrostripQuasiStaticModel,
    dispersion_model: MicrostripDispersionModel,
    compatibility_mode: CpwCompatibilityMode,
) -> MicrostripLine {
    let points = frequency.points();
    MicrostripLine::new(
        frequency,
        3.00e-3,
        1.55e-3,
        Some(35.0e-6),
        4.413,
        1.0,
        0.0182,
        Some(1.7e-8),
        0.15e-6,
        dielectric_model,
        quasi_static_model,
        dispersion_model,
        compatibility_mode,
        Some(Array1::from_elem(points, Complex64::new(50.0, 0.0))),
        None,
    )
    .expect("reference microstrip should be valid")
}

fn assert_network_ratio_within(
    actual: &Network,
    expected: &Network,
    reflection_db: f64,
    reflection_degrees: f64,
    transmission_db: f64,
    transmission_degrees: f64,
    fixture: &str,
) {
    assert_eq!(actual.s.dim(), expected.s.dim());
    for point in 0..actual.frequency_points() {
        for output in 0..actual.ports() {
            for input in 0..actual.ports() {
                let ratio = actual.s[(point, output, input)] / expected.s[(point, output, input)];
                let db = (20.0 * ratio.norm().log10()).abs();
                let degrees = ratio.arg().to_degrees().abs();
                let (db_limit, degree_limit) = if output == input {
                    (reflection_db, reflection_degrees)
                } else {
                    (transmission_db, transmission_degrees)
                };
                assert!(
                    db < db_limit,
                    "{fixture} S{}{} point {point} dB residual {db}",
                    output + 1,
                    input + 1
                );
                assert!(
                    degrees < degree_limit,
                    "{fixture} S{}{} point {point} phase residual {degrees}",
                    output + 1,
                    input + 1
                );
            }
        }
    }
}

fn reference_microstrip(
    quasi_static_model: MicrostripQuasiStaticModel,
    dispersion_model: MicrostripDispersionModel,
) -> MicrostripLine {
    let frequency =
        Frequency::from_values(Array1::from_vec(vec![1.0, 10.0, 100.0]), FrequencyUnit::GHz)
            .expect("reference frequency should be valid");
    MicrostripLine::new(
        frequency,
        3.0e-3,
        1.55e-3,
        Some(35.0e-6),
        4.2,
        1.0,
        0.0,
        Some(1.7e-8),
        0.15e-6,
        DielectricDispersionModel::FrequencyInvariant,
        quasi_static_model,
        dispersion_model,
        CpwCompatibilityMode::Qucs,
        None,
        None,
    )
    .expect("reference microstrip should be valid")
}
