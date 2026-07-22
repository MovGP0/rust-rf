#![allow(unused_imports)]

//! Microstrip-line regressions against analytical, Qucs, and ADS references.

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
use rust_rf::{Frequency, FrequencyUnit, Network, Result, SweepType};

/// Checks Hammerstad-Jensen impedance and effective permittivity against the
/// [mcalc reference](http://web.mit.edu/~geda/arch/i386_rhel3/versions/20050830/html/mcalc-1.5/),
/// including finite- and zero-thickness loss behavior.
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
    assert_relative_eq!(
        zero_thickness
            .attenuation()
            .expect("zero-thickness attenuation should be defined")
            .0[0],
        0.0,
        epsilon = f64::EPSILON
    );
}

/// Checks all upstream quasi-static model reference values.
#[test]
fn matches_upstream_microstrip_quasi_static_models() -> Result<()> {
    let cases = [
        (
            MicrostripQuasiStaticModel::Wheeler,
            50.253_269_936_647_97,
            3.290_855_371_164_564_5,
        ),
        (
            MicrostripQuasiStaticModel::Schneider,
            49.916_690_718_563_52,
            3.249_765_348_362_791_5,
        ),
        (
            MicrostripQuasiStaticModel::HammerstadJensen,
            50.250_686_925_360_405,
            3.174_965_337_163_942,
        ),
    ];
    for (model, expected_impedance, expected_effective) in cases {
        let line = reference_microstrip(model, MicrostripDispersionModel::None)?;
        let (impedance, effective, _) = line
            .quasi_static_characteristics()
            .expect("quasi-static analysis should succeed");
        assert_relative_eq!(impedance[0].re, expected_impedance, max_relative = 5.0e-8);
        assert_relative_eq!(effective[0].re, expected_effective, max_relative = 5.0e-8);
    }
    Ok(())
}

/// Checks the supported frequency-dispersion model reference values.
#[test]
fn matches_upstream_microstrip_dispersion_models() -> Result<()> {
    let cases = [
        (
            MicrostripDispersionModel::Schneider,
            [
                50.241_720_584_157_9,
                49.460_917_331_488_425,
                44.137_166_584_885_21,
            ],
            [
                3.176_098_672_663_985,
                3.277_167_660_288_808_5,
                4.115_419_341_098_31,
            ],
        ),
        (
            MicrostripDispersionModel::HammerstadJensen,
            [
                50.321_340_631_583_93,
                54.956_763_626_732_084,
                64.007_287_626_375_56,
            ],
            [
                3.179_619_018_135_380_5,
                3.496_027_915_825_514,
                4.178_006_987_788_865,
            ],
        ),
        (
            MicrostripDispersionModel::KirschningJansen,
            [
                50.230_968_123_140_26,
                52.702_735_566_148_12,
                80.142_580_505_261_86,
            ],
            [
                3.188_477_216_929_040_7,
                3.445_074_662_746_57,
                4.128_677_245_550_929_5,
            ],
        ),
        (
            MicrostripDispersionModel::Yamashita,
            [50.250_686_925_360_405; 3],
            [
                3.189_592_574_227_834_4,
                3.497_212_129_448_975,
                4.134_048_840_911_491_5,
            ],
        ),
        (
            MicrostripDispersionModel::Kobayashi,
            [50.250_686_925_360_405; 3],
            [
                3.188_493_392_139_607,
                3.451_644_245_197_029_7,
                4.108_645_706_642_019,
            ],
        ),
    ];
    for (model, expected_impedance, expected_effective) in cases {
        let line = reference_microstrip(MicrostripQuasiStaticModel::HammerstadJensen, model)?;
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
    Ok(())
}

/// Compares generated microstrip networks with Qucs fixtures.
#[test]
fn matches_mline_qucs_fixtures() -> Result<()> {
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
        let reference = read_mline_fixture("qucs", name)?;
        let mut line = reference_mline(
            reference.frequency.clone(),
            DielectricDispersionModel::FrequencyInvariant,
            quasi_static_model,
            dispersion_model,
            CpwCompatibilityMode::Qucs,
        )?;
        let actual = line
            .line(25.0e-3, LengthUnit::Meter)
            .expect("microstrip line should be constructed");
        assert_network_ratio_within(&actual, &reference, 0.1, 1.0, 0.1, 1.0, name);
        line.set_resistivity_material("cu")
            .expect("copper should resolve");
        assert!(line.to_string().contains("Microstrip Media"));
    }
    Ok(())
}

/// Compares generated microstrip networks with ADS fixtures.
#[test]
fn matches_mline_ads_fixtures() -> Result<()> {
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
        let reference = read_mline_fixture("ads", name)?;
        let line = reference_mline(
            reference.frequency.clone(),
            dielectric_model,
            MicrostripQuasiStaticModel::HammerstadJensen,
            dispersion_model,
            CpwCompatibilityMode::Native,
        )?;
        let actual = line
            .line(25.0e-3, LengthUnit::Meter)
            .expect("microstrip line should be constructed");
        assert_network_ratio_within(&actual, &reference, 1.0, 10.0, 0.1, 1.0, name);
    }
    Ok(())
}

const fn djordjevic_svensson() -> DielectricDispersionModel {
    DielectricDispersionModel::DjordjevicSvensson {
        low_frequency_hz: 1.0e3,
        high_frequency_hz: 1.0e12,
        specification_frequency_hz: 1.0e9,
    }
}

fn read_mline_fixture(simulator: &str, name: &str) -> Result<Network> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/media/mline")
        .join(simulator)
        .join(name);
    Network::read_touchstone(path)
}

fn reference_mline(
    frequency: Frequency,
    dielectric_model: DielectricDispersionModel,
    quasi_static_model: MicrostripQuasiStaticModel,
    dispersion_model: MicrostripDispersionModel,
    compatibility_mode: CpwCompatibilityMode,
) -> Result<MicrostripLine> {
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
) -> Result<MicrostripLine> {
    let frequency =
        Frequency::from_values(Array1::from_vec(vec![1.0, 10.0, 100.0]), FrequencyUnit::GHz)?;
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
}
