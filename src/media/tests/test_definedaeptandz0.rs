#![allow(unused_imports)]

//! Regression tests for media defined by attenuation, permittivity, loss tangent, and impedance.

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

const TOLERANCE: f64 = 1.0e-10;

/// Checks scalar nominal impedance dispersion and the resulting lossy propagation constant.
#[test]
fn calculates_defined_attenuation_and_impedance_dispersion() {
    let frequency = Frequency::new(1.0, 3.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let media = DefinedAEpTandZ0::from_scalars(
        frequency.clone(),
        0.067_018_872_231_560_5,
        frequency.values_hz()[0],
        3.253_542_864_282_65,
        0.013_393_675_893_949_3,
        75.0,
        Some(Complex64::new(50.0, 0.0)),
        DielectricDispersionModel::FrequencyInvariant,
    )
    .expect("defined medium should be valid");
    let permittivity = media
        .relative_permittivity_at_frequency()
        .expect("permittivity should be defined");
    assert_complex_close(
        permittivity[0],
        Complex64::new(
            3.253_542_864_282_65,
            -3.253_542_864_282_65 * 0.013_393_675_893_949_3,
        ),
    );
    let gamma = media
        .propagation_constant()
        .expect("propagation constant should be defined");
    assert!(gamma.iter().all(|value| value.re > 0.0 && value.im > 0.0));
    let impedance = media
        .characteristic_impedance()
        .expect("characteristic impedance should be defined");
    assert!(impedance.iter().all(|value| value.im.abs() > 0.1));
    assert_ne!(impedance[0], impedance[2]);
    let line = media
        .line(74.241_515_488_326_2e-3, LengthUnit::Meter)
        .expect("defined line should be constructed");
    assert_eq!(line.ports(), 2);
}

/// Checks that array-valued characteristic impedances remain raw values.
#[test]
fn preserves_raw_defined_characteristic_impedance() {
    let frequency = Frequency::new(1.0, 3.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    for raw in [
        Array1::from_vec(vec![Complex64::new(75.0, 0.0)]),
        Array1::from_vec(vec![
            Complex64::new(45.0, 0.0),
            Complex64::new(50.0, 0.0),
            Complex64::new(55.0, 0.0),
        ]),
    ] {
        let media = DefinedAEpTandZ0::new(
            frequency.clone(),
            Array1::zeros(3),
            1.0,
            Array1::from_elem(3, 3.0),
            Array1::zeros(3),
            DefinedCharacteristicImpedance::Raw(raw.clone()),
            None,
            DielectricDispersionModel::FrequencyInvariant,
        )
        .expect("raw-impedance medium should be valid");
        let actual = media
            .characteristic_impedance()
            .expect("raw impedance should be defined");
        for point in 0..3 {
            assert_eq!(actual[point], raw[point.min(raw.len() - 1)]);
        }
    }
}

/// Compares a generated line with the AWR reference network.
#[test]
fn matches_awr_defined_attenuation_line() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("media")
        .join("awr")
        .join("tlinp.s2p");
    let reference = Network::read_touchstone(fixture).expect("AWR fixture should load");
    let medium = DefinedAEpTandZ0::from_scalars(
        reference.frequency.clone(),
        0.067_018_872_231_560_5,
        reference.frequency.values_hz()[0],
        3.253_542_864_282_65,
        0.013_393_675_893_949_3,
        41.063_580_335_140_2,
        Some(Complex64::new(50.0, 0.0)),
        DielectricDispersionModel::FrequencyInvariant,
    )
    .expect("defined medium should construct");
    assert!(medium.to_string().starts_with("DefinedAEpTandZ0 Media."));
    let line = medium
        .line(74.241_515_488_326_2e-3, LengthUnit::Meter)
        .expect("defined line should construct");
    for (actual, expected) in line.s.iter().zip(reference.s.iter()) {
        let residual = actual / expected;
        assert!(20.0 * residual.norm().log10().abs() < 1.0e-5);
        assert!(residual.arg().to_degrees().abs() < 1.0e-4);
    }
}

/// Checks the Djordjevic-Svensson dielectric model at its specification frequency.
#[test]
fn matches_djordjevic_svensson_specification_point() {
    let frequency = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let media = DefinedAEpTandZ0::from_scalars(
        frequency,
        0.0,
        1.0,
        4.2,
        0.02,
        50.0,
        None,
        DielectricDispersionModel::DjordjevicSvensson {
            low_frequency_hz: 1.0e3,
            high_frequency_hz: 1.0e12,
            specification_frequency_hz: 1.0e9,
        },
    )
    .expect("dispersive medium should be valid");
    let actual = media
        .relative_permittivity_at_frequency()
        .expect("dispersive permittivity should be defined")[0];
    assert_relative_eq!(actual.re, 4.2, max_relative = 1.0e-12);
    assert_relative_eq!(actual.im, -4.2 * 0.02, max_relative = 1.0e-12);
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(actual.re, expected.re, epsilon = TOLERANCE);
    assert_relative_eq!(actual.im, expected.im, epsilon = TOLERANCE);
}
