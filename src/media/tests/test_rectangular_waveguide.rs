#![allow(unused_imports)]

//! Rectangular-waveguide cutoff, impedance, and wall-loss regressions.

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

/// Checks cutoff and propagation below and above the dominant-mode cutoff frequency.
#[test]
fn calculates_rectangular_waveguide_cutoff_and_propagation() {
    let width = 100.0 * 0.000_025_4;
    let frequency = Frequency::new(50.0, 100.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let waveguide = RectangularWaveguide::dominant_mode(frequency, width)
        .expect("dominant waveguide should be valid");
    let cutoff = waveguide.cutoff_frequency();
    assert_relative_eq!(
        cutoff[0],
        SPEED_OF_LIGHT / (2.0 * width),
        max_relative = 1.0e-9
    );
    let gamma = waveguide
        .propagation_constant()
        .expect("waveguide propagation should be defined");
    assert!(gamma[0].re > 0.0 && gamma[0].im == 0.0);
    assert!(gamma[1].re == 0.0 && gamma[1].im > 0.0);
    assert!(gamma[2].re == 0.0 && gamma[2].im > gamma[1].im);
    let impedance = waveguide
        .characteristic_impedance()
        .expect("TE impedance should be defined");
    assert!(impedance[0].im > 0.0);
    assert!(impedance[1].re > 0.0);
}

/// Checks derivation of broad-wall width from a target TE impedance.
#[test]
fn derives_rectangular_waveguide_width_from_impedance() {
    let frequency = Frequency::new(90.0, 90.0, 1, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let waveguide =
        RectangularWaveguide::from_characteristic_impedance(frequency, 500.0, 90.0e9, 1.0, 1.0)
            .expect("impedance-defined waveguide should be valid");
    let impedance = waveguide
        .characteristic_impedance()
        .expect("waveguide impedance should be defined");
    assert_relative_eq!(impedance[0].re, 500.0, max_relative = 1.0e-12);
    assert_relative_eq!(impedance[0].im, 0.0, epsilon = TOLERANCE);
}

/// Checks conductor attenuation and the additional loss caused by wall roughness.
#[test]
fn applies_rectangular_waveguide_conductor_and_roughness_loss() {
    let frequency = Frequency::new(75.0, 110.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let points = frequency.points();
    let smooth = RectangularWaveguide::new(
        frequency.clone(),
        100.0 * 0.000_025_4,
        None,
        WaveguideMode::TransverseElectric,
        1,
        0,
        Array1::ones(points),
        Array1::ones(points),
        Some(Array1::from_elem(points, 1.0 / 3.8e7)),
        None,
        None,
        Some(Array1::from_elem(points, Complex64::new(50.0, 0.0))),
    )
    .expect("smooth waveguide should be valid");
    let rough = RectangularWaveguide::new(
        frequency,
        100.0 * 0.000_025_4,
        None,
        WaveguideMode::TransverseElectric,
        1,
        0,
        Array1::ones(points),
        Array1::ones(points),
        Some(Array1::from_elem(points, 1.0 / 3.8e7)),
        Some(Array1::from_elem(points, 100.0e-9)),
        None,
        Some(Array1::from_elem(points, Complex64::new(50.0, 0.0))),
    )
    .expect("rough waveguide should be valid");
    let smooth_loss = smooth
        .conductor_attenuation()
        .expect("smooth conductor loss should be defined");
    let rough_loss = rough
        .conductor_attenuation()
        .expect("rough conductor loss should be defined");
    for point in 0..points {
        assert!(smooth_loss[point] > 0.0);
        assert!(rough_loss[point] > smooth_loss[point]);
    }
    assert!(
        rough
            .line(1.0, LengthUnit::Inch)
            .expect("rough waveguide line should be constructed")
            .s[(0, 1, 0)]
            .norm()
            < 1.0
    );
}

/// Compares transmission magnitude with smooth and rough SWG fixtures.
///
/// Only magnitude is compared because the loss approximation omits the
/// reactive field contribution at the sidewalls.
#[test]
fn matches_rectangular_waveguide_conductor_loss_fixtures() {
    for (fixture_name, roughness) in [
        ("wr1p5_1in_swg_Al_0rough.s2p", None),
        ("wr1p5_1in_swg_Al_100nm_rough.s2p", Some(100.0e-9)),
    ] {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/media/rectangular_waveguide")
            .join(fixture_name);
        let reference = Network::read_touchstone(fixture).expect("SWG fixture should load");
        let points = reference.frequency_points();
        let mut waveguide = RectangularWaveguide::new(
            reference.frequency.clone(),
            15.0 * 0.000_025_4,
            None,
            WaveguideMode::TransverseElectric,
            1,
            0,
            Array1::ones(points),
            Array1::ones(points),
            Some(Array1::from_elem(points, 1.0 / 3.8e7)),
            roughness.map(|value| Array1::from_elem(points, value)),
            None,
            Some(Array1::from_elem(points, Complex64::new(50.0, 0.0))),
        )
        .expect("rectangular waveguide should be valid");
        let actual = waveguide
            .line(1.0, LengthUnit::Inch)
            .expect("one-inch waveguide line should be constructed");
        let maximum_error = (0..points)
            .map(|point| (actual.s[(point, 1, 0)].norm() - reference.s[(point, 1, 0)].norm()).abs())
            .fold(0.0_f64, f64::max);
        assert!(maximum_error < 1.0e-3, "maximum error was {maximum_error}");
        waveguide
            .set_resistivity_material("aluminum")
            .expect("aluminum material should resolve");
        assert!(
            waveguide
                .to_string()
                .contains("Rectangular Waveguide Media")
        );
    }
}
