#![allow(unused_imports)]

//! Coaxial-media regression tests.
//!
//! These tests cover the Python suite's distributed-parameter identity,
//! frequency-dependent attenuation conversions, conductor loss, and line
//! construction, plus the Rust geometry constructor.

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

/// Verifies $LC=\mu\varepsilon'$ and constructs the corresponding lossy line.
#[test]
fn calculates_coaxial_distributed_parameters() {
    let frequency = Frequency::new(1.0, 10.0, 5, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let media = Coaxial::from_scalars(
        frequency,
        1.0e-3,
        3.0e-3,
        2.29,
        4.0e-4,
        1.0 / 1.68e-8,
        Some(Complex64::new(50.0, 0.0)),
    )
    .expect("coaxial medium should be valid");
    let inductance = media.inductance_per_meter();
    let capacitance = media.capacitance_per_meter();
    for point in 0..5 {
        assert_relative_eq!(
            inductance[point] * capacitance[point],
            FREE_SPACE_PERMEABILITY * FREE_SPACE_PERMITTIVITY * 2.29,
            max_relative = 1.0e-12
        );
        assert!(media.resistance_per_meter()[point] > 0.0);
        assert!(media.conductance_per_meter()[point] > 0.0);
    }
    let line = media
        .line(0.2, LengthUnit::Meter)
        .expect("coaxial line should be constructed");
    assert_eq!(line.z0[(0, 0)], Complex64::new(50.0, 0.0));
}

/// Verifies every supported dB/Np and meter/foot attenuation conversion.
#[test]
fn converts_coaxial_attenuation_units() {
    let frequency = Frequency::new(1.0, 2.0, 2, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let attenuation = Array1::from_elem(2, 3.0);
    let velocity_factor = Array1::from_elem(2, 0.8);
    for (unit, expected) in [
        (AttenuationUnit::DecibelsPerMeter, db_to_nepers(3.0)),
        (
            AttenuationUnit::DecibelsPerHundredMeters,
            db_to_nepers(3.0) / 100.0,
        ),
        (AttenuationUnit::DecibelsPerFoot, db_to_nepers(3.0) / 0.3048),
        (
            AttenuationUnit::DecibelsPerHundredFeet,
            db_to_nepers(3.0) / (100.0 * 0.3048),
        ),
        (AttenuationUnit::NepersPerMeter, 3.0),
        (AttenuationUnit::NepersPerFoot, 3.0 / 0.3048),
    ] {
        let media = Coaxial::from_attenuation_and_velocity_factor(
            frequency.clone(),
            attenuation.clone(),
            unit,
            velocity_factor.clone(),
            Complex64::new(50.0, 0.0),
            None,
        )
        .expect("attenuation-defined coaxial medium should be valid");
        assert_relative_eq!(media.gamma[0].re, expected, epsilon = TOLERANCE);
        assert_relative_eq!(
            media.gamma[0].im,
            frequency.angular()[0] / (SPEED_OF_LIGHT * 0.8),
            max_relative = 1.0e-12
        );
    }
}

/// Verifies that a requested impedance and outer diameter produce valid geometry.
#[test]
fn derives_coaxial_geometry_from_impedance() {
    let frequency = Frequency::new(1.0, 2.0, 2, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let media =
        Coaxial::from_characteristic_impedance_and_outer_diameter(frequency, 50.0, 5.0e-3, 1.0)
            .expect("impedance-defined coaxial medium should be valid");
    let impedance = media
        .characteristic_impedance()
        .expect("coaxial impedance should be defined");
    assert_relative_eq!(impedance[0].re, 50.0, max_relative = 1.0e-9);
    assert_relative_eq!(impedance[0].im, 0.0, epsilon = TOLERANCE);
}
