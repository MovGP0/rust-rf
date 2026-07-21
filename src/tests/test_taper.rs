use std::sync::Arc;

use approx::assert_relative_eq;
use ndarray::Array1;
use num_complex::Complex64;
use rust_rf::media::{DefinedGammaZ0, LengthUnit};
use rust_rf::taper::{Taper1D, TaperProfile};
use rust_rf::{Frequency, FrequencyUnit, SweepType};

#[test]
fn generates_standard_taper_profiles() {
    let linear = taper(TaperProfile::Linear, 10.0, 100.0, 4);
    assert_eq!(
        linear.value_vector().expect("linear values"),
        Array1::linspace(10.0, 100.0, 4)
    );

    let exponential = taper(TaperProfile::Exponential, 10.0, 80.0, 4);
    let values = exponential.value_vector().expect("exponential values");
    for (index, value) in values.iter().enumerate() {
        let normalized = index as f64 / 3.0;
        assert_relative_eq!(
            *value,
            10.0 * (normalized * 8.0_f64.ln()).exp(),
            epsilon = 1.0e-12
        );
    }

    let smooth = taper(TaperProfile::SmoothStep, 11.7, 4.6, 5);
    let values = smooth.value_vector().expect("smooth-step values");
    assert_relative_eq!(values[0], 11.7, epsilon = 1.0e-12);
    assert_relative_eq!(values[2], (11.7 + 4.6) / 2.0, epsilon = 1.0e-12);
    assert_relative_eq!(values[4], 4.6, epsilon = 1.0e-12);
}

#[test]
fn generates_custom_and_klopfenstein_profiles() {
    let frequency = test_frequency();
    let factory_frequency = Arc::new(frequency);
    let custom = Taper1D::custom_normalized(
        media_factory(Arc::clone(&factory_frequency)),
        10.0,
        100.0,
        5,
        1.0,
        LengthUnit::Millimeter,
        |value| 0.3 * value.powi(2),
    )
    .expect("custom taper should be valid");
    let values = custom.value_vector().expect("custom values");
    assert_relative_eq!(values[4], 37.0, epsilon = 1.0e-12);

    let first = Taper1D::klopfenstein(
        media_factory(Arc::clone(&factory_frequency)),
        250.0,
        10.0,
        10,
        0.5,
        LengthUnit::Millimeter,
        0.05,
    )
    .expect("Klopfenstein taper should be valid");
    let second = Taper1D::klopfenstein(
        media_factory(factory_frequency),
        250.0,
        10.0,
        10,
        0.5,
        LengthUnit::Millimeter,
        0.01,
    )
    .expect("second Klopfenstein taper should be valid");
    let first_values = first.value_vector().expect("first Klopfenstein values");
    let second_values = second.value_vector().expect("second Klopfenstein values");
    // The upstream formula deliberately samples below `start` at x=0.
    assert_relative_eq!(first_values[0], 230.670_208_647_749_43, epsilon = 1.0e-10);
    assert_relative_eq!(first_values[9], 10.0, epsilon = 1.0e-10);
    assert_ne!(first_values[4], second_values[4]);
}

#[test]
fn builds_sections_and_cascaded_taper_network() {
    let taper = taper(TaperProfile::Linear, 10.0, 100.0, 30);
    assert_relative_eq!(taper.section_length(), 1.0 / 30.0, epsilon = 1.0e-15);
    let section = taper.section_at(50.0).expect("section should be built");
    assert_eq!(section.z0[(0, 0)], Complex64::new(50.0, 0.0));
    let network = taper.network().expect("taper network should be built");
    assert_eq!(network.z0[(0, 0)], Complex64::new(10.0, 0.0));
    assert_eq!(network.z0[(0, 1)], Complex64::new(100.0, 0.0));
}

fn taper(profile: TaperProfile, start: f64, stop: f64, sections: usize) -> Taper1D<DefinedGammaZ0> {
    let frequency = Arc::new(test_frequency());
    Taper1D::new(
        media_factory(frequency),
        start,
        stop,
        sections,
        1.0,
        LengthUnit::Millimeter,
        profile,
    )
    .expect("taper should be valid")
}

fn media_factory(
    frequency: Arc<Frequency>,
) -> impl Fn(f64) -> rust_rf::Result<DefinedGammaZ0> + Send + Sync + 'static {
    move |impedance| {
        let points = frequency.points();
        DefinedGammaZ0::new(
            (*frequency).clone(),
            Array1::from_elem(points, Complex64::new(0.0, 10.0)),
            Array1::from_elem(points, Complex64::new(impedance, 0.0)),
            None,
        )
    }
}

fn test_frequency() -> Frequency {
    Frequency::new(1.0, 3.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid")
}
