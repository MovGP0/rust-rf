use approx::assert_relative_eq;
use ndarray::{Array1, array};
use rust_rf::{Frequency, FrequencyUnit, SweepType};

const TOLERANCE: f64 = 1.0e-12;

#[test]
fn creates_linear_sweep() {
    let frequency = Frequency::new(1.0, 10.0, 10, FrequencyUnit::GHz, SweepType::Linear)
        .expect("linear sweep should be valid");

    assert_eq!(frequency.points(), 10);
    assert_eq!(frequency.sweep_type(), SweepType::Linear);
    for (actual, expected) in frequency.scaled().iter().zip(1..=10) {
        assert_relative_eq!(*actual, f64::from(expected), epsilon = TOLERANCE);
    }
}

#[test]
fn creates_logarithmic_sweep() {
    let frequency = Frequency::new(1.0, 10.0, 10, FrequencyUnit::GHz, SweepType::Logarithmic)
        .expect("logarithmic sweep should be valid");

    assert_eq!(frequency.start(), Some(1.0e9));
    assert_eq!(frequency.stop(), Some(10.0e9));
    assert_eq!(frequency.sweep_type(), SweepType::Logarithmic);

    let adjacent_ratios = frequency
        .values_hz()
        .iter()
        .zip(frequency.values_hz().iter().skip(1))
        .map(|(left, right)| right / left)
        .collect::<Vec<_>>();
    assert!(adjacent_ratios.iter().all(|ratio| *ratio > 1.0));
    for ratio in adjacent_ratios.iter().skip(1) {
        assert_relative_eq!(*ratio, adjacent_ratios[0], epsilon = 1.0e-10);
    }
}

#[test]
fn creates_arbitrary_sweep_in_requested_unit() {
    let frequency = Frequency::from_values(array![1.0, 5.0, 200.0], FrequencyUnit::KHz)
        .expect("arbitrary sweep should be valid");

    assert_eq!(frequency.values_hz(), &array![1.0e3, 5.0e3, 200.0e3]);
    assert_eq!(frequency.scaled(), array![1.0, 5.0, 200.0]);
    assert_eq!(frequency.sweep_type(), SweepType::Arbitrary);
}

#[test]
fn slices_frequency_by_inclusive_value_range() {
    let frequency = Frequency::from_values(array![1.0, 2.0, 4.0, 5.0, 6.0], FrequencyUnit::GHz)
        .expect("frequency should be valid");

    let sliced = frequency
        .slice_range(2.0, 5.0, FrequencyUnit::GHz)
        .expect("slice should be valid");

    assert_eq!(sliced.values_hz(), &array![2.0e9, 4.0e9, 5.0e9]);
    assert_eq!(sliced.unit(), FrequencyUnit::GHz);
}

#[test]
fn detects_and_drops_non_monotonic_values() {
    let mut frequency = Frequency::from_values(array![1.0, 2.0, 2.0], FrequencyUnit::Hz)
        .expect("non-monotonic values are accepted for cleanup");

    assert!(!frequency.is_monotonic_increasing());
    assert_eq!(frequency.drop_non_monotonic_increasing(), vec![2]);
    assert_eq!(frequency.values_hz(), &array![1.0, 2.0]);
    assert!(frequency.is_monotonic_increasing());
}

#[test]
fn performs_frequency_arithmetic_with_broadcasting() {
    let frequency = Frequency::new(1.0, 10.0, 10, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let scalar = Frequency::from_hz(array![10.0]).expect("scalar axis should be valid");

    let added = frequency
        .try_add(&scalar)
        .expect("single-point axes should broadcast");
    for (actual, original) in added.values_hz().iter().zip(frequency.values_hz()) {
        assert_eq!(*actual, original + 10.0);
    }

    let multiplied = frequency
        .map_values(|value| value * 5.31)
        .expect("finite mapped values should be valid");
    for (actual, original) in multiplied.values_hz().iter().zip(frequency.values_hz()) {
        assert_eq!(*actual, original * 5.31);
    }

    let incompatible = Frequency::from_hz(Array1::linspace(10.0, 100.0, 20))
        .expect("comparison axis should be valid");
    assert!(frequency.try_add(&incompatible).is_err());

    let left = Frequency::from_hz(array![10.0, 21.0]).expect("left axis");
    let right = Frequency::from_hz(array![3.0, 4.0]).expect("right axis");
    assert_eq!(
        left.try_subtract(&right).expect("subtract").values_hz(),
        &array![7.0, 17.0]
    );
    assert_eq!(
        left.try_multiply(&right).expect("multiply").values_hz(),
        &array![30.0, 84.0]
    );
    assert_eq!(
        left.try_divide(&right).expect("divide").values_hz(),
        &array![10.0 / 3.0, 5.25]
    );
    assert_eq!(
        left.try_floor_divide(&right)
            .expect("floor divide")
            .values_hz(),
        &array![3.0, 5.0]
    );
    assert_eq!(
        left.try_remainder(&right).expect("remainder").values_hz(),
        &array![1.0, 1.0]
    );
}

#[test]
fn changes_display_units_without_mutating_hertz() {
    let mut frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency");
    let hertz = frequency.values_hz().clone();
    frequency.set_unit(FrequencyUnit::GHz);
    assert_eq!(frequency.values_hz(), &hertz);
    assert_eq!(frequency.scaled(), array![1.0, 2.0]);
    assert_eq!(frequency.to_string(), "1-2 GHz, 2 pts");
}

#[test]
fn calculates_overlap_on_the_left_axis_grid() {
    let left = Frequency::new(1.0, 10.0, 10, FrequencyUnit::GHz, SweepType::Linear)
        .expect("left axis should be valid");
    let right = Frequency::new(3.5, 6.5, 4, FrequencyUnit::GHz, SweepType::Linear)
        .expect("right axis should be valid");

    let overlap = left.overlap(&right).expect("axes should overlap");

    assert_eq!(overlap.values_hz(), &array![4.0e9, 5.0e9, 6.0e9]);
    assert_eq!(overlap.unit(), FrequencyUnit::GHz);
}

#[test]
fn exposes_derived_frequency_properties() {
    let frequency = Frequency::new(1.0, 5.0, 5, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");

    assert_eq!(frequency.center(), Some(3.0e9));
    assert_eq!(frequency.center_index(), Some(2));
    assert_eq!(frequency.span(), Some(4.0e9));
    assert_eq!(frequency.step(), Some(1.0e9));
    assert_eq!(
        frequency.gradient_hz().expect("gradient should exist"),
        array![1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9]
    );
}

#[test]
fn creates_centered_time_axis() {
    let frequency = Frequency::new(1.0, 5.0, 5, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");

    let time = frequency.time().expect("time axis should be defined");
    assert_eq!(time.len(), 5);
    assert_relative_eq!(time[0], -0.4e-9, epsilon = 1.0e-18);
    assert_relative_eq!(time[2], 0.0, epsilon = 1.0e-18);
    assert_relative_eq!(time[4], 0.4e-9, epsilon = 1.0e-18);
    assert_relative_eq!(
        frequency
            .time_nanoseconds()
            .expect("nanosecond time axis should be defined")[4],
        0.4,
        epsilon = TOLERANCE
    );
}

#[test]
fn rejects_invalid_construction_and_disjoint_overlap() {
    assert!(Frequency::new(0.0, 10.0, 10, FrequencyUnit::GHz, SweepType::Logarithmic,).is_err());
    assert!(Frequency::from_hz(array![f64::NAN]).is_err());

    let lower = Frequency::new(1.0, 2.0, 2, FrequencyUnit::GHz, SweepType::Linear)
        .expect("lower axis should be valid");
    let upper = Frequency::new(3.0, 4.0, 2, FrequencyUnit::GHz, SweepType::Linear)
        .expect("upper axis should be valid");
    assert!(lower.overlap(&upper).is_err());
}
