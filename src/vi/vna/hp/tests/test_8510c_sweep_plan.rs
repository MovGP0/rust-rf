#![cfg(feature = "visa")]

use rust_rf::vi::vna::hp8510c_sweep_plan::{SweepPlan, SweepSection};

fn linear_space(start: f64, stop: f64, points: usize) -> Vec<f64> {
    let step = (stop - start) / (points - 1) as f64;
    (0..points)
        .map(|index| start + index as f64 * step)
        .collect()
}

#[test]
fn plans_magic_and_typical_linear_sweep_sizes() {
    for points in [800, 801, 802, 1_001] {
        let frequencies = linear_space(100.0, 1_000.0, points);
        let plan = SweepPlan::from_hz(&frequencies).unwrap();
        assert!(plan.matches_frequency_list(&frequencies));
    }
}

#[test]
fn plans_multiple_frequency_blocks() {
    let mut frequencies = vec![1.0, 2.0, 3.0];
    frequencies.extend(linear_space(100.0, 1_000.0, 1_001));
    let plan = SweepPlan::from_hz(&frequencies).unwrap();
    assert!(plan.matches_frequency_list(&frequencies));
    assert!(plan.sections().len() > 1);
}

#[test]
fn plans_multiple_blocks_with_a_single_frequency() {
    let mut frequencies = vec![1.0, 2.0, 3.0];
    frequencies.extend(linear_space(100.0, 1_000.0, 1_001));
    frequencies.push(9_999.0);
    let plan = SweepPlan::from_hz(&frequencies).unwrap();
    assert!(plan.matches_frequency_list(&frequencies));
    assert!(
        plan.sections()
            .iter()
            .any(|section| matches!(section, SweepSection::Random(_)))
    );
}

#[test]
fn chunks_random_points_into_instrument_limits() {
    let frequencies = (0..60)
        .map(|index| (index * index + 1) as f64)
        .collect::<Vec<_>>();
    let plan = SweepPlan::from_hz(&frequencies).unwrap();
    assert!(plan.matches_frequency_list(&frequencies));
    assert!(plan.sections().iter().all(|section| {
        !matches!(section, SweepSection::Random(random) if random.frequencies_hz.len() > 29)
    }));
}
