#![cfg(feature = "visa")]

//! Verifies that HP 8510C sweep decomposition reproduces every requested
//! frequency while respecting the instrument's section-size constraints.

use num_traits::ToPrimitive;
use rust_rf::vi::vna::hp8510c_sweep_plan::{SweepPlan, SweepSection};

/// Generates a linearly spaced test frequency block.
fn linear_space(start: f64, stop: f64, points: usize) -> Vec<f64> {
    let interval_count = (points - 1).to_f64().unwrap_or(f64::INFINITY);
    let frequency_step = (stop - start) / interval_count;
    (0..points)
        .map(|index| {
            let frequency_offset = index.to_f64().unwrap_or(f64::INFINITY);
            frequency_offset.mul_add(frequency_step, start)
        })
        .collect()
}

#[test]
/// Plans the built-in 801-point size, the adjacent 800/802 sizes, and a typical
/// 1001-point compound sweep.
fn plans_magic_and_typical_linear_sweep_sizes() {
    for points in [800, 801, 802, 1_001] {
        let frequencies = linear_space(100.0, 1_000.0, points);
        let plan = SweepPlan::from_hz(&frequencies).unwrap();
        assert!(plan.matches_frequency_list(&frequencies));
    }
}

#[test]
/// Plans multiple differently spaced frequency blocks in one request.
fn plans_multiple_frequency_blocks() {
    let mut frequencies = vec![1.0, 2.0, 3.0];
    frequencies.extend(linear_space(100.0, 1_000.0, 1_001));
    let plan = SweepPlan::from_hz(&frequencies).unwrap();
    assert!(plan.matches_frequency_list(&frequencies));
    assert!(plan.sections().len() > 1);
}

#[test]
/// Plans multiple blocks when one block contains only a single frequency.
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
/// Splits non-linear remainder points into list sections within instrument limits.
fn chunks_random_points_into_instrument_limits() {
    let frequencies = (0..60)
        .map(|index| f64::from(index * index + 1))
        .collect::<Vec<_>>();
    let plan = SweepPlan::from_hz(&frequencies).unwrap();
    assert!(plan.matches_frequency_list(&frequencies));
    assert!(plan.sections().iter().all(|section| {
        !matches!(section, SweepSection::Random(random) if random.frequencies_hz.len() > 29)
    }));
}
