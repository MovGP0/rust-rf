//! Calibration-set generation and correction tests.

use approx::assert_relative_eq;
use ndarray::{Array2, Array3, array};
use num_complex::Complex64;
use rust_rf::calibration::{Calibration, OnePort};
use rust_rf::calibration_set::{DotCalibrationSet, cartesian_product};
use rust_rf::{Frequency, Network, NetworkSet, Result};

/// Ensures a dot-product calibration set can be generated and can correct a network.
#[test]
fn creates_dot_calibrations_and_corrected_standard_sets() -> Result<()> {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9, 3.0e9])?;
    let standards = [
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let ideals = standards
        .iter()
        .enumerate()
        .map(|(index, standard)| {
            let mut network = constant_one_port(frequency.clone(), *standard)?;
            network.name = Some(format!("standard-{index}"));
            Ok(network)
        })
        .collect::<Result<Vec<_>>>()?;
    let error_systems = [
        (
            Complex64::new(0.01, 0.02),
            Complex64::new(0.8, 0.1),
            Complex64::new(0.05, -0.01),
        ),
        (
            Complex64::new(-0.02, 0.01),
            Complex64::new(0.9, -0.05),
            Complex64::new(-0.03, 0.02),
        ),
    ];
    let measured_sets = standards
        .iter()
        .map(|standard| {
            let networks = error_systems
                .iter()
                .map(|(directivity, tracking, source_match)| {
                    constant_one_port(
                        frequency.clone(),
                        embed_value(*standard, *directivity, *tracking, *source_match),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            NetworkSet::new(networks, None)
        })
        .collect::<Result<Vec<_>>>()?;
    let calibration_set = DotCalibrationSet::new(
        ideals.clone(),
        measured_sets.clone(),
        Some("uncertainty".to_owned()),
        |measured, ideals| {
            let mut calibration = OnePort::new(measured, ideals)?;
            calibration.run()?;
            Ok(calibration)
        },
    )?;
    assert_eq!(calibration_set.len(), 2);

    let raw = constant_one_port(
        frequency,
        embed_value(
            Complex64::new(0.2, -0.1),
            error_systems[0].0,
            error_systems[0].1,
            error_systems[0].2,
        ),
    )?;
    let corrected = calibration_set.apply_cal(&raw)?;
    assert_eq!(corrected.name.as_deref(), Some("uncertainty"));
    assert_complex_close(
        corrected.networks[0].s[(0, 0, 0)],
        Complex64::new(0.2, -0.1),
    );

    let corrected_sets = calibration_set.corrected_sets()?;
    assert_eq!(corrected_sets.len(), 3);
    for (standard, set) in corrected_sets.iter().enumerate() {
        assert_eq!(set.len(), 2);
        for network in &set.networks {
            assert_complex_close(network.s[(0, 0, 0)], standards[standard]);
        }
    }

    let combinations = cartesian_product(&ideals, &measured_sets, &|measured, ideals| {
        OnePort::new(measured, ideals)
    })?;
    assert_eq!(combinations.len(), 8);
    Ok(())
}

#[test]
fn rejects_misaligned_dot_sets() -> Result<()> {
    let frequency = Frequency::from_hz(array![1.0e9])?;
    let ideals = vec![
        constant_one_port(frequency.clone(), Complex64::new(-1.0, 0.0))?,
        constant_one_port(frequency.clone(), Complex64::new(0.0, 0.0))?,
        constant_one_port(frequency, Complex64::new(1.0, 0.0))?,
    ];
    let measured_sets = vec![
        NetworkSet::new(vec![ideals[0].clone()], None)?,
        NetworkSet::new(vec![ideals[1].clone(), ideals[1].clone()], None)?,
        NetworkSet::new(vec![ideals[2].clone()], None)?,
    ];
    assert!(DotCalibrationSet::new(ideals, measured_sets, None, OnePort::new).is_err());
    Ok(())
}

fn embed_value(
    ideal: Complex64,
    directivity: Complex64,
    tracking: Complex64,
    source_match: Complex64,
) -> Complex64 {
    let a = tracking - directivity * source_match;
    (directivity + a * ideal) / (Complex64::new(1.0, 0.0) - source_match * ideal)
}

fn constant_one_port(frequency: Frequency, value: Complex64) -> Result<Network> {
    let points = frequency.points();
    Network::new(
        frequency,
        Array3::from_elem((points, 1, 1), value),
        Array2::from_elem((points, 1), Complex64::new(50.0, 0.0)),
    )
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-10);
    assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-10);
}
