use approx::assert_relative_eq;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rust_rf::io::write_sdatcv;
use rust_rf::{Frequency, Network, NetworkSet, Result};

#[test]
fn calculates_column_major_scalar_data_and_sample_covariance() -> Result<()> {
    let frequency = Frequency::from_hz(ndarray::arr1(&[1.0])).unwrap();
    let first = one_port(frequency.clone(), Complex64::new(1.0, 2.0))?;
    let second = one_port(frequency, Complex64::new(3.0, 4.0))?;
    let set = NetworkSet::new(vec![first, second], None).unwrap();

    let scalar = set.scalar_s_matrix().unwrap();
    assert_eq!(scalar.dim(), (1, 2, 2));
    assert_relative_eq!(scalar[(0, 0, 0)], 1.0, epsilon = 1.0e-12);
    assert_relative_eq!(scalar[(0, 1, 1)], 4.0, epsilon = 1.0e-12);
    let covariance = set.covariance_s().unwrap();
    assert_relative_eq!(covariance[(0, 0, 0)], 2.0, epsilon = 1.0e-12);
    assert_relative_eq!(covariance[(0, 0, 1)], 2.0, epsilon = 1.0e-12);
    assert_relative_eq!(covariance[(0, 1, 0)], 2.0, epsilon = 1.0e-12);
    assert_relative_eq!(covariance[(0, 1, 1)], 2.0, epsilon = 1.0e-12);
    Ok(())
}

#[test]
fn writes_metas_sdatcv_header_mean_and_covariance() -> Result<()> {
    let frequency = Frequency::from_hz(ndarray::arr1(&[1.0e9])).unwrap();
    let first = one_port(frequency.clone(), Complex64::new(1.0, 2.0))?;
    let second = one_port(frequency, Complex64::new(3.0, 4.0))?;
    let set = NetworkSet::new(vec![first, second], None).unwrap();
    let mut bytes = Vec::new();
    write_sdatcv(&set, &mut bytes).unwrap();
    let text = String::from_utf8(bytes).unwrap();

    assert!(text.starts_with("SDATCV\nPorts\n1\t\n"));
    assert!(text.contains("Zr[1]re\tZr[1]im"));
    assert!(text.contains("Freq\tS[1,1]re\tS[1,1]im\tCV[1,1]"));
    let data = text.lines().last().unwrap();
    let values = data
        .split('\t')
        .map(|value| value.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1.0e9, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0]);
    Ok(())
}

#[test]
fn rejects_empty_and_single_observation_sets() -> Result<()> {
    assert!(write_sdatcv(&NetworkSet::default(), Vec::new()).is_err());
    let frequency = Frequency::from_hz(ndarray::arr1(&[1.0])).unwrap();
    let set = NetworkSet::new(vec![one_port(frequency, Complex64::new(1.0, 0.0))?], None).unwrap();
    assert!(set.covariance_s().is_err());
    Ok(())
}

fn one_port(frequency: Frequency, value: Complex64) -> Result<Network> {
    let s = Array3::from_elem((frequency.points(), 1, 1), value);
    let z0 = Array2::from_elem((frequency.points(), 1), Complex64::new(50.0, 0.0));
    Network::new(frequency, s, z0)
}
