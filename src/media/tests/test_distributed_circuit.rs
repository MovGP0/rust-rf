#![allow(unused_imports)]

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

#[test]
fn calculates_distributed_circuit_wave_quantities() {
    let frequency = Frequency::new(1.0, 3.0, 3, FrequencyUnit::MHz, SweepType::Linear)
        .expect("frequency should be valid");
    let media = DistributedCircuit::from_scalars(frequency, 1.0, 0.01, 50.0e-9, 100.0e-12)
        .expect("distributed circuit should be valid");
    let impedance = media.distributed_impedance();
    let admittance = media.distributed_admittance();
    let gamma = media
        .propagation_constant()
        .expect("propagation constant should be defined");
    let z0 = media
        .characteristic_impedance()
        .expect("characteristic impedance should be defined");
    for point in 0..3 {
        assert_complex_close(gamma[point], (impedance[point] * admittance[point]).sqrt());
        assert_complex_close(z0[point], (impedance[point] / admittance[point]).sqrt());
    }
    let line = media
        .line(1.0, LengthUnit::Millimeter)
        .expect("distributed line should be constructed");
    assert_eq!(line.frequency_points(), 3);
    assert_eq!(line.ports(), 2);
}

#[test]
fn rejects_mismatched_distributed_circuit_arrays() {
    let frequency = Frequency::new(1.0, 3.0, 3, FrequencyUnit::MHz, SweepType::Linear)
        .expect("frequency should be valid");
    assert!(
        DistributedCircuit::new(
            frequency,
            Array1::from_elem(1, 1.0),
            Array1::from_elem(3, 0.01),
            Array1::from_elem(3, 50.0e-9),
            Array1::from_elem(3, 100.0e-12),
            None,
        )
        .is_err()
    );
}

#[test]
fn converts_distributed_circuit_to_and_from_media_csv() {
    let frequency = Frequency::new(1.0, 10.0, 5, FrequencyUnit::MHz, SweepType::Linear)
        .expect("frequency should be valid");
    let original = DistributedCircuit::new(
        frequency,
        Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        Array1::from_vec(vec![0.01, 0.02, 0.03, 0.04, 0.05]),
        Array1::from_vec(vec![50.0e-9, 51.0e-9, 52.0e-9, 53.0e-9, 54.0e-9]),
        Array1::from_vec(vec![100.0e-12, 101.0e-12, 102.0e-12, 103.0e-12, 104.0e-12]),
        Some(Array1::from_elem(5, Complex64::new(50.0, 0.0))),
    )
    .expect("distributed circuit should be valid");
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock should be after the epoch")
        .as_nanos();
    let directory = std::path::PathBuf::from(".temp")
        .join(format!("media-csv-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("temporary directory should be created");
    let path = directory.join("distributed.csv");
    original
        .write_csv(&path)
        .expect("distributed circuit CSV should be written");
    let restored =
        DistributedCircuit::from_csv(&path).expect("distributed circuit CSV should be restored");
    assert_eq!(restored.frequency, original.frequency);
    for point in 0..original.frequency.points() {
        assert_relative_eq!(
            restored.resistance_per_meter[point],
            original.resistance_per_meter[point],
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            restored.conductance_per_meter[point],
            original.conductance_per_meter[point],
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            restored.inductance_per_meter[point],
            original.inductance_per_meter[point],
            epsilon = 1.0e-18
        );
        assert_relative_eq!(
            restored.capacitance_per_meter[point],
            original.capacitance_per_meter[point],
            epsilon = 1.0e-18
        );
    }
    assert_eq!(
        restored.port_z0,
        Some(Array1::from_elem(5, Complex64::new(50.0, 0.0)))
    );
    assert!(original.to_string().contains("Distributed Circuit Media"));
    std::fs::remove_dir_all(directory).expect("temporary directory should be removed");
}

#[test]
fn matches_qucs_distributed_circuit_line() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/media/qucs/distributedCircuit,line1mm.s2p");
    let reference = Network::read_touchstone(fixture).expect("QUCS fixture should load");
    let media = DistributedCircuit::new(
        reference.frequency.clone(),
        Array1::from_elem(reference.frequency_points(), 1.0e5),
        Array1::from_elem(reference.frequency_points(), 1.0),
        Array1::from_elem(reference.frequency_points(), 1.0e-6),
        Array1::from_elem(reference.frequency_points(), 8.0e-12),
        Some(Array1::from_elem(
            reference.frequency_points(),
            Complex64::new(50.0, 0.0),
        )),
    )
    .expect("distributed circuit should be valid");
    let actual = media
        .line(1.0, LengthUnit::Millimeter)
        .expect("distributed line should be constructed");
    assert_eq!(actual.s.dim(), reference.s.dim());
    for (actual, expected) in actual.s.iter().zip(reference.s.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 2.0e-5);
        assert_relative_eq!(actual.im, expected.im, epsilon = 2.0e-5);
    }
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(actual.re, expected.re, epsilon = TOLERANCE);
    assert_relative_eq!(actual.im, expected.im, epsilon = TOLERANCE);
}
