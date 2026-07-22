//! CITI input/output regressions.
//!
//! These tests cover network conversion, parameter parsing, files whose only
//! variable is frequency, and representative one- and two-port values.

use std::io::Cursor;

use approx::assert_relative_eq;
use num_complex::Complex64;
use rust_rf::io::{Citi, CitiFormat};

/// Parses real/imaginary CITI data from a reader and converts it to networks.
#[test]
fn parses_scalar_ri_data_from_a_reader() {
    let text = "CITIFILE A.01.00\n\
# comment\n\
NAME MEMORY\n\
VAR FREQ MAG 2\n\
DATA S RI\n\
VAR_LIST_BEGIN\n1\n2\nVAR_LIST_END\n\
BEGIN\n1.0,-2.0\n3.0,4.0\nEND\n";

    let citi = Citi::from_reader(Cursor::new(text)).unwrap();
    assert_eq!(citi.name, "MEMORY");
    assert_eq!(citi.comments, vec!["# comment"]);
    assert!(citi.parameters().is_empty());
    assert_eq!(citi.data[0].format, CitiFormat::RealImaginary);

    let networks = citi.networks().unwrap();
    assert_eq!(networks.len(), 1);
    assert_eq!(networks[0].frequency.values_hz().to_vec(), vec![1.0, 2.0]);
    assert_eq!(networks[0].s[(0, 0, 0)], Complex64::new(1.0, -2.0));
    assert_eq!(networks[0].s[(1, 0, 0)], Complex64::new(3.0, 4.0));
}

/// Converts magnitude/angle and dB/angle data and preserves parameter values.
#[test]
fn converts_magangle_and_dbangle_and_builds_parameter_coordinates() {
    let text = "CITIFILE A.01.00\n\
NAME sweep\n\
VAR bias MAG 2\n\
VAR freq MAG 1\n\
DATA S[1,1] MAGANGLE\n\
DATA S[1,2] DBANGLE\n\
DATA S[2,1] RI\n\
DATA S[2,2] MAGANGLE\n\
DATA PortZ[1] RI\n\
DATA PORTZ[2] RI\n\
VAR_LIST_BEGIN\n1\n2\nVAR_LIST_END\n\
VAR_LIST_BEGIN\n1000000000\nVAR_LIST_END\n\
BEGIN\n1,90\n1,180\nEND\n\
BEGIN\n20,0\n0,90\nEND\n\
BEGIN\n3,4\n5,6\nEND\n\
BEGIN\n2,-90\n2,0\nEND\n\
BEGIN\n50,0\n51,0\nEND\n\
BEGIN\n75,0\n76,0\nEND\n";

    let citi = Citi::parse(text).unwrap();
    assert_eq!(citi.parameters(), vec!["bias"]);
    let set = citi.to_network_set().unwrap();
    assert_eq!(set.len(), 2);
    assert_eq!(set.parameters["bias"], vec![1.0, 2.0]);
    assert_relative_eq!(set.networks[0].s[(0, 0, 0)].re, 0.0, epsilon = 1.0e-12);
    assert_relative_eq!(set.networks[0].s[(0, 0, 0)].im, 1.0, epsilon = 1.0e-12);
    assert_relative_eq!(set.networks[0].s[(0, 0, 1)].re, 10.0, epsilon = 1.0e-12);
    assert_eq!(set.networks[0].s[(0, 1, 0)], Complex64::new(3.0, 4.0));
    assert_relative_eq!(set.networks[0].s[(0, 1, 1)].im, -2.0, epsilon = 1.0e-12);
    assert_eq!(set.networks[0].z0[(0, 0)], Complex64::new(50.0, 0.0));
    assert_eq!(set.networks[1].z0[(0, 1)], Complex64::new(76.0, 0.0));
}

/// Converts impedance parameters to scattering parameters.
#[test]
fn converts_impedance_parameters_to_scattering_parameters() {
    let text = "CITIFILE A.01.00\n\
VAR freq MAG 1\n\
DATA Z[1,1] RI\n\
VAR_LIST_BEGIN\n1\nVAR_LIST_END\n\
BEGIN\n50,0\nEND\n";

    let network = Citi::parse(text).unwrap().networks().unwrap().remove(0);
    assert_relative_eq!(network.s[(0, 0, 0)].norm(), 0.0, epsilon = 1.0e-12);
}

/// Rejects files without frequency data and incomplete parameter matrices.
#[test]
fn rejects_missing_frequency_and_incomplete_matrices() {
    let missing_frequency = "VAR bias MAG 1\nDATA S RI\nVAR_LIST_BEGIN\n1\nEND\nBEGIN\n0,0\nEND\n";
    assert!(Citi::parse(missing_frequency).unwrap().networks().is_err());

    let incomplete = "VAR freq MAG 1\nDATA S[1,1] RI\nDATA S[1,2] RI\nVAR_LIST_BEGIN\n1\nEND\nBEGIN\n0,0\nEND\nBEGIN\n0,0\nEND\n";
    assert!(Citi::parse(incomplete).unwrap().networks().is_err());
}
