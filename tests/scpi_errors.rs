#![cfg(feature = "visa")]

use rust_rf::vi::scpi_errors::{SCPI_ERROR_CODES, ScpiError, error_details};

#[test]
fn exposes_the_complete_upstream_error_table() {
    assert_eq!(SCPI_ERROR_CODES.len(), 121);
    assert_eq!(
        error_details(-113),
        Some(("UNDEFINED_HEADER", "The command is unrecognized"))
    );
    assert_eq!(error_details(-222).unwrap().0, "DATA_OUT_OF_RANGE");
    assert_eq!(error_details(-800).unwrap().0, "OPERATION_COMPLETE");
}

#[test]
fn formats_known_scpi_errors() {
    let error = ScpiError::new(-113);
    assert_eq!(error.error_code, -113);
    assert_eq!(error.abbreviation, "UNDEFINED_HEADER");
    assert_eq!(
        error.to_string(),
        "UNDEFINED_HEADER (-113): The command is unrecognized"
    );
}

#[test]
fn formats_unknown_scpi_errors() {
    let error = ScpiError::new(-999);
    assert_eq!(error.abbreviation, "UNKNOWN_ERR");
    assert_eq!(error.description, "An unknown error occurred.");
    assert_eq!(
        error.to_string(),
        "UNKNOWN_ERR (-999): An unknown error occurred."
    );
}
