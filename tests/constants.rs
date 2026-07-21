use approx::assert_relative_eq;
use rust_rf::constants::{
    BOLTZMANN_CONSTANT, DistanceUnit, INCH, MIL, REFERENCE_TEMPERATURE, SPEED_OF_LIGHT,
    distances_to_meters, to_meters,
};

#[test]
fn exposes_upstream_physical_constants() {
    assert_eq!(SPEED_OF_LIGHT, 299_792_458.0);
    assert_eq!(INCH, 0.0254);
    assert_eq!(MIL, 25.4e-6);
    assert_eq!(BOLTZMANN_CONSTANT, 1.380_648_52e-23);
    assert_eq!(REFERENCE_TEMPERATURE, 290.0);
}

#[test]
fn converts_length_and_time_units_to_meters() {
    assert_eq!(
        to_meters(2.0, DistanceUnit::Centimeter, None).unwrap(),
        0.02
    );
    assert_eq!(to_meters(1.0, DistanceUnit::Inch, None).unwrap(), INCH);
    assert_relative_eq!(
        to_meters(1.0, DistanceUnit::Nanosecond, None).unwrap(),
        0.299_792_458,
        epsilon = 1.0e-15
    );
    assert_eq!(
        to_meters(2.0, DistanceUnit::Microsecond, Some(2.0e8)).unwrap(),
        400.0
    );
    assert_eq!(
        distances_to_meters(&[1.0, 2.0], DistanceUnit::Millimeter, None).unwrap(),
        vec![0.001, 0.002]
    );
}

#[test]
fn parses_units_and_rejects_invalid_inputs() {
    assert!("GHz".parse::<DistanceUnit>().is_err());
    assert_eq!(
        "µm".parse::<DistanceUnit>().unwrap(),
        DistanceUnit::Micrometer
    );
    assert!(to_meters(f64::NAN, DistanceUnit::Meter, None).is_err());
    assert!(to_meters(1.0, DistanceUnit::Second, Some(-1.0)).is_err());
}
