#![cfg(feature = "visa")]

//! Integration tests for typed SCPI input and output validators.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use approx::assert_relative_eq;
use rust_rf::vi::validators::{
    BooleanValidator, DelimitedStringValidator, DictValidator, EnumValidator, FloatValidator,
    FrequencyValidator, IntValidator, SetValidator,
};

#[test]
/// Validates integer parsing and inclusive lower and upper bounds.
fn validates_integers_and_bounds() {
    let unbounded = IntValidator::default();
    for (input, expected) in [("0", 0), ("-1", -1), ("100000", 100_000), ("0.0", 0)] {
        assert_eq!(unbounded.validate_input(input).unwrap(), expected);
    }
    let bounded = IntValidator::new(Some(-1), Some(1));
    assert_eq!(bounded.validate_input("1").unwrap(), 1);
    assert!(bounded.validate_input("-2").is_err());
    assert!(bounded.validate_input("2").is_err());
}

#[test]
/// Validates floating-point parsing, bounds, and configured rounding.
fn validates_floats_bounds_and_rounding() {
    let validator = FloatValidator::new(Some(-0.5), Some(0.5), 2);
    assert_relative_eq!(
        validator.validate_input("0.254").unwrap(),
        0.25,
        epsilon = 1.0e-12
    );
    assert_relative_eq!(
        validator.validate_input("-0.255").unwrap(),
        -0.26,
        epsilon = 1.0e-12
    );
    assert!(validator.validate_input("-2.0").is_err());
    assert!(validator.validate_input("2.0").is_err());
}

#[test]
/// Validates frequency suffixes and numeric instrument responses.
fn validates_frequency_strings_and_numeric_responses() {
    let validator = FrequencyValidator;
    for (input, expected) in [
        ("100", 100),
        ("1hZ", 1),
        ("1 kHz", 1_000),
        ("1 MHz", 1_000_000),
        ("1.5 GHz", 1_500_000_000),
    ] {
        assert_eq!(validator.validate_input(input).unwrap(), expected);
    }
    assert_eq!(validator.validate_output("1000.0").unwrap(), 1_000);
    assert!(validator.validate_input("-1 GHz").is_err());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Foo {
    A,
    B,
}

impl Display for Foo {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::A => "A",
            Self::B => "B",
        })
    }
}

impl FromStr for Foo {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "A" => Ok(Self::A),
            "B" => Ok(Self::B),
            _ => Err(()),
        }
    }
}

#[test]
/// Converts enum values to SCPI strings and responses back to enum values.
fn validates_enums_in_both_directions() {
    let validator = EnumValidator::<Foo>::new();
    assert_eq!(validator.validate_input("A").unwrap(), "A");
    assert_eq!(validator.validate_input(Foo::B).unwrap(), "B");
    assert_eq!(validator.validate_output("A").unwrap(), Foo::A);
    assert!(validator.validate_input("C").is_err());
    assert!(validator.validate_output("C").is_err());
}

#[test]
/// Accepts only values contained in a non-empty typed set.
fn validates_membership_in_typed_set() {
    let validator = SetValidator::new([1, 2]).unwrap();
    assert_eq!(validator.validate_input("1").unwrap(), 1);
    assert_eq!(validator.validate_input(2).unwrap(), 2);
    assert!(validator.validate_input(3).is_err());
    assert!(SetValidator::<i32>::new([]).is_err());
}

#[test]
/// Formats dictionary commands and parses named response fields.
fn validates_dictionary_commands_and_responses() {
    let validator = DictValidator::new("{a},{b}", r"(?P<a>\d),(?P<b>\d)").unwrap();
    let arguments = BTreeMap::from([("a".to_owned(), 1), ("b".to_owned(), 2)]);
    assert_eq!(validator.validate_input(&arguments).unwrap(), "1,2");
    assert_eq!(
        validator.validate_output("1,2").unwrap(),
        BTreeMap::from([
            ("a".to_owned(), "1".to_owned()),
            ("b".to_owned(), "2".to_owned())
        ])
    );
    assert!(
        validator
            .validate_input(&BTreeMap::from([("a".to_owned(), 1)]))
            .is_err()
    );
    assert!(validator.validate_output("1,2,3").is_err());
}

#[test]
/// Formats and parses delimiter-separated typed values.
fn validates_delimited_values() {
    let validator = DelimitedStringValidator::<i32>::default();
    assert_eq!(validator.validate_input(&[1, 2, 3]), "1,2,3");
    assert_eq!(
        validator.validate_output("\"1,2,3\"").unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
/// Maps boolean values and aliases to configured SCPI settings and responses.
fn validates_boolean_settings_and_responses() {
    let validator = BooleanValidator::new(Some("enabled"), Some("disabled"), "ON", "OFF");
    assert_eq!(validator.validate_input(true).unwrap(), "ON");
    assert_eq!(validator.validate_input("enabled").unwrap(), "ON");
    assert_eq!(validator.validate_input("0").unwrap(), "OFF");
    assert!(validator.validate_input("perhaps").is_err());
    assert!(validator.validate_output("ON"));
    assert!(!validator.validate_output("OFF"));
}
