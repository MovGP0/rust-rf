use std::str::FromStr;

use crate::{Error, Result};

/// Origin: `skrf/constants.py::c`.
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
pub const INCH: f64 = 0.0254;
pub const MIL: f64 = 25.4e-6;
pub const FREE_SPACE_PERMEABILITY: f64 = 1.256_637_061_27e-6;
pub const FREE_SPACE_PERMITTIVITY: f64 = 8.854_187_818_8e-12;
pub const FREE_SPACE_IMPEDANCE: f64 = 376.730_313_412;
pub const NUMERICAL_INFINITY: f64 = 1.0e99;
pub const ALMOST_ZERO: f64 = 1.0e-12;
pub const ZERO: f64 = 1.0e-4;
pub const ONE: f64 = 1.0 + 1.0e-14;
pub const LOG_OF_NEGATIVE: f64 = -100.0;
pub const BOLTZMANN_CONSTANT: f64 = 1.380_648_52e-23;
pub const REFERENCE_TEMPERATURE: f64 = 290.0;
pub const MINIMUM_EIGENVALUE: f64 = 1.0e-12;
pub const MINIMUM_EIGENVALUE_RATIO: f64 = 1.0e-9;

/// Distance and propagation-time units accepted by `to_meters`.
///
/// Origin: `skrf/constants.py::get_distance_dict`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceUnit {
    Meter,
    Centimeter,
    Millimeter,
    Micrometer,
    Inch,
    Mil,
    Second,
    Microsecond,
    Nanosecond,
    Picosecond,
}

impl FromStr for DistanceUnit {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "m" => Ok(Self::Meter),
            "cm" => Ok(Self::Centimeter),
            "mm" => Ok(Self::Millimeter),
            "um" | "µm" => Ok(Self::Micrometer),
            "in" => Ok(Self::Inch),
            "mil" => Ok(Self::Mil),
            "s" => Ok(Self::Second),
            "us" | "µs" => Ok(Self::Microsecond),
            "ns" => Ok(Self::Nanosecond),
            "ps" => Ok(Self::Picosecond),
            _ => Err(Error::Unsupported(format!(
                "unknown distance unit '{value}'"
            ))),
        }
    }
}

impl DistanceUnit {
    pub fn meters_per_unit(self, group_velocity: Option<f64>) -> Result<f64> {
        let velocity = group_velocity.unwrap_or(SPEED_OF_LIGHT);
        if !velocity.is_finite() || velocity < 0.0 {
            return Err(Error::Unsupported(
                "group velocity must be finite and non-negative".to_owned(),
            ));
        }
        Ok(match self {
            Self::Meter => 1.0,
            Self::Centimeter => 1.0e-2,
            Self::Millimeter => 1.0e-3,
            Self::Micrometer => 1.0e-6,
            Self::Inch => INCH,
            Self::Mil => MIL,
            Self::Second => velocity,
            Self::Microsecond => 1.0e-6 * velocity,
            Self::Nanosecond => 1.0e-9 * velocity,
            Self::Picosecond => 1.0e-12 * velocity,
        })
    }
}

/// Port of `skrf.constants.to_meters` for a scalar value.
pub fn to_meters(value: f64, unit: DistanceUnit, group_velocity: Option<f64>) -> Result<f64> {
    if !value.is_finite() {
        return Err(Error::Unsupported(
            "distance value must be finite".to_owned(),
        ));
    }
    Ok(value * unit.meters_per_unit(group_velocity)?)
}

/// Array-like counterpart of `skrf.constants.to_meters`.
pub fn distances_to_meters(
    values: &[f64],
    unit: DistanceUnit,
    group_velocity: Option<f64>,
) -> Result<Vec<f64>> {
    let scale = unit.meters_per_unit(group_velocity)?;
    values
        .iter()
        .map(|value| {
            if value.is_finite() {
                Ok(*value * scale)
            } else {
                Err(Error::Unsupported(
                    "distance values must be finite".to_owned(),
                ))
            }
        })
        .collect()
}
