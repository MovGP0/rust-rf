//! Physical constants, numerical approximations, and unit conversions.

use std::str::FromStr;

use crate::{Error, Result};

/// Speed of light in vacuum, in meters per second.
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// International inch in meters.
pub const INCH: f64 = 0.0254;
/// One thousandth of an inch in meters.
pub const MIL: f64 = 25.4e-6;
/// Magnetic permeability of free space in henries per meter.
pub const FREE_SPACE_PERMEABILITY: f64 = 1.256_637_061_27e-6;
/// Electric permittivity of free space in farads per meter.
pub const FREE_SPACE_PERMITTIVITY: f64 = 8.854_187_818_8e-12;
/// Wave impedance of free space in ohms.
pub const FREE_SPACE_IMPEDANCE: f64 = 376.730_313_412;
/// High, finite value used in place of mathematical infinity.
pub const NUMERICAL_INFINITY: f64 = 1.0e99;
/// Very small, non-zero value used to handle numerical singularities.
pub const ALMOST_ZERO: f64 = 1.0e-12;
/// Small value commonly used for numerical comparisons.
pub const ZERO: f64 = 1.0e-4;
/// Value slightly greater than one, used to avoid numerical singularities.
pub const ONE: f64 = 1.0 + 1.0e-14;
/// Finite lower bound used in place of the logarithm of zero or a negative value.
pub const LOG_OF_NEGATIVE: f64 = -100.0;
/// Boltzmann constant in joules per kelvin.
pub const BOLTZMANN_CONSTANT: f64 = 1.380_648_52e-23;
/// Conventional room temperature in kelvin.
pub const REFERENCE_TEMPERATURE: f64 = 290.0;
/// Smallest eigenvalue accepted by the numerical eigenvalue adjustment routines.
pub const MINIMUM_EIGENVALUE: f64 = 1.0e-12;
/// Minimum eigenvalue ratio relative to the maximum eigenvalue.
pub const MINIMUM_EIGENVALUE_RATIO: f64 = 1.0e-9;

/// Distance and propagation-time units accepted by `to_meters`.
///
/// Origin: `skrf/constants.py::get_distance_dict`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceUnit {
    /// Meter (`m`).
    Meter,
    /// Centimeter (`cm`).
    Centimeter,
    /// Millimeter (`mm`).
    Millimeter,
    /// Micrometer (`um` or `µm`).
    Micrometer,
    /// International inch (`in`).
    Inch,
    /// One thousandth of an inch (`mil`).
    Mil,
    /// Propagation distance represented by one second at the supplied group velocity.
    Second,
    /// Propagation distance represented by one microsecond at the supplied group velocity.
    Microsecond,
    /// Propagation distance represented by one nanosecond at the supplied group velocity.
    Nanosecond,
    /// Propagation distance represented by one picosecond at the supplied group velocity.
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
    /// Returns this unit's conversion factor to meters.
    ///
    /// Time units use `group_velocity`, or [`SPEED_OF_LIGHT`] when it is not supplied.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when `group_velocity` is not finite or is
    /// negative.
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

/// Converts a scalar distance or propagation-time value to meters.
///
/// `group_velocity` is used for second-based units and defaults to [`SPEED_OF_LIGHT`].
///
/// # Errors
///
/// Returns [`Error::Unsupported`] when `value` is not finite or
/// `group_velocity` is not finite or is negative.
pub fn to_meters(value: f64, unit: DistanceUnit, group_velocity: Option<f64>) -> Result<f64> {
    if !value.is_finite() {
        return Err(Error::Unsupported(
            "distance value must be finite".to_owned(),
        ));
    }
    Ok(value * unit.meters_per_unit(group_velocity)?)
}

/// Converts a slice of distances or propagation-time values to meters.
///
/// `group_velocity` is used for second-based units and defaults to [`SPEED_OF_LIGHT`].
///
/// # Errors
///
/// Returns [`Error::Unsupported`] when any value is not finite or
/// `group_velocity` is not finite or is negative.
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
