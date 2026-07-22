//! Typed validation and conversion between Rust values and SCPI command text.
//!
//! Input validation converts caller values into instrument settings; output
//! validation converts instrument responses into useful Rust values.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::marker::PhantomData;
use std::str::FromStr;

use num_traits::ToPrimitive;
use regex::Regex;
use thiserror::Error;

/// Error returned when a value cannot be converted or violates a constraint.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// A value is invalid, with a description of the failed constraint.
    #[error("{0}")]
    Invalid(String),
}

/// Converts integer settings and responses with optional inclusive bounds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IntValidator {
    /// Optional inclusive minimum.
    pub min: Option<i64>,
    /// Optional inclusive maximum.
    pub max: Option<i64>,
}

impl IntValidator {
    /// Creates an integer validator with optional inclusive bounds.
    #[must_use]
    pub const fn new(min: Option<i64>, max: Option<i64>) -> Self {
        Self { min, max }
    }

    /// Converts an input to a bounded integer suitable for an instrument command.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not integral or is outside the bounds.
    pub fn validate_input(&self, arg: impl ToString) -> Result<i64, ValidationError> {
        let text = arg.to_string();
        drop(arg);
        let value = text
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite() && value.fract() == 0.0)
            .and_then(|value| value.to_i64())
            .ok_or_else(|| {
                ValidationError::Invalid(format!("Could not convert {text} to an int"))
            })?;
        self.check_bounds(value)?;
        Ok(value)
    }

    /// Converts an instrument response to a bounded integer.
    ///
    /// # Errors
    ///
    /// Returns the errors reported by [`validate_input`](Self::validate_input).
    pub fn validate_output(&self, arg: impl ToString) -> Result<i64, ValidationError> {
        self.validate_input(arg)
    }

    /// Checks an integer against the configured inclusive bounds.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` is below `min` or above `max`.
    pub fn check_bounds(&self, value: i64) -> Result<(), ValidationError> {
        if let Some(min) = self.min
            && value < min
        {
            return Err(ValidationError::Invalid(format!("{value} < {min}")));
        }
        if let Some(max) = self.max
            && value > max
        {
            return Err(ValidationError::Invalid(format!("{value} > {max}")));
        }
        Ok(())
    }
}

/// Converts floating-point settings and responses with bounds and rounding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatValidator {
    /// Optional inclusive minimum.
    pub min: Option<f64>,
    /// Optional inclusive maximum.
    pub max: Option<f64>,
    /// Number of decimal places retained during input conversion.
    pub decimal_places: u32,
}

impl Default for FloatValidator {
    fn default() -> Self {
        Self::new(None, None, 50)
    }
}

impl FloatValidator {
    /// Creates a floating-point validator with bounds and rounding precision.
    #[must_use]
    pub const fn new(min: Option<f64>, max: Option<f64>, decimal_places: u32) -> Self {
        Self {
            min,
            max,
            decimal_places,
        }
    }

    /// Converts, bounds-checks, and rounds an instrument setting.
    ///
    /// # Errors
    ///
    /// Returns an error when conversion fails or the value is outside the bounds.
    pub fn validate_input(&self, arg: impl ToString) -> Result<f64, ValidationError> {
        let text = arg.to_string();
        drop(arg);
        let value = text.parse::<f64>().map_err(|_| {
            ValidationError::Invalid(format!("Could not convert {text} to a float"))
        })?;
        self.check_bounds(value)?;

        let Ok(decimal_places) = i32::try_from(self.decimal_places) else {
            return Ok(value);
        };
        if decimal_places >= 16 || !value.is_finite() {
            return Ok(value);
        }
        let scale = 10_f64.powi(decimal_places);
        Ok((value * scale).round() / scale)
    }

    /// Converts an instrument response using the input validation rules.
    ///
    /// # Errors
    ///
    /// Returns the errors reported by [`validate_input`](Self::validate_input).
    pub fn validate_output(&self, arg: impl ToString) -> Result<f64, ValidationError> {
        self.validate_input(arg)
    }

    /// Checks a value against the configured inclusive bounds.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` is below `min` or above `max`.
    pub fn check_bounds(&self, value: f64) -> Result<(), ValidationError> {
        if let Some(min) = self.min
            && value < min
        {
            return Err(ValidationError::Invalid(format!("{value} < {min}")));
        }
        if let Some(max) = self.max
            && value > max
        {
            return Err(ValidationError::Invalid(format!("{value} > {max}")));
        }
        Ok(())
    }
}

/// Converts non-negative frequencies expressed in hertz or with SI prefixes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrequencyValidator;

/// Compatibility alias for the upstream validator name.
pub type FreqValidator = FrequencyValidator;

impl FrequencyValidator {
    /// Converts input such as `100`, `1 kHz`, `1 MHz`, or `1.5 GHz` to hertz.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is not a supported frequency string.
    pub fn validate_input(&self, arg: impl ToString) -> Result<u64, ValidationError> {
        let text = arg.to_string();
        drop(arg);
        let expression = Regex::new(
            r"^(?P<value>\d+(?:\.\d*)?)\s*(?P<prefix>[kMG])?(?:[hH][zZ])?$",
        )
        .map_err(|error| ValidationError::Invalid(format!("invalid frequency regex: {error}")))?;
        let captures = expression
            .captures(&text)
            .ok_or_else(|| ValidationError::Invalid("Invalid frequency string".into()))?;
        let value = captures["value"]
            .parse::<f64>()
            .map_err(|_| ValidationError::Invalid("Invalid frequency string".into()))?;
        let multiplier = match captures.name("prefix").map(|capture| capture.as_str()) {
            Some("k") => 1e3,
            Some("M") => 1e6,
            Some("G") => 1e9,
            _ => 1.0,
        };
        (value * multiplier).to_u64().ok_or_else(|| {
            ValidationError::Invalid(format!("Frequency is outside the supported range: {text}"))
        })
    }

    /// Converts a non-negative numeric instrument response to integer hertz.
    ///
    /// # Errors
    ///
    /// Returns an error when the response is not finite and non-negative.
    pub fn validate_output(&self, arg: impl ToString) -> Result<u64, ValidationError> {
        let text = arg.to_string();
        drop(arg);
        text.parse::<f64>()
            .ok()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .and_then(|value| value.to_u64())
            .ok_or_else(|| {
                ValidationError::Invalid(format!(
                    "Response from instrument ({text}) could not be converted to an int"
                ))
            })
    }
}

/// Converts values to and from a string-backed enum type.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnumValidator<E> {
    marker: PhantomData<E>,
}

impl<E> EnumValidator<E>
where
    E: FromStr + Display,
{
    /// Creates a validator for enum type `E`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    /// Validates a setting and returns its canonical display string.
    ///
    /// # Errors
    ///
    /// Returns an error when the input cannot be parsed as `E`.
    pub fn validate_input(&self, arg: impl ToString) -> Result<String, ValidationError> {
        let text = arg.to_string();
        drop(arg);
        text.parse::<E>()
            .map(|value| value.to_string())
            .map_err(|_| {
                ValidationError::Invalid(format!(
                    "{text} is not a valid {}",
                    std::any::type_name::<E>()
                ))
            })
    }

    /// Parses an instrument response as `E`.
    ///
    /// # Errors
    ///
    /// Returns an error when the response is not a valid enum value.
    pub fn validate_output(&self, arg: impl ToString) -> Result<E, ValidationError> {
        let text = arg.to_string();
        drop(arg);
        text.parse::<E>()
            .map_err(|_| ValidationError::Invalid(format!("Got unexpected response {text}")))
    }
}

/// Restricts settings to a non-empty homogeneous set of values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetValidator<T> {
    /// Values accepted by this validator.
    pub valid: Vec<T>,
}

impl<T> SetValidator<T>
where
    T: Clone + Display + FromStr + PartialEq,
{
    /// Creates a validator from the accepted values.
    ///
    /// # Errors
    ///
    /// Returns an error when the set is empty.
    pub fn new(valid: impl IntoIterator<Item = T>) -> Result<Self, ValidationError> {
        let valid = valid.into_iter().collect::<Vec<_>>();
        if valid.is_empty() {
            return Err(ValidationError::Invalid(
                "Set of valid values must not be empty".into(),
            ));
        }
        Ok(Self { valid })
    }

    /// Converts a setting and verifies that the set contains it.
    ///
    /// # Errors
    ///
    /// Returns an error when conversion fails or the value is not accepted.
    pub fn validate_input(&self, arg: impl ToString) -> Result<T, ValidationError> {
        let text = arg.to_string();
        drop(arg);
        let value = text.parse::<T>().map_err(|_| {
            ValidationError::Invalid(format!("Could not convert {text} to the set value type"))
        })?;
        if self.valid.contains(&value) {
            Ok(value)
        } else {
            Err(ValidationError::Invalid(format!(
                "{value} is not in the valid set"
            )))
        }
    }
}

/// Formats named command arguments and parses named response fields.
#[derive(Debug, Clone)]
pub struct DictValidator {
    /// Command template containing `{name}` placeholders.
    pub argument_format: String,
    /// Regular expression with named captures for complete responses.
    pub response_pattern: Regex,
}

impl DictValidator {
    /// Creates a dictionary validator from a command template and response regex.
    ///
    /// # Errors
    ///
    /// Returns an error when `response_pattern` is not a valid regular expression.
    pub fn new(
        argument_format: impl Into<String>,
        response_pattern: impl AsRef<str>,
    ) -> Result<Self, ValidationError> {
        let response_pattern = Regex::new(response_pattern.as_ref()).map_err(|error| {
            ValidationError::Invalid(format!("Invalid response regex: {error}"))
        })?;
        Ok(Self {
            argument_format: argument_format.into(),
            response_pattern,
        })
    }

    /// Replaces every named placeholder with the corresponding argument.
    ///
    /// # Errors
    ///
    /// Returns an error when a required argument is absent.
    pub fn validate_input<V>(
        &self,
        arguments: &BTreeMap<String, V>,
    ) -> Result<String, ValidationError>
    where
        V: Display,
    {
        let placeholder = Regex::new(r"\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}").map_err(|error| {
            ValidationError::Invalid(format!("invalid placeholder regex: {error}"))
        })?;
        let mut missing = None;
        let output =
            placeholder.replace_all(&self.argument_format, |captures: &regex::Captures<'_>| {
                let name = &captures["name"];
                arguments.get(name).map_or_else(
                    || {
                        missing = Some(name.to_owned());
                        captures[0].to_owned()
                    },
                    ToString::to_string,
                )
            });
        missing.map_or_else(
            || Ok(output.into_owned()),
            |name| {
                Err(ValidationError::Invalid(format!(
                    "Missing expected argument '{name}'"
                )))
            },
        )
    }

    /// Parses a complete response into its named capture groups.
    ///
    /// # Errors
    ///
    /// Returns an error when the response does not fully match the pattern.
    pub fn validate_output(
        &self,
        response: &str,
    ) -> Result<BTreeMap<String, String>, ValidationError> {
        let captures = self.response_pattern.captures(response).filter(|captures| {
            captures
                .get(0)
                .is_some_and(|capture| capture.as_str() == response)
        });
        let Some(captures) = captures else {
            return Err(ValidationError::Invalid(format!(
                "Response did not fit regex. Response: {response} Pattern: {}",
                self.response_pattern.as_str()
            )));
        };
        Ok(self
            .response_pattern
            .capture_names()
            .flatten()
            .filter_map(|name| {
                captures
                    .name(name)
                    .map(|capture| (name.to_owned(), capture.as_str().to_owned()))
            })
            .collect())
    }
}

/// Converts homogeneous values to and from delimiter-separated text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelimitedStringValidator<T> {
    /// Separator placed between values.
    pub separator: char,
    marker: PhantomData<T>,
}

/// Compatibility alias for the upstream validator name.
pub type DelimitedStrValidator<T> = DelimitedStringValidator<T>;

impl<T> DelimitedStringValidator<T>
where
    T: Display + FromStr,
{
    /// Creates a validator using `separator` between values.
    #[must_use]
    pub const fn new(separator: char) -> Self {
        Self {
            separator,
            marker: PhantomData,
        }
    }

    /// Formats values as delimiter-separated command text.
    pub fn validate_input(&self, values: &[T]) -> String {
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(&self.separator.to_string())
    }

    /// Parses delimiter-separated response text, ignoring double quotes.
    ///
    /// # Errors
    ///
    /// Returns an error when a field cannot be parsed as `T`.
    pub fn validate_output(&self, response: &str) -> Result<Vec<T>, ValidationError> {
        response
            .replace('"', "")
            .split(self.separator)
            .map(|value| {
                value
                    .parse::<T>()
                    .map_err(|_| ValidationError::Invalid(format!("Could not convert {value}")))
            })
            .collect()
    }
}

impl<T> Default for DelimitedStringValidator<T>
where
    T: Display + FromStr,
{
    fn default() -> Self {
        Self::new(',')
    }
}

/// Converts truthy and falsey aliases to configured instrument settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanValidator {
    truthy: Vec<String>,
    falsey: Vec<String>,
    /// Instrument command value representing `true`.
    pub true_setting: String,
    /// Instrument command value representing `false`.
    pub false_setting: String,
}

impl Default for BooleanValidator {
    fn default() -> Self {
        Self::new(None, None, "1", "0")
    }
}

impl BooleanValidator {
    /// Creates a boolean validator with optional response aliases and command values.
    pub fn new(
        true_response: Option<&str>,
        false_response: Option<&str>,
        true_setting: impl Into<String>,
        false_setting: impl Into<String>,
    ) -> Self {
        let mut truthy = vec!["1".into(), "on".into(), "true".into()];
        let mut falsey = vec!["0".into(), "off".into(), "false".into()];
        if let Some(value) = true_response {
            truthy.push(value.to_lowercase());
        }
        if let Some(value) = false_response {
            falsey.push(value.to_lowercase());
        }
        Self {
            truthy,
            falsey,
            true_setting: true_setting.into(),
            false_setting: false_setting.into(),
        }
    }

    /// Converts a truthy or falsey input to its configured command value.
    ///
    /// Accepted built-ins are `1`, `on`, `true`, `0`, `off`, and `false`,
    /// case-insensitively, plus the optional aliases supplied at construction.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is neither truthy nor falsey.
    pub fn validate_input(&self, arg: impl ToString) -> Result<String, ValidationError> {
        let value = arg.to_string().to_lowercase();
        drop(arg);
        if self.truthy.contains(&value) {
            Ok(self.true_setting.clone())
        } else if self.falsey.contains(&value) {
            Ok(self.false_setting.clone())
        } else {
            Err(ValidationError::Invalid(
                "Argument must be a truthy or falsey value".into(),
            ))
        }
    }

    /// Returns whether an instrument response is one of the truthy aliases.
    pub fn validate_output(&self, arg: impl ToString) -> bool {
        let value = arg.to_string().to_lowercase();
        drop(arg);
        self.truthy.contains(&value)
    }
}
