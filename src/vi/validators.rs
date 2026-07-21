//! SCPI value validators.
//!
//! Origin: `skrf/vi/validators.py`.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::marker::PhantomData;
use std::str::FromStr;

use regex::Regex;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ValidationError {
    #[error("{0}")]
    Invalid(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IntValidator {
    pub min: Option<i64>,
    pub max: Option<i64>,
}

impl IntValidator {
    pub const fn new(min: Option<i64>, max: Option<i64>) -> Self {
        Self { min, max }
    }

    pub fn validate_input(&self, arg: impl ToString) -> Result<i64, ValidationError> {
        let text = arg.to_string();
        let value = text
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite() && value.fract() == 0.0)
            .map(|value| value as i64)
            .ok_or_else(|| {
                ValidationError::Invalid(format!("Could not convert {text} to an int"))
            })?;
        self.check_bounds(value)?;
        Ok(value)
    }

    pub fn validate_output(&self, arg: impl ToString) -> Result<i64, ValidationError> {
        self.validate_input(arg)
    }

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatValidator {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub decimal_places: u32,
}

impl Default for FloatValidator {
    fn default() -> Self {
        Self::new(None, None, 50)
    }
}

impl FloatValidator {
    pub const fn new(min: Option<f64>, max: Option<f64>, decimal_places: u32) -> Self {
        Self {
            min,
            max,
            decimal_places,
        }
    }

    pub fn validate_input(&self, arg: impl ToString) -> Result<f64, ValidationError> {
        let text = arg.to_string();
        let value = text.parse::<f64>().map_err(|_| {
            ValidationError::Invalid(format!("Could not convert {text} to a float"))
        })?;
        self.check_bounds(value)?;

        if self.decimal_places >= 16 || !value.is_finite() {
            return Ok(value);
        }
        let scale = 10_f64.powi(self.decimal_places as i32);
        Ok((value * scale).round() / scale)
    }

    pub fn validate_output(&self, arg: impl ToString) -> Result<f64, ValidationError> {
        self.validate_input(arg)
    }

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrequencyValidator;

/// Python-compatible alias for `FreqValidator`.
pub type FreqValidator = FrequencyValidator;

impl FrequencyValidator {
    pub fn validate_input(&self, arg: impl ToString) -> Result<u64, ValidationError> {
        let text = arg.to_string();
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
        Ok((value * multiplier) as u64)
    }

    pub fn validate_output(&self, arg: impl ToString) -> Result<u64, ValidationError> {
        let text = arg.to_string();
        text.parse::<f64>()
            .ok()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map(|value| value as u64)
            .ok_or_else(|| {
                ValidationError::Invalid(format!(
                    "Response from instrument ({text}) could not be converted to an int"
                ))
            })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EnumValidator<E> {
    marker: PhantomData<E>,
}

impl<E> EnumValidator<E>
where
    E: FromStr + Display,
{
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    pub fn validate_input(&self, arg: impl ToString) -> Result<String, ValidationError> {
        let text = arg.to_string();
        text.parse::<E>()
            .map(|value| value.to_string())
            .map_err(|_| {
                ValidationError::Invalid(format!(
                    "{text} is not a valid {}",
                    std::any::type_name::<E>()
                ))
            })
    }

    pub fn validate_output(&self, arg: impl ToString) -> Result<E, ValidationError> {
        let text = arg.to_string();
        text.parse::<E>()
            .map_err(|_| ValidationError::Invalid(format!("Got unexpected response {text}")))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetValidator<T> {
    pub valid: Vec<T>,
}

impl<T> SetValidator<T>
where
    T: Clone + Display + FromStr + PartialEq,
{
    pub fn new(valid: impl IntoIterator<Item = T>) -> Result<Self, ValidationError> {
        let valid = valid.into_iter().collect::<Vec<_>>();
        if valid.is_empty() {
            return Err(ValidationError::Invalid(
                "Set of valid values must not be empty".into(),
            ));
        }
        Ok(Self { valid })
    }

    pub fn validate_input(&self, arg: impl ToString) -> Result<T, ValidationError> {
        let text = arg.to_string();
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

#[derive(Debug, Clone)]
pub struct DictValidator {
    pub argument_format: String,
    pub response_pattern: Regex,
}

impl DictValidator {
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
                match arguments.get(name) {
                    Some(value) => value.to_string(),
                    None => {
                        missing = Some(name.to_owned());
                        captures[0].to_owned()
                    }
                }
            });
        if let Some(name) = missing {
            Err(ValidationError::Invalid(format!(
                "Missing expected argument '{name}'"
            )))
        } else {
            Ok(output.into_owned())
        }
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelimitedStringValidator<T> {
    pub separator: char,
    marker: PhantomData<T>,
}

/// Python-compatible alias for `DelimitedStrValidator`.
pub type DelimitedStrValidator<T> = DelimitedStringValidator<T>;

impl<T> DelimitedStringValidator<T>
where
    T: Display + FromStr,
{
    pub const fn new(separator: char) -> Self {
        Self {
            separator,
            marker: PhantomData,
        }
    }

    pub fn validate_input(&self, values: &[T]) -> String {
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(&self.separator.to_string())
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanValidator {
    truthy: Vec<String>,
    falsey: Vec<String>,
    pub true_setting: String,
    pub false_setting: String,
}

impl Default for BooleanValidator {
    fn default() -> Self {
        Self::new(None, None, "1", "0")
    }
}

impl BooleanValidator {
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

    pub fn validate_input(&self, arg: impl ToString) -> Result<String, ValidationError> {
        let value = arg.to_string().to_lowercase();
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

    pub fn validate_output(&self, arg: impl ToString) -> bool {
        self.truthy.contains(&arg.to_string().to_lowercase())
    }
}
