//! Frequency-axis construction, scaling, slicing, and time-domain helpers.

use std::fmt;

use ndarray::Array1;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const FREQUENCY_EQUALITY_TOLERANCE_HZ: f64 = 1.0e-4;
const SWEEP_RELATIVE_TOLERANCE: f64 = 0.05;

/// A frequency band or arbitrary frequency axis.
///
/// Values are stored internally in hertz while [`Self::unit`] controls their
/// preferred display and input scaling. Linear or logarithmic bands can be
/// created with [`Self::new`]; arbitrary vectors use [`Self::from_values`].
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rust_rf::{Frequency, FrequencyUnit, SweepType};
///
/// # fn main() -> rust_rf::Result<()> {
/// let wr1p5 = Frequency::new(500.0, 750.0, 401, FrequencyUnit::GHz, SweepType::Linear)?;
/// let measured = Frequency::from_values(array![75.0, 80.0, 100.0], FrequencyUnit::GHz)?;
/// assert_eq!(wr1p5.points(), 401);
/// assert_eq!(measured.start_scaled(), Some(75.0));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frequency {
    values_hz: Array1<f64>,
    unit: FrequencyUnit,
    sweep_type: SweepType,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
/// Scaling unit for frequency input and display.
pub enum FrequencyUnit {
    /// Hertz.
    Hz,
    /// Kilohertz.
    KHz,
    /// Megahertz.
    MHz,
    /// Gigahertz.
    #[default]
    GHz,
    /// Terahertz.
    THz,
}

impl FrequencyUnit {
    /// Returns the multiplier that converts this unit to hertz.
    #[must_use]
    pub const fn multiplier(self) -> f64 {
        match self {
            Self::Hz => 1.0,
            Self::KHz => 1.0e3,
            Self::MHz => 1.0e6,
            Self::GHz => 1.0e9,
            Self::THz => 1.0e12,
        }
    }

    /// Returns the correctly capitalized unit symbol.
    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Hz => "Hz",
            Self::KHz => "kHz",
            Self::MHz => "MHz",
            Self::GHz => "GHz",
            Self::THz => "THz",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
/// Frequency sweep classification.
pub enum SweepType {
    /// Evenly spaced frequency values.
    #[default]
    Linear,
    /// Geometrically spaced positive frequency values.
    Logarithmic,
    /// Explicit values that are neither linear nor logarithmic.
    Arbitrary,
}

#[derive(Clone, Copy)]
enum BinaryOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    FloorDivide,
    Remainder,
}

impl Frequency {
    /// Creates a frequency band from start, stop, point count, and unit.
    ///
    /// `start` and `stop` are expressed in `unit`. [`SweepType::Arbitrary`]
    /// must instead be constructed with [`Self::from_values`]. A logarithmic
    /// sweep requires positive endpoints.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when an endpoint or its scaled value
    /// is not finite, a logarithmic endpoint is not positive, or an arbitrary
    /// sweep is requested without explicit values.
    pub fn new(
        start: f64,
        stop: f64,
        points: usize,
        unit: FrequencyUnit,
        sweep_type: SweepType,
    ) -> Result<Self> {
        validate_finite(&[start, stop])?;

        if points == 0 {
            return Ok(Self {
                values_hz: Array1::default(0),
                unit,
                sweep_type,
            });
        }

        let start_hz = scale_to_hz(start, unit)?;
        let stop_hz = scale_to_hz(stop, unit)?;
        let values_hz = match sweep_type {
            SweepType::Linear => linear_space(start_hz, stop_hz, points),
            SweepType::Logarithmic if start_hz > 0.0 && stop_hz > 0.0 => {
                logarithmic_space(start_hz, stop_hz, points)
            }
            SweepType::Logarithmic => {
                return Err(Error::InvalidFrequency(
                    "a logarithmic sweep requires positive start and stop values".to_owned(),
                ));
            }
            SweepType::Arbitrary => {
                return Err(Error::InvalidFrequency(
                    "an arbitrary sweep must be constructed from explicit values".to_owned(),
                ));
            }
        };

        Ok(Self {
            values_hz,
            unit,
            sweep_type,
        })
    }

    /// Creates a frequency axis from explicit values already expressed in hertz.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when a value is not finite.
    pub fn from_hz(values_hz: Array1<f64>) -> Result<Self> {
        Self::from_values(values_hz, FrequencyUnit::Hz)
    }

    /// Creates a frequency axis from an explicit vector expressed in `unit`.
    ///
    /// The sweep type is inferred from the resulting values.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when an input value or its scaled
    /// value is not finite.
    pub fn from_values(values: Array1<f64>, unit: FrequencyUnit) -> Result<Self> {
        validate_finite(values.as_slice().unwrap_or(&[]))?;

        let multiplier = unit.multiplier();
        let values_hz = values.mapv_into(|value| value * multiplier);
        validate_finite(values_hz.as_slice().unwrap_or(&[]))?;
        let sweep_type = classify_sweep(&values_hz);

        Ok(Self {
            values_hz,
            unit,
            sweep_type,
        })
    }

    /// Returns the frequency vector in hertz.
    #[must_use]
    pub const fn values_hz(&self) -> &Array1<f64> {
        &self.values_hz
    }

    /// Returns the preferred scaling unit.
    #[must_use]
    pub const fn unit(&self) -> FrequencyUnit {
        self.unit
    }

    /// Changes the preferred display/input scaling unit.
    ///
    /// Stored values remain unchanged in hertz.
    pub const fn set_unit(&mut self, unit: FrequencyUnit) {
        self.unit = unit;
    }

    /// Returns the classified sweep type.
    #[must_use]
    pub const fn sweep_type(&self) -> SweepType {
        self.sweep_type
    }

    /// Returns the starting frequency in hertz, or `None` for an empty axis.
    #[must_use]
    pub fn start(&self) -> Option<f64> {
        self.values_hz.first().copied()
    }

    /// Returns the final frequency in hertz, or `None` for an empty axis.
    #[must_use]
    pub fn stop(&self) -> Option<f64> {
        self.values_hz.last().copied()
    }

    /// Returns the starting frequency in [`Self::unit`].
    #[must_use]
    pub fn start_scaled(&self) -> Option<f64> {
        self.start().map(|value| value / self.unit.multiplier())
    }

    /// Returns the final frequency in [`Self::unit`].
    #[must_use]
    pub fn stop_scaled(&self) -> Option<f64> {
        self.stop().map(|value| value / self.unit.multiplier())
    }

    /// Returns the number of frequency points.
    #[must_use]
    pub fn points(&self) -> usize {
        self.values_hz.len()
    }

    /// Returns the exact center frequency in hertz.
    #[must_use]
    pub fn center(&self) -> Option<f64> {
        Some(self.start()? + (self.stop()? - self.start()?) / 2.0)
    }

    /// Returns the index closest to the center frequency.
    #[must_use]
    pub fn center_index(&self) -> Option<usize> {
        (!self.values_hz.is_empty()).then_some(self.points() / 2)
    }

    /// Returns the exact center frequency in [`Self::unit`].
    #[must_use]
    pub fn center_scaled(&self) -> Option<f64> {
        self.center().map(|value| value / self.unit.multiplier())
    }

    /// Returns the inter-frequency step in hertz for an evenly spaced sweep.
    ///
    /// Use [`Self::gradient_hz`] for a general, nonuniform axis.
    #[must_use]
    pub fn step(&self) -> Option<f64> {
        let span = self.span()?;
        if span == 0.0 {
            Some(0.0)
        } else {
            Some(span / (self.points() - 1).to_f64().unwrap_or(f64::INFINITY))
        }
    }

    /// Returns the inter-frequency step in [`Self::unit`].
    #[must_use]
    pub fn step_scaled(&self) -> Option<f64> {
        self.step().map(|value| value / self.unit.multiplier())
    }

    /// Returns the absolute frequency span in hertz.
    #[must_use]
    pub fn span(&self) -> Option<f64> {
        Some((self.stop()? - self.start()?).abs())
    }

    /// Returns the absolute frequency span in [`Self::unit`].
    #[must_use]
    pub fn span_scaled(&self) -> Option<f64> {
        self.span().map(|value| value / self.unit.multiplier())
    }

    /// Returns the frequency vector in [`Self::unit`].
    #[must_use]
    pub fn scaled(&self) -> Array1<f64> {
        self.values_hz.mapv(|value| value / self.unit.multiplier())
    }

    /// Returns angular frequency in radians per second.
    ///
    /// Angular frequency is $\omega = 2\pi f$.
    #[must_use]
    pub fn angular(&self) -> Array1<f64> {
        self.values_hz.mapv(|value| std::f64::consts::TAU * value)
    }

    /// Returns the numerical gradient of the frequency vector in hertz.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axis has fewer than two
    /// points.
    pub fn gradient_hz(&self) -> Result<Array1<f64>> {
        gradient(&self.values_hz)
    }

    /// Returns the numerical gradient in [`Self::unit`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axis has fewer than two
    /// points.
    pub fn gradient_scaled(&self) -> Result<Array1<f64>> {
        Ok(self.gradient_hz()? / self.unit.multiplier())
    }

    /// Returns the numerical gradient of angular frequency in radians per second.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axis has fewer than two
    /// points.
    pub fn angular_gradient(&self) -> Result<Array1<f64>> {
        Ok(self.gradient_hz()? * std::f64::consts::TAU)
    }

    /// Returns the time vector corresponding to a padded frequency axis.
    ///
    /// `padding_points` extends the default output length, while
    /// `output_points` overrides it. If `bandpass` is omitted, an axis that
    /// starts above DC is treated as band-pass.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the point-count calculation
    /// overflows, the axis is empty, or its frequency step is not positive.
    pub fn padded_time(
        &self,
        padding_points: usize,
        output_points: Option<usize>,
        bandpass: Option<bool>,
    ) -> Result<Array1<f64>> {
        let bandpass = bandpass.unwrap_or_else(|| self.start() != Some(0.0));
        let mut points = output_points.unwrap_or_else(|| self.points() + padding_points);
        if !bandpass {
            points = points
                .checked_mul(2)
                .and_then(|value| value.checked_sub(1))
                .ok_or_else(|| {
                    Error::InvalidFrequency("time-axis point count overflowed".to_owned())
                })?;
        }
        if points == 0 {
            return Ok(Array1::default(0));
        }

        let step = self.step().ok_or_else(|| {
            Error::InvalidFrequency("an empty axis has no time transform".to_owned())
        })?;
        if step <= 0.0 {
            return Err(Error::InvalidFrequency(
                "a positive frequency step is required for a time transform".to_owned(),
            ));
        }
        let time_step = 1.0 / (points.to_f64().unwrap_or(f64::INFINITY) * step);

        if bandpass {
            let positive_extent = ((points - 1) / 2).to_f64().unwrap_or(f64::INFINITY) * time_step;
            let start = if points % 2 == 0 {
                -positive_extent - time_step
            } else {
                -(points / 2).to_f64().unwrap_or(f64::INFINITY) * time_step
            };
            Ok(linear_space(start, positive_extent, points))
        } else {
            let extent = time_step * (points / 2).to_f64().unwrap_or(f64::INFINITY);
            Ok(linear_space(-extent, extent, points))
        }
    }

    /// Returns the time vector in seconds.
    ///
    /// For an evenly spaced sweep, the time period is
    /// $2(N-1)/\Delta f$.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axis is empty or its
    /// frequency step is not positive.
    pub fn time(&self) -> Result<Array1<f64>> {
        self.padded_time(0, None, Some(true))
    }

    /// Returns the time vector in nanoseconds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axis is empty or its
    /// frequency step is not positive.
    pub fn time_nanoseconds(&self) -> Result<Array1<f64>> {
        Ok(self.time()? * 1.0e9)
    }

    /// Returns whether every frequency is strictly greater than its predecessor.
    #[must_use]
    pub fn is_monotonic_increasing(&self) -> bool {
        self.values_hz
            .iter()
            .zip(self.values_hz.iter().skip(1))
            .all(|(left, right)| right > left)
    }

    /// Removes duplicate and non-increasing frequency values.
    ///
    /// Returns the original zero-based indices that were dropped.
    pub fn drop_non_monotonic_increasing(&mut self) -> Vec<usize> {
        if self.values_hz.len() < 2 {
            return Vec::new();
        }

        let mut invalid_indices = Vec::new();
        let mut retained_values = Vec::with_capacity(self.values_hz.len());
        retained_values.push(self.values_hz[0]);

        for index in 1..self.values_hz.len() {
            if self.values_hz[index] <= self.values_hz[index - 1] {
                invalid_indices.push(index);
            } else {
                retained_values.push(self.values_hz[index]);
            }
        }

        self.values_hz = Array1::from_vec(retained_values);
        self.sweep_type = classify_sweep(&self.values_hz);
        invalid_indices
    }

    /// Returns the inclusive range between `start` and `stop`.
    ///
    /// This is the typed Rust counterpart of a scikit-rf string slice such as
    /// `"2-5ghz"`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when a boundary or its scaled value
    /// is not finite, or when `start` exceeds `stop`.
    pub fn slice_range(&self, start: f64, stop: f64, unit: FrequencyUnit) -> Result<Self> {
        validate_finite(&[start, stop])?;
        let start_hz = scale_to_hz(start, unit)?;
        let stop_hz = scale_to_hz(stop, unit)?;
        if start_hz > stop_hz {
            return Err(Error::InvalidFrequency(
                "slice start must not exceed slice stop".to_owned(),
            ));
        }

        let values = self
            .values_hz
            .iter()
            .copied()
            .filter(|value| *value >= start_hz && *value <= stop_hz)
            .collect::<Vec<_>>();
        let mut frequency = Self::from_hz(Array1::from_vec(values))?;
        frequency.unit = self.unit;
        Ok(frequency)
    }

    /// Returns the frequency point nearest `value` in `unit`.
    ///
    /// This is the typed Rust counterpart of a one-value scikit-rf frequency
    /// slice.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axis is empty or `value`
    /// cannot be represented as a finite frequency in `unit`.
    pub fn nearest(&self, value: f64, unit: FrequencyUnit) -> Result<Self> {
        if self.values_hz.is_empty() {
            return Err(Error::InvalidFrequency(
                "cannot select from an empty frequency axis".to_owned(),
            ));
        }
        let target_hz = scale_to_hz(value, unit)?;
        let nearest = self
            .values_hz
            .iter()
            .copied()
            .min_by(|left, right| {
                (left - target_hz)
                    .abs()
                    .total_cmp(&(right - target_hz).abs())
            })
            .ok_or_else(|| {
                Error::InvalidFrequency("cannot select from an empty frequency axis".to_owned())
            })?;
        let mut frequency = Self::from_hz(Array1::from_vec(vec![nearest]))?;
        frequency.unit = self.unit;
        Ok(frequency)
    }

    /// Returns the overlapping band between this axis and `other`.
    ///
    /// Values are retained from `self`; an error is returned if the axes do not
    /// overlap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when either axis is empty or the two
    /// axes do not overlap.
    pub fn overlap(&self, other: &Self) -> Result<Self> {
        let self_start = self.start().ok_or_else(|| {
            Error::InvalidFrequency("cannot overlap an empty frequency axis".to_owned())
        })?;
        let self_stop = self.stop().ok_or_else(|| {
            Error::InvalidFrequency("cannot overlap an empty frequency axis".to_owned())
        })?;
        let other_start = other.start().ok_or_else(|| {
            Error::InvalidFrequency("cannot overlap an empty frequency axis".to_owned())
        })?;
        let other_stop = other.stop().ok_or_else(|| {
            Error::InvalidFrequency("cannot overlap an empty frequency axis".to_owned())
        })?;

        if self_start > other_stop || other_start > self_stop {
            return Err(Error::InvalidFrequency(
                "frequency axes do not overlap".to_owned(),
            ));
        }

        let start = self_start.max(other_start);
        let stop = self_stop.min(other_stop);
        let values = self
            .values_hz
            .iter()
            .copied()
            .filter(|value| *value >= start && *value <= stop)
            .collect::<Vec<_>>();
        let mut overlap = Self::from_hz(Array1::from_vec(values))?;
        overlap.unit = self.unit;
        Ok(overlap)
    }

    /// Rounds frequency values to `precision_hz`.
    ///
    /// This is useful for finite-precision limitations in VNAs and other
    /// measurement software.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when `precision_hz` is not finite or
    /// is not positive.
    pub fn round_to(&mut self, precision_hz: f64) -> Result<()> {
        if !precision_hz.is_finite() || precision_hz <= 0.0 {
            return Err(Error::InvalidFrequency(
                "rounding precision must be finite and positive".to_owned(),
            ));
        }

        self.values_hz
            .mapv_inplace(|value| (value / precision_hz).round() * precision_hz);
        self.sweep_type = classify_sweep(&self.values_hz);
        Ok(())
    }

    /// Adds two frequency vectors element-wise with scalar broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axes cannot be broadcast or
    /// the resulting values are not finite.
    pub fn try_add(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::Add)
    }

    /// Subtracts two frequency vectors element-wise with scalar broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axes cannot be broadcast or
    /// the resulting values are not finite.
    pub fn try_subtract(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::Subtract)
    }

    /// Multiplies two frequency vectors element-wise with scalar broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axes cannot be broadcast or
    /// the resulting values are not finite.
    pub fn try_multiply(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::Multiply)
    }

    /// Divides two frequency vectors element-wise with scalar broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axes cannot be broadcast or
    /// division produces a non-finite value.
    pub fn try_divide(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::Divide)
    }

    /// Floor-divides two frequency vectors element-wise with scalar broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axes cannot be broadcast or
    /// division produces a non-finite value.
    pub fn try_floor_divide(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::FloorDivide)
    }

    /// Calculates element-wise remainders with scalar broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when the axes cannot be broadcast or
    /// the operation produces a non-finite value.
    pub fn try_remainder(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::Remainder)
    }

    /// Applies `operation` to every stored hertz value.
    ///
    /// The preferred unit is retained and the sweep type is reclassified.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFrequency`] when `operation` produces a
    /// non-finite value.
    pub fn map_values(&self, operation: impl Fn(f64) -> f64) -> Result<Self> {
        let values_hz = self.values_hz.mapv(operation);
        validate_finite(values_hz.as_slice().unwrap_or(&[]))?;
        let mut result = Self::from_hz(values_hz)?;
        result.unit = self.unit;
        Ok(result)
    }

    fn binary_operation(&self, other: &Self, operation: BinaryOperation) -> Result<Self> {
        let output_length = broadcast_length(self.points(), other.points())?;
        let mut values = Vec::with_capacity(output_length);
        for index in 0..output_length {
            let left = broadcast_value(&self.values_hz, index);
            let right = broadcast_value(&other.values_hz, index);
            values.push(apply_binary_operation(left, right, operation));
        }

        let mut result = Self::from_hz(Array1::from_vec(values))?;
        result.unit = self.unit;
        Ok(result)
    }
}

impl PartialEq for Frequency {
    fn eq(&self, other: &Self) -> bool {
        self.points() == other.points()
            && self
                .values_hz
                .iter()
                .zip(other.values_hz.iter())
                .all(|(left, right)| (left - right).abs() < FREQUENCY_EQUALITY_TOLERANCE_HZ)
    }
}

impl fmt::Display for Frequency {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.start_scaled(), self.stop_scaled()) {
            (Some(start), Some(stop)) => write!(
                formatter,
                "{start}-{stop} {}, {} pts",
                self.unit.symbol(),
                self.points()
            ),
            _ => formatter.write_str("[no freqs]"),
        }
    }
}

fn validate_finite(values: &[f64]) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Error::InvalidFrequency(
            "frequency values must be finite".to_owned(),
        ))
    }
}

fn scale_to_hz(value: f64, unit: FrequencyUnit) -> Result<f64> {
    let scaled = value * unit.multiplier();
    validate_finite(&[scaled])?;
    Ok(scaled)
}

fn linear_space(start: f64, stop: f64, points: usize) -> Array1<f64> {
    if points == 1 {
        return Array1::from_vec(vec![start]);
    }

    let increment = (stop - start) / (points - 1).to_f64().unwrap_or(f64::INFINITY);
    Array1::from_iter((0..points).map(|index| {
        if index + 1 == points {
            stop
        } else {
            index
                .to_f64()
                .unwrap_or(f64::INFINITY)
                .mul_add(increment, start)
        }
    }))
}

fn logarithmic_space(start: f64, stop: f64, points: usize) -> Array1<f64> {
    if points == 1 {
        return Array1::from_vec(vec![start]);
    }

    let logarithmic_step =
        (stop.ln() - start.ln()) / (points - 1).to_f64().unwrap_or(f64::INFINITY);
    Array1::from_iter((0..points).map(|index| {
        if index == 0 {
            start
        } else if index + 1 == points {
            stop
        } else {
            index
                .to_f64()
                .unwrap_or(f64::INFINITY)
                .mul_add(logarithmic_step, start.ln())
                .exp()
        }
    }))
}

fn classify_sweep(values: &Array1<f64>) -> SweepType {
    if values.len() <= 1 {
        return SweepType::Linear;
    }

    let linear = linear_space(values[0], values[values.len() - 1], values.len());
    if approximately_equal(values, &linear, SWEEP_RELATIVE_TOLERANCE) {
        return SweepType::Linear;
    }

    if values[0] > 0.0 && values[values.len() - 1] > 0.0 {
        let logarithmic = logarithmic_space(values[0], values[values.len() - 1], values.len());
        if approximately_equal(values, &logarithmic, SWEEP_RELATIVE_TOLERANCE) {
            return SweepType::Logarithmic;
        }
    }

    SweepType::Arbitrary
}

fn approximately_equal(left: &Array1<f64>, right: &Array1<f64>, relative_tolerance: f64) -> bool {
    left.iter().zip(right.iter()).all(|(left, right)| {
        let scale = left.abs().max(right.abs());
        (left - right).abs() <= relative_tolerance * scale
    })
}

fn gradient(values: &Array1<f64>) -> Result<Array1<f64>> {
    match values.len() {
        0 | 1 => Err(Error::InvalidFrequency(
            "at least two points are required to calculate a gradient".to_owned(),
        )),
        2 => {
            let difference = values[1] - values[0];
            Ok(Array1::from_vec(vec![difference, difference]))
        }
        length => {
            let mut result = Array1::zeros(length);
            result[0] = values[1] - values[0];
            result[length - 1] = values[length - 1] - values[length - 2];
            for index in 1..length - 1 {
                result[index] = (values[index + 1] - values[index - 1]) / 2.0;
            }
            Ok(result)
        }
    }
}

fn broadcast_length(left: usize, right: usize) -> Result<usize> {
    if left == right {
        Ok(left)
    } else if left == 1 {
        Ok(right)
    } else if right == 1 {
        Ok(left)
    } else {
        Err(Error::InvalidFrequency(format!(
            "cannot broadcast frequency axes with {left} and {right} points"
        )))
    }
}

fn broadcast_value(values: &Array1<f64>, index: usize) -> f64 {
    if values.len() == 1 {
        values[0]
    } else {
        values[index]
    }
}

fn apply_binary_operation(left: f64, right: f64, operation: BinaryOperation) -> f64 {
    match operation {
        BinaryOperation::Add => left + right,
        BinaryOperation::Subtract => left - right,
        BinaryOperation::Multiply => left * right,
        BinaryOperation::Divide => left / right,
        BinaryOperation::FloorDivide => (left / right).floor(),
        BinaryOperation::Remainder => (left / right).floor().mul_add(-right, left),
    }
}
