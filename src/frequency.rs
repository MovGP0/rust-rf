use std::fmt;

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const FREQUENCY_EQUALITY_TOLERANCE_HZ: f64 = 1.0e-4;
const SWEEP_RELATIVE_TOLERANCE: f64 = 0.05;

/// Origin: `skrf/frequency.py::Frequency`.
///
/// Frequency values are stored internally in hertz. The fields are private so
/// callers cannot create an inconsistent axis by changing individual values.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frequency {
    values_hz: Array1<f64>,
    unit: FrequencyUnit,
    sweep_type: SweepType,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum FrequencyUnit {
    Hz,
    KHz,
    MHz,
    #[default]
    GHz,
    THz,
}

impl FrequencyUnit {
    pub const fn multiplier(self) -> f64 {
        match self {
            Self::Hz => 1.0,
            Self::KHz => 1.0e3,
            Self::MHz => 1.0e6,
            Self::GHz => 1.0e9,
            Self::THz => 1.0e12,
        }
    }

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
pub enum SweepType {
    #[default]
    Linear,
    Logarithmic,
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
    /// Port of `skrf.frequency.Frequency.__init__`.
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

    /// Port of `skrf.frequency.Frequency.from_f` for values already in hertz.
    pub fn from_hz(values_hz: Array1<f64>) -> Result<Self> {
        Self::from_values(values_hz, FrequencyUnit::Hz)
    }

    /// Port of `skrf.frequency.Frequency.from_f`.
    pub fn from_values(values: Array1<f64>, unit: FrequencyUnit) -> Result<Self> {
        validate_finite(values.as_slice().unwrap_or(&[]))?;

        let multiplier = unit.multiplier();
        let values_hz = values.mapv(|value| value * multiplier);
        validate_finite(values_hz.as_slice().unwrap_or(&[]))?;
        let sweep_type = classify_sweep(&values_hz);

        Ok(Self {
            values_hz,
            unit,
            sweep_type,
        })
    }

    pub fn values_hz(&self) -> &Array1<f64> {
        &self.values_hz
    }

    pub const fn unit(&self) -> FrequencyUnit {
        self.unit
    }

    /// Changes only the preferred display/input scaling unit; stored values remain in hertz.
    ///
    /// Origin: `skrf.frequency.Frequency.unit` setter.
    pub fn set_unit(&mut self, unit: FrequencyUnit) {
        self.unit = unit;
    }

    pub const fn sweep_type(&self) -> SweepType {
        self.sweep_type
    }

    pub fn start(&self) -> Option<f64> {
        self.values_hz.first().copied()
    }

    pub fn stop(&self) -> Option<f64> {
        self.values_hz.last().copied()
    }

    pub fn start_scaled(&self) -> Option<f64> {
        self.start().map(|value| value / self.unit.multiplier())
    }

    pub fn stop_scaled(&self) -> Option<f64> {
        self.stop().map(|value| value / self.unit.multiplier())
    }

    pub fn points(&self) -> usize {
        self.values_hz.len()
    }

    pub fn center(&self) -> Option<f64> {
        Some(self.start()? + (self.stop()? - self.start()?) / 2.0)
    }

    pub fn center_index(&self) -> Option<usize> {
        (!self.values_hz.is_empty()).then_some(self.points() / 2)
    }

    pub fn center_scaled(&self) -> Option<f64> {
        self.center().map(|value| value / self.unit.multiplier())
    }

    pub fn step(&self) -> Option<f64> {
        let span = self.span()?;
        if span == 0.0 {
            Some(0.0)
        } else {
            Some(span / (self.points() - 1) as f64)
        }
    }

    pub fn step_scaled(&self) -> Option<f64> {
        self.step().map(|value| value / self.unit.multiplier())
    }

    pub fn span(&self) -> Option<f64> {
        Some((self.stop()? - self.start()?).abs())
    }

    pub fn span_scaled(&self) -> Option<f64> {
        self.span().map(|value| value / self.unit.multiplier())
    }

    pub fn scaled(&self) -> Array1<f64> {
        self.values_hz.mapv(|value| value / self.unit.multiplier())
    }

    pub fn angular(&self) -> Array1<f64> {
        self.values_hz.mapv(|value| std::f64::consts::TAU * value)
    }

    pub fn gradient_hz(&self) -> Result<Array1<f64>> {
        gradient(&self.values_hz)
    }

    pub fn gradient_scaled(&self) -> Result<Array1<f64>> {
        Ok(self.gradient_hz()? / self.unit.multiplier())
    }

    pub fn angular_gradient(&self) -> Result<Array1<f64>> {
        Ok(self.gradient_hz()? * std::f64::consts::TAU)
    }

    /// Port of `skrf.frequency.Frequency._t_padded`.
    pub fn padded_time(
        &self,
        padding_points: usize,
        output_points: Option<usize>,
        bandpass: Option<bool>,
    ) -> Result<Array1<f64>> {
        let bandpass = bandpass.unwrap_or_else(|| self.start() != Some(0.0));
        let mut points = output_points.unwrap_or(self.points() + padding_points);
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
        let time_step = 1.0 / (points as f64 * step);

        if bandpass {
            let stop = ((points - 1) / 2) as f64 * time_step;
            let start = if points % 2 == 0 {
                -stop - time_step
            } else {
                -((points / 2) as f64) * time_step
            };
            Ok(linear_space(start, stop, points))
        } else {
            let extent = time_step * (points / 2) as f64;
            Ok(linear_space(-extent, extent, points))
        }
    }

    /// Port of `skrf.frequency.Frequency.t`.
    pub fn time(&self) -> Result<Array1<f64>> {
        self.padded_time(0, None, Some(true))
    }

    /// Port of `skrf.frequency.Frequency.t_ns`.
    pub fn time_nanoseconds(&self) -> Result<Array1<f64>> {
        Ok(self.time()? * 1.0e9)
    }

    pub fn is_monotonic_increasing(&self) -> bool {
        self.values_hz
            .iter()
            .zip(self.values_hz.iter().skip(1))
            .all(|(left, right)| right > left)
    }

    /// Port of `Frequency.drop_non_monotonic_increasing`.
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

    /// Inclusive value-domain slice corresponding to a string such as
    /// `"2-5ghz"` in scikit-rf.
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

    /// Select the nearest frequency point, corresponding to a one-value
    /// scikit-rf frequency slice.
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

    /// Port of `skrf.frequency.overlap_freq`, retaining points from `self`.
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

    pub fn try_add(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::Add)
    }

    pub fn try_subtract(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::Subtract)
    }

    pub fn try_multiply(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::Multiply)
    }

    pub fn try_divide(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::Divide)
    }

    pub fn try_floor_divide(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::FloorDivide)
    }

    pub fn try_remainder(&self, other: &Self) -> Result<Self> {
        self.binary_operation(other, BinaryOperation::Remainder)
    }

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

    let step = (stop - start) / (points - 1) as f64;
    Array1::from_iter((0..points).map(|index| {
        if index + 1 == points {
            stop
        } else {
            start + index as f64 * step
        }
    }))
}

fn logarithmic_space(start: f64, stop: f64, points: usize) -> Array1<f64> {
    if points == 1 {
        return Array1::from_vec(vec![start]);
    }

    let logarithmic_step = (stop.ln() - start.ln()) / (points - 1) as f64;
    Array1::from_iter((0..points).map(|index| {
        if index == 0 {
            start
        } else if index + 1 == points {
            stop
        } else {
            (start.ln() + index as f64 * logarithmic_step).exp()
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
        BinaryOperation::Remainder => left - (left / right).floor() * right,
    }
}
