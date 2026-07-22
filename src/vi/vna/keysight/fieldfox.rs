//! Keysight `FieldFox` handheld analyzer driver.
//!
//! The `FieldFox` supports several operating modes; this module implements its
//! vector network analyzer mode and typed display, frequency, calibration,
//! trace, sweep, and data-transfer operations.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use num_traits::ToPrimitive;

use crate::vi::validators::{
    BooleanValidator, FloatValidator, FrequencyValidator, IntValidator, SetValidator,
};
use crate::{Error, Frequency, FrequencyUnit, Network, Result, SweepType as FrequencySweepType};

use super::super::{InstrumentSession, ValuesFormat, Vna};

macro_rules! scpi_enum {
    ($name:ident { $($variant:ident => $value:literal),+ $(,)? }) => {
        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(match self {
                    $(Self::$variant => $value),+
                })
            }
        }

        impl FromStr for $name {
            type Err = Error;

            fn from_str(value: &str) -> Result<Self> {
                match value.trim() {
                    $($value => Ok(Self::$variant)),+,
                    _ => Err(Error::Parse(format!("unexpected {} value {value:?}", stringify!($name)))),
                }
            }
        }
    };
}

/// Arrangement of traces in the `FieldFox` display window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFormat {
    /// One full-window trace.
    OneTrace,
    /// Two traces.
    TwoTraces,
    /// Three traces.
    ThreeTraces,
    /// Two traces arranged vertically.
    TwoVertical,
    /// One trace in the first row and two in the second row.
    OneFirstRowTwoSecondRow,
    /// Four traces in a two-by-two grid.
    TwoByTwo,
}

scpi_enum!(WindowFormat {
    OneTrace => "D1",
    TwoTraces => "D2",
    ThreeTraces => "D3",
    TwoVertical => "D12H",
    OneFirstRowTwoSecondRow => "D11_23",
    TwoByTwo => "D12_34",
});

const FIELD_FOX_BANDWIDTHS: [i64; 8] = [10, 30, 100, 300, 1_000, 10_000, 30_000, 100_000];

const CALIBRATION_TERMS: [(&str, &str); 12] = [
    ("forward directivity", "ed,1,1"),
    ("reverse directivity", "ed,2,2"),
    ("forward source match", "es,1,1"),
    ("reverse source match", "es,2,2"),
    ("forward reflection tracking", "er,1,1"),
    ("reverse reflection tracking", "er,2,2"),
    ("forward transmission tracking", "et,2,1"),
    ("reverse transmission tracking", "et,1,2"),
    ("forward load match", "el,2,1"),
    ("reverse load match", "el,1,2"),
    ("forward isolation", "ex,2,1"),
    ("reverse isolation", "ex,1,2"),
];

/// Driver for a Keysight `FieldFox` in network-analyzer mode.
pub struct FieldFox<S: InstrumentSession> {
    /// Shared VNA transport and value-transfer functionality.
    pub vna: Vna<S>,
}

impl<S: InstrumentSession> FieldFox<S> {
    /// Creates a driver and synchronizes its value-transfer format.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument transfer format cannot be queried or parsed.
    pub fn new(address: impl Into<String>, session: S) -> Result<Self> {
        let mut field_fox = Self {
            vna: Vna::new(address, session, None),
        };
        field_fox.query_format()?;
        Ok(field_fox)
    }

    /// Wraps an existing VNA session without querying the instrument.
    pub const fn from_vna(vna: Vna<S>) -> Self {
        Self { vna }
    }

    /// Returns the number of supported RF ports.
    pub const fn ports(&self) -> usize {
        2
    }

    /// Returns the start frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the start frequency cannot be queried or parsed.
    pub fn frequency_start(&mut self) -> Result<u64> {
        self.query_frequency("SENS:FREQ:STAR?")
    }

    /// Sets the start frequency, accepting hertz or a supported SI suffix.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is invalid or the instrument rejects it.
    pub fn set_frequency_start(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("SENS:FREQ:STAR", value)
    }

    /// Returns the stop frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the stop frequency cannot be queried or parsed.
    pub fn frequency_stop(&mut self) -> Result<u64> {
        self.query_frequency("SENS:FREQ:STOP?")
    }

    /// Sets the stop frequency, accepting hertz or a supported SI suffix.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is invalid or the instrument rejects it.
    pub fn set_frequency_stop(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("SENS:FREQ:STOP", value)
    }

    /// Returns the center frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the center frequency cannot be queried or parsed.
    pub fn frequency_center(&mut self) -> Result<u64> {
        self.query_frequency("SENS:FREQ:CENT?")
    }

    /// Sets the center frequency, accepting hertz or a supported SI suffix.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is invalid or the instrument rejects it.
    pub fn set_frequency_center(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("SENS:FREQ:CENT", value)
    }

    /// Returns the frequency span in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency span cannot be queried or parsed.
    pub fn frequency_span(&mut self) -> Result<u64> {
        self.query_frequency("SENS:FREQ:SPAN?")
    }

    /// Sets the frequency span, accepting hertz or a supported SI suffix.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is invalid or the instrument rejects it.
    pub fn set_frequency_span(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("SENS:FREQ:SPAN", value)
    }

    /// Returns the number of sweep points.
    ///
    /// # Errors
    ///
    /// Returns an error if the point count cannot be queried, parsed, or represented as `usize`.
    pub fn points(&mut self) -> Result<usize> {
        parse_integer(self.vna.query("SENS:SWE:POIN?")?, None, None)
            .and_then(|value| integer_to_usize(value, "FieldFox sweep point count"))
    }

    /// Sets the number of sweep points.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the point-count command.
    pub fn set_points(&mut self, points: usize) -> Result<()> {
        self.vna.write(&format!("SENS:SWE:POIN {points}"))
    }

    /// Returns the sweep time in seconds.
    ///
    /// # Errors
    ///
    /// Returns an error if the sweep time cannot be queried or parsed.
    pub fn sweep_time(&mut self) -> Result<f64> {
        parse_float(self.vna.query("SENS:SWE:TIME?")?, None, None, 50)
    }

    /// Sets the sweep time in seconds.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the sweep-time command.
    pub fn set_sweep_time(&mut self, seconds: f64) -> Result<()> {
        self.vna.write(&format!("SENS:SWE:TIME {seconds}"))
    }

    /// Returns the intermediate-frequency bandwidth in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the bandwidth cannot be queried or is not supported.
    pub fn if_bandwidth(&mut self) -> Result<i64> {
        let value = self.vna.query("SENS:BWID?")?;
        SetValidator::new(FIELD_FOX_BANDWIDTHS)
            .map_err(validation_error)?
            .validate_input(value)
            .map_err(validation_error)
    }

    /// Sets one of the supported intermediate-frequency bandwidths.
    ///
    /// # Errors
    ///
    /// Returns an error if `bandwidth_hz` is unsupported or cannot be sent.
    pub fn set_if_bandwidth(&mut self, bandwidth_hz: i64) -> Result<()> {
        let bandwidth = SetValidator::new(FIELD_FOX_BANDWIDTHS)
            .map_err(validation_error)?
            .validate_input(bandwidth_hz)
            .map_err(validation_error)?;
        self.vna.write(&format!("SENS:BWID {bandwidth}"))
    }

    /// Returns the current display-window arrangement.
    ///
    /// # Errors
    ///
    /// Returns an error if the window format cannot be queried or parsed.
    pub fn window_format(&mut self) -> Result<WindowFormat> {
        self.vna.query("DISP:WIND:SPL?")?.parse()
    }

    /// Sets the display-window arrangement.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the window-format command.
    pub fn set_window_format(&mut self, format: WindowFormat) -> Result<()> {
        self.vna.write(&format!("DISP:WIND:SPL {format}"))
    }

    /// Returns the number of configured traces, from one through four.
    ///
    /// # Errors
    ///
    /// Returns an error if the trace count cannot be queried, parsed, or represented as `usize`.
    pub fn trace_count(&mut self) -> Result<usize> {
        parse_integer(self.vna.query("CALC:PAR:COUN?")?, Some(1), Some(4))
            .and_then(|value| integer_to_usize(value, "FieldFox trace count"))
    }

    /// Sets the trace count from one through four.
    ///
    /// # Errors
    ///
    /// Returns an error if `count` is outside the supported range or cannot be sent.
    pub fn set_trace_count(&mut self, count: usize) -> Result<()> {
        parse_integer(count.to_string(), Some(1), Some(4))?;
        self.vna.write(&format!("CALC:PAR:COUN {count}"))
    }

    /// Selects the active trace, numbered one through four.
    ///
    /// # Errors
    ///
    /// Returns an error if `trace` is outside the supported range or cannot be selected.
    pub fn set_active_trace(&mut self, trace: usize) -> Result<()> {
        parse_integer(trace.to_string(), Some(1), Some(4))?;
        self.vna.write(&format!("CALC:PAR{trace}:SEL"))
    }

    /// Returns complex S-data for the active trace.
    ///
    /// # Errors
    ///
    /// Returns an error if the active trace data cannot be queried or parsed.
    pub fn active_trace_s_data(&mut self) -> Result<Vec<Complex64>> {
        self.vna.query_complex_values("CALC:DATA:SDATA?")
    }

    /// Returns whether continuous sweep mode is enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the continuous-sweep state cannot be queried.
    pub fn is_continuous(&mut self) -> Result<bool> {
        Ok(BooleanValidator::default().validate_output(self.vna.query("INIT:CONT?")?))
    }

    /// Enables or disables continuous sweep mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the state is invalid or the instrument rejects it.
    pub fn set_continuous(&mut self, continuous: bool) -> Result<()> {
        let value = BooleanValidator::default()
            .validate_input(continuous)
            .map_err(validation_error)?;
        self.vna.write(&format!("INIT:CONT {value}"))
    }

    /// Returns the linear frequency step in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency axis cannot be queried or its step cannot be represented as `u64`.
    pub fn frequency_step(&mut self) -> Result<u64> {
        self.frequency()?
            .step()
            .unwrap_or(0.0)
            .to_u64()
            .ok_or_else(|| {
                Error::InvalidFrequency("FieldFox frequency step is outside the u64 range".into())
            })
    }

    /// Changes the point count to obtain the requested positive frequency step.
    ///
    /// # Errors
    ///
    /// Returns an error if `step_hz` is zero or the required point count cannot be represented.
    pub fn set_frequency_step(&mut self, step_hz: u64) -> Result<()> {
        if step_hz == 0 {
            return Err(Error::InvalidFrequency(
                "FieldFox frequency step must be positive".into(),
            ));
        }
        let frequency = self.frequency()?;
        let span = (frequency.stop().unwrap_or(0.0) - frequency.start().unwrap_or(0.0))
            .to_u64()
            .ok_or_else(|| {
                Error::InvalidFrequency("FieldFox frequency span is outside the u64 range".into())
            })?;
        let points = usize::try_from(span / step_hz + 1).map_err(|error| {
            Error::InvalidFrequency(format!("FieldFox point count is too large: {error}"))
        })?;
        self.set_points(points)
    }

    /// Returns the current linear frequency axis.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency limits or point count cannot be queried or represented.
    pub fn frequency(&mut self) -> Result<Frequency> {
        let start = self.frequency_start()?.to_f64().ok_or_else(|| {
            Error::InvalidFrequency("FieldFox start frequency cannot be represented as f64".into())
        })?;
        let stop = self.frequency_stop()?.to_f64().ok_or_else(|| {
            Error::InvalidFrequency("FieldFox stop frequency cannot be represented as f64".into())
        })?;
        Frequency::new(
            start,
            stop,
            self.points()?,
            FrequencyUnit::Hz,
            FrequencySweepType::Linear,
        )
    }

    /// Programs start, stop, and point count from a frequency axis.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency limits or point count cannot be applied.
    pub fn set_frequency(&mut self, frequency: &Frequency) -> Result<()> {
        self.set_frequency_start(frequency.start().unwrap_or(0.0))?;
        self.set_frequency_stop(frequency.stop().unwrap_or(0.0))?;
        self.set_points(frequency.points())
    }

    /// Reads the twelve currently defined two-port calibration error terms.
    ///
    /// # Errors
    ///
    /// Returns an error if any calibration coefficient cannot be queried or parsed.
    pub fn calibration_coefficients(&mut self) -> Result<BTreeMap<String, Array1<Complex64>>> {
        CALIBRATION_TERMS
            .iter()
            .map(|(name, term)| {
                self.vna
                    .query_complex_values(&format!("SENS:CORR:COEF? {term}"))
                    .map(|values| ((*name).to_owned(), Array1::from(values)))
            })
            .collect()
    }

    /// Writes all twelve two-port calibration error terms.
    ///
    /// # Errors
    ///
    /// Returns an error when a required named coefficient array is missing or a
    /// transfer fails.
    pub fn set_calibration_coefficients(
        &mut self,
        coefficients: &BTreeMap<String, Array1<Complex64>>,
    ) -> Result<()> {
        for (name, term) in CALIBRATION_TERMS {
            let values = coefficients.get(name).ok_or_else(|| {
                Error::Unsupported(format!("missing FieldFox calibration term {name}"))
            })?;
            self.vna.write_complex_values(
                &format!("SENS:CORR:COEF {term},"),
                values.as_slice().unwrap_or(&[]),
            )?;
        }
        Ok(())
    }

    /// Queries how numeric values are transferred and updates the session format.
    ///
    /// ASCII is readable, while 32- or 64-bit binary transfer is substantially
    /// faster for large traces.
    ///
    /// # Errors
    ///
    /// Returns an error if the transfer format cannot be queried or parsed.
    pub fn query_format(&mut self) -> Result<ValuesFormat> {
        let format = parse_values_format(&self.vna.query("FORM?")?)?;
        self.vna.values_format = format;
        Ok(format)
    }

    /// Selects ASCII, 32-bit binary, or 64-bit binary value transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the transfer-format command.
    pub fn set_query_format(&mut self, format: ValuesFormat) -> Result<()> {
        self.vna.write(match format {
            ValuesFormat::Ascii => "FORM ASC,0",
            ValuesFormat::Binary32 => "FORM REAL,32",
            ValuesFormat::Binary64 => "FORM REAL,64",
        })?;
        self.vna.values_format = format;
        Ok(())
    }

    /// Returns the measurement parameter assigned to a trace.
    ///
    /// Parameters include values such as `S11`, `S21`, `A`, `B`, or `R1`.
    ///
    /// # Errors
    ///
    /// Returns an error if `trace` is outside the supported range or cannot be queried.
    pub fn measurement_parameter(&mut self, trace: usize) -> Result<String> {
        if !(1..=4).contains(&trace) {
            return Err(Error::Unsupported("Trace must be between 1 and 4".into()));
        }
        self.vna.query(&format!("CALC:PAR{trace}:DEF?"))
    }

    /// Defines the measurement parameter for a trace, increasing the trace count
    /// when necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if `trace` is outside the supported range or cannot be configured.
    pub fn define_measurement(&mut self, trace: usize, parameter: &str) -> Result<()> {
        if trace == 0 || trace > 4 {
            return Err(Error::Unsupported("Trace must be between 1 and 4".into()));
        }
        if trace > self.trace_count()? {
            self.set_trace_count(trace)?;
        }
        self.vna.write(&format!("CALC:PAR{trace}:DEF {parameter}"))
    }

    /// Triggers a fresh single sweep and restores the previous continuous state.
    ///
    /// # Errors
    ///
    /// Returns an error if the sweep cannot be triggered or the continuous state cannot be restored.
    pub fn sweep(&mut self) -> Result<()> {
        self.vna.clear()?;
        let was_continuous = self.is_continuous()?;
        self.set_continuous(false)?;
        let result = self.vna.write("INIT");
        self.set_continuous(was_continuous)?;
        result
    }

    /// Acquires the requested one- or two-port S-parameters as a [`Network`].
    ///
    /// `ports` may contain port 1, port 2, or both. When `restore_settings` is
    /// `true`, the original trace count, parameters, and window layout are saved
    /// and restored; disabling restoration is faster for repeated measurements.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid ports, invalid trace data, or instrument
    /// communication failure.
    pub fn get_snp_network(&mut self, ports: &[usize], restore_settings: bool) -> Result<Network> {
        if ports.is_empty() || ports.iter().any(|port| !matches!(port, 1 | 2)) {
            return Err(Error::Unsupported(
                "FieldFox ports must contain only port 1 or 2".into(),
            ));
        }
        let measurements = ports
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(output_index, output)| {
                ports
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(input_index, input)| (output_index, input_index, output, input))
            })
            .collect::<Vec<_>>();
        let original = if restore_settings {
            let count = self.trace_count()?;
            Some((
                count,
                self.window_format()?,
                (1..=count)
                    .map(|trace| self.measurement_parameter(trace))
                    .collect::<Result<Vec<_>>>()?,
            ))
        } else {
            None
        };

        self.set_trace_count(measurements.len())?;
        for (index, (_, _, output, input)) in measurements.iter().enumerate() {
            self.define_measurement(index + 1, &format!("S{output}{input}"))?;
        }
        let frequency = self.frequency()?;
        let mut s = Array3::zeros((frequency.points(), ports.len(), ports.len()));
        self.sweep()?;
        for (trace, (output_index, input_index, _, _)) in measurements.iter().enumerate() {
            self.set_active_trace(trace + 1)?;
            let values = self.active_trace_s_data()?;
            if values.len() != frequency.points() {
                return Err(Error::IncompatibleShape(format!(
                    "FieldFox trace has {} points, expected {}",
                    values.len(),
                    frequency.points()
                )));
            }
            for point in 0..frequency.points() {
                s[[point, *output_index, *input_index]] = values[point];
            }
        }

        if let Some((count, window, parameters)) = original {
            for (index, parameter) in parameters.iter().enumerate() {
                self.define_measurement(index + 1, parameter)?;
            }
            self.set_trace_count(count)?;
            self.set_window_format(window)?;
        }
        network(frequency, s)
    }

    fn query_frequency(&mut self, command: &str) -> Result<u64> {
        FrequencyValidator
            .validate_output(self.vna.query(command)?)
            .map_err(validation_error)
    }

    fn set_frequency_value(&mut self, command: &str, value: impl ToString) -> Result<()> {
        let value = FrequencyValidator
            .validate_input(value)
            .map_err(validation_error)?;
        self.vna.write(&format!("{command} {value}"))
    }
}

fn network(frequency: Frequency, s: Array3<Complex64>) -> Result<Network> {
    let ports = s.dim().1;
    let points = frequency.points();
    Network::new(
        frequency,
        s,
        Array2::from_elem((points, ports), Complex64::new(50.0, 0.0)),
    )
}

fn validation_error(error: impl Display) -> Error {
    Error::Parse(error.to_string())
}

fn parse_float(value: String, min: Option<f64>, max: Option<f64>, decimals: u32) -> Result<f64> {
    FloatValidator::new(min, max, decimals)
        .validate_output(value)
        .map_err(validation_error)
}

fn parse_integer(value: String, min: Option<i64>, max: Option<i64>) -> Result<i64> {
    IntValidator::new(min, max)
        .validate_output(value)
        .map_err(validation_error)
}

fn integer_to_usize(value: i64, description: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|error| Error::Parse(format!("{description} is outside the usize range: {error}")))
}

fn parse_values_format(value: &str) -> Result<ValuesFormat> {
    match value.trim() {
        "ASC,0" => Ok(ValuesFormat::Ascii),
        "REAL,32" => Ok(ValuesFormat::Binary32),
        "REAL,64" => Ok(ValuesFormat::Binary64),
        value => Err(Error::Parse(format!(
            "unexpected Keysight data format {value:?}"
        ))),
    }
}
