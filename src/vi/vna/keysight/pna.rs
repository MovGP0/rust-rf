//! Driver for Keysight PNA, PNA-L, and PNA-X vector network analyzers.
//!
//! Supported PNA models include E8361A/B/C through E8364A/B/C, E8356A through
//! E8358A, E8801A through E8803A, N3381A through N3383A, N5221A/B through
//! N5227A/B, and N5250C. PNA-L support covers the N5230, N5231, N5232, N5234,
//! N5235, and N5239 families; PNA-X support covers the N5241, N5242, N5244,
//! N5245, N5247, N5249, and N5264 families.

use std::fmt::{Display, Formatter};
use std::str::FromStr;

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use num_traits::ToPrimitive;

use crate::vi::validators::{BooleanValidator, FloatValidator, FrequencyValidator, IntValidator};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Sweep-axis type used by a PNA channel.
pub enum PnaSweepType {
    /// Linearly spaced frequency sweep.
    Linear,
    /// Logarithmically spaced frequency sweep.
    Log,
    /// Power sweep.
    Power,
    /// Continuous-wave sweep.
    ContinuousWave,
    /// Segmented sweep.
    Segment,
    /// Phase sweep.
    Phase,
}

scpi_enum!(PnaSweepType {
    Linear => "LIN",
    Log => "LOG",
    Power => "POW",
    ContinuousWave => "CW",
    Segment => "SEGM",
    Phase => "PHAS",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Trigger behavior for a PNA channel sweep.
pub enum PnaSweepMode {
    /// Hold the channel without sweeping.
    Hold,
    /// Sweep continuously.
    Continuous,
    /// Acquire the configured number of sweep groups.
    Groups,
    /// Acquire one sweep.
    Single,
}

scpi_enum!(PnaSweepMode {
    Hold => "HOLD",
    Continuous => "CONT",
    Groups => "GRO",
    Single => "SING",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Source of the sweep trigger signal.
pub enum TriggerSource {
    /// Trigger from the external input.
    External,
    /// Trigger immediately.
    Immediate,
    /// Trigger only when requested manually.
    Manual,
}

scpi_enum!(TriggerSource {
    External => "EXT",
    Immediate => "IMM",
    Manual => "MAN",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// How measurements are combined when averaging is enabled.
pub enum PnaAveragingMode {
    /// Average each point across repeated measurements.
    Point,
    /// Average complete sweeps.
    Sweep,
}

scpi_enum!(PnaAveragingMode {
    Point => "POIN",
    Sweep => "SWE",
});

/// Keysight PNA, PNA-L, or PNA-X instrument.
pub struct Pna<S: InstrumentSession> {
    /// Shared SCPI instrument connection and channel registry.
    pub vna: Vna<S>,
    /// Model identifier reported by the instrument.
    pub model: String,
}

impl<S: InstrumentSession> Pna<S> {
    /// Connects to a PNA and derives its model from the identification response.
    ///
    /// # Errors
    ///
    /// Returns an error if instrument communication, identification parsing, or channel setup fails.
    pub fn new(address: impl Into<String>, session: S) -> Result<Self> {
        let mut vna = Vna::new(address, session, None);
        let model = vna
            .id()?
            .split(',')
            .nth(1)
            .unwrap_or("unknown")
            .trim()
            .to_owned();
        Self::from_model(vna, model)
    }

    /// Builds a PNA driver from an existing VNA connection and known model.
    ///
    /// # Errors
    ///
    /// Returns an error if the default instrument channel cannot be created.
    pub fn from_model(vna: Vna<S>, model: impl Into<String>) -> Result<Self> {
        let mut pna = Self {
            vna,
            model: model.into(),
        };
        pna.create_channel(1, "Channel 1")?;
        Ok(pna)
    }

    /// Creates a numbered channel and its default S11 measurement.
    ///
    /// # Errors
    ///
    /// Returns an error if the channel or its default measurement cannot be created.
    pub fn create_channel(&mut self, number: usize, name: impl Into<String>) -> Result<()> {
        self.vna.create_channel(number, name)?;
        if number != 1 {
            self.create_measurement(number, &format!("CH{number}_S11_1"), "S11")?;
        }
        Ok(())
    }

    /// Deletes a numbered channel when it exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the channel deletion command.
    pub fn delete_channel(&mut self, number: usize) -> Result<()> {
        if self
            .vna
            .channels()
            .iter()
            .any(|channel| channel.number == number)
        {
            self.vna.write(&format!("SYST:CHAN:DEL {number}"))?;
            self.vna.delete_channel(number);
        }
        Ok(())
    }

    /// Returns a controller for an existing numbered channel.
    ///
    /// # Errors
    ///
    /// Returns an error if `number` does not identify an existing channel.
    pub fn channel(&mut self, number: usize) -> Result<PnaChannel<'_, S>> {
        if !self
            .vna
            .channels()
            .iter()
            .any(|channel| channel.number == number)
        {
            return Err(Error::Unsupported(format!(
                "Channel {number} does not exist"
            )));
        }
        Ok(PnaChannel {
            parent: self,
            number,
        })
    }

    /// Returns the source of the sweep trigger signal.
    ///
    /// # Errors
    ///
    /// Returns an error if the trigger source cannot be queried or parsed.
    pub fn trigger_source(&mut self) -> Result<TriggerSource> {
        self.vna.query("TRIG:SOUR?")?.parse()
    }

    /// Selects the source of the sweep trigger signal.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the trigger source command.
    pub fn set_trigger_source(&mut self, source: TriggerSource) -> Result<()> {
        self.vna.write(&format!("TRIG:SOUR {source}"))
    }

    /// Returns the number of errors since the error queue was last cleared.
    ///
    /// # Errors
    ///
    /// Returns an error if the error count cannot be queried, parsed, or represented as `usize`.
    pub fn error_count(&mut self) -> Result<usize> {
        parse_integer(self.vna.query("SYST:ERR:COUN?")?, Some(0), None)
            .and_then(|value| integer_to_usize(value, "PNA error count"))
    }

    /// Returns the channel numbers currently in use.
    ///
    /// # Errors
    ///
    /// Returns an error if the channel catalog cannot be queried or parsed.
    pub fn channel_numbers(&mut self) -> Result<Vec<usize>> {
        parse_integer_list(&self.vna.query("SYST:CHAN:CAT?")?)
    }

    /// Returns the number of physical test ports.
    ///
    /// # Errors
    ///
    /// Returns an error if the hardware port count cannot be queried, parsed, or represented as `usize`.
    pub fn ports(&mut self) -> Result<usize> {
        if self.supports("nports") {
            parse_integer(self.vna.query("SYST:CAP:HARD:PORT:COUN?")?, Some(1), None)
                .and_then(|value| integer_to_usize(value, "PNA port count"))
        } else {
            Ok(self.model_ports())
        }
    }

    /// Returns the active channel number, if it is registered by this driver.
    ///
    /// # Errors
    ///
    /// Returns an error if the active channel cannot be queried, parsed, or represented as `usize`.
    pub fn active_channel_number(&mut self) -> Result<Option<usize>> {
        let number = integer_to_usize(
            parse_integer(self.vna.query("SYST:ACT:CHAN?")?, Some(0), None)?,
            "active PNA channel number",
        )?;
        Ok(self
            .vna
            .channels()
            .iter()
            .any(|channel| channel.number == number)
            .then_some(number))
    }

    /// Activates a channel by selecting its first measurement.
    ///
    /// # Errors
    ///
    /// Returns an error if the channel has no measurement or cannot be selected.
    pub fn set_active_channel(&mut self, number: usize) -> Result<()> {
        if self.active_channel_number()? == Some(number) {
            return Ok(());
        }
        let measurement = self
            .measurement_numbers(number)?
            .first()
            .copied()
            .ok_or_else(|| Error::Unsupported(format!("Channel {number} has no measurements")))?;
        self.vna
            .write(&format!("CALC{number}:PAR:MNUM {measurement}"))?;
        self.vna.set_active_channel(number)
    }

    /// Queries the numeric transfer format used for instrument responses.
    ///
    /// # Errors
    ///
    /// Returns an error if the transfer format cannot be queried or parsed.
    pub fn query_format(&mut self) -> Result<ValuesFormat> {
        let format = parse_values_format(&self.vna.query("FORM?")?.replace('+', ""))?;
        self.vna.values_format = format;
        Ok(format)
    }

    /// Sets the numeric transfer format used for instrument responses.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects a transfer-format command.
    pub fn set_query_format(&mut self, format: ValuesFormat) -> Result<()> {
        match format {
            ValuesFormat::Ascii => self.vna.write("FORM ASC,0")?,
            ValuesFormat::Binary32 => {
                self.vna.write("FORM:BORD SWAP")?;
                self.vna.write("FORM REAL,32")?;
            }
            ValuesFormat::Binary64 => {
                self.vna.write("FORM:BORD SWAP")?;
                self.vna.write("FORM REAL,64")?;
            }
        }
        self.vna.values_format = format;
        Ok(())
    }

    /// Returns the name of the active measurement.
    ///
    /// # Errors
    ///
    /// Returns an error if the active measurement cannot be queried.
    pub fn active_measurement(&mut self) -> Result<String> {
        Ok(self.vna.query("SYST:ACT:MEAS?")?.replace('"', ""))
    }

    /// Selects an existing measurement by name.
    ///
    /// # Errors
    ///
    /// Returns an error if channel measurements cannot be queried or `name` does not exist.
    pub fn set_active_measurement(&mut self, name: &str) -> Result<()> {
        for index in 0..self.vna.channels().len() {
            let number = self.vna.channels()[index].number;
            if self
                .measurement_names(number)?
                .iter()
                .any(|candidate| candidate == name)
            {
                let suffix = if self.supports("fast_sweep") {
                    ",fast"
                } else {
                    ""
                };
                return self
                    .vna
                    .write(&format!("CALC{number}:PAR:SEL '{name}'{suffix}"));
            }
        }
        Err(Error::Unsupported(format!(
            "measurement {name} does not exist"
        )))
    }

    /// Returns `(name, parameter)` pairs for a channel's measurements.
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement catalog cannot be queried.
    pub fn measurements(&mut self, channel: usize) -> Result<Vec<(String, String)>> {
        let values = self
            .vna
            .query(&format!("CALC{channel}:PAR:CAT:EXT?"))?
            .replace('"', "")
            .split(',')
            .map(str::to_owned)
            .collect::<Vec<_>>();
        Ok(values
            .chunks_exact(2)
            .map(|pair| (pair[0].clone(), pair[1].clone()))
            .collect())
    }

    /// Returns the measurement names defined on a channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement catalog cannot be queried.
    pub fn measurement_names(&mut self, channel: usize) -> Result<Vec<String>> {
        Ok(self
            .measurements(channel)?
            .into_iter()
            .map(|(name, _)| name)
            .collect())
    }

    /// Returns the instrument-assigned measurement numbers on a channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement catalog cannot be queried or parsed.
    pub fn measurement_numbers(&mut self, channel: usize) -> Result<Vec<usize>> {
        parse_integer_list(&self.vna.query(&format!("SYST:MEAS:CAT? {channel}"))?)
    }

    /// Creates and displays a measurement on a channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement cannot be created or assigned to a trace.
    pub fn create_measurement(
        &mut self,
        channel: usize,
        name: &str,
        parameter: &str,
    ) -> Result<()> {
        self.vna
            .write(&format!("CALC{channel}:PAR:EXT '{name}',{parameter}"))?;
        let traces = self.vna.query("DISP:WIND:CAT?")?.replace('"', "");
        let next_trace = if traces == "EMPTY" || traces.is_empty() {
            1
        } else {
            parse_integer_list(&traces)?.last().copied().unwrap_or(0) + 1
        };
        self.vna
            .write(&format!("DISP:WIND:TRAC{next_trace}:FEED '{name}'"))
    }

    /// Deletes a named measurement from a channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the measurement deletion command.
    pub fn delete_measurement(&mut self, channel: usize, name: &str) -> Result<()> {
        self.vna.write(&format!("CALC{channel}:PAR:DEL '{name}'"))
    }

    fn supports(&self, feature: &str) -> bool {
        self.model != "E8362C" || !matches!(feature, "nports" | "freq_step" | "fast_sweep")
    }

    fn model_ports(&self) -> usize {
        if self.model == "N5227B" { 4 } else { 2 }
    }
}

/// Controller for one channel of a [`Pna`].
pub struct PnaChannel<'a, S: InstrumentSession> {
    parent: &'a mut Pna<S>,
    /// Instrument channel number.
    pub number: usize,
}

impl<S: InstrumentSession> PnaChannel<'_, S> {
    /// Returns the start frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the start frequency cannot be queried or parsed.
    pub fn frequency_start(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:STAR?")
    }

    /// Sets the start frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is invalid or the instrument rejects it.
    pub fn set_frequency_start(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("FREQ:STAR", value)
    }

    /// Returns the stop frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the stop frequency cannot be queried or parsed.
    pub fn frequency_stop(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:STOP?")
    }

    /// Sets the stop frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is invalid or the instrument rejects it.
    pub fn set_frequency_stop(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("FREQ:STOP", value)
    }

    /// Returns the frequency span in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency span cannot be queried or parsed.
    pub fn frequency_span(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:SPAN?")
    }

    /// Sets the frequency span in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is invalid or the instrument rejects it.
    pub fn set_frequency_span(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("FREQ:SPAN", value)
    }

    /// Returns the center frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the center frequency cannot be queried or parsed.
    pub fn frequency_center(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:CENT?")
    }

    /// Sets the center frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is invalid or the instrument rejects it.
    pub fn set_frequency_center(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("FREQ:CENT", value)
    }

    /// Returns the number of frequency points.
    ///
    /// Changing this value also changes the frequency step.
    ///
    /// # Errors
    ///
    /// Returns an error if the point count cannot be queried, parsed, or represented as `usize`.
    pub fn points(&mut self) -> Result<usize> {
        parse_integer(
            self.parent
                .vna
                .query(&format!("SENS{}:SWE:POIN?", self.number))?,
            Some(1),
            None,
        )
        .and_then(|value| integer_to_usize(value, "PNA frequency point count"))
    }

    /// Sets the number of frequency points.
    ///
    /// This also changes the frequency step.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the point-count command.
    pub fn set_points(&mut self, points: usize) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:POIN {points}", self.number))
    }

    /// Returns the intermediate-frequency bandwidth in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the bandwidth cannot be queried or parsed.
    pub fn if_bandwidth(&mut self) -> Result<u64> {
        self.query_frequency("BWID?")
    }

    /// Sets the intermediate-frequency bandwidth in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is invalid or the instrument rejects it.
    pub fn set_if_bandwidth(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("BWID", value)
    }

    /// Returns the duration of one sweep in seconds.
    ///
    /// # Errors
    ///
    /// Returns an error if the sweep time cannot be queried or parsed.
    pub fn sweep_time(&mut self) -> Result<f64> {
        parse_float(
            self.parent
                .vna
                .query(&format!("SENS{}:SWE:TIME?", self.number))?,
            None,
            None,
            2,
        )
    }

    /// Sets the duration of one sweep in seconds.
    ///
    /// # Errors
    ///
    /// Returns an error if `seconds` is invalid or the instrument rejects it.
    pub fn set_sweep_time(&mut self, seconds: f64) -> Result<()> {
        let seconds = FloatValidator::new(None, None, 2)
            .validate_input(seconds)
            .map_err(validation_error)?;
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:TIME {seconds}", self.number))
    }

    /// Returns the channel's sweep-axis type.
    ///
    /// # Errors
    ///
    /// Returns an error if the sweep type cannot be queried or parsed.
    pub fn sweep_type(&mut self) -> Result<PnaSweepType> {
        self.parent
            .vna
            .query(&format!("SENS{}:SWE:TYPE?", self.number))?
            .parse()
    }

    /// Sets the channel's sweep-axis type.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the sweep-type command.
    pub fn set_sweep_type(&mut self, sweep_type: PnaSweepType) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:TYPE {sweep_type}", self.number))
    }

    /// Returns this channel's trigger mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the sweep mode cannot be queried or parsed.
    pub fn sweep_mode(&mut self) -> Result<PnaSweepMode> {
        self.parent
            .vna
            .query(&format!("SENS{}:SWE:MODE?", self.number))?
            .parse()
    }

    /// Sets this channel's trigger mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the sweep-mode command.
    pub fn set_sweep_mode(&mut self, mode: PnaSweepMode) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:MODE {mode}", self.number))
    }

    /// Returns the instrument-assigned measurement numbers on this channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement catalog cannot be queried or parsed.
    pub fn measurement_numbers(&mut self) -> Result<Vec<usize>> {
        self.parent.measurement_numbers(self.number)
    }

    /// Returns whether measurement averaging is enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the averaging state cannot be queried.
    pub fn averaging_on(&mut self) -> Result<bool> {
        Ok(BooleanValidator::default().validate_output(
            self.parent
                .vna
                .query(&format!("SENS{}:AVER:STATE?", self.number))?,
        ))
    }

    /// Enables or disables measurement averaging.
    ///
    /// # Errors
    ///
    /// Returns an error if the state is invalid or the instrument rejects it.
    pub fn set_averaging_on(&mut self, enabled: bool) -> Result<()> {
        let value = BooleanValidator::default()
            .validate_input(enabled)
            .map_err(validation_error)?;
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:STATE {value}", self.number))
    }

    /// Returns the number of measurements combined for an average.
    ///
    /// # Errors
    ///
    /// Returns an error if the averaging count cannot be queried, parsed, or represented as `usize`.
    pub fn averaging_count(&mut self) -> Result<usize> {
        parse_integer(
            self.parent
                .vna
                .query(&format!("SENS{}:AVER:COUN?", self.number))?,
            Some(1),
            Some(65_536),
        )
        .and_then(|value| integer_to_usize(value, "PNA averaging count"))
    }

    /// Sets the number of measurements combined for an average.
    ///
    /// # Errors
    ///
    /// Returns an error if `count` is outside the supported range or cannot be sent.
    pub fn set_averaging_count(&mut self, count: usize) -> Result<()> {
        parse_integer(count.to_string(), Some(1), Some(65_536))?;
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:COUN {count}", self.number))
    }

    /// Returns how measurements are averaged together.
    ///
    /// # Errors
    ///
    /// Returns an error if the averaging mode cannot be queried or parsed.
    pub fn averaging_mode(&mut self) -> Result<PnaAveragingMode> {
        self.parent
            .vna
            .query(&format!("SENS{}:AVER:MODE?", self.number))?
            .parse()
    }

    /// Selects how measurements are averaged together.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the averaging-mode command.
    pub fn set_averaging_mode(&mut self, mode: PnaAveragingMode) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:MODE {mode}", self.number))
    }

    /// Returns the number of triggers issued by one grouped trigger command.
    ///
    /// # Errors
    ///
    /// Returns an error if the group count cannot be queried, parsed, or represented as `usize`.
    pub fn sweep_groups(&mut self) -> Result<usize> {
        parse_integer(
            self.parent
                .vna
                .query(&format!("SENS{}:SWE:GRO:COUN?", self.number))?,
            Some(1),
            Some(2_000_000),
        )
        .and_then(|value| integer_to_usize(value, "PNA sweep group count"))
    }

    /// Sets the number of triggers issued by one grouped trigger command.
    ///
    /// # Errors
    ///
    /// Returns an error if `groups` is outside the supported range or cannot be sent.
    pub fn set_sweep_groups(&mut self, groups: usize) -> Result<()> {
        parse_integer(groups.to_string(), Some(1), Some(2_000_000))?;
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:GRO:COUN {groups}", self.number))
    }

    /// Returns the frequency step in hertz.
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
                Error::InvalidFrequency("PNA frequency step is outside the u64 range".into())
            })
    }

    /// Sets the frequency step by adjusting the number of points.
    ///
    /// This calculation is used because not every instrument supports the
    /// `SENS:FREQ:STEP` command.
    ///
    /// # Errors
    ///
    /// Returns an error if `step` is invalid or the required point count cannot be represented.
    pub fn set_frequency_step(&mut self, step: impl ToString) -> Result<()> {
        let step = FrequencyValidator
            .validate_input(step)
            .map_err(validation_error)?;
        if step == 0 {
            return Err(Error::InvalidFrequency(
                "PNA frequency step must be positive".into(),
            ));
        }
        let frequency = self.frequency()?;
        let span = (frequency.stop().unwrap_or(0.0) - frequency.start().unwrap_or(0.0))
            .to_u64()
            .ok_or_else(|| {
                Error::InvalidFrequency("PNA frequency span is outside the u64 range".into())
            })?;
        let points = usize::try_from(span / step + 1).map_err(|error| {
            Error::InvalidFrequency(format!("PNA point count is too large: {error}"))
        })?;
        self.set_points(points)
    }

    /// Returns the channel's frequency axis.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency limits or point count cannot be queried or represented.
    pub fn frequency(&mut self) -> Result<Frequency> {
        let start = self.frequency_start()?.to_f64().ok_or_else(|| {
            Error::InvalidFrequency("PNA start frequency cannot be represented as f64".into())
        })?;
        let stop = self.frequency_stop()?.to_f64().ok_or_else(|| {
            Error::InvalidFrequency("PNA stop frequency cannot be represented as f64".into())
        })?;
        Frequency::new(
            start,
            stop,
            self.points()?,
            FrequencyUnit::Hz,
            FrequencySweepType::Linear,
        )
    }

    /// Configures the channel from a frequency axis.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency limits or point count cannot be applied.
    pub fn set_frequency(&mut self, frequency: &Frequency) -> Result<()> {
        self.set_frequency_start(frequency.start().unwrap_or(0.0))?;
        self.set_frequency_stop(frequency.stop().unwrap_or(0.0))?;
        self.set_points(frequency.points())
    }

    /// Returns `(name, parameter)` pairs for measurements on this channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement catalog cannot be queried.
    pub fn measurements(&mut self) -> Result<Vec<(String, String)>> {
        self.parent.measurements(self.number)
    }

    /// Returns the measurement names defined on this channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement catalog cannot be queried.
    pub fn measurement_names(&mut self) -> Result<Vec<String>> {
        self.parent.measurement_names(self.number)
    }

    /// Reads complex S-parameter data from the active trace.
    ///
    /// # Errors
    ///
    /// Returns an error if no trace is active or its data cannot be queried.
    pub fn active_trace_s_data(&mut self) -> Result<Vec<Complex64>> {
        let selected = self
            .parent
            .vna
            .query(&format!("CALC{}:PAR:SEL?", self.number))?
            .replace('"', "");
        if selected.split(',').next().unwrap_or("").is_empty() {
            return Err(Error::Unsupported(
                "No trace is active. Must select measurement first.".into(),
            ));
        }
        self.parent
            .vna
            .query_complex_values(&format!("CALC{}:DATA? SDATA", self.number))
    }

    /// Clears accumulated averaging data.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the averaging-clear command.
    pub fn clear_averaging(&mut self) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:CLE", self.number))
    }

    /// Creates and displays a named measurement.
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement cannot be created or displayed.
    pub fn create_measurement(&mut self, name: &str, parameter: &str) -> Result<()> {
        self.parent.create_measurement(self.number, name, parameter)
    }

    /// Deletes a named measurement.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the measurement deletion command.
    pub fn delete_measurement(&mut self, name: &str) -> Result<()> {
        self.parent.delete_measurement(self.number, name)
    }

    /// Sweeps and returns the active trace as a one-port network.
    ///
    /// # Errors
    ///
    /// Returns an error if acquisition, transfer-format restoration, or network construction fails.
    pub fn get_active_trace(&mut self) -> Result<Network> {
        self.sweep()?;
        let original = self.parent.vna.values_format;
        self.parent.set_query_format(ValuesFormat::Binary64)?;
        let result = (|| {
            let frequency = self.frequency()?;
            let values = self.active_trace_s_data()?;
            one_port_network(frequency, values)
        })();
        self.parent.set_query_format(original)?;
        result
    }

    /// Selects, sweeps, and returns a named measurement.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` does not exist or acquisition fails.
    pub fn get_measurement(&mut self, name: &str) -> Result<Network> {
        if !self
            .measurement_names()?
            .iter()
            .any(|candidate| candidate == name)
        {
            return Err(Error::Unsupported(format!(
                "measurement {name} does not exist"
            )));
        }
        self.parent.set_active_measurement(name)?;
        let mut network = self.get_active_trace()?;
        network.name = Some(name.to_owned());
        Ok(network)
    }

    /// Acquires one S-parameter as a one-port network.
    ///
    /// `output` and `input` identify the driven and measured ports in the
    /// parameter name $S_{output,input}$.
    ///
    /// # Errors
    ///
    /// Returns an error if the temporary measurement cannot be acquired or converted to a network.
    pub fn get_s_data(&mut self, output: impl Display, input: impl Display) -> Result<Network> {
        let original = self.parent.vna.values_format;
        self.parent.set_query_format(ValuesFormat::Binary64)?;
        let result = (|| {
            let parameter = format!("S{output}{input}");
            self.create_measurement("SKRF_TMP", &parameter)?;
            self.parent.set_active_measurement("SKRF_TMP")?;
            self.sweep()?;
            let frequency = self.frequency()?;
            let values = self.active_trace_s_data()?;
            self.delete_measurement("SKRF_TMP")?;
            one_port_network(frequency, values)
        })();
        self.parent.set_query_format(original)?;
        result
    }

    /// Acquires an $n$-port network for the requested physical ports.
    ///
    /// # Errors
    ///
    /// Returns an error if `ports` is empty, acquisition fails, or the response shape is invalid.
    pub fn get_snp_network(&mut self, ports: &[usize]) -> Result<Network> {
        if ports.is_empty() {
            return Err(Error::Unsupported("PNA SNP port list is empty".into()));
        }
        let original_format = self.parent.vna.values_format;
        self.parent.set_query_format(ValuesFormat::Binary64)?;
        self.parent.set_active_channel(self.number)?;
        let original_snp_format = self.parent.vna.query("MMEM:STOR:TRAC:FORM:SNP?")?;
        self.parent.vna.write("MMEM:STOR:TRACE:FORM:SNP RI")?;

        let parameters = ports
            .iter()
            .flat_map(|output| ports.iter().map(move |input| format!("S{output}{input}")))
            .collect::<Vec<_>>();
        let mut names = Vec::with_capacity(parameters.len());
        for parameter in &parameters {
            let name = format!("CH{}_SKRF_{parameter}", self.number);
            self.create_measurement(&name, parameter)?;
            names.push(name);
        }
        self.sweep()?;
        let port_list = ports
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let raw = self.parent.vna.query_values(&format!(
            "CALC{}:DATA:SNP:PORTS? '{port_list}'",
            self.number
        ))?;
        self.parent.vna.wait_for_complete()?;
        for name in &names {
            self.delete_measurement(name)?;
        }

        let frequency = self.frequency()?;
        let points = frequency.points();
        let expected_rows = 1 + 2 * ports.len() * ports.len();
        if raw.len() != expected_rows * points {
            return Err(Error::IncompatibleShape(format!(
                "PNA SNP response has {} values, expected {}",
                raw.len(),
                expected_rows * points
            )));
        }
        let mut s = Array3::zeros((points, ports.len(), ports.len()));
        for output in 0..ports.len() {
            for input in 0..ports.len() {
                let parameter = output * ports.len() + input;
                let real_row = 1 + parameter * 2;
                let imaginary_row = real_row + 1;
                for point in 0..points {
                    s[[point, output, input]] = Complex64::new(
                        raw[real_row * points + point],
                        raw[imaginary_row * points + point],
                    );
                }
            }
        }
        self.parent.set_query_format(original_format)?;
        self.parent
            .vna
            .write(&format!("MMEM:STOR:TRACE:FORM:SNP {original_snp_format}"))?;
        network(frequency, s)
    }

    /// Performs a complete acquisition and restores the prior sweep settings.
    ///
    /// Sweep averaging uses grouped triggering; other modes use a single sweep.
    /// The instrument timeout is expanded to cover all ports and averages.
    ///
    /// # Errors
    ///
    /// Returns an error if acquisition or restoration of the previous sweep settings fails.
    pub fn sweep(&mut self) -> Result<()> {
        self.parent.set_trigger_source(TriggerSource::Immediate)?;
        self.parent.vna.clear()?;
        let original_mode = self.sweep_mode()?;
        let original_time = self.sweep_time()?;
        let original_averaging = self.averaging_on()?;
        let original_averaging_mode = self.averaging_mode()?;
        let original_timeout = self.parent.vna.timeout_ms;
        let ports = self.parent.ports()?;

        let sweeps = if original_averaging && original_averaging_mode == PnaAveragingMode::Sweep {
            self.set_sweep_mode(PnaSweepMode::Groups)?;
            let count = self.averaging_count()?;
            self.set_sweep_groups(count)?;
            count * ports
        } else {
            self.set_sweep_mode(PnaSweepMode::Single)?;
            ports
        };
        let sweeps = sweeps
            .to_f64()
            .ok_or_else(|| Error::Parse("PNA sweep count cannot be represented as f64".into()))?;
        let timeout_ms = (original_time * sweeps * 1_000.0)
            .to_u64()
            .ok_or_else(|| Error::Parse("PNA sweep timeout is outside the u64 range".into()))?;
        self.parent.vna.timeout_ms = Some(timeout_ms.max(5_000));
        let acquisition = self.parent.vna.wait_for_complete().map(|_| ());
        self.parent.vna.clear()?;
        self.parent.vna.timeout_ms = original_timeout;
        self.set_sweep_mode(original_mode)?;
        self.set_sweep_time(original_time)?;
        self.set_averaging_on(original_averaging)?;
        self.set_averaging_mode(original_averaging_mode)?;
        acquisition
    }

    fn query_frequency(&mut self, command: &str) -> Result<u64> {
        FrequencyValidator
            .validate_output(
                self.parent
                    .vna
                    .query(&format!("SENS{}:{command}", self.number))?,
            )
            .map_err(validation_error)
    }

    fn set_frequency_value(&mut self, command: &str, value: impl ToString) -> Result<()> {
        let value = FrequencyValidator
            .validate_input(value)
            .map_err(validation_error)?;
        self.parent
            .vna
            .write(&format!("SENS{}:{command} {value}", self.number))
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

fn one_port_network(frequency: Frequency, values: Vec<Complex64>) -> Result<Network> {
    let points = frequency.points();
    if values.len() != points {
        return Err(Error::IncompatibleShape(format!(
            "trace has {} values for {points} frequency points",
            values.len()
        )));
    }
    let s = Array3::from_shape_vec((points, 1, 1), values)
        .map_err(|error| Error::IncompatibleShape(error.to_string()))?;
    network(frequency, s)
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

fn parse_integer_list(value: &str) -> Result<Vec<usize>> {
    value
        .replace('"', "")
        .split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .map_err(|error| Error::Parse(format!("invalid integer list value: {error}")))
        })
        .collect()
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
