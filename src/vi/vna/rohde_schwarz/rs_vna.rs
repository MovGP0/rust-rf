//! Rohde & Schwarz VNA family.
//!
//! Origins: `skrf/vi/vna/rohde_schwarz/rs_vna.py`, `zna.py`, and `zva.py`.
//!
//! The driver provides channel, sweep, averaging, measurement, and network
//! acquisition operations used by the ZNA and ZVA model wrappers.

use std::fmt::{Display, Formatter};
use std::str::FromStr;

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use num_traits::ToPrimitive;

use crate::vi::validators::{BooleanValidator, FrequencyValidator, IntValidator};
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
/// Sweep-axis type supported by Rohde & Schwarz analyzers.
pub enum SweepType {
    /// Linearly spaced frequency sweep.
    Linear,
    /// Logarithmically spaced frequency sweep.
    Log,
    /// Segmented sweep.
    Segment,
    /// Power sweep.
    Power,
    /// Continuous-wave sweep.
    ContinuousWave,
    /// Point sweep.
    Point,
    /// Pulse sweep.
    Pulse,
    /// Intermodulation-amplitude sweep.
    IAmplitude,
    /// Intermodulation-phase sweep.
    IPhase,
}

scpi_enum!(SweepType {
    Linear => "LIN",
    Log => "LOG",
    Segment => "SEGM",
    Power => "POW",
    ContinuousWave => "CW",
    Point => "POIN",
    Pulse => "PULS",
    IAmplitude => "IAMP",
    IPhase => "IPH",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Channel sweep mode.
pub enum SweepMode {
    /// Acquire one sweep when initiated.
    Single,
    /// Sweep continuously.
    Continuous,
}

scpi_enum!(SweepMode {
    Single => "0",
    Continuous => "1",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Averaging algorithm used by a channel.
pub enum AveragingMode {
    /// Let the instrument select the averaging algorithm.
    Auto,
    /// Flatten averaging.
    Flatten,
    /// Reduce averaging.
    Reduce,
    /// Moving average.
    Moving,
}

scpi_enum!(AveragingMode {
    Auto => "AUTO",
    Flatten => "FLAT",
    Reduce => "RED",
    Moving => "MOV",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Rohde & Schwarz instrument family and its command capabilities.
pub enum RsFamily {
    /// ZNA family.
    Zna,
    /// ZVA family.
    Zva,
    /// Generic instrument using the shared command set.
    Generic,
}

/// Shared driver for Rohde & Schwarz vector network analyzers.
pub struct RohdeSchwarzVna<S>
where
    S: InstrumentSession,
{
    /// Underlying SCPI transport and channel registry.
    pub vna: Vna<S>,
    /// Model identifier reported by the instrument.
    pub model: String,
    /// Instrument family used for capability selection.
    pub family: RsFamily,
}

impl<S> RohdeSchwarzVna<S>
where
    S: InstrumentSession,
{
    /// Connects to an analyzer and derives its model from identification data.
    ///
    /// # Errors
    ///
    /// Returns an error if identification or initial channel setup fails.
    pub fn new(address: impl Into<String>, session: S, family: RsFamily) -> Result<Self> {
        let mut vna = Vna::new(address, session, None);
        let identification = vna.id()?;
        let model = identification
            .split(',')
            .nth(1)
            .unwrap_or("unknown")
            .trim()
            .to_owned();
        Self::from_model(vna, family, model)
    }

    /// Builds a driver from an existing VNA connection and known model.
    ///
    /// # Errors
    ///
    /// Returns an error if the default channel cannot be created.
    pub fn from_model(vna: Vna<S>, family: RsFamily, model: impl Into<String>) -> Result<Self> {
        let mut driver = Self {
            vna,
            model: model.into(),
            family,
        };
        driver.create_channel(1, "Channel 1")?;
        Ok(driver)
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
            self.vna.write(&format!("CONF:CHAN{number} OFF"))?;
            self.vna.delete_channel(number);
        }
        Ok(())
    }

    /// Returns a controller for an existing numbered channel.
    ///
    /// # Errors
    ///
    /// Returns an error if `number` does not identify an existing channel.
    pub fn channel(&mut self, number: usize) -> Result<RsChannel<'_, S>> {
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
        Ok(RsChannel {
            parent: self,
            number,
        })
    }

    /// Returns the number of physical test ports.
    ///
    /// # Errors
    ///
    /// Returns an error if the port-count response cannot be queried or parsed.
    pub fn nports(&mut self) -> Result<usize> {
        if self.supports("nports") {
            IntValidator::new(Some(1), None)
                .validate_output(self.vna.query("INST:PORT:COUN?")?)
                .map_err(|error| Error::Parse(error.to_string()))
                .and_then(|value| {
                    usize::try_from(value).map_err(|error| Error::Parse(error.to_string()))
                })
        } else {
            Ok(self.model_ports())
        }
    }

    /// Returns the active channel number.
    ///
    /// # Errors
    ///
    /// Returns an error if the active-channel response cannot be queried or parsed.
    pub fn active_channel_number(&mut self) -> Result<usize> {
        IntValidator::new(Some(1), None)
            .validate_output(self.vna.query("INST:NSEL?")?)
            .map_err(|error| Error::Parse(error.to_string()))
            .and_then(|value| {
                usize::try_from(value).map_err(|error| Error::Parse(error.to_string()))
            })
    }

    /// Selects an existing channel as active.
    ///
    /// # Errors
    ///
    /// Returns an error if the channel does not exist or cannot be selected.
    pub fn set_active_channel(&mut self, number: usize) -> Result<()> {
        if self.active_channel_number()? != number {
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
            self.vna.write(&format!("INST:NSEL {number}"))?;
        }
        self.vna.set_active_channel(number)
    }

    /// Queries the numeric transfer format used for instrument responses.
    ///
    /// # Errors
    ///
    /// Returns an error if the format cannot be queried or is not recognized.
    pub fn query_format(&mut self) -> Result<ValuesFormat> {
        let format = match self.vna.query("FORM?")?.as_str() {
            "ASC,0" => ValuesFormat::Ascii,
            "REAL,32" => ValuesFormat::Binary32,
            "REAL,64" => ValuesFormat::Binary64,
            value => {
                return Err(Error::Parse(format!(
                    "unexpected R&S data format {value:?}"
                )));
            }
        };
        self.vna.values_format = format;
        Ok(format)
    }

    /// Sets the numeric transfer format used for instrument responses.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects a format command.
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

    /// Returns the active measurement name on the active channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the active channel or measurement cannot be queried.
    pub fn active_measurement(&mut self) -> Result<String> {
        let channel = self.active_channel_number()?;
        self.vna.query(&format!("CALC{channel}:PAR:SEL?"))
    }

    /// Selects an existing measurement by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement does not exist or cannot be selected.
    pub fn set_active_measurement(&mut self, name: &str) -> Result<()> {
        let mut channel_index = 0;
        while channel_index < self.vna.channels().len() {
            let channel = self.vna.channels()[channel_index].number;
            if self
                .measurement_names(channel)?
                .iter()
                .any(|candidate| candidate == name)
            {
                return self.vna.write(&format!("CALC{channel}:PAR:SEL '{name}'"));
            }
            channel_index += 1;
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
        let response = self
            .vna
            .query(&format!("CALC{channel}:PAR:CAT?"))?
            .trim_matches('\'')
            .to_owned();
        let values = response.split(',').map(str::to_owned).collect::<Vec<_>>();
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

    /// Creates and displays a measurement on a channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects either measurement command.
    pub fn create_measurement(
        &mut self,
        channel: usize,
        name: &str,
        parameter: &str,
    ) -> Result<()> {
        self.vna
            .write(&format!("CALC{channel}:PAR:SDEF '{name}','{parameter}'"))?;
        self.vna.write(&format!("DISP:TRAC:EFE '{name}'"))
    }

    /// Deletes a named measurement from a channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the deletion command.
    pub fn delete_measurement(&mut self, channel: usize, name: &str) -> Result<()> {
        self.vna.write(&format!("CALC{channel}:PAR:DEL '{name}'"))
    }

    fn supports(&self, feature: &str) -> bool {
        feature != "nports" || !matches!(self.family, RsFamily::Zva)
    }

    fn model_ports(&self) -> usize {
        match self.model.as_str() {
            model if model.ends_with("-4Port") => 4,
            _ => 2,
        }
    }
}

/// Controller for one channel of a [`RohdeSchwarzVna`].
pub struct RsChannel<'a, S>
where
    S: InstrumentSession,
{
    parent: &'a mut RohdeSchwarzVna<S>,
    /// Instrument channel number.
    pub number: usize,
}

impl<S> RsChannel<'_, S>
where
    S: InstrumentSession,
{
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
    /// Returns an error if the instrument rejects the start-frequency command.
    pub fn set_frequency_start(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("FREQ:STAR", value)
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
    /// Returns an error if the instrument rejects the stop-frequency command.
    pub fn set_frequency_stop(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("FREQ:STOP", value)
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
    /// Returns an error if the instrument rejects the frequency-span command.
    pub fn set_frequency_span(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("FREQ:SPAN", value)
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
    /// Returns an error if the instrument rejects the center-frequency command.
    pub fn set_frequency_center(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("FREQ:CENT", value)
    }

    /// Returns the fixed frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed frequency cannot be queried or parsed.
    pub fn frequency_fixed(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:FIX?")
    }

    /// Sets the fixed frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the fixed-frequency command.
    pub fn set_frequency_fixed(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("FREQ:FIX", value)
    }

    /// Returns the number of frequency points.
    ///
    /// Changing this value also changes the frequency step.
    ///
    /// # Errors
    ///
    /// Returns an error if the point count cannot be queried or parsed.
    pub fn points(&mut self) -> Result<usize> {
        IntValidator::default()
            .validate_output(
                self.parent
                    .vna
                    .query(&format!("SENS{}:SWE:POIN?", self.number))?,
            )
            .map_err(|error| Error::Parse(error.to_string()))
            .and_then(|value| {
                usize::try_from(value).map_err(|error| Error::Parse(error.to_string()))
            })
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
    /// Returns an error if the instrument rejects the bandwidth command.
    pub fn set_if_bandwidth(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("BWID", value)
    }

    /// Returns the frequency step in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency step cannot be queried or parsed.
    pub fn frequency_step(&mut self) -> Result<u64> {
        self.query_frequency("SWE:STEP?")
    }

    /// Sets the frequency step in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the frequency-step command.
    pub fn set_frequency_step(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("SWE:STEP", value)
    }

    /// Returns the duration of one sweep in seconds.
    ///
    /// # Errors
    ///
    /// Returns an error if the sweep time cannot be queried or parsed.
    pub fn sweep_time(&mut self) -> Result<f64> {
        self.parent
            .vna
            .query(&format!("SENS{}:SWE:TIME?", self.number))?
            .parse()
            .map_err(|error| Error::Parse(format!("invalid sweep time: {error}")))
    }

    /// Sets the duration of one sweep in seconds.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the sweep-time command.
    pub fn set_sweep_time(&mut self, seconds: f64) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:TIME {seconds}", self.number))
    }

    /// Returns the sweep-axis type.
    ///
    /// # Errors
    ///
    /// Returns an error if the sweep type cannot be queried or parsed.
    pub fn sweep_type(&mut self) -> Result<SweepType> {
        self.parent
            .vna
            .query(&format!("SENS{}:SWE:TYPE?", self.number))?
            .parse()
    }

    /// Sets the sweep-axis type.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the sweep-type command.
    pub fn set_sweep_type(&mut self, sweep_type: SweepType) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:TYPE {sweep_type}", self.number))
    }

    /// Returns whether measurement averaging is enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the averaging state cannot be queried or parsed.
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
    /// Returns an error if the instrument rejects the averaging-state command.
    pub fn set_averaging_on(&mut self, enabled: bool) -> Result<()> {
        let value = BooleanValidator::default()
            .validate_input(enabled)
            .map_err(|error| Error::Parse(error.to_string()))?;
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:STATE {value}", self.number))
    }

    /// Returns the number of measurements combined for an average.
    ///
    /// # Errors
    ///
    /// Returns an error if the averaging count cannot be queried or parsed.
    pub fn averaging_count(&mut self) -> Result<usize> {
        IntValidator::new(Some(1), Some(1_000))
            .validate_output(
                self.parent
                    .vna
                    .query(&format!("SENS{}:AVER:COUN?", self.number))?,
            )
            .map_err(|error| Error::Parse(error.to_string()))
            .and_then(|value| {
                usize::try_from(value).map_err(|error| Error::Parse(error.to_string()))
            })
    }

    /// Sets the number of measurements combined for an average.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the averaging-count command.
    pub fn set_averaging_count(&mut self, count: usize) -> Result<()> {
        IntValidator::new(Some(1), Some(1_000))
            .validate_input(count)
            .map_err(|error| Error::Parse(error.to_string()))?;
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:COUN {count}", self.number))
    }

    /// Returns how measurements are averaged together.
    ///
    /// # Errors
    ///
    /// Returns an error if the averaging mode cannot be queried or parsed.
    pub fn averaging_mode(&mut self) -> Result<AveragingMode> {
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
    pub fn set_averaging_mode(&mut self, mode: AveragingMode) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:MODE {mode}", self.number))
    }

    /// Returns this channel's sweep mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the sweep mode cannot be queried or parsed.
    pub fn sweep_mode(&mut self) -> Result<SweepMode> {
        self.parent
            .vna
            .query(&format!("INIT{}:CONT?", self.number))?
            .parse()
    }

    /// Sets this channel's sweep mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the sweep-mode command.
    pub fn set_sweep_mode(&mut self, mode: SweepMode) -> Result<()> {
        self.parent
            .vna
            .write(&format!("INIT{}:CONT {mode}", self.number))
    }

    /// Returns the channel's frequency axis.
    ///
    /// # Errors
    ///
    /// Returns an error if the axis settings cannot be queried, parsed, or combined.
    pub fn frequency(&mut self) -> Result<Frequency> {
        let start_hz = self
            .frequency_start()?
            .to_f64()
            .ok_or_else(|| Error::Parse("start frequency cannot be represented as f64".into()))?;
        let stop_hz = self
            .frequency_stop()?
            .to_f64()
            .ok_or_else(|| Error::Parse("stop frequency cannot be represented as f64".into()))?;
        Frequency::new(
            start_hz,
            stop_hz,
            self.points()?,
            FrequencyUnit::Hz,
            FrequencySweepType::Linear,
        )
    }

    /// Configures the channel from a frequency axis.
    ///
    /// # Errors
    ///
    /// Returns an error if any frequency-axis command is rejected.
    pub fn set_frequency_axis(&mut self, frequency: &Frequency) -> Result<()> {
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

    /// Clears accumulated averaging data.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the clear command.
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
    /// Returns an error if the instrument rejects the deletion command.
    pub fn delete_measurement(&mut self, name: &str) -> Result<()> {
        self.parent.delete_measurement(self.number, name)
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
            .query(&format!("CALC{}:PAR:SEL?", self.number))?;
        if selected
            .trim_matches('\'')
            .split(',')
            .next()
            .unwrap_or("")
            .is_empty()
        {
            return Err(Error::Unsupported(
                "No trace is active. Must select measurement first.".into(),
            ));
        }
        self.parent
            .vna
            .query_complex_values(&format!("CALC{}:DATA? SDATA", self.number))
    }

    /// Creates an S-parameter group for the requested physical ports.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument rejects the group-definition command.
    pub fn create_s_parameter_group(&mut self, ports: &[usize]) -> Result<()> {
        let ports = ports
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        self.parent
            .vna
            .write(&format!("CALC{}:PAR:DEF:SGR {ports}", self.number))
    }

    /// Performs one channel sweep and restores the prior sweep mode.
    ///
    /// # Errors
    ///
    /// Returns an error if any sweep-control command fails.
    pub fn sweep(&mut self) -> Result<()> {
        let original = self.sweep_mode()?;
        self.set_sweep_mode(SweepMode::Single)?;
        self.parent.vna.clear()?;
        self.parent.vna.write(&format!("INIT{}:IMM", self.number))?;
        self.parent.vna.wait_for_complete()?;
        self.set_sweep_mode(original)
    }

    /// Acquires an $n$-port network for the requested physical ports.
    ///
    /// # Errors
    ///
    /// Returns an error if acquisition, data conversion, or network construction fails.
    pub fn get_snp_network(&mut self, ports: &[usize]) -> Result<Network> {
        let original_format = self.parent.vna.values_format;
        self.parent.set_query_format(ValuesFormat::Binary64)?;
        self.create_s_parameter_group(ports)?;
        self.sweep()?;
        let raw = self
            .parent
            .vna
            .query_complex_values(&format!("CALC{}:DATA:SGR? SDAT", self.number))?;
        let frequency = self.frequency()?;
        let points = frequency.points();
        if raw.len() != points * ports.len() * ports.len() {
            return Err(Error::IncompatibleShape(format!(
                "R&S SNP response has {} values, expected {}",
                raw.len(),
                points * ports.len() * ports.len()
            )));
        }
        let mut s = Array3::zeros((points, ports.len(), ports.len()));
        for output in 0..ports.len() {
            for input in 0..ports.len() {
                for point in 0..points {
                    s[[point, output, input]] =
                        raw[(output * ports.len() + input) * points + point];
                }
            }
        }
        self.parent.set_query_format(original_format)?;
        Network::new(
            frequency,
            s,
            Array2::from_elem((points, ports.len()), Complex64::new(50.0, 0.0)),
        )
    }

    fn query_frequency(&mut self, command: &str) -> Result<u64> {
        FrequencyValidator
            .validate_output(
                self.parent
                    .vna
                    .query(&format!("SENS{}:{command}", self.number))?,
            )
            .map_err(|error| Error::Parse(error.to_string()))
    }

    fn set_frequency(&mut self, command: &str, value: impl ToString) -> Result<()> {
        let value = FrequencyValidator
            .validate_input(value)
            .map_err(|error| Error::Parse(error.to_string()))?;
        self.parent
            .vna
            .write(&format!("SENS{}:{command} {value}", self.number))
    }
}
