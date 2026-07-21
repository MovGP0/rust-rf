//! Keysight VNA driver implementation.
//!
//! Origin: `skrf/vi/vna/keysight/pna.py`.

use std::fmt::{Display, Formatter};
use std::str::FromStr;

use ndarray::{Array2, Array3};
use num_complex::Complex64;

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
pub enum PnaSweepType {
    Linear,
    Log,
    Power,
    ContinuousWave,
    Segment,
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
pub enum PnaSweepMode {
    Hold,
    Continuous,
    Groups,
    Single,
}

scpi_enum!(PnaSweepMode {
    Hold => "HOLD",
    Continuous => "CONT",
    Groups => "GRO",
    Single => "SING",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerSource {
    External,
    Immediate,
    Manual,
}

scpi_enum!(TriggerSource {
    External => "EXT",
    Immediate => "IMM",
    Manual => "MAN",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PnaAveragingMode {
    Point,
    Sweep,
}

scpi_enum!(PnaAveragingMode {
    Point => "POIN",
    Sweep => "SWE",
});

pub struct Pna<S: InstrumentSession> {
    pub vna: Vna<S>,
    pub model: String,
}

impl<S: InstrumentSession> Pna<S> {
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

    pub fn from_model(vna: Vna<S>, model: impl Into<String>) -> Result<Self> {
        let mut pna = Self {
            vna,
            model: model.into(),
        };
        pna.create_channel(1, "Channel 1")?;
        Ok(pna)
    }

    pub fn create_channel(&mut self, number: usize, name: impl Into<String>) -> Result<()> {
        self.vna.create_channel(number, name)?;
        if number != 1 {
            self.create_measurement(number, &format!("CH{number}_S11_1"), "S11")?;
        }
        Ok(())
    }

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

    pub fn trigger_source(&mut self) -> Result<TriggerSource> {
        self.vna.query("TRIG:SOUR?")?.parse()
    }

    pub fn set_trigger_source(&mut self, source: TriggerSource) -> Result<()> {
        self.vna.write(&format!("TRIG:SOUR {source}"))
    }

    pub fn error_count(&mut self) -> Result<usize> {
        parse_integer(self.vna.query("SYST:ERR:COUN?")?, Some(0), None).map(|value| value as usize)
    }

    pub fn channel_numbers(&mut self) -> Result<Vec<usize>> {
        parse_integer_list(&self.vna.query("SYST:CHAN:CAT?")?)
    }

    pub fn ports(&mut self) -> Result<usize> {
        if self.supports("nports") {
            parse_integer(self.vna.query("SYST:CAP:HARD:PORT:COUN?")?, Some(1), None)
                .map(|value| value as usize)
        } else {
            Ok(self.model_ports())
        }
    }

    pub fn active_channel_number(&mut self) -> Result<Option<usize>> {
        let number = parse_integer(self.vna.query("SYST:ACT:CHAN?")?, Some(0), None)? as usize;
        Ok(self
            .vna
            .channels()
            .iter()
            .any(|channel| channel.number == number)
            .then_some(number))
    }

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

    pub fn query_format(&mut self) -> Result<ValuesFormat> {
        let format = parse_values_format(&self.vna.query("FORM?")?.replace('+', ""))?;
        self.vna.values_format = format;
        Ok(format)
    }

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

    pub fn active_measurement(&mut self) -> Result<String> {
        Ok(self.vna.query("SYST:ACT:MEAS?")?.replace('"', ""))
    }

    pub fn set_active_measurement(&mut self, name: &str) -> Result<()> {
        for number in self
            .vna
            .channels()
            .iter()
            .map(|channel| channel.number)
            .collect::<Vec<_>>()
        {
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

    pub fn measurement_names(&mut self, channel: usize) -> Result<Vec<String>> {
        Ok(self
            .measurements(channel)?
            .into_iter()
            .map(|(name, _)| name)
            .collect())
    }

    pub fn measurement_numbers(&mut self, channel: usize) -> Result<Vec<usize>> {
        parse_integer_list(&self.vna.query(&format!("SYST:MEAS:CAT? {channel}"))?)
    }

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

pub struct PnaChannel<'a, S: InstrumentSession> {
    parent: &'a mut Pna<S>,
    pub number: usize,
}

impl<S: InstrumentSession> PnaChannel<'_, S> {
    pub fn frequency_start(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:STAR?")
    }

    pub fn set_frequency_start(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("FREQ:STAR", value)
    }

    pub fn frequency_stop(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:STOP?")
    }

    pub fn set_frequency_stop(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("FREQ:STOP", value)
    }

    pub fn frequency_span(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:SPAN?")
    }

    pub fn set_frequency_span(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("FREQ:SPAN", value)
    }

    pub fn frequency_center(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:CENT?")
    }

    pub fn set_frequency_center(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("FREQ:CENT", value)
    }

    pub fn points(&mut self) -> Result<usize> {
        parse_integer(
            self.parent
                .vna
                .query(&format!("SENS{}:SWE:POIN?", self.number))?,
            Some(1),
            None,
        )
        .map(|value| value as usize)
    }

    pub fn set_points(&mut self, points: usize) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:POIN {points}", self.number))
    }

    pub fn if_bandwidth(&mut self) -> Result<u64> {
        self.query_frequency("BWID?")
    }

    pub fn set_if_bandwidth(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency_value("BWID", value)
    }

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

    pub fn set_sweep_time(&mut self, seconds: f64) -> Result<()> {
        let seconds = FloatValidator::new(None, None, 2)
            .validate_input(seconds)
            .map_err(validation_error)?;
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:TIME {seconds}", self.number))
    }

    pub fn sweep_type(&mut self) -> Result<PnaSweepType> {
        self.parent
            .vna
            .query(&format!("SENS{}:SWE:TYPE?", self.number))?
            .parse()
    }

    pub fn set_sweep_type(&mut self, sweep_type: PnaSweepType) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:TYPE {sweep_type}", self.number))
    }

    pub fn sweep_mode(&mut self) -> Result<PnaSweepMode> {
        self.parent
            .vna
            .query(&format!("SENS{}:SWE:MODE?", self.number))?
            .parse()
    }

    pub fn set_sweep_mode(&mut self, mode: PnaSweepMode) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:MODE {mode}", self.number))
    }

    pub fn measurement_numbers(&mut self) -> Result<Vec<usize>> {
        self.parent.measurement_numbers(self.number)
    }

    pub fn averaging_on(&mut self) -> Result<bool> {
        Ok(BooleanValidator::default().validate_output(
            self.parent
                .vna
                .query(&format!("SENS{}:AVER:STATE?", self.number))?,
        ))
    }

    pub fn set_averaging_on(&mut self, enabled: bool) -> Result<()> {
        let value = BooleanValidator::default()
            .validate_input(enabled)
            .map_err(validation_error)?;
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:STATE {value}", self.number))
    }

    pub fn averaging_count(&mut self) -> Result<usize> {
        parse_integer(
            self.parent
                .vna
                .query(&format!("SENS{}:AVER:COUN?", self.number))?,
            Some(1),
            Some(65_536),
        )
        .map(|value| value as usize)
    }

    pub fn set_averaging_count(&mut self, count: usize) -> Result<()> {
        parse_integer(count.to_string(), Some(1), Some(65_536))?;
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:COUN {count}", self.number))
    }

    pub fn averaging_mode(&mut self) -> Result<PnaAveragingMode> {
        self.parent
            .vna
            .query(&format!("SENS{}:AVER:MODE?", self.number))?
            .parse()
    }

    pub fn set_averaging_mode(&mut self, mode: PnaAveragingMode) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:MODE {mode}", self.number))
    }

    pub fn sweep_groups(&mut self) -> Result<usize> {
        parse_integer(
            self.parent
                .vna
                .query(&format!("SENS{}:SWE:GRO:COUN?", self.number))?,
            Some(1),
            Some(2_000_000),
        )
        .map(|value| value as usize)
    }

    pub fn set_sweep_groups(&mut self, groups: usize) -> Result<()> {
        parse_integer(groups.to_string(), Some(1), Some(2_000_000))?;
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:GRO:COUN {groups}", self.number))
    }

    pub fn frequency_step(&mut self) -> Result<u64> {
        Ok(self.frequency()?.step().unwrap_or(0.0) as u64)
    }

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
        let span = frequency.stop().unwrap_or(0.0) as u64 - frequency.start().unwrap_or(0.0) as u64;
        self.set_points((span / step + 1) as usize)
    }

    pub fn frequency(&mut self) -> Result<Frequency> {
        Frequency::new(
            self.frequency_start()? as f64,
            self.frequency_stop()? as f64,
            self.points()?,
            FrequencyUnit::Hz,
            FrequencySweepType::Linear,
        )
    }

    pub fn set_frequency(&mut self, frequency: &Frequency) -> Result<()> {
        self.set_frequency_start(frequency.start().unwrap_or(0.0))?;
        self.set_frequency_stop(frequency.stop().unwrap_or(0.0))?;
        self.set_points(frequency.points())
    }

    pub fn measurements(&mut self) -> Result<Vec<(String, String)>> {
        self.parent.measurements(self.number)
    }

    pub fn measurement_names(&mut self) -> Result<Vec<String>> {
        self.parent.measurement_names(self.number)
    }

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

    pub fn clear_averaging(&mut self) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:CLE", self.number))
    }

    pub fn create_measurement(&mut self, name: &str, parameter: &str) -> Result<()> {
        self.parent.create_measurement(self.number, name, parameter)
    }

    pub fn delete_measurement(&mut self, name: &str) -> Result<()> {
        self.parent.delete_measurement(self.number, name)
    }

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
        self.parent.vna.timeout_ms =
            Some(((original_time * sweeps as f64 * 1_000.0) as u64).max(5_000));
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
