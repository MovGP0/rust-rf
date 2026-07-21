//! Rohde & Schwarz VNA family.
//!
//! Origins: `skrf/vi/vna/rohde_schwarz/rs_vna.py`, `zna.py`, and `zva.py`.

use std::fmt::{Display, Formatter};
use std::str::FromStr;

use ndarray::{Array2, Array3};
use num_complex::Complex64;

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
pub enum SweepType {
    Linear,
    Log,
    Segment,
    Power,
    ContinuousWave,
    Point,
    Pulse,
    IAmplitude,
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
pub enum SweepMode {
    Single,
    Continuous,
}

scpi_enum!(SweepMode {
    Single => "0",
    Continuous => "1",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AveragingMode {
    Auto,
    Flatten,
    Reduce,
    Moving,
}

scpi_enum!(AveragingMode {
    Auto => "AUTO",
    Flatten => "FLAT",
    Reduce => "RED",
    Moving => "MOV",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsFamily {
    Zna,
    Zva,
    Generic,
}

pub struct RohdeSchwarzVna<S>
where
    S: InstrumentSession,
{
    pub vna: Vna<S>,
    pub model: String,
    pub family: RsFamily,
}

impl<S> RohdeSchwarzVna<S>
where
    S: InstrumentSession,
{
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

    pub fn from_model(vna: Vna<S>, family: RsFamily, model: impl Into<String>) -> Result<Self> {
        let mut driver = Self {
            vna,
            model: model.into(),
            family,
        };
        driver.create_channel(1, "Channel 1")?;
        Ok(driver)
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
            self.vna.write(&format!("CONF:CHAN{number} OFF"))?;
            self.vna.delete_channel(number);
        }
        Ok(())
    }

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

    pub fn nports(&mut self) -> Result<usize> {
        if self.supports("nports") {
            IntValidator::new(Some(1), None)
                .validate_output(self.vna.query("INST:PORT:COUN?")?)
                .map(|value| value as usize)
                .map_err(|error| Error::Parse(error.to_string()))
        } else {
            Ok(self.model_ports())
        }
    }

    pub fn active_channel_number(&mut self) -> Result<usize> {
        IntValidator::new(Some(1), None)
            .validate_output(self.vna.query("INST:NSEL?")?)
            .map(|value| value as usize)
            .map_err(|error| Error::Parse(error.to_string()))
    }

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
        let channel = self.active_channel_number()?;
        self.vna.query(&format!("CALC{channel}:PAR:SEL?"))
    }

    pub fn set_active_measurement(&mut self, name: &str) -> Result<()> {
        for channel in self
            .vna
            .channels()
            .iter()
            .map(|channel| channel.number)
            .collect::<Vec<_>>()
        {
            if self
                .measurement_names(channel)?
                .iter()
                .any(|candidate| candidate == name)
            {
                return self.vna.write(&format!("CALC{channel}:PAR:SEL '{name}'"));
            }
        }
        Err(Error::Unsupported(format!(
            "measurement {name} does not exist"
        )))
    }

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

    pub fn measurement_names(&mut self, channel: usize) -> Result<Vec<String>> {
        Ok(self
            .measurements(channel)?
            .into_iter()
            .map(|(name, _)| name)
            .collect())
    }

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

pub struct RsChannel<'a, S>
where
    S: InstrumentSession,
{
    parent: &'a mut RohdeSchwarzVna<S>,
    pub number: usize,
}

impl<S> RsChannel<'_, S>
where
    S: InstrumentSession,
{
    pub fn frequency_start(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:STAR?")
    }

    pub fn set_frequency_start(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("FREQ:STAR", value)
    }

    pub fn frequency_stop(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:STOP?")
    }

    pub fn set_frequency_stop(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("FREQ:STOP", value)
    }

    pub fn frequency_span(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:SPAN?")
    }

    pub fn set_frequency_span(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("FREQ:SPAN", value)
    }

    pub fn frequency_center(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:CENT?")
    }

    pub fn set_frequency_center(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("FREQ:CENT", value)
    }

    pub fn frequency_fixed(&mut self) -> Result<u64> {
        self.query_frequency("FREQ:FIX?")
    }

    pub fn set_frequency_fixed(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("FREQ:FIX", value)
    }

    pub fn points(&mut self) -> Result<usize> {
        IntValidator::default()
            .validate_output(
                self.parent
                    .vna
                    .query(&format!("SENS{}:SWE:POIN?", self.number))?,
            )
            .map(|value| value as usize)
            .map_err(|error| Error::Parse(error.to_string()))
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
        self.set_frequency("BWID", value)
    }

    pub fn frequency_step(&mut self) -> Result<u64> {
        self.query_frequency("SWE:STEP?")
    }

    pub fn set_frequency_step(&mut self, value: impl ToString) -> Result<()> {
        self.set_frequency("SWE:STEP", value)
    }

    pub fn sweep_time(&mut self) -> Result<f64> {
        self.parent
            .vna
            .query(&format!("SENS{}:SWE:TIME?", self.number))?
            .parse()
            .map_err(|error| Error::Parse(format!("invalid sweep time: {error}")))
    }

    pub fn set_sweep_time(&mut self, seconds: f64) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:TIME {seconds}", self.number))
    }

    pub fn sweep_type(&mut self) -> Result<SweepType> {
        self.parent
            .vna
            .query(&format!("SENS{}:SWE:TYPE?", self.number))?
            .parse()
    }

    pub fn set_sweep_type(&mut self, sweep_type: SweepType) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:SWE:TYPE {sweep_type}", self.number))
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
            .map_err(|error| Error::Parse(error.to_string()))?;
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:STATE {value}", self.number))
    }

    pub fn averaging_count(&mut self) -> Result<usize> {
        IntValidator::new(Some(1), Some(1_000))
            .validate_output(
                self.parent
                    .vna
                    .query(&format!("SENS{}:AVER:COUN?", self.number))?,
            )
            .map(|value| value as usize)
            .map_err(|error| Error::Parse(error.to_string()))
    }

    pub fn set_averaging_count(&mut self, count: usize) -> Result<()> {
        IntValidator::new(Some(1), Some(1_000))
            .validate_input(count)
            .map_err(|error| Error::Parse(error.to_string()))?;
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:COUN {count}", self.number))
    }

    pub fn averaging_mode(&mut self) -> Result<AveragingMode> {
        self.parent
            .vna
            .query(&format!("SENS{}:AVER:MODE?", self.number))?
            .parse()
    }

    pub fn set_averaging_mode(&mut self, mode: AveragingMode) -> Result<()> {
        self.parent
            .vna
            .write(&format!("SENS{}:AVER:MODE {mode}", self.number))
    }

    pub fn sweep_mode(&mut self) -> Result<SweepMode> {
        self.parent
            .vna
            .query(&format!("INIT{}:CONT?", self.number))?
            .parse()
    }

    pub fn set_sweep_mode(&mut self, mode: SweepMode) -> Result<()> {
        self.parent
            .vna
            .write(&format!("INIT{}:CONT {mode}", self.number))
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

    pub fn set_frequency_axis(&mut self, frequency: &Frequency) -> Result<()> {
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

    pub fn sweep(&mut self) -> Result<()> {
        let original = self.sweep_mode()?;
        self.set_sweep_mode(SweepMode::Single)?;
        self.parent.vna.clear()?;
        self.parent.vna.write(&format!("INIT{}:IMM", self.number))?;
        self.parent.vna.wait_for_complete()?;
        self.set_sweep_mode(original)
    }

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
