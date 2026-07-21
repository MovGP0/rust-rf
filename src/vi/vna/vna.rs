//! Vector network analyzer support.
//!
//! Origin: `skrf/vi/vna/vna.py`.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use num_complex::Complex64;
use regex::Regex;
use thiserror::Error;

use crate::{Error, Result};

use crate::vi::scpi_errors::ScpiError;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ValuesFormat {
    Binary32,
    Binary64,
    #[default]
    Ascii,
}

pub trait InstrumentSession: Read + Write + Send {
    fn clear(&mut self) -> Result<()>;

    fn query(&mut self, command: &str) -> Result<Vec<u8>>;

    fn read_raw(&mut self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.read_to_end(&mut bytes)?;
        Ok(bytes)
    }
}

#[derive(Debug, Error)]
pub enum VnaError {
    #[error(transparent)]
    Port(#[from] Error),
    #[error(transparent)]
    Scpi(#[from] ScpiError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Channel {
    pub number: usize,
    pub name: String,
}

/// A transport-independent vector network analyzer base.
pub struct Vna<S>
where
    S: InstrumentSession,
{
    pub session: S,
    pub address: String,
    pub values_format: ValuesFormat,
    pub echo: bool,
    pub timeout_ms: Option<u64>,
    channels: Vec<Channel>,
    active_channel: Option<usize>,
}

impl<S> Vna<S>
where
    S: InstrumentSession,
{
    pub fn new(address: impl Into<String>, session: S, timeout_ms: Option<u64>) -> Self {
        Self {
            session,
            address: address.into(),
            values_format: ValuesFormat::Ascii,
            echo: false,
            timeout_ms,
            channels: Vec::new(),
            active_channel: None,
        }
    }

    pub fn into_session(self) -> S {
        self.session
    }

    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    pub fn create_channel(&mut self, number: usize, name: impl Into<String>) -> Result<&Channel> {
        if self.channels.iter().any(|channel| channel.number == number) {
            return Err(Error::Unsupported(format!(
                "Channel {number} already exists"
            )));
        }
        self.channels.push(Channel {
            number,
            name: name.into(),
        });
        self.channels.sort_by_key(|channel| channel.number);
        self.active_channel.get_or_insert(number);
        Ok(self
            .channels
            .iter()
            .find(|channel| channel.number == number)
            .ok_or_else(|| Error::Unsupported("newly created VNA channel is missing".to_owned()))?)
    }

    pub fn delete_channel(&mut self, number: usize) -> Option<Channel> {
        let index = self
            .channels
            .iter()
            .position(|channel| channel.number == number)?;
        let channel = self.channels.remove(index);
        if self.active_channel == Some(number) {
            self.active_channel = self.channels.first().map(|channel| channel.number);
        }
        Some(channel)
    }

    pub fn set_active_channel(&mut self, number: usize) -> Result<()> {
        if !self.channels.iter().any(|channel| channel.number == number) {
            return Err(Error::Unsupported(format!(
                "Channel {number} does not exist"
            )));
        }
        self.active_channel = Some(number);
        Ok(())
    }

    pub fn active_channel(&self) -> Option<&Channel> {
        let number = self.active_channel?;
        self.channels
            .iter()
            .find(|channel| channel.number == number)
    }

    pub fn read(&mut self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.session.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    pub fn write(&mut self, command: &str) -> Result<()> {
        self.session.write_all(command.as_bytes())?;
        self.session.flush()?;
        Ok(())
    }

    pub fn query(&mut self, command: &str) -> Result<String> {
        let response = self.session.query(command)?;
        String::from_utf8(response)
            .map(|response| response.trim().to_owned())
            .map_err(|error| Error::Parse(format!("instrument returned non-UTF-8 text: {error}")))
    }

    pub fn clear(&mut self) -> Result<()> {
        self.session.clear()
    }

    pub fn wait_for_complete(&mut self) -> Result<String> {
        self.query("*OPC?")
    }

    pub fn status(&mut self) -> Result<String> {
        self.query("*STB?")
    }

    pub fn options(&mut self) -> Result<String> {
        self.query("*OPT?")
    }

    pub fn id(&mut self) -> Result<String> {
        self.query("*IDN?")
    }

    pub fn clear_errors(&mut self) -> Result<()> {
        self.write("*CLS")
    }

    pub fn check_errors(&mut self) -> std::result::Result<(), VnaError> {
        let response = self.query("SYST:ERR?")?;
        let error_code = response
            .split(',')
            .next()
            .ok_or_else(|| Error::Parse("empty SCPI error response".into()))?
            .trim()
            .parse::<i32>()
            .map_err(|error| Error::Parse(format!("invalid SCPI error response: {error}")))?;
        if error_code == 0 {
            Ok(())
        } else {
            Err(ScpiError::new(error_code).into())
        }
    }

    pub fn write_values(&mut self, command: &str, values: &[f64]) -> Result<()> {
        match self.values_format {
            ValuesFormat::Ascii => {
                let values = values
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                self.write(&format!("{command} {values}"))
            }
            ValuesFormat::Binary32 => {
                let payload = values
                    .iter()
                    .flat_map(|value| (*value as f32).to_le_bytes())
                    .collect::<Vec<_>>();
                self.write_binary_block(command, &payload)
            }
            ValuesFormat::Binary64 => {
                let payload = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                self.write_binary_block(command, &payload)
            }
        }
    }

    pub fn write_complex_values(&mut self, command: &str, values: &[Complex64]) -> Result<()> {
        let interleaved = values
            .iter()
            .flat_map(|value| [value.re, value.im])
            .collect::<Vec<_>>();
        self.write_values(command, &interleaved)
    }

    pub fn query_values(&mut self, command: &str) -> Result<Vec<f64>> {
        let response = self.session.query(command)?;
        match self.values_format {
            ValuesFormat::Ascii => parse_ascii_values(&response),
            ValuesFormat::Binary32 => {
                parse_binary_values::<4>(&response, |bytes| f32::from_le_bytes(bytes) as f64)
            }
            ValuesFormat::Binary64 => parse_binary_values::<8>(&response, f64::from_le_bytes),
        }
    }

    pub fn query_complex_values(&mut self, command: &str) -> Result<Vec<Complex64>> {
        let values = self.query_values(command)?;
        if !values.chunks_exact(2).remainder().is_empty() {
            return Err(Error::Parse(format!(
                "complex value response contained {} scalar values",
                values.len()
            )));
        }
        Ok(values
            .chunks_exact(2)
            .map(|pair| Complex64::new(pair[0], pair[1]))
            .collect())
    }

    fn write_binary_block(&mut self, command: &str, payload: &[u8]) -> Result<()> {
        let length = payload.len().to_string();
        self.session.write_all(command.as_bytes())?;
        self.session.write_all(b" ")?;
        self.session
            .write_all(format!("#{}{length}", length.len()).as_bytes())?;
        self.session.write_all(payload)?;
        self.session.flush()?;
        Ok(())
    }
}

pub fn format_command(command: &str, parameters: &BTreeMap<String, String>) -> Result<String> {
    let expression = Regex::new(r"<(?:(?P<prefix>\w+):)?(?P<attribute>\w+)>")
        .map_err(|error| Error::Parse(format!("invalid VNA command regex: {error}")))?;
    let mut missing = None;
    let formatted = expression.replace_all(command, |captures: &regex::Captures<'_>| {
        let key = captures
            .name("prefix")
            .map(|prefix| format!("{}:{}", prefix.as_str(), &captures["attribute"]))
            .unwrap_or_else(|| captures["attribute"].to_owned());
        match parameters.get(&key) {
            Some(value) => value.clone(),
            None => {
                missing = Some(key);
                captures[0].to_owned()
            }
        }
    });
    if let Some(key) = missing {
        Err(Error::Parse(format!("missing VNA command parameter {key}")))
    } else {
        Ok(formatted.into_owned())
    }
}

fn parse_ascii_values(response: &[u8]) -> Result<Vec<f64>> {
    let response = std::str::from_utf8(response)
        .map_err(|error| Error::Parse(format!("instrument returned non-UTF-8 values: {error}")))?;
    response
        .trim()
        .split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value.trim().parse::<f64>().map_err(|error| {
                Error::Parse(format!("invalid instrument value {value:?}: {error}"))
            })
        })
        .collect()
}

fn parse_binary_values<const WIDTH: usize>(
    response: &[u8],
    convert: impl Fn([u8; WIDTH]) -> f64,
) -> Result<Vec<f64>> {
    let payload = definite_block_payload(response)?;
    if !payload.chunks_exact(WIDTH).remainder().is_empty() {
        return Err(Error::Parse(format!(
            "binary value payload has {} bytes, not a multiple of {WIDTH}",
            payload.len()
        )));
    }
    payload
        .chunks_exact(WIDTH)
        .map(|chunk| {
            chunk.try_into().map(|bytes| convert(bytes)).map_err(|_| {
                Error::Parse(format!("binary value chunk does not contain {WIDTH} bytes"))
            })
        })
        .collect()
}

fn definite_block_payload(response: &[u8]) -> Result<&[u8]> {
    if response.len() < 2 || response[0] != b'#' || !response[1].is_ascii_digit() {
        return Err(Error::Parse("invalid SCPI definite-length block".into()));
    }
    let digit_count = (response[1] - b'0') as usize;
    if digit_count == 0 || response.len() < 2 + digit_count {
        return Err(Error::Parse("invalid SCPI definite-length header".into()));
    }
    let payload_length = std::str::from_utf8(&response[2..2 + digit_count])
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| Error::Parse("invalid SCPI definite-length size".into()))?;
    let start = 2 + digit_count;
    let end = start + payload_length;
    response
        .get(start..end)
        .ok_or_else(|| Error::Parse("truncated SCPI definite-length block".into()))
}
