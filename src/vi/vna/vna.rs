//! Vector network analyzer support.
//!
//! Origin: `skrf/vi/vna/vna.py`.
//!
//! [`Vna`] provides the shared channel registry, SCPI common commands, numeric
//! transfer formats, and scalar/complex value conversion used by instrument
//! drivers. Rust drivers expose typed methods directly instead of constructing
//! dynamic properties at runtime.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use num_complex::Complex64;
use num_traits::ToPrimitive;
use regex::Regex;
use thiserror::Error;

use crate::{Error, Result};

use crate::vi::scpi_errors::ScpiError;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// How numeric values are written to and queried from an instrument.
pub enum ValuesFormat {
    /// IEEE-754 binary values with 32 bits per value.
    Binary32,
    /// IEEE-754 binary values with 64 bits per value.
    Binary64,
    /// Comma-separated ASCII values.
    #[default]
    Ascii,
}

/// Byte-oriented transport required by the VNA drivers.
///
/// Implementations may wrap VISA, a serial connection, a network socket, or a
/// deterministic test double.
pub trait InstrumentSession: Read + Write + Send {
    /// Clears the instrument or transport buffers.
    ///
    /// # Errors
    ///
    /// Returns an error when the transport cannot clear the instrument or its
    /// buffers.
    fn clear(&mut self) -> Result<()>;

    /// Writes a textual command and returns its raw response bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when the command cannot be written or its response
    /// cannot be read from the transport.
    fn query(&mut self, command: &str) -> Result<Vec<u8>>;

    /// Reads the available raw response bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when the transport cannot be read to completion.
    fn read_raw(&mut self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.read_to_end(&mut bytes)?;
        Ok(bytes)
    }
}

#[derive(Debug, Error)]
/// Error returned by shared VNA operations that also check the SCPI queue.
pub enum VnaError {
    /// Transport, parsing, validation, or port error.
    #[error(transparent)]
    Port(#[from] Error),
    /// Error reported by the instrument's SCPI error queue.
    #[error(transparent)]
    Scpi(#[from] ScpiError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// A single logical channel of an instrument.
///
/// Model-specific channel controllers borrow the parent driver and add the
/// commands supported by that instrument.
pub struct Channel {
    /// Instrument channel number.
    pub number: usize,
    /// Human-readable channel name.
    pub name: String,
}

/// A transport-independent vector network analyzer base.
pub struct Vna<S>
where
    S: InstrumentSession,
{
    /// Instrument transport session.
    pub session: S,
    /// Resource address used to open the instrument.
    pub address: String,
    /// Numeric transfer format currently selected on the instrument.
    pub values_format: ValuesFormat,
    /// Whether command echoing is enabled by the caller.
    pub echo: bool,
    /// Optional transport timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    channels: Vec<Channel>,
    active_channel: Option<usize>,
}

impl<S> Vna<S>
where
    S: InstrumentSession,
{
    /// Creates a transport-independent VNA with ASCII transfers by default.
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

    /// Consumes the driver and returns its transport session.
    pub fn into_session(self) -> S {
        self.session
    }

    /// Returns the logical channels currently registered by the driver.
    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    /// Registers a numbered channel.
    ///
    /// The first channel created becomes active automatically.
    ///
    /// # Errors
    ///
    /// Returns an error when `number` is already registered or the inserted
    /// channel cannot be found after sorting the registry.
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
        self.channels
            .iter()
            .find(|channel| channel.number == number)
            .ok_or_else(|| Error::Unsupported("newly created VNA channel is missing".to_owned()))
    }

    /// Removes a numbered channel and returns its metadata when present.
    ///
    /// If the active channel is removed, the first remaining channel becomes
    /// active.
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

    /// Marks an existing channel as active.
    ///
    /// # Errors
    ///
    /// Returns an error when `number` is not registered.
    pub fn set_active_channel(&mut self, number: usize) -> Result<()> {
        if !self.channels.iter().any(|channel| channel.number == number) {
            return Err(Error::Unsupported(format!(
                "Channel {number} does not exist"
            )));
        }
        self.active_channel = Some(number);
        Ok(())
    }

    /// Returns the active channel, if one is registered.
    pub fn active_channel(&self) -> Option<&Channel> {
        let number = self.active_channel?;
        self.channels
            .iter()
            .find(|channel| channel.number == number)
    }

    /// Reads all currently available raw bytes from the session.
    ///
    /// # Errors
    ///
    /// Returns an error when the session cannot be read to completion.
    pub fn read(&mut self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.session.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    /// Writes a command and flushes the transport.
    ///
    /// # Errors
    ///
    /// Returns an error when the command cannot be written or the session
    /// cannot be flushed.
    pub fn write(&mut self, command: &str) -> Result<()> {
        self.session.write_all(command.as_bytes())?;
        self.session.flush()?;
        Ok(())
    }

    /// Sends a textual query and returns its trimmed UTF-8 response.
    ///
    /// # Errors
    ///
    /// Returns an error when the session query fails or the response is not
    /// valid UTF-8.
    pub fn query(&mut self, command: &str) -> Result<String> {
        let response = self.session.query(command)?;
        String::from_utf8(response)
            .map(|response| response.trim().to_owned())
            .map_err(|error| Error::Parse(format!("instrument returned non-UTF-8 text: {error}")))
    }

    /// Clears the instrument or transport buffers.
    ///
    /// # Errors
    ///
    /// Returns an error when the session cannot clear the instrument or its
    /// buffers.
    pub fn clear(&mut self) -> Result<()> {
        self.session.clear()
    }

    /// Waits for pending instrument operations to complete using `*OPC?`.
    ///
    /// # Errors
    ///
    /// Returns an error when the `*OPC?` query fails or its response is invalid.
    pub fn wait_for_complete(&mut self) -> Result<String> {
        self.query("*OPC?")
    }

    /// Queries the SCPI status byte using `*STB?`.
    ///
    /// # Errors
    ///
    /// Returns an error when the `*STB?` query fails or its response is invalid.
    pub fn status(&mut self) -> Result<String> {
        self.query("*STB?")
    }

    /// Queries the installed instrument options using `*OPT?`.
    ///
    /// # Errors
    ///
    /// Returns an error when the `*OPT?` query fails or its response is invalid.
    pub fn options(&mut self) -> Result<String> {
        self.query("*OPT?")
    }

    /// Queries the instrument identification string using `*IDN?`.
    ///
    /// # Errors
    ///
    /// Returns an error when the `*IDN?` query fails or its response is invalid.
    pub fn id(&mut self) -> Result<String> {
        self.query("*IDN?")
    }

    /// Clears the SCPI status and error queues using `*CLS`.
    ///
    /// # Errors
    ///
    /// Returns an error when the `*CLS` command cannot be written or flushed.
    pub fn clear_errors(&mut self) -> Result<()> {
        self.write("*CLS")
    }

    /// Queries the SCPI error queue and returns the reported error, if any.
    ///
    /// # Errors
    ///
    /// Returns an error when the `SYST:ERR?` query fails, its response cannot be
    /// parsed, or the instrument reports a non-zero SCPI error code.
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

    /// Writes scalar values in the selected [`ValuesFormat`].
    ///
    /// Binary transfers use an IEEE 488.2 definite-length block and
    /// little-endian IEEE-754 values.
    ///
    /// # Errors
    ///
    /// Returns an error when the formatted command or binary block cannot be
    /// written and flushed to the session.
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
                    .map(|value| {
                        value.to_f32().ok_or_else(|| {
                            Error::Unsupported(
                                "value cannot be represented in Binary32 format".to_owned(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
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

    /// Writes complex values as interleaved real and imaginary scalars.
    ///
    /// # Errors
    ///
    /// Returns an error when a component cannot be represented in the selected
    /// format or the interleaved values cannot be written to the session.
    pub fn write_complex_values(&mut self, command: &str, values: &[Complex64]) -> Result<()> {
        let interleaved = values
            .iter()
            .flat_map(|value| [value.re, value.im])
            .collect::<Vec<_>>();
        self.write_values(command, &interleaved)
    }

    /// Queries scalar values in the selected [`ValuesFormat`].
    ///
    /// # Errors
    ///
    /// Returns an error when the session query fails or the response cannot be
    /// parsed in the selected transfer format.
    pub fn query_values(&mut self, command: &str) -> Result<Vec<f64>> {
        let response = self.session.query(command)?;
        match self.values_format {
            ValuesFormat::Ascii => parse_ascii_values(&response),
            ValuesFormat::Binary32 => {
                parse_binary_values::<4>(&response, |bytes| f64::from(f32::from_le_bytes(bytes)))
            }
            ValuesFormat::Binary64 => parse_binary_values::<8>(&response, f64::from_le_bytes),
        }
    }

    /// Queries interleaved real and imaginary scalars as complex values.
    ///
    /// A response `[real(0), imag(0), real(1), imag(1), ...]` becomes
    /// `[complex(0), complex(1), ...]`.
    ///
    /// # Errors
    ///
    /// Returns an error when the scalar query fails or the response contains an
    /// odd number of scalar components.
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

/// Substitutes angle-bracket placeholders in a VNA command template.
///
/// A placeholder such as `<channel>` reads the `channel` key, while
/// `<self:number>` reads the `self:number` key. Missing parameters are reported
/// as parsing errors.
///
/// # Errors
///
/// Returns an error when the VNA command-template expression is invalid or a
/// referenced parameter is missing.
pub fn format_command(command: &str, parameters: &BTreeMap<String, String>) -> Result<String> {
    let expression = Regex::new(r"<(?:(?P<prefix>\w+):)?(?P<attribute>\w+)>")
        .map_err(|error| Error::Parse(format!("invalid VNA command regex: {error}")))?;
    let mut missing = None;
    let formatted = expression.replace_all(command, |captures: &regex::Captures<'_>| {
        let key = captures.name("prefix").map_or_else(
            || captures["attribute"].to_owned(),
            |prefix| format!("{}:{}", prefix.as_str(), &captures["attribute"]),
        );
        parameters.get(&key).map_or_else(
            || {
                missing = Some(key);
                captures[0].to_owned()
            },
            Clone::clone,
        )
    });
    missing.map_or_else(
        || Ok(formatted.into_owned()),
        |key| Err(Error::Parse(format!("missing VNA command parameter {key}"))),
    )
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
            chunk.try_into().map(&convert).map_err(|_| {
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
