//! Native VISA transport for vector network analyzers.
//!
//! This is the Rust integration counterpart of the PyVISA resource-manager
//! setup used by `skrf/vi/vna/vna.py`.

use std::ffi::CString;
use std::io::{Read, Write};
use std::time::Duration;

use visa_rs::prelude::{AccessMode, AsResourceManager, DefaultRM};
use visa_rs::{Instrument, VisaString};

use crate::vi::vna::{InstrumentSession, Vna};
use crate::{Error, Result};

const DEFAULT_TIMEOUT_MS: u64 = 2_000;
const READ_BUFFER_SIZE: usize = 64 * 1024;

/// A native VISA resource and its owning resource-manager session.
///
/// Keeping the resource manager alive is required because dropping it closes
/// every instrument session opened through it.
#[derive(Debug)]
pub struct VisaSession {
    resource_manager: DefaultRM,
    instrument: Instrument,
}

impl VisaSession {
    /// Opens a native VISA instrument session.
    ///
    /// # Errors
    ///
    /// Returns an error when the address contains an interior NUL byte, the
    /// VISA resource manager cannot be created, or the instrument cannot be
    /// opened.
    pub fn open(address: &str, timeout_ms: Option<u64>) -> Result<Self> {
        let resource = CString::new(address)
            .map(VisaString::from)
            .map_err(|error| Error::Parse(format!("invalid VISA resource address: {error}")))?;
        let resource_manager = DefaultRM::new().map_err(map_visa_error)?;
        let timeout = Duration::from_millis(timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS));
        let instrument = resource_manager
            .open(&resource, AccessMode::NO_LOCK, timeout)
            .map_err(map_visa_error)?;

        Ok(Self {
            resource_manager,
            instrument,
        })
    }

    /// Returns the native VISA instrument handle.
    #[must_use]
    pub const fn instrument(&self) -> &Instrument {
        &self.instrument
    }

    /// Returns the owning VISA resource manager.
    #[must_use]
    pub const fn resource_manager(&self) -> &DefaultRM {
        &self.resource_manager
    }

    fn read_message(&mut self) -> Result<Vec<u8>> {
        let mut response = Vec::new();
        let mut buffer = vec![0_u8; READ_BUFFER_SIZE];

        loop {
            let count = self.instrument.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            response.extend_from_slice(&buffer[..count]);

            if count < buffer.len() || response.last() == Some(&b'\n') {
                break;
            }
        }

        Ok(response)
    }
}

impl Read for VisaSession {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.instrument.read(buffer)
    }
}

impl Write for VisaSession {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.instrument.write(buffer)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.instrument.flush()
    }
}

impl InstrumentSession for VisaSession {
    fn clear(&mut self) -> Result<()> {
        self.instrument.clear().map_err(map_visa_error)
    }

    fn query(&mut self, command: &str) -> Result<Vec<u8>> {
        self.instrument.write_all(command.as_bytes())?;
        if !command.ends_with('\n') {
            self.instrument.write_all(b"\n")?;
        }
        self.instrument.flush()?;
        self.read_message()
    }

    fn read_raw(&mut self) -> Result<Vec<u8>> {
        self.read_message()
    }
}

impl Vna<VisaSession> {
    /// Opens a transport-backed VNA directly from a VISA resource address.
    ///
    /// # Errors
    ///
    /// Returns an error when the native VISA resource manager or instrument
    /// session cannot be opened.
    pub fn open_visa(address: impl Into<String>, timeout_ms: Option<u64>) -> Result<Self> {
        let address = address.into();
        let session = VisaSession::open(&address, timeout_ms)?;
        Ok(Self::new(address, session, timeout_ms))
    }
}

fn map_visa_error(error: visa_rs::Error) -> Error {
    Error::Io(std::io::Error::other(error))
}
