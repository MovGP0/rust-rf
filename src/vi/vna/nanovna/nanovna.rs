//! NanoVNA V2 binary-protocol driver.
//!
//! This driver connects to a NanoVNA V2 over USB using the
//! [NanoVNA V2 binary protocol](https://nanorfe.com/nanovna-v2-user-manual.html#__RefHeading___Toc2537_2953165397).
//! Devices that use the same protocol, such as the
//! [LiteVNA](https://www.zeenko.tech/litevna), may also be compatible. NanoVNA
//! variants that use a text protocol are not supported.
//!
//! > **Warning:** The device returns uncalibrated networks regardless of the
//! > calibration stored on the device. Apply calibration to the returned
//! > [`crate::Network`] in software.
//!
//! ## Example
//!
//! The native VISA transport must be available for this example.
//!
//! ```ignore
//! use rust_rf::{Frequency, FrequencyUnit, SweepType};
//! use rust_rf::vi::{NanoVnaV2, VisaSession};
//!
//! # fn main() -> rust_rf::Result<()> {
//! let address = "ASRL1::INSTR";
//! let session = VisaSession::open(address, None)?;
//! let mut vna = NanoVnaV2::new(address, session)?;
//! let frequency = Frequency::new(1.0, 2.0, 101, FrequencyUnit::GHz, SweepType::Linear)?;
//! vna.set_frequency(frequency)?;
//! let (s11, s21) = vna.get_s11_s21()?;
//! # let _ = (s11, s21);
//! # Ok(())
//! # }
//! ```

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use num_traits::ToPrimitive;

use crate::{Error, Frequency, FrequencyUnit, Network, Result, SweepType};

use super::super::{InstrumentSession, Vna};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
/// Operation byte in the `NanoVNA` V2 binary protocol.
pub enum Op {
    /// No operation.
    Nop = 0x00,
    /// Request a visual indication from the device.
    Indicate = 0x0d,
    /// Read one byte from a register.
    Read = 0x10,
    /// Read two bytes from a register.
    Read2 = 0x11,
    /// Read four bytes from a register.
    Read4 = 0x12,
    /// Read records from a FIFO register.
    ReadFifo = 0x18,
    /// Write one byte to a register.
    Write = 0x20,
    /// Write two bytes to a register.
    Write2 = 0x21,
    /// Write four bytes to a register.
    Write4 = 0x22,
    /// Write eight bytes to a register.
    Write8 = 0x23,
    /// Write records to a FIFO register.
    WriteFifo = 0x28,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
/// Register address in the `NanoVNA` V2 binary protocol.
pub enum RegisterAddress {
    /// Sweep start frequency.
    SweepStart = 0x00,
    /// Sweep frequency step.
    SweepStep = 0x10,
    /// Number of sweep points.
    SweepPoints = 0x20,
    /// Number of values returned per frequency.
    ValuesPerFrequency = 0x22,
    /// Raw-sample acquisition mode.
    RawSamplesMode = 0x26,
    /// FIFO containing measured values.
    ValuesFifo = 0x30,
    /// Device variant identifier.
    DeviceVariant = 0xf0,
    /// Binary-protocol version.
    ProtocolVersion = 0xf1,
    /// Hardware revision.
    HardwareRevision = 0xf2,
    /// Firmware major version.
    FirmwareMajor = 0xf3,
    /// Firmware minor version.
    FirmwareMinor = 0xf4,
}

/// `NanoVNA` V2 instrument using the USB binary protocol.
pub struct NanoVnaV2<S>
where
    S: InstrumentSession,
{
    /// Underlying instrument transport.
    pub vna: Vna<S>,
    frequency: Frequency,
}

impl<S> NanoVnaV2<S>
where
    S: InstrumentSession,
{
    /// Creates a driver, resets the binary protocol, and programs the default sweep.
    ///
    /// # Errors
    ///
    /// Returns an error when the default frequency axis is invalid or device
    /// communication and sweep programming fail.
    pub fn new(address: impl Into<String>, session: S) -> Result<Self> {
        let frequency = Frequency::new(1.0e6, 10.0e6, 201, FrequencyUnit::Hz, SweepType::Linear)?;
        let mut device = Self {
            vna: Vna::new(address, session, None),
            frequency: frequency.clone(),
        };
        device.reset_protocol()?;
        device.set_frequency(frequency)?;
        Ok(device)
    }

    /// Resets protocol framing by sending eight zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when writing or flushing the reset sequence fails.
    pub fn reset_protocol(&mut self) -> Result<()> {
        self.vna.session.write_all(&[0; 8])?;
        self.vna.session.flush()?;
        Ok(())
    }

    /// Executes a binary register read and returns exactly `bytes` response bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when writing, flushing, or reading the register transaction fails.
    pub fn query_register(
        &mut self,
        operation: Op,
        address: RegisterAddress,
        bytes: usize,
    ) -> Result<Vec<u8>> {
        self.vna
            .session
            .write_all(&[operation as u8, address as u8])?;
        self.vna.session.flush()?;
        let mut response = vec![0; bytes];
        self.vna.session.read_exact(&mut response)?;
        Ok(response)
    }

    /// Writes an integer argument to a binary-protocol register.
    ///
    /// The argument is encoded little-endian using `bytes` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported register width or when writing and
    /// flushing the transaction fails.
    pub fn write_register(
        &mut self,
        operation: Op,
        address: RegisterAddress,
        bytes: usize,
        argument: u64,
    ) -> Result<()> {
        if !(1..=8).contains(&bytes) {
            return Err(Error::Unsupported(format!(
                "NanoVNA register writes require 1 through 8 bytes, got {bytes}"
            )));
        }
        let mut command = vec![operation as u8, address as u8];
        if operation == Op::WriteFifo {
            let byte_count = u8::try_from(bytes).map_err(|_| {
                Error::Unsupported(format!("NanoVNA FIFO byte count is out of range: {bytes}"))
            })?;
            command.push(byte_count);
        }
        command.extend_from_slice(&argument.to_le_bytes()[..bytes]);
        self.vna.session.write_all(&command)?;
        self.vna.session.flush()?;
        Ok(())
    }

    /// Returns the device-variant identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when the device-variant register cannot be read.
    pub fn id(&mut self) -> Result<String> {
        Ok(self.query_register(Op::Read, RegisterAddress::DeviceVariant, 1)?[0].to_string())
    }

    /// Returns the device variant and protocol, hardware, and firmware versions.
    ///
    /// # Errors
    ///
    /// Returns an error when any device-information register cannot be read.
    pub fn device_info(&mut self) -> Result<String> {
        let variant = self.id()?;
        let protocol = self.query_register(Op::Read, RegisterAddress::ProtocolVersion, 1)?[0];
        let hardware = self.query_register(Op::Read, RegisterAddress::HardwareRevision, 1)?[0];
        let firmware_major = self.query_register(Op::Read, RegisterAddress::FirmwareMajor, 1)?[0];
        let firmware_minor = self.query_register(Op::Read, RegisterAddress::FirmwareMinor, 1)?[0];
        Ok(format!(
            "NanoVNAv2\n\tVariant:{variant}\n\tProtocol Version:{protocol}\n\tHardware Version: {hardware}\n\tFirmware Version: {firmware_major}.{firmware_minor}"
        ))
    }

    /// Returns the configured frequency axis.
    pub const fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    /// Returns the start frequency in hertz.
    pub fn frequency_start(&self) -> f64 {
        self.frequency.start().unwrap_or(0.0)
    }

    /// Returns the stop frequency in hertz.
    pub fn frequency_stop(&self) -> f64 {
        self.frequency.stop().unwrap_or(0.0)
    }

    /// Returns the frequency step in hertz.
    pub fn frequency_step(&self) -> f64 {
        self.frequency.step().unwrap_or(0.0)
    }

    /// Returns the number of sweep points.
    pub fn points(&self) -> usize {
        self.frequency.points()
    }

    /// Programs the sweep start, step, and point-count registers.
    ///
    /// # Errors
    ///
    /// Returns an error when frequencies cannot be encoded as non-negative integer
    /// hertz or when a register write fails.
    pub fn set_frequency(&mut self, frequency: Frequency) -> Result<()> {
        let start = nonnegative_integer(frequency.start().unwrap_or(0.0), "start frequency")?;
        let step = nonnegative_integer(frequency.step().unwrap_or(0.0), "frequency step")?;
        self.write_register(Op::Write8, RegisterAddress::SweepStart, 8, start)?;
        self.write_register(Op::Write8, RegisterAddress::SweepStep, 8, step)?;
        self.write_register(
            Op::Write2,
            RegisterAddress::SweepPoints,
            2,
            frequency.points() as u64,
        )?;
        self.frequency = frequency;
        Ok(())
    }

    /// Sets the start frequency in hertz while preserving stop and point count.
    ///
    /// # Errors
    ///
    /// Returns an error when the frequency axis is invalid or sweep programming fails.
    pub fn set_frequency_start(&mut self, start_hz: f64) -> Result<()> {
        let frequency = Frequency::new(
            start_hz,
            self.frequency_stop(),
            self.points(),
            FrequencyUnit::Hz,
            SweepType::Linear,
        )?;
        self.set_frequency(frequency)
    }

    /// Sets the stop frequency in hertz while preserving start and point count.
    ///
    /// # Errors
    ///
    /// Returns an error when the frequency axis is invalid or sweep programming fails.
    pub fn set_frequency_stop(&mut self, stop_hz: f64) -> Result<()> {
        let frequency = Frequency::new(
            self.frequency_start(),
            stop_hz,
            self.points(),
            FrequencyUnit::Hz,
            SweepType::Linear,
        )?;
        self.set_frequency(frequency)
    }

    /// Sets the frequency step by recalculating the number of sweep points.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-positive step, an invalid recalculated axis, or
    /// sweep-programming failure.
    pub fn set_frequency_step(&mut self, step_hz: f64) -> Result<()> {
        if !step_hz.is_finite() || step_hz <= 0.0 {
            return Err(Error::InvalidFrequency(
                "NanoVNA frequency step must be positive".into(),
            ));
        }
        // Preserve the NanoVNA point-count formula `(stop - start + step) / step`,
        // which includes both endpoints, while validating its integer conversion.
        let point_count =
            ((self.frequency_stop() - self.frequency_start() + step_hz) / step_hz).round();
        let points = point_count.to_usize().ok_or_else(|| {
            Error::InvalidFrequency(format!(
                "NanoVNA frequency step produces an invalid point count: {point_count}"
            ))
        })?;
        self.set_points(points)
    }

    /// Sets the number of sweep points while preserving start and stop.
    ///
    /// # Errors
    ///
    /// Returns an error when the frequency axis is invalid or sweep programming fails.
    pub fn set_points(&mut self, points: usize) -> Result<()> {
        let frequency = Frequency::new(
            self.frequency_start(),
            self.frequency_stop(),
            points,
            FrequencyUnit::Hz,
            SweepType::Linear,
        )?;
        self.set_frequency(frequency)
    }

    /// Clears all pending measurement records from the values FIFO.
    ///
    /// # Errors
    ///
    /// Returns an error when the values-FIFO clear command cannot be written.
    pub fn clear_fifo(&mut self) -> Result<()> {
        self.write_register(Op::Write, RegisterAddress::ValuesFifo, 1, 0)
    }

    /// Decodes 32-byte `NanoVNA` records into S11 and S21 samples.
    ///
    /// Each record contains the forward wave `a_1`, reflected wave `b_1`,
    /// transmitted wave `b_2`, and its frequency index. The returned values are
    /// calculated as `S_{11} = b_1/a_1` and `S_{21} = b_2/a_1`.
    ///
    /// # Errors
    ///
    /// Returns an error when the buffer is short, a forward wave is zero, or a
    /// record contains an out-of-range frequency index.
    pub fn convert_bytes_to_s_parameters(
        points: usize,
        raw: &[u8],
    ) -> Result<(Vec<Complex64>, Vec<Complex64>)> {
        if raw.len() < points * 32 {
            return Err(Error::Parse(format!(
                "NanoVNA returned {} bytes for {points} points; expected {}",
                raw.len(),
                points * 32
            )));
        }
        let mut s11 = vec![Complex64::new(0.0, 0.0); points];
        let mut s21 = s11.clone();
        for chunk in raw.chunks_exact(32).take(points) {
            let signed = |offset: usize| {
                f64::from(i32::from_le_bytes([
                    chunk[offset],
                    chunk[offset + 1],
                    chunk[offset + 2],
                    chunk[offset + 3],
                ]))
            };
            let forward = Complex64::new(signed(0), signed(4));
            if forward == Complex64::new(0.0, 0.0) {
                return Err(Error::Parse("NanoVNA forward wave is zero".into()));
            }
            let reverse_port_1 = Complex64::new(signed(8), signed(12));
            let reverse_port_2 = Complex64::new(signed(16), signed(20));
            let frequency_index = u16::from_le_bytes([chunk[24], chunk[25]]) as usize;
            if frequency_index >= points {
                return Err(Error::Parse(format!(
                    "NanoVNA frequency index {frequency_index} exceeds {points} points"
                )));
            }
            s11[frequency_index] = reverse_port_1 / forward;
            s21[frequency_index] = reverse_port_2 / forward;
        }
        Ok((s11, s21))
    }

    /// Acquires the uncalibrated S11 and S21 networks.
    ///
    /// FIFO reads are split into segments of at most 255 records, as required by
    /// the binary protocol.
    ///
    /// # Errors
    ///
    /// Returns an error when FIFO control or reads fail, returned records are
    /// malformed, or the one-port networks cannot be constructed.
    pub fn get_s11_s21(&mut self) -> Result<(Network, Network)> {
        let points = self.points();
        self.clear_fifo()?;
        let mut raw = Vec::with_capacity(points * 32);
        let mut remaining = points;
        while remaining > 0 {
            let segment = remaining.min(255);
            remaining -= segment;
            self.write_register(Op::ReadFifo, RegisterAddress::ValuesFifo, 1, segment as u64)?;
            let start = raw.len();
            raw.resize(start + segment * 32, 0);
            self.vna.session.read_exact(&mut raw[start..])?;
        }
        let (s11, s21) = Self::convert_bytes_to_s_parameters(points, &raw)?;
        Ok((self.one_port_network(s11)?, self.one_port_network(s21)?))
    }

    /// Acquires one measurable S-parameter as a one-port [`Network`].
    ///
    /// `input_port` may be 1 or 2 and `output_port` must be 1, corresponding to
    /// S11 or S21 respectively.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported port pair or when acquisition and
    /// network construction fail.
    pub fn get_s_data(&mut self, input_port: usize, output_port: usize) -> Result<Network> {
        if !matches!((input_port, output_port), (1 | 2, 1)) {
            return Err(Error::Unsupported(
                "NanoVNA V2 can only measure S11 and S21".into(),
            ));
        }
        let (s11, s21) = self.get_s11_s21()?;
        Ok(if input_port == 1 { s11 } else { s21 })
    }

    /// Acquires an uncalibrated network with a custom port mapping.
    ///
    /// Each entry may be port 1, port 2, or [`None`]. Only S11 and S21 are
    /// measurable; all other matrix entries remain zero.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid port, acquisition failure, or incompatible
    /// network shape.
    pub fn get_snp_network(&mut self, ports: &[Option<usize>]) -> Result<Network> {
        if let Some(&port) = ports.iter().flatten().find(|port| !matches!(port, 1 | 2)) {
            return Err(Error::InvalidPort { port, ports: 2 });
        }
        let (s11, s21) = self.get_s11_s21()?;
        let mut s = Array3::zeros((self.points(), ports.len(), ports.len()));
        for (output, output_port) in ports.iter().enumerate() {
            for (input, input_port) in ports.iter().enumerate() {
                let source = match (output_port, input_port) {
                    (Some(1), Some(1)) => Some(&s11.s),
                    (Some(2), Some(1)) => Some(&s21.s),
                    _ => None,
                };
                if let Some(source) = source {
                    for point in 0..self.points() {
                        s[[point, output, input]] = source[[point, 0, 0]];
                    }
                }
            }
        }
        Network::new(
            self.frequency.clone(),
            s,
            Array2::from_elem((self.points(), ports.len()), Complex64::new(50.0, 0.0)),
        )
    }

    fn one_port_network(&self, values: Vec<Complex64>) -> Result<Network> {
        let s = Array3::from_shape_vec((self.points(), 1, 1), values)
            .map_err(|error| Error::IncompatibleShape(error.to_string()))?;
        Network::new(
            self.frequency.clone(),
            s,
            Array2::from_elem((self.points(), 1), Complex64::new(50.0, 0.0)),
        )
    }
}

fn nonnegative_integer(value: f64, name: &str) -> Result<u64> {
    if !value.is_finite() || value < 0.0 {
        return Err(Error::InvalidFrequency(format!(
            "NanoVNA {name} must be a non-negative integer"
        )));
    }
    // NanoVNA frequency registers store integer hertz; preserve the original
    // round-to-nearest behavior while checking the target range.
    value.round().to_u64().ok_or_else(|| {
        Error::InvalidFrequency(format!("NanoVNA {name} is outside the u64 register range"))
    })
}
