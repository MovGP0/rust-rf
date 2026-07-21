//! NanoVNA V2 binary-protocol driver.
//!
//! Origin: `skrf/vi/vna/nanovna/nanovna.py`.

use ndarray::{Array2, Array3};
use num_complex::Complex64;

use crate::{Error, Frequency, FrequencyUnit, Network, Result, SweepType};

use super::super::{InstrumentSession, Vna};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Op {
    Nop = 0x00,
    Indicate = 0x0d,
    Read = 0x10,
    Read2 = 0x11,
    Read4 = 0x12,
    ReadFifo = 0x18,
    Write = 0x20,
    Write2 = 0x21,
    Write4 = 0x22,
    Write8 = 0x23,
    WriteFifo = 0x28,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RegisterAddress {
    SweepStart = 0x00,
    SweepStep = 0x10,
    SweepPoints = 0x20,
    ValuesPerFrequency = 0x22,
    RawSamplesMode = 0x26,
    ValuesFifo = 0x30,
    DeviceVariant = 0xf0,
    ProtocolVersion = 0xf1,
    HardwareRevision = 0xf2,
    FirmwareMajor = 0xf3,
    FirmwareMinor = 0xf4,
}

pub struct NanoVnaV2<S>
where
    S: InstrumentSession,
{
    pub vna: Vna<S>,
    frequency: Frequency,
}

impl<S> NanoVnaV2<S>
where
    S: InstrumentSession,
{
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

    pub fn reset_protocol(&mut self) -> Result<()> {
        self.vna.session.write_all(&[0; 8])?;
        self.vna.session.flush()?;
        Ok(())
    }

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
            command.push(bytes as u8);
        }
        command.extend_from_slice(&argument.to_le_bytes()[..bytes]);
        self.vna.session.write_all(&command)?;
        self.vna.session.flush()?;
        Ok(())
    }

    pub fn id(&mut self) -> Result<String> {
        Ok(self.query_register(Op::Read, RegisterAddress::DeviceVariant, 1)?[0].to_string())
    }

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

    pub fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    pub fn frequency_start(&self) -> f64 {
        self.frequency.start().unwrap_or(0.0)
    }

    pub fn frequency_stop(&self) -> f64 {
        self.frequency.stop().unwrap_or(0.0)
    }

    pub fn frequency_step(&self) -> f64 {
        self.frequency.step().unwrap_or(0.0)
    }

    pub fn points(&self) -> usize {
        self.frequency.points()
    }

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

    pub fn set_frequency_step(&mut self, step_hz: f64) -> Result<()> {
        if !step_hz.is_finite() || step_hz <= 0.0 {
            return Err(Error::InvalidFrequency(
                "NanoVNA frequency step must be positive".into(),
            ));
        }
        let points =
            ((self.frequency_stop() - self.frequency_start() + step_hz) / step_hz).round() as usize;
        self.set_points(points)
    }

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

    pub fn clear_fifo(&mut self) -> Result<()> {
        self.write_register(Op::Write, RegisterAddress::ValuesFifo, 1, 0)
    }

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
                i32::from_le_bytes([
                    chunk[offset],
                    chunk[offset + 1],
                    chunk[offset + 2],
                    chunk[offset + 3],
                ]) as f64
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

    pub fn get_s_data(&mut self, input_port: usize, output_port: usize) -> Result<Network> {
        if !matches!((input_port, output_port), (1 | 2, 1)) {
            return Err(Error::Unsupported(
                "NanoVNA V2 can only measure S11 and S21".into(),
            ));
        }
        let (s11, s21) = self.get_s11_s21()?;
        Ok(if input_port == 1 { s11 } else { s21 })
    }

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
    if !value.is_finite() || value < 0.0 || value > u64::MAX as f64 {
        return Err(Error::InvalidFrequency(format!(
            "NanoVNA {name} must be a non-negative integer"
        )));
    }
    Ok(value.round() as u64)
}
