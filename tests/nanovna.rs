#![cfg(feature = "visa")]

use std::collections::VecDeque;
use std::io::{Read, Write};

use rust_rf::Result;
use rust_rf::vi::vna::InstrumentSession;
use rust_rf::vi::vna::nanovna::{NanoVnaV2, Op, RegisterAddress};

#[derive(Default)]
struct SerialMock {
    reads: VecDeque<u8>,
    writes: Vec<u8>,
}

impl Read for SerialMock {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = buffer.len().min(self.reads.len());
        for target in &mut buffer[..count] {
            *target = self.reads.pop_front().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "NanoVNA scripted read queue was exhausted",
                )
            })?;
        }
        Ok(count)
    }
}

impl Write for SerialMock {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.writes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl InstrumentSession for SerialMock {
    fn clear(&mut self) -> Result<()> {
        Ok(())
    }

    fn query(&mut self, _command: &str) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
}

#[test]
fn initializes_and_encodes_little_endian_register_writes() {
    let mut device = NanoVnaV2::new("serial", SerialMock::default()).unwrap();
    assert_eq!(&device.vna.session.writes[..8], &[0; 8]);
    let previous = device.vna.session.writes.len();
    device
        .write_register(Op::Write4, RegisterAddress::SweepStep, 4, 0x1234_5678)
        .unwrap();
    assert_eq!(
        &device.vna.session.writes[previous..],
        &[0x22, 0x10, 0x78, 0x56, 0x34, 0x12]
    );
}

#[test]
fn decodes_s11_and_s21_samples_by_frequency_index() {
    let mut raw = vec![0_u8; 64];
    encode_sample(&mut raw[0..32], [(2, 0), (1, 0), (0, 1)], 1);
    encode_sample(&mut raw[32..64], [(4, 0), (0, 2), (8, 0)], 0);
    let (s11, s21) = NanoVnaV2::<SerialMock>::convert_bytes_to_s_parameters(2, &raw).unwrap();
    assert_eq!(s11[0], rust_rf::Complex64::new(0.0, 0.5));
    assert_eq!(s21[0], rust_rf::Complex64::new(2.0, 0.0));
    assert_eq!(s11[1], rust_rf::Complex64::new(0.5, 0.0));
    assert_eq!(s21[1], rust_rf::Complex64::new(0.0, 0.5));
}

#[test]
fn rejects_unsupported_s_parameters() {
    let mut device = NanoVnaV2::new("serial", SerialMock::default()).unwrap();
    assert!(device.get_s_data(1, 2).is_err());
    assert!(device.get_s_data(3, 1).is_err());
}

fn encode_sample(target: &mut [u8], waves: [(i32, i32); 3], index: u16) {
    let [
        (forward_real, forward_imaginary),
        (reverse_1_real, reverse_1_imaginary),
        (reverse_2_real, reverse_2_imaginary),
    ] = waves;
    target[0..4].copy_from_slice(&forward_real.to_le_bytes());
    target[4..8].copy_from_slice(&forward_imaginary.to_le_bytes());
    target[8..12].copy_from_slice(&reverse_1_real.to_le_bytes());
    target[12..16].copy_from_slice(&reverse_1_imaginary.to_le_bytes());
    target[16..20].copy_from_slice(&reverse_2_real.to_le_bytes());
    target[20..24].copy_from_slice(&reverse_2_imaginary.to_le_bytes());
    target[24..26].copy_from_slice(&index.to_le_bytes());
}
