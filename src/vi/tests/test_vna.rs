#![cfg(feature = "visa")]

//! Integration tests for the shared VNA session, command, channel, and value APIs.

use std::collections::BTreeMap;
use std::io::{Cursor, Read, Write};

use rust_rf::Result;
use rust_rf::vi::scpi_errors::ScpiError;
use rust_rf::vi::vna::{InstrumentSession, ValuesFormat, Vna, VnaError, format_command};

#[derive(Default)]
struct MockSession {
    read: Cursor<Vec<u8>>,
    written: Vec<u8>,
    responses: BTreeMap<String, Vec<u8>>,
    cleared: bool,
}

impl Read for MockSession {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.read.read(buffer)
    }
}

impl Write for MockSession {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.written.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl InstrumentSession for MockSession {
    fn clear(&mut self) -> Result<()> {
        self.cleared = true;
        Ok(())
    }

    fn query(&mut self, command: &str) -> Result<Vec<u8>> {
        Ok(self.responses.get(command).cloned().unwrap_or_default())
    }
}

#[test]
/// Substitutes channel and argument placeholders in VNA command templates.
fn formats_vna_command_placeholders() {
    for (command, parameters, expected) in [
        ("*IDN?", BTreeMap::new(), "*IDN?"),
        (
            "SENS<self:cnum>",
            BTreeMap::from([("self:cnum".into(), "1".into())]),
            "SENS1",
        ),
        (
            "SENS<self:cnum>:STAR <arg>",
            BTreeMap::from([
                ("self:cnum".into(), "1".into()),
                ("arg".into(), "100".into()),
            ]),
            "SENS1:STAR 100",
        ),
    ] {
        assert_eq!(format_command(command, &parameters).unwrap(), expected);
    }
}

#[test]
/// Creates, orders, activates, and deletes VNA channels.
fn creates_activates_and_deletes_channels() {
    let mut vna = Vna::new("mock", MockSession::default(), None);
    vna.create_channel(2, "channel 2").unwrap();
    vna.create_channel(1, "channel 1").unwrap();
    assert_eq!(vna.channels()[0].number, 1);
    assert_eq!(vna.active_channel().unwrap().number, 2);
    vna.set_active_channel(1).unwrap();
    assert_eq!(vna.active_channel().unwrap().name, "channel 1");
    assert_eq!(vna.delete_channel(1).unwrap().name, "channel 1");
    assert_eq!(vna.active_channel().unwrap().number, 2);
}

#[test]
/// Exercises standard identification, error-query, and error-clear commands.
fn implements_standard_scpi_commands_and_error_checking() {
    let session = MockSession {
        responses: BTreeMap::from([
            ("*IDN?".into(), b"rust-rf,mock\n".to_vec()),
            ("SYST:ERR?".into(), b"-222,data out of range\n".to_vec()),
        ]),
        ..MockSession::default()
    };
    let mut vna = Vna::new("mock", session, Some(1_000));
    assert_eq!(vna.id().unwrap(), "rust-rf,mock");
    assert!(matches!(
        vna.check_errors(),
        Err(VnaError::Scpi(ScpiError {
            error_code: -222,
            ..
        }))
    ));
    vna.clear_errors().unwrap();
    assert_eq!(vna.session.written, b"*CLS");
}

#[test]
/// Reads ASCII scalar and complex values and writes interleaved complex values.
fn reads_and_writes_ascii_and_complex_values() {
    let session = MockSession {
        responses: BTreeMap::from([
            ("DATA?".into(), b"1,2.5,-3".to_vec()),
            ("COMPLEX?".into(), b"1,2,3,4".to_vec()),
        ]),
        ..MockSession::default()
    };
    let mut vna = Vna::new("mock", session, None);
    assert_eq!(vna.query_values("DATA?").unwrap(), vec![1.0, 2.5, -3.0]);
    let complex = vna.query_complex_values("COMPLEX?").unwrap();
    assert_eq!(complex[0], rust_rf::Complex64::new(1.0, 2.0));
    vna.write_complex_values("DATA", &complex).unwrap();
    assert_eq!(vna.session.written, b"DATA 1,2,3,4");
}

#[test]
/// Decodes IEEE 488.2 definite-length binary floating-point blocks.
fn reads_scpi_binary_value_blocks() {
    let payload = [1.5_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat();
    let mut response = b"#18".to_vec();
    response.extend(payload);
    let session = MockSession {
        responses: BTreeMap::from([("DATA?".into(), response)]),
        ..MockSession::default()
    };
    let mut vna = Vna::new("mock", session, None);
    vna.values_format = ValuesFormat::Binary32;
    assert_eq!(vna.query_values("DATA?").unwrap(), vec![1.5, -2.0]);
}
