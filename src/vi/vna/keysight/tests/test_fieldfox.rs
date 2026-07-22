#![allow(dead_code)]
#![cfg(feature = "visa")]
//! `FieldFox` driver tests using a deterministic SCPI session.

use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};

use approx::assert_relative_eq;
use rust_rf::Result;
use rust_rf::vi::vna::keysight::{FieldFox, Pna, WindowFormat};
use rust_rf::vi::vna::{InstrumentSession, ValuesFormat, Vna};

#[derive(Default)]
/// In-memory SCPI session that records writes and returns queued responses.
struct ScpiMock {
    responses: BTreeMap<String, VecDeque<Vec<u8>>>,
    writes: Vec<u8>,
    clears: usize,
}

impl ScpiMock {
    /// Creates a mock with one text response for each command.
    fn with_text(responses: impl IntoIterator<Item = (&'static str, &'static str)>) -> Self {
        let mut session = Self::default();
        for (command, response) in responses {
            session
                .responses
                .entry(command.into())
                .or_default()
                .push_back(response.as_bytes().to_vec());
        }
        session
    }

    /// Queues another response for a command.
    fn push(&mut self, command: &str, response: impl Into<Vec<u8>>) {
        self.responses
            .entry(command.into())
            .or_default()
            .push_back(response.into());
    }
}

impl Read for ScpiMock {
    fn read(&mut self, _buffer: &mut [u8]) -> std::io::Result<usize> {
        Ok(0)
    }
}

impl Write for ScpiMock {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.writes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl InstrumentSession for ScpiMock {
    fn clear(&mut self) -> Result<()> {
        self.clears += 1;
        Ok(())
    }

    fn query(&mut self, command: &str) -> Result<Vec<u8>> {
        Ok(self
            .responses
            .get_mut(command)
            .and_then(VecDeque::pop_front)
            .unwrap_or_default())
    }
}

/// Creates a `FieldFox` backed by the supplied mock session.
fn field_fox(session: ScpiMock) -> FieldFox<ScpiMock> {
    FieldFox::from_vna(Vna::new("mock", session, None))
}

/// Creates a PNA backed by the supplied mock session.
fn make_pna(session: ScpiMock, model: &str) -> Result<Pna<ScpiMock>> {
    Pna::from_model(Vna::new("mock", session, None), model)
}

#[test]
/// Verifies typed `FieldFox` parameter queries and writes.
fn field_fox_queries_and_sets_typed_parameters() {
    let session = ScpiMock::with_text([
        ("SENS:FREQ:STAR?", "100"),
        ("SENS:SWE:POIN?", "11"),
        ("SENS:BWID?", "100"),
        ("DISP:WIND:SPL?", "D1"),
        ("CALC:PAR:COUN?", "1"),
    ]);
    let mut field_fox = field_fox(session);
    assert_eq!(field_fox.frequency_start().unwrap(), 100);
    assert_eq!(field_fox.points().unwrap(), 11);
    assert_eq!(field_fox.if_bandwidth().unwrap(), 100);
    assert_eq!(field_fox.window_format().unwrap(), WindowFormat::OneTrace);
    assert_eq!(field_fox.trace_count().unwrap(), 1);
    field_fox.set_frequency_start("100 kHz").unwrap();
    field_fox.set_if_bandwidth(1_000).unwrap();
    field_fox.set_window_format(WindowFormat::TwoByTwo).unwrap();
    field_fox.set_active_trace(1).unwrap();
    assert_eq!(
        String::from_utf8(field_fox.vna.session.writes).unwrap(),
        "SENS:FREQ:STAR 100000SENS:BWID 1000DISP:WIND:SPL D12_34CALC:PAR1:SEL"
    );
}

#[test]
/// Verifies frequency, transfer-format, measurement, and sweep commands.
fn field_fox_handles_frequency_formats_measurements_and_sweep() {
    let session = ScpiMock::with_text([
        ("SENS:FREQ:STAR?", "100"),
        ("SENS:FREQ:STOP?", "200"),
        ("SENS:SWE:POIN?", "11"),
        ("FORM?", "REAL,32"),
        ("CALC:PAR:COUN?", "1"),
        ("CALC:PAR1:DEF?", "S11"),
        ("INIT:CONT?", "1"),
    ]);
    let mut field_fox = field_fox(session);
    assert_eq!(field_fox.frequency().unwrap().points(), 11);
    assert_eq!(field_fox.query_format().unwrap(), ValuesFormat::Binary32);
    field_fox.set_query_format(ValuesFormat::Ascii).unwrap();
    field_fox.define_measurement(1, "S11").unwrap();
    assert_eq!(field_fox.measurement_parameter(1).unwrap(), "S11");
    field_fox.sweep().unwrap();
    assert_eq!(field_fox.vna.session.clears, 1);
    let writes = String::from_utf8(field_fox.vna.session.writes).unwrap();
    assert!(writes.contains("FORM ASC,0"));
    assert!(writes.contains("CALC:PAR1:DEF S11"));
    assert!(writes.contains("INIT:CONT 0INITINIT:CONT 1"));
}

#[test]
/// Verifies that four acquired S-parameters form a two-port network.
fn field_fox_assembles_a_two_port_network() {
    let mut session = ScpiMock::default();
    for _ in 0..4 {
        session.push("CALC:PAR:COUN?", b"4".to_vec());
    }
    session.push("SENS:FREQ:STAR?", b"100".to_vec());
    session.push("SENS:FREQ:STOP?", b"200".to_vec());
    session.push("SENS:SWE:POIN?", b"2".to_vec());
    session.push("INIT:CONT?", b"1".to_vec());
    for value in 1..=4 {
        session.push(
            "CALC:DATA:SDATA?",
            format!("{value},0,{value},0").into_bytes(),
        );
    }
    let mut field_fox = field_fox(session);
    let network = field_fox.get_snp_network(&[1, 2], false).unwrap();
    assert_eq!(network.s.dim(), (2, 2, 2));
    assert_relative_eq!(network.s[[0, 0, 0]].re, 1.0, epsilon = f64::EPSILON);
    assert_relative_eq!(network.s[[0, 0, 1]].re, 2.0, epsilon = f64::EPSILON);
    assert_relative_eq!(network.s[[0, 1, 0]].re, 3.0, epsilon = f64::EPSILON);
    assert_relative_eq!(network.s[[0, 1, 1]].re, 4.0, epsilon = f64::EPSILON);
}
