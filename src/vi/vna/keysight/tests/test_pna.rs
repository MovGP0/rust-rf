#![allow(dead_code)]
#![cfg(feature = "visa")]
//! Keysight PNA driver tests using a deterministic SCPI session.

use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};

use rust_rf::Result;
use rust_rf::vi::vna::keysight::{FieldFox, Pna, PnaSweepMode, PnaSweepType, TriggerSource};
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

/// Creates a PNA backed by the supplied mock session and model identifier.
fn make_pna(session: ScpiMock, model: &str) -> rust_rf::Result<Pna<ScpiMock>> {
    Pna::from_model(Vna::new("mock", session, None), model)
}

#[test]
/// Verifies typed queries and writes for PNA channel properties.
fn pna_queries_and_sets_channel_parameters() -> rust_rf::Result<()> {
    let session = ScpiMock::with_text([
        ("SENS1:FREQ:STAR?", "100"),
        ("SENS1:SWE:POIN?", "11"),
        ("SENS1:SWE:TYPE?", "LIN"),
        ("SENS1:SWE:MODE?", "SING"),
        ("SYST:MEAS:CAT? 1", "1,2,3"),
    ]);
    let mut pna = make_pna(session, "N5227B")?;
    {
        let mut channel = pna.channel(1).unwrap();
        assert_eq!(channel.frequency_start().unwrap(), 100);
        assert_eq!(channel.points().unwrap(), 11);
        assert_eq!(channel.sweep_type().unwrap(), PnaSweepType::Linear);
        assert_eq!(channel.sweep_mode().unwrap(), PnaSweepMode::Single);
        assert_eq!(channel.measurement_numbers().unwrap(), vec![1, 2, 3]);
        channel.set_frequency_start("1 MHz").unwrap();
        channel.set_sweep_type(PnaSweepType::Log).unwrap();
    }
    assert_eq!(
        String::from_utf8(pna.vna.session.writes).unwrap(),
        "SENS1:FREQ:STAR 1000000SENS1:SWE:TYPE LOG"
    );
    Ok(())
}

#[test]
/// Verifies transfer formats, measurements, triggers, and model capabilities.
fn pna_manages_formats_measurements_and_model_capabilities() -> rust_rf::Result<()> {
    let session = ScpiMock::with_text([
        ("FORM?", "REAL,+64"),
        ("CALC1:PAR:CAT:EXT?", "CH1_S11_1,S11,CH1_S12_1,S12"),
        ("DISP:WIND:CAT?", "1"),
        ("SYST:CAP:HARD:PORT:COUN?", "4"),
    ]);
    let mut pna = make_pna(session, "N5227B")?;
    assert_eq!(pna.query_format().unwrap(), ValuesFormat::Binary64);
    assert_eq!(pna.ports().unwrap(), 4);
    assert_eq!(
        pna.measurement_names(1).unwrap(),
        vec!["CH1_S11_1", "CH1_S12_1"]
    );
    pna.create_measurement(1, "TRACE", "S21").unwrap();
    pna.delete_measurement(1, "TRACE").unwrap();
    pna.set_trigger_source(TriggerSource::Immediate).unwrap();
    let writes = String::from_utf8(pna.vna.session.writes).unwrap();
    assert!(writes.contains("CALC1:PAR:EXT 'TRACE',S21"));
    assert!(writes.contains("DISP:WIND:TRAC2:FEED 'TRACE'"));
    assert!(writes.contains("CALC1:PAR:DEL 'TRACE'"));
    assert!(writes.contains("TRIG:SOUR IMM"));

    let mut legacy = make_pna(ScpiMock::default(), "E8362C")?;
    assert_eq!(legacy.ports().unwrap(), 2);
    Ok(())
}

#[test]
/// Verifies active-trace acquisition and restoration of instrument state.
fn pna_acquires_an_active_trace_network_and_restores_state() -> rust_rf::Result<()> {
    let mut session = ScpiMock::default();
    session.push("SENS1:SWE:MODE?", b"SING".to_vec());
    session.push("SENS1:SWE:TIME?", b"1.0".to_vec());
    session.push("SENS1:AVER:STATE?", b"0".to_vec());
    session.push("SENS1:AVER:MODE?", b"POIN".to_vec());
    session.push("*OPC?", b"1".to_vec());
    session.push("SENS1:FREQ:STAR?", b"100".to_vec());
    session.push("SENS1:FREQ:STOP?", b"200".to_vec());
    session.push("SENS1:SWE:POIN?", b"11".to_vec());
    session.push("CALC1:PAR:SEL?", b"TRACE".to_vec());
    session.push(
        "CALC1:DATA? SDATA",
        binary_f64_block(&(0..11).flat_map(|_| [1.0, -1.0]).collect::<Vec<_>>()),
    );
    let mut pna = make_pna(session, "E8362C")?;
    let network = pna.channel(1).unwrap().get_active_trace().unwrap();
    assert_eq!(network.s.dim(), (11, 1, 1));
    assert_eq!(network.s[[0, 0, 0]], rust_rf::Complex64::new(1.0, -1.0));
    assert_eq!(pna.vna.values_format, ValuesFormat::Ascii);
    assert_eq!(pna.vna.session.clears, 2);
    Ok(())
}

/// Encodes floating-point values as an IEEE 488.2 binary block.
fn binary_f64_block(values: &[f64]) -> Vec<u8> {
    let payload = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let length = payload.len().to_string();
    let mut block = format!("#{}{length}", length.len()).into_bytes();
    block.extend(payload);
    block
}
