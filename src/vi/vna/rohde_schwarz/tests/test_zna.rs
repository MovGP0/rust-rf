#![allow(dead_code)]
#![cfg(feature = "visa")]
//! Rohde & Schwarz ZNA command tests using a deterministic SCPI session.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use rust_rf::Result;
use rust_rf::vi::vna::rohde_schwarz::{RohdeSchwarzVna, RsFamily, SweepMode, SweepType};
use rust_rf::vi::vna::{InstrumentSession, ValuesFormat, Vna};

#[derive(Default)]
/// In-memory SCPI session that records commands and returns canned responses.
struct ScpiMock {
    responses: BTreeMap<String, Vec<u8>>,
    writes: Vec<u8>,
    cleared: bool,
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
        self.cleared = true;
        Ok(())
    }

    fn query(&mut self, command: &str) -> Result<Vec<u8>> {
        Ok(self.responses.get(command).cloned().unwrap_or_default())
    }
}

/// Creates a ZNA-family driver backed by a mock SCPI session.
fn driver(
    responses: impl IntoIterator<Item = (&'static str, &'static str)>,
) -> Result<RohdeSchwarzVna<ScpiMock>> {
    let session = ScpiMock {
        responses: responses
            .into_iter()
            .map(|(command, response)| (command.to_owned(), response.as_bytes().to_vec()))
            .collect(),
        ..ScpiMock::default()
    };
    RohdeSchwarzVna::from_model(
        Vna::new("mock", session, None),
        RsFamily::Zna,
        "ZNA26-2Port",
    )
}

#[test]
/// Verifies typed channel property queries and writes.
fn queries_and_sets_typed_channel_parameters() -> Result<()> {
    let mut driver = driver([
        ("SENS1:FREQ:STAR?", "100"),
        ("SENS1:SWE:POIN?", "11"),
        ("SENS1:SWE:TYPE?", "LIN"),
        ("INIT1:CONT?", "0"),
    ])?;
    {
        let mut channel = driver.channel(1).unwrap();
        assert_eq!(channel.frequency_start().unwrap(), 100);
        assert_eq!(channel.points().unwrap(), 11);
        assert_eq!(channel.sweep_type().unwrap(), SweepType::Linear);
        assert_eq!(channel.sweep_mode().unwrap(), SweepMode::Single);
        channel.set_frequency_start("100 kHz").unwrap();
        channel.set_points(101).unwrap();
        channel.set_sweep_type(SweepType::Log).unwrap();
    }
    assert_eq!(
        String::from_utf8(driver.vna.session.writes).unwrap(),
        "SENS1:FREQ:STAR 100000SENS1:SWE:POIN 101SENS1:SWE:TYPE LOG"
    );
    Ok(())
}

#[test]
/// Verifies measurement creation, deletion, enumeration, and averaging reset.
fn manages_measurements_and_averaging_commands() -> Result<()> {
    let mut driver = driver([("CALC1:PAR:CAT?", "CH1_S11_1,S11,CH1_S12_1,S12")])?;
    {
        let mut channel = driver.channel(1).unwrap();
        assert_eq!(
            channel.measurements().unwrap(),
            vec![
                ("CH1_S11_1".into(), "S11".into()),
                ("CH1_S12_1".into(), "S12".into())
            ]
        );
        channel.clear_averaging().unwrap();
        channel.create_measurement("TRACE", "S21").unwrap();
        channel.delete_measurement("TRACE").unwrap();
    }
    assert_eq!(
        String::from_utf8(driver.vna.session.writes).unwrap(),
        "SENS1:AVER:CLECALC1:PAR:SDEF 'TRACE','S21'DISP:TRAC:EFE 'TRACE'CALC1:PAR:DEL 'TRACE'"
    );
    Ok(())
}

#[test]
/// Verifies ASCII and binary transfer-format queries and writes.
fn queries_and_sets_data_formats() -> Result<()> {
    let mut driver = driver([("FORM?", "REAL,64")])?;
    assert_eq!(driver.query_format().unwrap(), ValuesFormat::Binary64);
    driver.set_query_format(ValuesFormat::Binary32).unwrap();
    assert_eq!(driver.vna.values_format, ValuesFormat::Binary32);
    assert_eq!(
        String::from_utf8(driver.vna.session.writes).unwrap(),
        "FORM:BORD SWAPFORM REAL,32"
    );
    Ok(())
}
