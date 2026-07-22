#![allow(dead_code)]
#![cfg(feature = "visa")]
//! Rohde & Schwarz ZVA model-capability tests using a mock SCPI session.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use rust_rf::Result;
use rust_rf::vi::vna::rohde_schwarz::{RohdeSchwarzVna, RsFamily, Zna, Zva};
use rust_rf::vi::vna::{InstrumentSession, Vna};

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

/// Creates a shared Rohde & Schwarz driver backed by a mock session.
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
/// Verifies family-specific physical-port discovery for ZNA and ZVA models.
fn zna_and_zva_apply_model_specific_port_behavior() {
    // The upstream ZNA case queries the instrument for its physical port count.
    let port_count_query_session = ScpiMock {
        responses: BTreeMap::from([
            ("*IDN?".into(), b"Rohde&Schwarz,ZNA26-4Port,123,1".to_vec()),
            ("INST:PORT:COUN?".into(), b"4".to_vec()),
        ]),
        ..ScpiMock::default()
    };
    let mut zna = Zna::new("mock", port_count_query_session).unwrap();
    assert_eq!(zna.nports().unwrap(), 4);

    // The upstream ZVA case derives its physical port count from the model.
    let model_derived_port_session = ScpiMock {
        responses: BTreeMap::from([("*IDN?".into(), b"Rohde&Schwarz,ZVA40,123,1".to_vec())]),
        ..ScpiMock::default()
    };
    let mut zva = Zva::new("mock", model_derived_port_session).unwrap();
    assert_eq!(zva.nports().unwrap(), 2);
}
