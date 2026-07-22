#![allow(dead_code)]
#![cfg(feature = "visa")]

//! Mock-session integration tests for the HP 8510C driver.

use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};

use approx::assert_relative_eq;
use ndarray::Array1;
use rust_rf::vi::vna::hp::{Hp8510C, Hp8720B};
use rust_rf::vi::vna::{InstrumentSession, Vna};
use rust_rf::{Frequency, Result};

#[derive(Default)]
struct HpMock {
    responses: BTreeMap<String, VecDeque<Vec<u8>>>,
    raw: VecDeque<Vec<u8>>,
    writes: Vec<u8>,
    clears: usize,
}

impl HpMock {
    fn push_text(&mut self, command: &str, response: &str) {
        self.responses
            .entry(command.into())
            .or_default()
            .push_back(response.as_bytes().to_vec());
    }
}

impl Read for HpMock {
    fn read(&mut self, _buffer: &mut [u8]) -> std::io::Result<usize> {
        Ok(0)
    }
}

impl Write for HpMock {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.writes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl InstrumentSession for HpMock {
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

    fn read_raw(&mut self) -> Result<Vec<u8>> {
        Ok(self.raw.pop_front().unwrap_or_default())
    }
}

/// Wraps a mock session as an HP 8510C without hardware initialization.
fn hp8510(session: HpMock) -> Hp8510C<HpMock> {
    Hp8510C::from_vna(Vna::new("mock", session, None))
}

/// Wraps a mock session as an HP 8720B for shared HP test support.
fn hp8720(session: HpMock) -> Hp8720B<HpMock> {
    Hp8720B::from_vna(Vna::new("mock", session, None))
}

#[test]
/// Queries identification and frequency state and writes core sweep settings.
fn hp8510_queries_and_sets_core_parameters() {
    let mut session = HpMock::default();
    session.push_text("OUTPIDEN;", "HP8510C.07.14: Aug 26  1998");
    session.push_text("STAR;OUTPACTI;", "100");
    session.push_text("STOP;OUTPACTI;", "200");
    session.push_text("GROU?", "\"CONTINUAL\"");
    let mut hp = hp8510(session);
    assert_eq!(hp.id().unwrap(), "HP8510C.07.14: Aug 26  1998");
    assert_relative_eq!(hp.frequency_start().unwrap(), 100.0, epsilon = 1.0e-12);
    assert_relative_eq!(hp.frequency_stop().unwrap(), 200.0, epsilon = 1.0e-12);
    assert!(hp.is_continuous().unwrap());
    hp.set_frequency_start(100.0).unwrap();
    hp.set_frequency_stop(200.0).unwrap();
    hp.set_continuous(true).unwrap();
    assert_eq!(
        String::from_utf8(hp.vna.session.writes).unwrap(),
        "STEP; STAR 100;STEP; STOP 200;CONT;"
    );
}

#[test]
/// Resets the instrument and chooses native or compound sweep programming.
fn hp8510_resets_and_builds_native_or_compound_sweeps() {
    let mut session = HpMock::default();
    session.push_text("OUTPIDEN;", "HP8510C");
    let mut hp = hp8510(session);
    hp.reset().unwrap();
    hp.set_frequency_step(100.0, 1_000.0, 801).unwrap();
    assert!(hp.compound_sweep_plan.is_none());
    hp.set_frequency_step(100.0, 1_000.0, 802).unwrap();
    assert_eq!(
        hp.compound_sweep_plan
            .as_ref()
            .unwrap()
            .frequencies_hz()
            .len(),
        802
    );
    assert!(Hp8510C::<HpMock>::supports_native_step(792));
    assert!(!Hp8510C::<HpMock>::supports_native_step(793));
    assert_eq!(hp.vna.session.clears, 1);
    assert!(
        String::from_utf8(hp.vna.session.writes)
            .unwrap()
            .contains("FACTPRES;STEP; STAR 100; STOP 1000; POIN801;")
    );
}

#[test]
/// Uses a requested sweep plan as the public frequency and point-count state.
fn hp8510_frequency_property_uses_the_sweep_plan() {
    let frequency = Frequency::from_hz(Array1::linspace(100.0, 200.0, 51)).unwrap();
    let mut hp = hp8510(HpMock::default());
    hp.minimum_hz = Some(100.0);
    hp.maximum_hz = Some(200.0);
    hp.set_frequency(&frequency).unwrap();
    assert_eq!(hp.frequency().unwrap(), frequency);
    assert_eq!(hp.points().unwrap(), 51);
}

#[test]
/// Decodes FORM2 traces and assembles them into the correct two-port matrix.
fn hp8510_decodes_binary_data_and_assembles_two_ports() {
    let mut session = HpMock::default();
    for _ in 0..4 {
        session.push_text("OUTPSTAT", "0,1");
    }
    for value in [1.0, 2.0, 4.0, 3.0] {
        session
            .raw
            .push_back(hp_form2(&[(value, -value), (value + 0.5, -(value + 0.5))]));
    }
    let mut hp = hp8510(session);
    let network = hp
        .two_port_native(false, Some(&[100.0, 200.0]), true)
        .unwrap();
    assert_eq!(network.s.dim(), (2, 2, 2));
    assert_eq!(network.s[[0, 0, 0]], rust_rf::Complex64::new(1.0, -1.0));
    assert_eq!(network.s[[0, 0, 1]], rust_rf::Complex64::new(2.0, -2.0));
    assert_eq!(network.s[[0, 1, 0]], rust_rf::Complex64::new(3.0, -3.0));
    assert_eq!(network.s[[0, 1, 1]], rust_rf::Complex64::new(4.0, -4.0));
}

/// Encodes complex test samples in the HP FORM2 binary representation.
fn hp_form2(values: &[(f32, f32)]) -> Vec<u8> {
    let mut bytes = vec![0, 0, 0, 0];
    for (real, imaginary) in values {
        bytes.extend(real.to_be_bytes());
        bytes.extend(imaginary.to_be_bytes());
    }
    bytes
}
