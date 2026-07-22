#![allow(dead_code)]
#![cfg(feature = "visa")]

use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};

use approx::assert_relative_eq;
use rust_rf::Result;
use rust_rf::vi::vna::hp::{Hp8510C, Hp8720B};
use rust_rf::vi::vna::{InstrumentSession, Vna};

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

fn hp8510(session: HpMock) -> Hp8510C<HpMock> {
    Hp8510C::from_vna(Vna::new("mock", session, None))
}

fn hp8720(session: HpMock) -> Hp8720B<HpMock> {
    Hp8720B::from_vna(Vna::new("mock", session, None))
}

#[test]
fn hp8720_controls_bandwidth_averaging_and_sweep_modes() {
    let mut session = HpMock::default();
    session.push_text("IFBW?", "3000");
    session.push_text("AVERFACT?", "16");
    session.push_text("TRIG?", "0");
    let mut hp = hp8720(session);
    assert_relative_eq!(hp.if_bandwidth().unwrap(), 3_000.0, epsilon = 1.0e-12);
    assert_eq!(hp.averaging().unwrap(), 16);
    assert!(hp.is_continuous().unwrap());
    hp.set_if_bandwidth(100).unwrap();
    hp.set_averaging(Some(8)).unwrap();
    hp.set_continuous(false).unwrap();
    assert_eq!(hp.vna.timeout_ms, Some(60_000));
    assert_eq!(
        String::from_utf8(hp.vna.session.writes).unwrap(),
        "IFBW 100AVERON; AVERFACT 8;SING;"
    );
}

#[test]
fn hp8720_acquires_a_one_port_network() {
    let mut session = HpMock::default();
    session.push_text("TRIG?", "0");
    session.push_text("STAR;OUTPACTI;", "100");
    session.push_text("STOP;OUTPACTI;", "200");
    session.push_text("POIN;OUTPACTI;", "2");
    session.raw.push_back(hp_form2(&[(1.0, 2.0), (3.0, 4.0)]));
    let mut hp = hp8720(session);
    let network = hp.one_port().unwrap();
    assert_eq!(network.s.dim(), (2, 1, 1));
    assert_eq!(network.s[[1, 0, 0]], rust_rf::Complex64::new(3.0, 4.0));
    assert_eq!(hp.vna.session.clears, 0);
    assert!(
        String::from_utf8(hp.vna.session.writes)
            .unwrap()
            .contains("SING;FORM2;OUTPDATACONT;")
    );
}

fn hp_form2(values: &[(f32, f32)]) -> Vec<u8> {
    let mut bytes = vec![0, 0, 0, 0];
    for (real, imaginary) in values {
        bytes.extend(real.to_be_bytes());
        bytes.extend(imaginary.to_be_bytes());
    }
    bytes
}
