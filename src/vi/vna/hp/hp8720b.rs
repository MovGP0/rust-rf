//! Hewlett-Packard VNA driver implementation.
//!
//! Origin: `skrf/vi/vna/hp/hp8720b.py`.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

use crate::{Error, Frequency, Network, Result};

use super::super::{InstrumentSession, Vna};
const HP8720_NATIVE_POINTS: [usize; 9] = [3, 11, 21, 51, 101, 201, 401, 801, 1_601];

pub struct Hp8720B<S: InstrumentSession> {
    pub vna: Vna<S>,
    pub minimum_hz: f64,
    pub maximum_hz: f64,
}

impl<S: InstrumentSession> Hp8720B<S> {
    pub fn new(address: impl Into<String>, session: S) -> Result<Self> {
        let mut driver = Self::from_vna(Vna::new(address, session, None));
        let bandwidth = driver.if_bandwidth()?;
        driver.vna.timeout_ms = Some((2_000.0 * (3_000.0 / bandwidth)) as u64);
        let identification = driver.id()?;
        if !identification.contains("8720") {
            return Err(Error::Unsupported(format!(
                "instrument identification is not an HP8720: {identification}"
            )));
        }
        driver.minimum_hz = driver.frequency_start()?;
        driver.maximum_hz = driver.frequency_stop()?;
        driver.vna.write("CONT;")?;
        driver.vna.write("DEBUON;")?;
        Ok(driver)
    }

    pub fn from_vna(vna: Vna<S>) -> Self {
        Self {
            vna,
            minimum_hz: 130.0e6,
            maximum_hz: 20.0e9,
        }
    }

    pub fn id(&mut self) -> Result<String> {
        self.vna.query("OUTPIDEN;")
    }

    pub fn reset(&mut self) -> Result<()> {
        self.vna.write("PRES;")?;
        self.wait_until_finished()
    }

    pub fn wait_until_finished(&mut self) -> Result<()> {
        self.vna.query("OUTPIDEN;").map(|_| ())
    }

    pub fn error(&mut self) -> Result<String> {
        self.vna.query("OUTPERRO")
    }

    pub fn if_bandwidth(&mut self) -> Result<f64> {
        parse_float(&self.vna.query("IFBW?")?)
    }

    pub fn set_if_bandwidth(&mut self, bandwidth_hz: usize) -> Result<()> {
        if ![3, 10, 30, 100, 300, 1_000, 3_000].contains(&bandwidth_hz) {
            return Err(Error::Unsupported(format!(
                "unsupported HP8720 IF bandwidth {bandwidth_hz}"
            )));
        }
        self.vna.write(&format!("IFBW {bandwidth_hz}"))?;
        self.vna.timeout_ms = Some((2_000.0 * (3_000.0 / bandwidth_hz as f64)) as u64);
        Ok(())
    }

    pub fn is_continuous(&mut self) -> Result<bool> {
        match self.vna.query("TRIG?")?.as_str() {
            "0" => Ok(true),
            "1" => Ok(false),
            value => Err(Error::Parse(format!(
                "unexpected HP8720 trigger state {value:?}"
            ))),
        }
    }

    pub fn set_continuous(&mut self, continuous: bool) -> Result<()> {
        self.vna.write(if continuous { "CONT;" } else { "SING;" })
    }

    pub fn averaging(&mut self) -> Result<usize> {
        parse_float(&self.vna.query("AVERFACT?")?).map(|value| value as usize)
    }

    pub fn set_averaging(&mut self, factor: Option<usize>) -> Result<()> {
        match factor {
            None | Some(0) => self.vna.write("AVEROFF"),
            Some(factor) => self.vna.write(&format!("AVERON; AVERFACT {factor};")),
        }
    }

    pub fn frequency_start(&mut self) -> Result<f64> {
        parse_float(&self.vna.query("STAR;OUTPACTI;")?)
    }

    pub fn set_frequency_start(&mut self, frequency_hz: f64) -> Result<()> {
        self.vna.write(&format!("STAR {frequency_hz};"))
    }

    pub fn frequency_stop(&mut self) -> Result<f64> {
        parse_float(&self.vna.query("STOP;OUTPACTI;")?)
    }

    pub fn set_frequency_stop(&mut self, frequency_hz: f64) -> Result<()> {
        self.vna.write(&format!("STOP {frequency_hz};"))
    }

    pub fn points(&mut self) -> Result<usize> {
        parse_float(&self.vna.query("POIN;OUTPACTI;")?).map(|value| value as usize)
    }

    pub fn frequency(&mut self) -> Result<Frequency> {
        Frequency::from_hz(Array1::from(linear_space(
            self.frequency_start()?,
            self.frequency_stop()?,
            self.points()?,
        )))
    }

    pub fn set_frequency_step(&mut self, start_hz: f64, stop_hz: f64, points: usize) -> Result<()> {
        if !Self::supports_native_step(points) {
            return Err(Error::Unsupported(format!(
                "HP8720 cannot perform a {points}-point sweep"
            )));
        }
        self.vna.clear()?;
        if HP8720_NATIVE_POINTS.contains(&points) {
            self.vna
                .write(&format!("STAR {start_hz}; STOP {stop_hz}; POIN {points};"))
        } else {
            for command in [
                "EDITLIST;".to_owned(),
                "SDEL;".to_owned(),
                "SADD;".to_owned(),
                format!("STAR {start_hz}; STOP {stop_hz}; POIN {points};"),
                "SDON; EDITDONE; LISFREQ;".to_owned(),
            ] {
                self.vna.write(&command)?;
            }
            Ok(())
        }
    }

    pub fn supports_native_step(points: usize) -> bool {
        points > 0 && points <= 1_601
    }

    pub fn set_instrument_cw_step_state(&mut self, frequencies_hz: &[f64]) -> Result<()> {
        if frequencies_hz.len() > 30 {
            return Err(Error::Unsupported(
                "HP8720 CW list supports at most 30 points".into(),
            ));
        }
        for command in ["EDITLIST;", "CLEL;"] {
            self.vna.write(command)?;
        }
        for frequency in frequencies_hz {
            self.vna.write("SADD;")?;
            self.vna.write(&format!("CWFREQ {};", *frequency as u64))?;
        }
        self.vna.write("SDON; EDITDONE; LISFREQ;")
    }

    pub fn ask_for_complex(&mut self, output_command: &str) -> Result<Vec<Complex64>> {
        self.vna.write("FORM2;")?;
        self.vna.write(output_command)?;
        parse_hp_binary(&self.vna.session.read_raw()?)
    }

    pub fn one_port_native(
        &mut self,
        expected_hz: Option<&[f64]>,
        fresh_sweep: bool,
    ) -> Result<Network> {
        let was_continuous = self.is_continuous()?;
        if fresh_sweep {
            self.vna.write("SING;")?;
        }
        let values = self.ask_for_complex("OUTPDATA")?;
        let frequency = match expected_hz {
            Some(values) => Frequency::from_hz(Array1::from(values.to_vec()))?,
            None => self.frequency()?,
        };
        self.set_continuous(was_continuous)?;
        one_port_network(frequency, values)
    }

    pub fn one_port(&mut self) -> Result<Network> {
        self.one_port_native(None, true)
    }

    pub fn two_port(&mut self) -> Result<Network> {
        let s11 = self.parameter_sweep("S11;")?;
        let s12 = self.parameter_sweep("S12;")?;
        let s22 = self.parameter_sweep("S22;")?;
        let s21 = self.parameter_sweep("S21;")?;
        assemble_two_port(s11, s12, s21, s22)
    }

    pub fn get_snp_network(&mut self, ports: &[usize]) -> Result<Network> {
        match ports {
            [1] => {
                self.vna.write("S11;")?;
                self.one_port()
            }
            [2] => {
                self.vna.write("S22;")?;
                self.one_port()
            }
            [1, 2] | [2, 1] => self.two_port(),
            _ => Err(Error::Unsupported(format!(
                "invalid HP8720 ports {ports:?}; expected [1], [2], or [1, 2]"
            ))),
        }
    }

    pub fn switch_terms(&mut self) -> Result<(Network, Network)> {
        self.vna
            .write("USER2;DRIVPORT1;LOCKA1;NUMEB2;DENOA2;CONV1S;")?;
        let mut forward = self.one_port()?;
        forward.name = Some("forward switch term".into());
        self.vna
            .write("USER1;DRIVPORT2;LOCKA2;NUMEB1;DENOA1;CONV1S;")?;
        let mut reverse = self.one_port()?;
        reverse.name = Some("reverse switch term".into());
        Ok((forward, reverse))
    }

    fn parameter_sweep(&mut self, command: &str) -> Result<Network> {
        self.vna.write(command)?;
        self.one_port_native(None, true)
    }
}

fn assemble_two_port(s11: Network, s12: Network, s21: Network, s22: Network) -> Result<Network> {
    if s11.frequency != s12.frequency
        || s11.frequency != s21.frequency
        || s11.frequency != s22.frequency
    {
        return Err(Error::InvalidFrequency(
            "HP two-port parameter sweeps have different frequencies".into(),
        ));
    }
    let points = s11.frequency_points();
    let s = Array3::from_shape_fn((points, 2, 2), |(point, output, input)| {
        match (output, input) {
            (0, 0) => s11.s[[point, 0, 0]],
            (0, 1) => s12.s[[point, 0, 0]],
            (1, 0) => s21.s[[point, 0, 0]],
            (1, 1) => s22.s[[point, 0, 0]],
            _ => unreachable!(),
        }
    });
    Network::new(
        s11.frequency,
        s,
        Array2::from_elem((points, 2), Complex64::new(50.0, 0.0)),
    )
}

fn linear_space(start: f64, stop: f64, points: usize) -> Vec<f64> {
    match points {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (points - 1) as f64;
            (0..points)
                .map(|index| start + index as f64 * step)
                .collect()
        }
    }
}

fn one_port_network(frequency: Frequency, values: Vec<Complex64>) -> Result<Network> {
    let points = frequency.points();
    if values.len() != points {
        return Err(Error::IncompatibleShape(format!(
            "HP trace has {} values for {points} frequency points",
            values.len()
        )));
    }
    Network::new(
        frequency,
        Array3::from_shape_vec((points, 1, 1), values)
            .map_err(|error| Error::IncompatibleShape(error.to_string()))?,
        Array2::from_elem((points, 1), Complex64::new(50.0, 0.0)),
    )
}

fn parse_float(value: &str) -> Result<f64> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|error| Error::Parse(format!("invalid HP numeric response {value:?}: {error}")))
}

fn parse_hp_binary(buffer: &[u8]) -> Result<Vec<Complex64>> {
    if buffer.len() < 4 {
        return Err(Error::Parse(
            "HP FORM2 response is shorter than its header".into(),
        ));
    }
    let mut payload = &buffer[4..];
    while !payload.chunks_exact(8).remainder().is_empty()
        && payload.last().is_some_and(u8::is_ascii_whitespace)
    {
        payload = &payload[..payload.len() - 1];
    }
    if !payload.chunks_exact(8).remainder().is_empty() {
        return Err(Error::Parse(format!(
            "HP FORM2 payload has {} bytes, expected complex f32 pairs",
            payload.len()
        )));
    }
    Ok(payload
        .chunks_exact(8)
        .map(|chunk| {
            Complex64::new(
                f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64,
                f32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]) as f64,
            )
        })
        .collect())
}
