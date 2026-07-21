//! Hewlett-Packard VNA driver implementation.
//!
//! Origin: `skrf/vi/vna/hp/hp8510c.py`.

use ndarray::{Array1, Array2, Array3, Axis, concatenate};
use num_complex::Complex64;

use crate::{Error, Frequency, Network, Result};

use super::super::{InstrumentSession, Vna};
use super::hp8510c_sweep_plan::{Hp8510SweepProgrammer, SweepPlan};

const HP8510_NATIVE_POINTS: [usize; 5] = [51, 101, 201, 401, 801];
pub struct Hp8510C<S: InstrumentSession> {
    pub vna: Vna<S>,
    pub minimum_hz: Option<f64>,
    pub maximum_hz: Option<f64>,
    pub compound_sweep_plan: Option<SweepPlan>,
}

impl<S: InstrumentSession> Hp8510C<S> {
    pub fn new(address: impl Into<String>, session: S) -> Result<Self> {
        let mut driver = Self::from_vna(Vna::new(address, session, Some(2_000)));
        let identification = driver.id()?;
        if !identification.contains("HP8510") {
            return Err(Error::Unsupported(format!(
                "instrument identification is not an HP8510: {identification}"
            )));
        }
        driver.vna.timeout_ms = Some(60_000);
        driver.reset()?;
        driver.minimum_hz = Some(driver.frequency_start()?);
        driver.maximum_hz = Some(driver.frequency_stop()?);
        driver.vna.write("STEP;")?;
        Ok(driver)
    }

    pub fn from_vna(vna: Vna<S>) -> Self {
        Self {
            vna,
            minimum_hz: None,
            maximum_hz: None,
            compound_sweep_plan: None,
        }
    }

    pub fn id(&mut self) -> Result<String> {
        self.vna.query("OUTPIDEN;")
    }

    pub fn reset(&mut self) -> Result<()> {
        self.vna.write("FACTPRES;")?;
        self.wait_until_finished()
    }

    pub fn clear(&mut self) -> Result<()> {
        self.vna.clear()
    }

    pub fn wait_until_finished(&mut self) -> Result<()> {
        self.vna.query("OUTPIDEN;").map(|_| ())
    }

    pub fn error(&mut self) -> Result<String> {
        self.vna.query("OUTPERRO")
    }

    pub fn is_continuous(&mut self) -> Result<bool> {
        match self.vna.query("GROU?")?.as_str() {
            "\"HOLD\"" => Ok(false),
            "\"CONTINUAL\"" => Ok(true),
            value => Err(Error::Parse(format!(
                "unexpected HP8510 sweep state {value:?}"
            ))),
        }
    }

    pub fn set_continuous(&mut self, continuous: bool) -> Result<()> {
        self.vna.write(if continuous { "CONT;" } else { "SING;" })
    }

    pub fn set_averaging(&mut self, factor: usize) -> Result<()> {
        self.vna.write(&format!("AVERON {factor};"))
    }

    pub fn frequency_start(&mut self) -> Result<f64> {
        parse_float(&self.vna.query("STAR;OUTPACTI;")?)
    }

    pub fn set_frequency_start(&mut self, frequency_hz: f64) -> Result<()> {
        self.vna.write(&format!("STEP; STAR {frequency_hz};"))
    }

    pub fn frequency_stop(&mut self) -> Result<f64> {
        parse_float(&self.vna.query("STOP;OUTPACTI;")?)
    }

    pub fn set_frequency_stop(&mut self, frequency_hz: f64) -> Result<()> {
        self.vna.write(&format!("STEP; STOP {frequency_hz};"))
    }

    pub fn native_points(&mut self) -> Result<usize> {
        parse_float(&self.vna.query("POIN;OUTPACTI;")?).map(|value| value as usize)
    }

    pub fn points(&mut self) -> Result<usize> {
        match &self.compound_sweep_plan {
            Some(plan) => Ok(plan.frequencies_hz().len()),
            None => self.native_points(),
        }
    }

    pub fn set_points(&mut self, points: usize) -> Result<()> {
        let start = self.frequency_start()?;
        let stop = self.frequency_stop()?;
        self.set_frequency_step(start, stop, points)
    }

    pub fn native_frequency(&mut self) -> Result<Frequency> {
        Frequency::from_hz(Array1::from(linear_space(
            self.frequency_start()?,
            self.frequency_stop()?,
            self.native_points()?,
        )))
    }

    pub fn frequency(&mut self) -> Result<Frequency> {
        match &self.compound_sweep_plan {
            Some(plan) => Frequency::from_hz(Array1::from(plan.frequencies_hz())),
            None => self.native_frequency(),
        }
    }

    pub fn set_frequency(&mut self, frequency: &Frequency) -> Result<()> {
        let values = frequency
            .values_hz()
            .iter()
            .copied()
            .filter(|value| {
                self.minimum_hz.is_none_or(|minimum| *value >= minimum)
                    && self.maximum_hz.is_none_or(|maximum| *value <= maximum)
            })
            .collect::<Vec<_>>();
        self.compound_sweep_plan = Some(SweepPlan::from_hz(&values)?);
        Ok(())
    }

    pub fn set_frequency_single_point(&mut self, frequency_hz: f64, points: usize) -> Result<()> {
        if !HP8510_NATIVE_POINTS.contains(&points) {
            return Err(Error::Unsupported(format!(
                "HP8510 single-point sweep does not support {points} points"
            )));
        }
        self.vna.clear()?;
        self.vna
            .write(&format!("SINP; CWFREQ {frequency_hz};POIN{points};"))?;
        self.compound_sweep_plan = None;
        Ok(())
    }

    pub fn set_frequency_sweep(
        &mut self,
        start_hz: f64,
        stop_hz: f64,
        points: usize,
    ) -> Result<()> {
        self.set_frequency_step(start_hz, stop_hz, points)
    }

    pub fn set_frequency_step(&mut self, start_hz: f64, stop_hz: f64, points: usize) -> Result<()> {
        if Self::supports_native_step(points) {
            self.compound_sweep_plan = None;
            self.set_instrument_step_state(start_hz, stop_hz, points)
        } else {
            self.compound_sweep_plan = Some(SweepPlan::from_start_stop_points(
                start_hz, stop_hz, points,
            )?);
            Ok(())
        }
    }

    pub fn set_frequency_ramp(&mut self, start_hz: f64, stop_hz: f64, points: usize) -> Result<()> {
        if !HP8510_NATIVE_POINTS.contains(&points) {
            return Err(Error::Unsupported(format!(
                "HP8510 ramp sweep does not support {points} points"
            )));
        }
        self.vna.clear()?;
        self.vna.write(&format!(
            "RAMP; STAR {start_hz}; STOP {stop_hz}; POIN{points};"
        ))
    }

    pub fn supports_native_step(points: usize) -> bool {
        points > 0 && (HP8510_NATIVE_POINTS.contains(&points) || points <= 792)
    }

    pub fn set_instrument_step_state(
        &mut self,
        start_hz: f64,
        stop_hz: f64,
        points: usize,
    ) -> Result<()> {
        if !Self::supports_native_step(points) {
            return Err(Error::Unsupported(format!(
                "HP8510 cannot perform a native {points}-point sweep"
            )));
        }
        self.vna.clear()?;
        if HP8510_NATIVE_POINTS.contains(&points) {
            self.vna.write(&format!(
                "STEP; STAR {start_hz}; STOP {stop_hz}; POIN{points};"
            ))
        } else {
            for command in [
                "STEP;".to_owned(),
                "EDITLIST;".to_owned(),
                "CLEL;".to_owned(),
                "SADD;".to_owned(),
                format!("STAR {start_hz}; STOP {stop_hz}; POIN {points};"),
                "SDON; EDITDONE; LISFREQ;".to_owned(),
            ] {
                self.vna.write(&command)?;
            }
            Ok(())
        }
    }

    pub fn set_instrument_cw_step_state(&mut self, frequencies_hz: &[f64]) -> Result<()> {
        if frequencies_hz.len() > 30 {
            return Err(Error::Unsupported(
                "HP8510 CW list supports at most 30 points".into(),
            ));
        }
        for command in ["STEP;", "EDITLIST;", "CLEL;"] {
            self.vna.write(command)?;
        }
        for frequency in frequencies_hz {
            self.vna.write("SADD;")?;
            self.vna.write(&format!("CWFREQ {};", *frequency as u64))?;
        }
        self.vna.write("SDON; EDITDONE; LISFREQ;")
    }

    pub fn wait_for_status(&mut self) -> Result<(i32, i32)> {
        let status = self.vna.query("OUTPSTAT")?;
        let values = status
            .trim()
            .split(',')
            .map(|value| value.parse::<i32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| Error::Parse(format!("invalid HP8510 status: {error}")))?;
        match values.as_slice() {
            [first, second] => Ok((*first, *second)),
            _ => Err(Error::Parse(format!("invalid HP8510 status {status:?}"))),
        }
    }

    pub fn ask_for_complex(&mut self, output_command: &str) -> Result<Vec<Complex64>> {
        self.wait_for_status()?;
        self.vna.write("FORM2;")?;
        self.vna.write(output_command)?;
        parse_hp_binary(&self.vna.session.read_raw()?)
    }

    pub fn one_port_native(
        &mut self,
        continuous_wave: bool,
        expected_hz: Option<&[f64]>,
        fresh_sweep: bool,
    ) -> Result<Network> {
        if fresh_sweep {
            self.vna
                .write(if continuous_wave { "SINP;" } else { "SING;" })?;
        }
        let values = self.ask_for_complex("OUTPDATA")?;
        let frequency = match expected_hz {
            Some(values) => Frequency::from_hz(Array1::from(values.to_vec()))?,
            None => self.native_frequency()?,
        };
        one_port_network(frequency, values)
    }

    pub fn one_port(&mut self, continuous_wave: bool) -> Result<Network> {
        let Some(plan) = self.compound_sweep_plan.clone() else {
            return self.one_port_native(continuous_wave, None, true);
        };
        let old_start = self.frequency_start()?;
        let old_stop = self.frequency_stop()?;
        let mut result = None;
        for section in plan.sections() {
            section.apply(self)?;
            let raw_hz = section.raw_frequencies_hz();
            let chunk = self.one_port_native(continuous_wave, Some(&raw_hz), true)?;
            let chunk = section.mask_network(&chunk)?;
            result = Some(match result {
                None => chunk,
                Some(previous) => stitch(&previous, &chunk)?,
            });
        }
        self.set_frequency_start(old_start)?;
        self.set_frequency_stop(old_stop)?;
        result.ok_or_else(|| Error::InvalidFrequency("HP8510 sweep plan is empty".into()))
    }

    pub fn two_port_native(
        &mut self,
        continuous_wave: bool,
        expected_hz: Option<&[f64]>,
        fresh_sweep: bool,
    ) -> Result<Network> {
        let s11 = self.parameter_sweep("s11;", continuous_wave, expected_hz, fresh_sweep)?;
        let s12 = self.parameter_sweep("s12;", continuous_wave, expected_hz, fresh_sweep)?;
        let s22 = self.parameter_sweep("s22;", continuous_wave, expected_hz, fresh_sweep)?;
        let s21 = self.parameter_sweep("s21;", continuous_wave, expected_hz, fresh_sweep)?;
        assemble_two_port(s11, s12, s21, s22)
    }

    pub fn two_port(&mut self, continuous_wave: bool) -> Result<Network> {
        let Some(plan) = self.compound_sweep_plan.clone() else {
            return self.two_port_native(continuous_wave, None, true);
        };
        let old_start = self.frequency_start()?;
        let old_stop = self.frequency_stop()?;
        let mut result = None;
        for section in plan.sections() {
            section.apply(self)?;
            let raw_hz = section.raw_frequencies_hz();
            let chunk = self.two_port_native(continuous_wave, Some(&raw_hz), true)?;
            let chunk = section.mask_network(&chunk)?;
            result = Some(match result {
                None => chunk,
                Some(previous) => stitch(&previous, &chunk)?,
            });
        }
        self.set_frequency_start(old_start)?;
        self.set_frequency_stop(old_stop)?;
        result.ok_or_else(|| Error::InvalidFrequency("HP8510 sweep plan is empty".into()))
    }

    pub fn get_snp_network(&mut self, ports: &[usize], continuous_wave: bool) -> Result<Network> {
        match ports {
            [1] => {
                self.vna.write("s11;")?;
                self.one_port(continuous_wave)
            }
            [2] => {
                self.vna.write("s22;")?;
                self.one_port(continuous_wave)
            }
            [1, 2] | [2, 1] => self.two_port(continuous_wave),
            _ => Err(Error::Unsupported(format!(
                "invalid HP8510 ports {ports:?}; expected [1], [2], or [1, 2]"
            ))),
        }
    }

    pub fn switch_terms(&mut self) -> Result<(Network, Network)> {
        self.vna
            .write("USER2;DRIVPORT1;LOCKA1;NUMEB2;DENOA2;CONV1S;")?;
        let mut forward = self.one_port(false)?;
        forward.name = Some("forward switch term".into());
        self.vna
            .write("USER1;DRIVPORT2;LOCKA2;NUMEB1;DENOA1;CONV1S;")?;
        let mut reverse = self.one_port(false)?;
        reverse.name = Some("reverse switch term".into());
        Ok((forward, reverse))
    }

    fn parameter_sweep(
        &mut self,
        command: &str,
        continuous_wave: bool,
        expected_hz: Option<&[f64]>,
        fresh_sweep: bool,
    ) -> Result<Network> {
        self.vna.write(command)?;
        self.one_port_native(continuous_wave, expected_hz, fresh_sweep)
    }
}

impl<S: InstrumentSession> Hp8510SweepProgrammer for Hp8510C<S> {
    fn set_instrument_step_state(
        &mut self,
        start_hz: f64,
        stop_hz: f64,
        points: usize,
    ) -> Result<()> {
        Self::set_instrument_step_state(self, start_hz, stop_hz, points)
    }

    fn set_instrument_cw_step_state(&mut self, frequencies_hz: &[f64]) -> Result<()> {
        Self::set_instrument_cw_step_state(self, frequencies_hz)
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

fn stitch(left: &Network, right: &Network) -> Result<Network> {
    if left.ports() != right.ports() {
        return Err(Error::IncompatibleShape(
            "cannot stitch HP sweeps with different port counts".into(),
        ));
    }
    let frequency_values = concatenate(
        Axis(0),
        &[
            left.frequency.values_hz().view(),
            right.frequency.values_hz().view(),
        ],
    )
    .map_err(|error| Error::IncompatibleShape(error.to_string()))?;
    let s = concatenate(Axis(0), &[left.s.view(), right.s.view()])
        .map_err(|error| Error::IncompatibleShape(error.to_string()))?;
    let z0 = concatenate(Axis(0), &[left.z0.view(), right.z0.view()])
        .map_err(|error| Error::IncompatibleShape(error.to_string()))?;
    Network::new(Frequency::from_hz(frequency_values)?, s, z0)
}
