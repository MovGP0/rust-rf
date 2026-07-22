//! Hewlett-Packard 8720B vector network analyzer driver.
//!
//! The instrument natively supports 3, 11, 21, 51, 101, 201, 401, 801, and
//! 1601 points. Other counts up to 1601 use its frequency-segment mode. Complex
//! trace data is transferred efficiently with the HP FORM2 binary format.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use num_traits::ToPrimitive;

use crate::{Error, Frequency, Network, Result};

use super::super::{InstrumentSession, Vna};
const HP8720_NATIVE_POINTS: [usize; 9] = [3, 11, 21, 51, 101, 201, 401, 801, 1_601];

/// Driver for the HP 8720B vector network analyzer.
pub struct Hp8720B<S: InstrumentSession> {
    /// Shared VNA transport and value-transfer functionality.
    pub vna: Vna<S>,
    /// Minimum supported frequency in hertz.
    pub minimum_hz: f64,
    /// Maximum supported frequency in hertz.
    pub maximum_hz: f64,
}

impl<S: InstrumentSession> Hp8720B<S> {
    /// Creates and identifies an HP 8720B session and configures its timeout.
    ///
    /// # Errors
    ///
    /// Returns an error for communication failure or a non-8720 identification.
    pub fn new(address: impl Into<String>, session: S) -> Result<Self> {
        let mut driver = Self::from_vna(Vna::new(address, session, None));
        let bandwidth = driver.if_bandwidth()?;
        let timeout_ms = (2_000.0 * (3_000.0 / bandwidth))
            .to_u64()
            .ok_or_else(|| Error::Parse(format!("invalid HP8720 IF bandwidth {bandwidth}")))?;
        driver.vna.timeout_ms = Some(timeout_ms);
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

    /// Wraps an existing VNA session with default HP 8720B frequency limits.
    pub const fn from_vna(vna: Vna<S>) -> Self {
        Self {
            vna,
            minimum_hz: 130.0e6,
            maximum_hz: 20.0e9,
        }
    }

    /// Returns the instrument identification string.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument query fails.
    pub fn id(&mut self) -> Result<String> {
        self.vna.query("OUTPIDEN;")
    }

    /// Presets the instrument and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error if the preset command or completion query fails.
    pub fn reset(&mut self) -> Result<()> {
        self.vna.write("PRES;")?;
        self.wait_until_finished()
    }

    /// Blocks until the instrument responds to an identification query.
    ///
    /// # Errors
    ///
    /// Returns an error if the completion query fails.
    pub fn wait_until_finished(&mut self) -> Result<()> {
        self.vna.query("OUTPIDEN;").map(|_| ())
    }

    /// Returns the instrument error response from `OUTPERRO`.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument query fails.
    pub fn error(&mut self) -> Result<String> {
        self.vna.query("OUTPERRO")
    }

    /// Returns the current intermediate-frequency bandwidth in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails or the response is not numeric.
    pub fn if_bandwidth(&mut self) -> Result<f64> {
        parse_float(&self.vna.query("IFBW?")?)
    }

    /// Sets the intermediate-frequency bandwidth and adjusts the session timeout.
    ///
    /// Supported values are 3, 10, 30, 100, 300, 1000, and 3000 Hz.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported bandwidth or a failed instrument command.
    pub fn set_if_bandwidth(&mut self, bandwidth_hz: usize) -> Result<()> {
        let timeout_ms = match bandwidth_hz {
            3 => 2_000_000,
            10 => 600_000,
            30 => 200_000,
            100 => 60_000,
            300 => 20_000,
            1_000 => 6_000,
            3_000 => 2_000,
            _ => {
                return Err(Error::Unsupported(format!(
                    "unsupported HP8720 IF bandwidth {bandwidth_hz}"
                )));
            }
        };
        self.vna.write(&format!("IFBW {bandwidth_hz}"))?;
        self.vna.timeout_ms = Some(timeout_ms);
        Ok(())
    }

    /// Returns whether the instrument is in continuous sweep mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails or the trigger state is unexpected.
    pub fn is_continuous(&mut self) -> Result<bool> {
        match self.vna.query("TRIG?")?.as_str() {
            "0" => Ok(true),
            "1" => Ok(false),
            value => Err(Error::Parse(format!(
                "unexpected HP8720 trigger state {value:?}"
            ))),
        }
    }

    /// Selects continuous or single-sweep operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument command fails.
    pub fn set_continuous(&mut self, continuous: bool) -> Result<()> {
        self.vna.write(if continuous { "CONT;" } else { "SING;" })
    }

    /// Returns the current averaging factor.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails or the response is not numeric.
    pub fn averaging(&mut self) -> Result<usize> {
        let value = parse_float(&self.vna.query("AVERFACT?")?)?;
        value
            .to_usize()
            .ok_or_else(|| Error::Parse(format!("invalid HP8720 averaging factor {value}")))
    }

    /// Enables averaging with a factor, or disables it for `None` or zero.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument command fails.
    pub fn set_averaging(&mut self, factor: Option<usize>) -> Result<()> {
        match factor {
            None | Some(0) => self.vna.write("AVEROFF"),
            Some(factor) => self.vna.write(&format!("AVERON; AVERFACT {factor};")),
        }
    }

    /// Returns the sweep start frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails or the response is not numeric.
    pub fn frequency_start(&mut self) -> Result<f64> {
        parse_float(&self.vna.query("STAR;OUTPACTI;")?)
    }

    /// Sets the sweep start frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument command fails.
    pub fn set_frequency_start(&mut self, frequency_hz: f64) -> Result<()> {
        self.vna.write(&format!("STAR {frequency_hz};"))
    }

    /// Returns the sweep stop frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails or the response is not numeric.
    pub fn frequency_stop(&mut self) -> Result<f64> {
        parse_float(&self.vna.query("STOP;OUTPACTI;")?)
    }

    /// Sets the sweep stop frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error if the instrument command fails.
    pub fn set_frequency_stop(&mut self, frequency_hz: f64) -> Result<()> {
        self.vna.write(&format!("STOP {frequency_hz};"))
    }

    /// Returns the number of sweep points.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails or the response is not numeric.
    pub fn points(&mut self) -> Result<usize> {
        let value = parse_float(&self.vna.query("POIN;OUTPACTI;")?)?;
        value
            .to_usize()
            .ok_or_else(|| Error::Parse(format!("invalid HP8720 point count {value}")))
    }

    /// Returns the linearly spaced frequency axis of the current sweep.
    ///
    /// # Errors
    ///
    /// Returns an error if a frequency or point-count query fails or is invalid.
    pub fn frequency(&mut self) -> Result<Frequency> {
        Frequency::from_hz(Array1::from(linear_space(
            self.frequency_start()?,
            self.frequency_stop()?,
            self.points()?,
        )?))
    }

    /// Programs a synthesized step or segmented sweep.
    ///
    /// # Errors
    ///
    /// Returns an error for zero or more than 1601 points.
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

    /// Returns whether the requested point count can be programmed.
    #[must_use]
    pub const fn supports_native_step(points: usize) -> bool {
        points > 0 && points <= 1_601
    }

    /// Programs an arbitrary continuous-wave list of at most 30 frequencies.
    ///
    /// # Errors
    ///
    /// Returns an error for more than 30 frequencies, an invalid frequency, or a failed
    /// instrument command.
    pub fn set_instrument_cw_step_state(&mut self, frequencies_hz: &[f64]) -> Result<()> {
        if frequencies_hz.len() > 30 {
            return Err(Error::Unsupported(
                "HP8720 CW list supports at most 30 points".into(),
            ));
        }
        for command in ["EDITLIST;", "CLEL;"] {
            self.vna.write(command)?;
        }
        for &frequency_hz in frequencies_hz {
            let whole_hz = frequency_hz.to_u64().ok_or_else(|| {
                Error::InvalidFrequency(format!("invalid HP8720 CW frequency {frequency_hz}"))
            })?;
            self.vna.write("SADD;")?;
            self.vna.write(&format!("CWFREQ {whole_hz};"))?;
        }
        self.vna.write("SDON; EDITDONE; LISFREQ;")
    }

    /// Reads complex trace values using fast HP FORM2 binary transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if an instrument operation fails or the binary response is invalid.
    pub fn ask_for_complex(&mut self, output_command: &str) -> Result<Vec<Complex64>> {
        self.vna.write("FORM2;")?;
        self.vna.write(output_command)?;
        parse_hp_binary(&self.vna.session.read_raw()?)
    }

    /// Performs a native one-port sweep and restores the previous trigger mode.
    ///
    /// # Errors
    ///
    /// Returns an error if instrument I/O, frequency construction, or network construction fails.
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

    /// Performs a one-port sweep and returns the measured network.
    ///
    /// # Errors
    ///
    /// Returns an error if the sweep or network construction fails.
    pub fn one_port(&mut self) -> Result<Network> {
        self.one_port_native(None, true)
    }

    /// Performs four parameter sweeps and assembles a two-port network.
    ///
    /// # Errors
    ///
    /// Returns an error if a sweep fails or the parameter networks are incompatible.
    pub fn two_port(&mut self) -> Result<Network> {
        let s11 = self.parameter_sweep("S11;")?;
        let s12 = self.parameter_sweep("S12;")?;
        let s22 = self.parameter_sweep("S22;")?;
        let s21 = self.parameter_sweep("S21;")?;
        assemble_two_port(s11, &s12, &s21, &s22)
    }

    /// Obtains S-parameter data for port 1, port 2, or both ports.
    ///
    /// # Errors
    ///
    /// Returns an error unless `ports` is `[1]`, `[2]`, `[1, 2]`, or `[2, 1]`.
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

    /// Measures forward and reverse switch terms with a thru connected.
    ///
    /// The returned networks characterize reflection from the imperfect switched
    /// termination on the non-stimulated port.
    ///
    /// # Errors
    ///
    /// Returns an error if an instrument command, sweep, or network construction fails.
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

fn assemble_two_port(s11: Network, s12: &Network, s21: &Network, s22: &Network) -> Result<Network> {
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

fn linear_space(start: f64, stop: f64, points: usize) -> Result<Vec<f64>> {
    match points {
        0 => Ok(Vec::new()),
        1 => Ok(vec![start]),
        _ => {
            let interval_count = (points - 1).to_f64().ok_or_else(|| {
                Error::InvalidFrequency(format!("HP8720 point count {points} is too large"))
            })?;
            let increment = (stop - start) / interval_count;
            (0..points)
                .map(|index| {
                    let floating_index = index.to_f64().ok_or_else(|| {
                        Error::InvalidFrequency(format!("HP8720 point index {index} is too large"))
                    })?;
                    Ok(floating_index.mul_add(increment, start))
                })
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
                f64::from(f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
                f64::from(f32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]])),
            )
        })
        .collect())
}
