//! Hewlett-Packard 8510C vector network analyzer driver.
//!
//! The driver supports compound and segmented sweeps plus fast FORM2 binary
//! transfer. Requests exceeding native 51-, 101-, 201-, 401-, or 801-point
//! sweeps are automatically decomposed into shorter sweeps and stitched back
//! together. Short or irregular frequency lists use segmented sweep modes.

use ndarray::{Array1, Array2, Array3, Axis, concatenate};
use num_complex::Complex64;
use num_traits::ToPrimitive;

use crate::{Error, Frequency, Network, Result};

use super::super::{InstrumentSession, Vna};
use super::hp8510c_sweep_plan::{Hp8510SweepProgrammer, SweepPlan};

const HP8510_NATIVE_POINTS: [usize; 5] = [51, 101, 201, 401, 801];
/// Driver for the HP 8510C and compatible variants.
pub struct Hp8510C<S: InstrumentSession> {
    /// Shared VNA transport and value-transfer functionality.
    pub vna: Vna<S>,
    /// Optional lower instrument frequency limit in hertz.
    pub minimum_hz: Option<f64>,
    /// Optional upper instrument frequency limit in hertz.
    pub maximum_hz: Option<f64>,
    /// Planned sections for a compound or segmented sweep.
    pub compound_sweep_plan: Option<SweepPlan>,
}

impl<S: InstrumentSession> Hp8510C<S> {
    /// Creates, identifies, resets, and configures an HP 8510C session.
    ///
    /// # Errors
    ///
    /// Returns an error when communication fails or the identification response
    /// does not describe an HP 8510 instrument.
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

    /// Wraps an existing VNA session without querying or resetting the instrument.
    pub const fn from_vna(vna: Vna<S>) -> Self {
        Self {
            vna,
            minimum_hz: None,
            maximum_hz: None,
            compound_sweep_plan: None,
        }
    }

    /// Returns the instrument identification string.
    ///
    /// # Errors
    ///
    /// Returns an error when the identification query cannot be completed.
    pub fn id(&mut self) -> Result<String> {
        self.vna.query("OUTPIDEN;")
    }

    /// Presets the instrument and waits for it to finish.
    ///
    /// # Errors
    ///
    /// Returns an error when the preset command or completion query fails.
    pub fn reset(&mut self) -> Result<()> {
        self.vna.write("FACTPRES;")?;
        self.wait_until_finished()
    }

    /// Clears the instrument session.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying session cannot be cleared.
    pub fn clear(&mut self) -> Result<()> {
        self.vna.clear()
    }

    /// Blocks until the instrument accepts a completed identification query.
    ///
    /// # Errors
    ///
    /// Returns an error when the completion query fails.
    pub fn wait_until_finished(&mut self) -> Result<()> {
        self.vna.query("OUTPIDEN;").map(|_| ())
    }

    /// Returns the instrument error response from `OUTPERRO`.
    ///
    /// # Errors
    ///
    /// Returns an error when the instrument error query fails.
    pub fn error(&mut self) -> Result<String> {
        self.vna.query("OUTPERRO")
    }

    /// Returns whether the sweep mode is continuous.
    ///
    /// # Errors
    ///
    /// Returns an error for communication failure or an unknown sweep-state response.
    pub fn is_continuous(&mut self) -> Result<bool> {
        match self.vna.query("GROU?")?.as_str() {
            "\"HOLD\"" => Ok(false),
            "\"CONTINUAL\"" => Ok(true),
            value => Err(Error::Parse(format!(
                "unexpected HP8510 sweep state {value:?}"
            ))),
        }
    }

    /// Selects continuous or single-sweep operation.
    ///
    /// # Errors
    ///
    /// Returns an error when the sweep-mode command cannot be written.
    pub fn set_continuous(&mut self, continuous: bool) -> Result<()> {
        self.vna.write(if continuous { "CONT;" } else { "SING;" })
    }

    /// Enables averaging with the requested averaging factor.
    ///
    /// # Errors
    ///
    /// Returns an error when the averaging command cannot be written.
    pub fn set_averaging(&mut self, factor: usize) -> Result<()> {
        self.vna.write(&format!("AVERON {factor};"))
    }

    /// Returns the native sweep start frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error when the query fails or its response is not numeric.
    pub fn frequency_start(&mut self) -> Result<f64> {
        parse_float(&self.vna.query("STAR;OUTPACTI;")?)
    }

    /// Sets the native sweep start frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error when the frequency command cannot be written.
    pub fn set_frequency_start(&mut self, frequency_hz: f64) -> Result<()> {
        self.vna.write(&format!("STEP; STAR {frequency_hz};"))
    }

    /// Returns the native sweep stop frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error when the query fails or its response is not numeric.
    pub fn frequency_stop(&mut self) -> Result<f64> {
        parse_float(&self.vna.query("STOP;OUTPACTI;")?)
    }

    /// Sets the native sweep stop frequency in hertz.
    ///
    /// # Errors
    ///
    /// Returns an error when the frequency command cannot be written.
    pub fn set_frequency_stop(&mut self, frequency_hz: f64) -> Result<()> {
        self.vna.write(&format!("STEP; STOP {frequency_hz};"))
    }

    /// Returns the point count of the currently programmed native sweep.
    ///
    /// # Errors
    ///
    /// Returns an error when the query fails or the response cannot be converted
    /// to a point count.
    pub fn native_points(&mut self) -> Result<usize> {
        let value = parse_float(&self.vna.query("POIN;OUTPACTI;")?)?;
        if value.fract() != 0.0 {
            return Err(Error::Parse(format!(
                "HP point count is not an integer: {value}"
            )));
        }
        value
            .to_usize()
            .ok_or_else(|| Error::Parse(format!("HP point count is out of range: {value}")))
    }

    /// Returns the compound plan point count, or the native point count when no
    /// compound sweep is configured.
    ///
    /// # Errors
    ///
    /// Returns an error when the native point count cannot be queried or converted.
    pub fn points(&mut self) -> Result<usize> {
        match &self.compound_sweep_plan {
            Some(plan) => Ok(plan.frequencies_hz().len()),
            None => self.native_points(),
        }
    }

    /// Reconfigures the current start/stop interval for `points` samples.
    ///
    /// # Errors
    ///
    /// Returns an error when querying the current interval or programming the
    /// requested sweep fails.
    pub fn set_points(&mut self, points: usize) -> Result<()> {
        let start = self.frequency_start()?;
        let stop = self.frequency_stop()?;
        self.set_frequency_step(start, stop, points)
    }

    /// Returns the frequency axis of the currently programmed native sweep.
    ///
    /// # Errors
    ///
    /// Returns an error when querying or parsing the native sweep settings fails,
    /// or when the resulting frequency axis is invalid.
    pub fn native_frequency(&mut self) -> Result<Frequency> {
        Frequency::from_hz(Array1::from(linear_space(
            self.frequency_start()?,
            self.frequency_stop()?,
            self.native_points()?,
        )))
    }

    /// Returns the complete compound frequency axis, or the native axis when no
    /// compound plan is configured.
    ///
    /// # Errors
    ///
    /// Returns an error when the planned axis is invalid or the native sweep
    /// settings cannot be queried and converted.
    pub fn frequency(&mut self) -> Result<Frequency> {
        match &self.compound_sweep_plan {
            Some(plan) => Frequency::from_hz(Array1::from(plan.frequencies_hz())),
            None => self.native_frequency(),
        }
    }

    /// Plans a list sweep for the supplied frequency axis.
    ///
    /// Frequencies outside the optional instrument limits are discarded before
    /// planning.
    ///
    /// # Errors
    ///
    /// Returns an error when the retained frequencies cannot form a valid HP 8510
    /// sweep plan.
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

    /// Configures continuous-wave measurement at one frequency using a native
    /// point count.
    ///
    /// # Errors
    ///
    /// Returns an error when `points` is not one of the native HP 8510 counts or
    /// instrument communication fails.
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

    /// Configures a synthesized step sweep, using a compound plan when required.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested interval cannot form a valid sweep plan
    /// or the native sweep cannot be programmed.
    pub fn set_frequency_sweep(
        &mut self,
        start_hz: f64,
        stop_hz: f64,
        points: usize,
    ) -> Result<()> {
        self.set_frequency_step(start_hz, stop_hz, points)
    }

    /// Configures a slow synthesized step sweep.
    ///
    /// Native point counts and custom counts up to 792 are programmed directly;
    /// larger requests are represented by a [`SweepPlan`].
    ///
    /// # Errors
    ///
    /// Returns an error when the interval cannot form a valid sweep plan or the
    /// native sweep cannot be programmed.
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

    /// Configures a fast, non-synthesized ramp sweep.
    ///
    /// # Errors
    ///
    /// Returns an error unless `points` is a native HP 8510 point count.
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

    /// Returns whether the instrument can execute a step sweep directly.
    #[must_use]
    pub fn supports_native_step(points: usize) -> bool {
        points > 0 && (HP8510_NATIVE_POINTS.contains(&points) || points <= 792)
    }

    /// Programs a direct built-in or segmented linear step sweep.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported point count or communication failure.
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

    /// Programs a segmented continuous-wave list sweep.
    ///
    /// # Errors
    ///
    /// Returns an error when more than 30 raw frequencies are supplied or
    /// instrument communication fails.
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
            // HP 8510 CWFREQ accepts integer hertz; preserve the original driver's
            // truncation while rejecting non-finite, negative, or out-of-range values.
            let frequency_hz = frequency.trunc().to_u64().ok_or_else(|| {
                Error::InvalidFrequency(format!("invalid HP8510 CW frequency {frequency}"))
            })?;
            self.vna.write(&format!("CWFREQ {frequency_hz};"))?;
        }
        self.vna.write("SDON; EDITDONE; LISFREQ;")
    }

    /// Reads the two integer instrument status values.
    ///
    /// # Errors
    ///
    /// Returns an error when the status response is malformed.
    pub fn wait_for_status(&mut self) -> Result<(i32, i32)> {
        let status = self.vna.query("OUTPSTAT")?;
        let values = status
            .trim()
            .split(',')
            .map(str::parse::<i32>)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| Error::Parse(format!("invalid HP8510 status: {error}")))?;
        match values.as_slice() {
            [first, second] => Ok((*first, *second)),
            _ => Err(Error::Parse(format!("invalid HP8510 status {status:?}"))),
        }
    }

    /// Reads complex values using the fast HP FORM2 binary transfer format.
    ///
    /// # Errors
    ///
    /// Returns an error for communication failure or an invalid binary response.
    pub fn ask_for_complex(&mut self, output_command: &str) -> Result<Vec<Complex64>> {
        self.wait_for_status()?;
        self.vna.write("FORM2;")?;
        self.vna.write(output_command)?;
        parse_hp_binary(&self.vna.session.read_raw()?)
    }

    /// Performs one native one-port sweep and returns its network data.
    ///
    /// `expected_hz` supplies the frequency axis for a sweep-plan section.
    ///
    /// # Errors
    ///
    /// Returns an error when the sweep or binary transfer fails, the frequency
    /// axis is invalid, or the returned trace length does not match the axis.
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

    /// Performs a one-port native or compound sweep and returns stitched data.
    ///
    /// # Errors
    ///
    /// Returns an error when sweep programming, acquisition, masking, stitching,
    /// or restoration of the original sweep interval fails.
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

    /// Performs the four native parameter sweeps required for a two-port network.
    ///
    /// # Errors
    ///
    /// Returns an error when a parameter sweep fails or the four traces cannot be
    /// assembled into a frequency-aligned two-port network.
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
        assemble_two_port(s11, &s12, &s21, &s22)
    }

    /// Performs a two-port native or compound sweep and returns stitched data.
    ///
    /// # Errors
    ///
    /// Returns an error when sweep programming, acquisition, masking, stitching,
    /// or restoration of the original sweep interval fails.
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

    /// Obtains S-parameter network data for port 1, port 2, or both ports.
    ///
    /// # Errors
    ///
    /// Returns an error unless `ports` is `[1]`, `[2]`, `[1, 2]`, or `[2, 1]`,
    /// or if the measurement fails.
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

    /// Measures the forward and reverse switch terms.
    ///
    /// The first network is the forward term `Gamma_f` = `b_2/a_2` while port 1
    /// drives the source. The second is the reverse term `Gamma_r` = `b_1/a_1`
    /// while port 2 drives the source.
    ///
    /// # Errors
    ///
    /// Returns an error when configuring or acquiring either switch-term sweep fails.
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

fn linear_space(start: f64, stop: f64, points: usize) -> Vec<f64> {
    match points {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let point_intervals = (points - 1).to_f64().unwrap_or(f64::NAN);
            let frequency_increment = (stop - start) / point_intervals;
            (0..points)
                .map(|index| {
                    index
                        .to_f64()
                        .unwrap_or(f64::NAN)
                        .mul_add(frequency_increment, start)
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
