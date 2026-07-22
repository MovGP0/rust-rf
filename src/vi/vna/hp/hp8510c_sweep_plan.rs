//! Decomposes requested frequency lists into sweeps supported by the HP 8510C.
//!
//! The instrument supports smaller fixed-spacing and list sweeps rather than an
//! arbitrary large frequency plan. [`SweepPlan`] partitions the requested points
//! into executable [`SweepSection`] values. The planner does not support
//! overlapping linear sweeps or logarithmic sweeps.

use ndarray::{Array1, Axis};
use num_traits::ToPrimitive;

use crate::{Error, Frequency, Network, Result};

/// Operations required to program an HP 8510C sweep section.
pub trait Hp8510SweepProgrammer {
    /// Programs a linearly spaced sweep.
    ///
    /// # Errors
    ///
    /// Returns an error when the instrument rejects the sweep configuration.
    fn set_instrument_step_state(
        &mut self,
        start_hz: f64,
        stop_hz: f64,
        points: usize,
    ) -> Result<()>;

    /// Programs an arbitrary continuous-wave frequency list.
    ///
    /// # Errors
    ///
    /// Returns an error when the instrument rejects the frequency list.
    fn set_instrument_cw_step_state(&mut self, frequencies_hz: &[f64]) -> Result<()>;
}

/// One sweep the HP 8510C can execute.
#[derive(Debug, Clone, PartialEq)]
pub enum SweepSection {
    /// A linear sweep using one of the instrument's built-in point counts.
    LinearBuiltin(LinearBuiltinSweepSection),
    /// A linear sweep whose measured points are filtered by a mask.
    LinearMasked(LinearMaskedSweepSection),
    /// A linear sweep using a caller-selected point count.
    LinearCustom(LinearCustomSweepSection),
    /// An arbitrary list sweep.
    Random(RandomSweepSection),
}

impl SweepSection {
    /// Returns the requested frequencies represented after masking.
    #[must_use]
    pub fn frequencies_hz(&self) -> Vec<f64> {
        match self {
            Self::LinearBuiltin(section) => section.frequencies_hz(),
            Self::LinearMasked(section) => section.frequencies_hz(),
            Self::LinearCustom(section) => section.frequencies_hz(),
            Self::Random(section) => section.frequencies_hz(),
        }
    }

    /// Returns all frequencies fetched from the instrument before masking.
    #[must_use]
    pub fn raw_frequencies_hz(&self) -> Vec<f64> {
        match self {
            Self::LinearBuiltin(section) => section.frequencies_hz(),
            Self::LinearMasked(section) => section.raw_frequencies_hz(),
            Self::LinearCustom(section) => section.raw_frequencies_hz(),
            Self::Random(section) => section.raw_frequencies_hz(),
        }
    }

    /// Programs this section on an HP 8510C-compatible instrument.
    ///
    /// # Errors
    ///
    /// Returns an error when the programmer rejects the sweep configuration.
    pub fn apply(&self, programmer: &mut impl Hp8510SweepProgrammer) -> Result<()> {
        match self {
            Self::LinearBuiltin(section) => programmer.set_instrument_step_state(
                section.start_hz,
                section.stop_hz,
                section.points,
            ),
            Self::LinearMasked(section) => programmer.set_instrument_step_state(
                section.start_hz,
                section.stop_hz,
                section.points,
            ),
            Self::LinearCustom(section) => {
                let points = if section.points == 1 {
                    2
                } else {
                    section.points
                };
                programmer.set_instrument_step_state(section.start_hz, section.stop_hz, points)
            }
            Self::Random(section) => {
                programmer.set_instrument_cw_step_state(&section.raw_frequencies_hz())
            }
        }
    }

    /// Applies this section's point mask to a measured network.
    ///
    /// # Errors
    ///
    /// Returns an error when a mask index is outside the network frequency axis
    /// or the selected data cannot form a valid network.
    pub fn mask_network(&self, network: &Network) -> Result<Network> {
        let indices = match self {
            Self::LinearMasked(section) => section
                .mask
                .iter()
                .enumerate()
                .filter_map(|(index, include)| include.then_some(index))
                .collect(),
            Self::LinearCustom(section) if section.points == 1 => vec![0],
            Self::Random(section) if section.frequencies_hz.len() == 1 => vec![0],
            _ => return Ok(network.clone()),
        };
        select_frequency_points(network, &indices)
    }
}

/// Linear sweep using a built-in HP 8510C point count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearBuiltinSweepSection {
    /// First frequency in hertz.
    pub start_hz: f64,
    /// Last frequency in hertz.
    pub stop_hz: f64,
    /// Number of points.
    pub points: usize,
}

impl LinearBuiltinSweepSection {
    /// Returns the linearly spaced frequencies represented by the section.
    #[must_use]
    pub fn frequencies_hz(&self) -> Vec<f64> {
        linear_space(self.start_hz, self.stop_hz, self.points)
    }
}

/// Linear sweep with a mask selecting requested measurement points.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearMaskedSweepSection {
    /// First raw frequency in hertz.
    pub start_hz: f64,
    /// Last raw frequency in hertz.
    pub stop_hz: f64,
    /// Number of raw points measured by the instrument.
    pub points: usize,
    /// Per-point inclusion mask applied after measurement.
    pub mask: Vec<bool>,
}

impl LinearMaskedSweepSection {
    /// Creates a masked linear section.
    ///
    /// # Errors
    ///
    /// Returns an error when the mask length differs from `points`.
    pub fn new(start_hz: f64, stop_hz: f64, points: usize, mask: Vec<bool>) -> Result<Self> {
        if mask.len() != points {
            return Err(Error::IncompatibleShape(format!(
                "sweep mask has {} entries for {points} points",
                mask.len()
            )));
        }
        Ok(Self {
            start_hz,
            stop_hz,
            points,
            mask,
        })
    }

    /// Returns the requested frequencies retained by the mask.
    #[must_use]
    pub fn frequencies_hz(&self) -> Vec<f64> {
        self.raw_frequencies_hz()
            .into_iter()
            .zip(&self.mask)
            .filter_map(|(frequency, include)| include.then_some(frequency))
            .collect()
    }

    /// Returns all linearly spaced frequencies measured by the instrument.
    #[must_use]
    pub fn raw_frequencies_hz(&self) -> Vec<f64> {
        linear_space(self.start_hz, self.stop_hz, self.points)
    }
}

/// Linear sweep using a custom point count supported by the HP 8510C.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearCustomSweepSection {
    /// First frequency in hertz.
    pub start_hz: f64,
    /// Last frequency in hertz.
    pub stop_hz: f64,
    /// Number of requested points.
    pub points: usize,
}

impl LinearCustomSweepSection {
    /// Returns the requested linearly spaced frequencies.
    #[must_use]
    pub fn frequencies_hz(&self) -> Vec<f64> {
        linear_space(self.start_hz, self.stop_hz, self.points)
    }

    /// Returns the frequencies actually fetched from the instrument.
    ///
    /// A single requested point is programmed as two points because the HP 8510C
    /// treats one- and two-point sweeps identically; the extra point is masked.
    #[must_use]
    pub fn raw_frequencies_hz(&self) -> Vec<f64> {
        if self.points == 1 {
            vec![self.start_hz, self.start_hz + 1.0]
        } else {
            self.frequencies_hz()
        }
    }
}

/// Arbitrary list sweep for points that do not belong to a linear section.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomSweepSection {
    /// Requested frequencies in hertz.
    pub frequencies_hz: Vec<f64>,
}

impl RandomSweepSection {
    /// Returns the requested frequencies.
    #[must_use]
    pub fn frequencies_hz(&self) -> Vec<f64> {
        self.frequencies_hz.clone()
    }

    /// Returns the frequencies actually fetched from the instrument.
    ///
    /// A single requested point is expanded to two raw points for HP 8510C
    /// compatibility and reduced again by [`SweepSection::mask_network`].
    #[must_use]
    pub fn raw_frequencies_hz(&self) -> Vec<f64> {
        if self.frequencies_hz.len() == 1 {
            vec![self.frequencies_hz[0], self.frequencies_hz[0] + 1.0]
        } else {
            self.frequencies_hz()
        }
    }
}

/// Executable HP 8510C sections covering a requested frequency list.
///
/// Each section is a sweep the instrument can perform. Together, all sections
/// reproduce the requested points. The planner prefers built-in 801- and
/// 401-point sweeps, then custom linear sweeps of up to 792 points, and finally
/// list sweeps for non-linear remainder points.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepPlan {
    sections: Vec<SweepSection>,
}

impl SweepPlan {
    /// Creates a plan from precomputed sweep sections.
    #[must_use]
    pub const fn new(sections: Vec<SweepSection>) -> Self {
        Self { sections }
    }

    /// Decomposes an arbitrary frequency list into HP 8510C sweep sections.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite frequencies or if the generated plan does
    /// not reproduce the requested points.
    pub fn from_hz(frequencies_hz: &[f64]) -> Result<Self> {
        if frequencies_hz
            .iter()
            .any(|frequency| !frequency.is_finite())
        {
            return Err(Error::InvalidFrequency(
                "sweep frequencies must be finite".into(),
            ));
        }
        let sections = sweep_sections_from_hz(frequencies_hz);
        let plan = Self::new(sections);
        if !plan.matches_frequency_list(frequencies_hz) {
            return Err(Error::InvalidFrequency(
                "generated HP 8510C sweep plan does not match requested points".into(),
            ));
        }
        Ok(plan)
    }

    /// Creates a plan for linearly spaced start, stop, and point-count settings.
    ///
    /// # Errors
    ///
    /// Returns the errors reported by [`from_hz`](Self::from_hz).
    pub fn from_start_stop_points(start_hz: f64, stop_hz: f64, points: usize) -> Result<Self> {
        Self::from_hz(&linear_space(start_hz, stop_hz, points))
    }

    /// Returns the executable sweep sections in order.
    #[must_use]
    pub fn sections(&self) -> &[SweepSection] {
        &self.sections
    }

    /// Returns every requested frequency represented by the plan.
    pub fn frequencies_hz(&self) -> Vec<f64> {
        self.sections
            .iter()
            .flat_map(SweepSection::frequencies_hz)
            .collect()
    }

    /// Returns whether the plan represents exactly the supplied frequency list,
    /// allowing numeric tolerance and ignoring order.
    pub fn matches_frequency_list(&self, frequencies_hz: &[f64]) -> bool {
        let mut expected = frequencies_hz.to_vec();
        let mut actual = self.frequencies_hz();
        expected.sort_by(f64::total_cmp);
        actual.sort_by(f64::total_cmp);
        expected.len() == actual.len()
            && expected
                .iter()
                .zip(actual)
                .all(|(expected, actual)| approximately_equal(*expected, actual))
    }
}

fn sweep_sections_from_hz(frequencies_hz: &[f64]) -> Vec<SweepSection> {
    let mut frequencies_hz = frequencies_hz.to_vec();
    frequencies_hz.sort_by(f64::total_cmp);
    let mut misfits = Vec::new();
    let mut window = Vec::new();
    let mut window_step = -1.0;
    let mut sections = Vec::new();

    for (index, frequency) in frequencies_hz.iter().copied().enumerate() {
        match window.len() {
            0 => window.push(frequency),
            1 => {
                window_step = frequency - frequencies_hz[index - 1];
                window.push(frequency);
            }
            _ if ((frequency - frequencies_hz[index - 1]) - window_step).abs() < 0.5 => {
                window.push(frequency);
            }
            _ => {
                finalize_window(&mut window, &mut misfits, &mut sections);
                window.push(frequency);
            }
        }
    }
    finalize_window(&mut window, &mut misfits, &mut sections);

    for chunk in misfits.chunks(29) {
        sections.push(SweepSection::Random(RandomSweepSection {
            frequencies_hz: chunk.to_vec(),
        }));
    }
    sections
}

fn finalize_window(
    window: &mut Vec<f64>,
    misfits: &mut Vec<f64>,
    sections: &mut Vec<SweepSection>,
) {
    if window.len() <= 2 {
        misfits.append(window);
        return;
    }

    for builtin_points in [801, 401] {
        while window.len() >= builtin_points {
            let chunk = window.drain(..builtin_points).collect::<Vec<_>>();
            sections.push(SweepSection::LinearBuiltin(LinearBuiltinSweepSection {
                start_hz: chunk[0],
                stop_hz: chunk[builtin_points - 1],
                points: builtin_points,
            }));
        }
    }

    while !window.is_empty() {
        let take = window.len().min(792);
        let chunk = window.drain(..take).collect::<Vec<_>>();
        sections.push(SweepSection::LinearCustom(LinearCustomSweepSection {
            start_hz: chunk[0],
            stop_hz: chunk[take - 1],
            points: take,
        }));
    }
}

fn select_frequency_points(network: &Network, indices: &[usize]) -> Result<Network> {
    if indices
        .iter()
        .any(|index| *index >= network.frequency_points())
    {
        return Err(Error::IncompatibleShape(
            "sweep mask index exceeds network frequency axis".into(),
        ));
    }
    let frequency = Frequency::from_hz(Array1::from_iter(
        indices
            .iter()
            .map(|index| network.frequency.values_hz()[*index]),
    ))?;
    let mut selected = Network::new(
        frequency,
        network.s.select(Axis(0), indices),
        network.z0.select(Axis(0), indices),
    )?;
    selected.name.clone_from(&network.name);
    selected.comments.clone_from(&network.comments);
    selected.port_names.clone_from(&network.port_names);
    selected.variables = network.variables.clone();
    selected.s_definition = network.s_definition;
    Ok(selected)
}

fn linear_space(start: f64, stop: f64, points: usize) -> Vec<f64> {
    match points {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let point_intervals = (points - 1).to_f64().unwrap_or(f64::NAN);
            let frequency_step = (stop - start) / point_intervals;
            (0..points)
                .map(|index| {
                    index
                        .to_f64()
                        .unwrap_or(f64::NAN)
                        .mul_add(frequency_step, start)
                })
                .collect()
        }
    }
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-8_f64.max(1e-5 * left.abs().max(right.abs()))
}
