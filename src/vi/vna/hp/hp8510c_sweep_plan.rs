//! HP 8510C sweep decomposition.
//!
//! Origin: `skrf/vi/vna/hp/hp8510c_sweep_plan.py`.

use ndarray::{Array1, Axis};

use crate::{Error, Frequency, Network, Result};

pub trait Hp8510SweepProgrammer {
    fn set_instrument_step_state(
        &mut self,
        start_hz: f64,
        stop_hz: f64,
        points: usize,
    ) -> Result<()>;

    fn set_instrument_cw_step_state(&mut self, frequencies_hz: &[f64]) -> Result<()>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum SweepSection {
    LinearBuiltin(LinearBuiltinSweepSection),
    LinearMasked(LinearMaskedSweepSection),
    LinearCustom(LinearCustomSweepSection),
    Random(RandomSweepSection),
}

impl SweepSection {
    pub fn frequencies_hz(&self) -> Vec<f64> {
        match self {
            Self::LinearBuiltin(section) => section.frequencies_hz(),
            Self::LinearMasked(section) => section.frequencies_hz(),
            Self::LinearCustom(section) => section.frequencies_hz(),
            Self::Random(section) => section.frequencies_hz(),
        }
    }

    pub fn raw_frequencies_hz(&self) -> Vec<f64> {
        match self {
            Self::LinearBuiltin(section) => section.frequencies_hz(),
            Self::LinearMasked(section) => section.raw_frequencies_hz(),
            Self::LinearCustom(section) => section.raw_frequencies_hz(),
            Self::Random(section) => section.raw_frequencies_hz(),
        }
    }

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearBuiltinSweepSection {
    pub start_hz: f64,
    pub stop_hz: f64,
    pub points: usize,
}

impl LinearBuiltinSweepSection {
    pub fn frequencies_hz(&self) -> Vec<f64> {
        linear_space(self.start_hz, self.stop_hz, self.points)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearMaskedSweepSection {
    pub start_hz: f64,
    pub stop_hz: f64,
    pub points: usize,
    pub mask: Vec<bool>,
}

impl LinearMaskedSweepSection {
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

    pub fn frequencies_hz(&self) -> Vec<f64> {
        self.raw_frequencies_hz()
            .into_iter()
            .zip(&self.mask)
            .filter_map(|(frequency, include)| include.then_some(frequency))
            .collect()
    }

    pub fn raw_frequencies_hz(&self) -> Vec<f64> {
        linear_space(self.start_hz, self.stop_hz, self.points)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearCustomSweepSection {
    pub start_hz: f64,
    pub stop_hz: f64,
    pub points: usize,
}

impl LinearCustomSweepSection {
    pub fn frequencies_hz(&self) -> Vec<f64> {
        linear_space(self.start_hz, self.stop_hz, self.points)
    }

    pub fn raw_frequencies_hz(&self) -> Vec<f64> {
        if self.points == 1 {
            vec![self.start_hz, self.start_hz + 1.0]
        } else {
            self.frequencies_hz()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RandomSweepSection {
    pub frequencies_hz: Vec<f64>,
}

impl RandomSweepSection {
    pub fn frequencies_hz(&self) -> Vec<f64> {
        self.frequencies_hz.clone()
    }

    pub fn raw_frequencies_hz(&self) -> Vec<f64> {
        if self.frequencies_hz.len() == 1 {
            vec![self.frequencies_hz[0], self.frequencies_hz[0] + 1.0]
        } else {
            self.frequencies_hz()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SweepPlan {
    sections: Vec<SweepSection>,
}

impl SweepPlan {
    pub fn new(sections: Vec<SweepSection>) -> Self {
        Self { sections }
    }

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

    pub fn from_start_stop_points(start_hz: f64, stop_hz: f64, points: usize) -> Result<Self> {
        Self::from_hz(&linear_space(start_hz, stop_hz, points))
    }

    pub fn sections(&self) -> &[SweepSection] {
        &self.sections
    }

    pub fn frequencies_hz(&self) -> Vec<f64> {
        self.sections
            .iter()
            .flat_map(SweepSection::frequencies_hz)
            .collect()
    }

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
    selected.name = network.name.clone();
    selected.comments = network.comments.clone();
    selected.port_names = network.port_names.clone();
    selected.variables = network.variables.clone();
    selected.s_definition = network.s_definition;
    Ok(selected)
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

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-8_f64.max(1e-5 * left.abs().max(right.abs()))
}
