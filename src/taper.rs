use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

use crate::math::modified_bessel_i1;
use crate::media::{LengthUnit, Media};
use crate::{Error, Network, Result, SParameterDefinition};

/// Origin: `skrf/taper.py::Taper1D` and its concrete subclasses.
pub struct Taper1D<M>
where
    M: Media,
{
    pub media_at: Box<dyn Fn(f64) -> Result<M> + Send + Sync>,
    pub start: f64,
    pub stop: f64,
    pub length: f64,
    pub length_unit: LengthUnit,
    pub section_count: usize,
    pub profile: TaperProfile,
    pub maximum_reflection: f64,
    custom_normalized_profile: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaperProfile {
    Linear,
    Exponential,
    SmoothStep,
    Klopfenstein,
    CustomNormalized,
}

impl<M> Taper1D<M>
where
    M: Media,
{
    pub fn new(
        media_at: impl Fn(f64) -> Result<M> + Send + Sync + 'static,
        start: f64,
        stop: f64,
        section_count: usize,
        length: f64,
        length_unit: LengthUnit,
        profile: TaperProfile,
    ) -> Result<Self> {
        Self::new_with_options(
            media_at,
            start,
            stop,
            section_count,
            length,
            length_unit,
            profile,
            0.05,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_options(
        media_at: impl Fn(f64) -> Result<M> + Send + Sync + 'static,
        start: f64,
        stop: f64,
        section_count: usize,
        length: f64,
        length_unit: LengthUnit,
        profile: TaperProfile,
        maximum_reflection: f64,
        custom_normalized_profile: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
    ) -> Result<Self> {
        if section_count == 0 {
            return Err(Error::Unsupported(
                "a taper requires at least one section".to_owned(),
            ));
        }
        if !start.is_finite() || !stop.is_finite() || !length.is_finite() || length <= 0.0 {
            return Err(Error::Unsupported(
                "taper endpoints must be finite and length must be positive".to_owned(),
            ));
        }
        if matches!(
            profile,
            TaperProfile::Exponential | TaperProfile::Klopfenstein
        ) && (start <= 0.0 || stop <= 0.0)
        {
            return Err(Error::Unsupported(
                "exponential and Klopfenstein tapers require positive endpoints".to_owned(),
            ));
        }
        if profile == TaperProfile::Klopfenstein
            && (!maximum_reflection.is_finite()
                || maximum_reflection <= 0.0
                || maximum_reflection >= 1.0)
        {
            return Err(Error::Unsupported(
                "Klopfenstein maximum reflection must satisfy 0 < rmax < 1".to_owned(),
            ));
        }
        if (profile == TaperProfile::CustomNormalized) != custom_normalized_profile.is_some() {
            return Err(Error::Unsupported(
                "a custom normalized taper requires exactly one custom profile function".to_owned(),
            ));
        }
        Ok(Self {
            media_at: Box::new(media_at),
            start,
            stop,
            length,
            length_unit,
            section_count,
            profile,
            maximum_reflection,
            custom_normalized_profile,
        })
    }

    pub fn custom_normalized(
        media_at: impl Fn(f64) -> Result<M> + Send + Sync + 'static,
        start: f64,
        stop: f64,
        section_count: usize,
        length: f64,
        length_unit: LengthUnit,
        profile: impl Fn(f64) -> f64 + Send + Sync + 'static,
    ) -> Result<Self> {
        Self::new_with_options(
            media_at,
            start,
            stop,
            section_count,
            length,
            length_unit,
            TaperProfile::CustomNormalized,
            0.05,
            Some(Box::new(profile)),
        )
    }

    pub fn klopfenstein(
        media_at: impl Fn(f64) -> Result<M> + Send + Sync + 'static,
        start: f64,
        stop: f64,
        section_count: usize,
        length: f64,
        length_unit: LengthUnit,
        maximum_reflection: f64,
    ) -> Result<Self> {
        Self::new_with_options(
            media_at,
            start,
            stop,
            section_count,
            length,
            length_unit,
            TaperProfile::Klopfenstein,
            maximum_reflection,
            None,
        )
    }

    pub fn section_length(&self) -> f64 {
        self.length / self.section_count as f64
    }

    pub fn value_vector(&self) -> Result<Array1<f64>> {
        let denominator = self.section_count.saturating_sub(1).max(1) as f64;
        let mut values = Array1::zeros(self.section_count);
        for index in 0..self.section_count {
            let normalized = index as f64 / denominator;
            values[index] = match self.profile {
                TaperProfile::Linear => self.start + normalized * (self.stop - self.start),
                TaperProfile::Exponential => {
                    self.start * (normalized * (self.stop / self.start).ln()).exp()
                }
                TaperProfile::SmoothStep => {
                    let shaped = 3.0 * normalized.powi(2) - 2.0 * normalized.powi(3);
                    self.start + shaped * (self.stop - self.start)
                }
                TaperProfile::Klopfenstein => self.klopfenstein_value(normalized)?,
                TaperProfile::CustomNormalized => {
                    let profile = self.custom_normalized_profile.as_ref().ok_or_else(|| {
                        Error::Unsupported("custom taper profile is missing".to_owned())
                    })?;
                    let shaped = profile(normalized);
                    if !shaped.is_finite() {
                        return Err(Error::Unsupported(
                            "custom taper profile returned a non-finite value".to_owned(),
                        ));
                    }
                    self.start + shaped * (self.stop - self.start)
                }
            };
        }
        Ok(values)
    }

    pub fn medium_at(&self, value: f64) -> Result<M> {
        (self.media_at)(value)
    }

    pub fn section_at(&self, value: f64) -> Result<Network> {
        self.medium_at(value)?
            .line(self.section_length(), self.length_unit)
    }

    pub fn media(&self) -> Result<Vec<M>> {
        self.value_vector()?
            .iter()
            .map(|value| self.medium_at(*value))
            .collect()
    }

    pub fn sections(&self) -> Result<Vec<Network>> {
        self.value_vector()?
            .iter()
            .map(|value| self.section_at(*value))
            .collect()
    }

    pub fn network(&self) -> Result<Network> {
        let sections = self.sections()?;
        let mut result = sections[0].clone();
        for next in &sections[1..] {
            let mismatch = taper_junction(&result, next)?;
            result = result.cascade(&mismatch)?.cascade(next)?;
        }
        Ok(result)
    }

    fn klopfenstein_value(&self, normalized: f64) -> Result<f64> {
        let position = normalized * self.length;
        let centered = position - self.length / 2.0;
        let log_ratio = (self.stop / self.start).ln() / 2.0;
        let parameter = (1.0 / self.maximum_reflection).acosh();
        let phi = klopfenstein_phi(2.0 * centered / self.length, parameter)?;
        let endpoint_step = if position >= self.length { 1.0 } else { 0.0 };
        let log_value = (self.start * self.stop).ln() / 2.0
            + log_ratio / parameter.cosh() * (parameter.powi(2) * phi + endpoint_step);
        Ok(log_value.exp())
    }
}

fn klopfenstein_phi(endpoint: f64, parameter: f64) -> Result<f64> {
    let intervals = 512_usize;
    let step = endpoint / intervals as f64;
    let integrand = |value: f64| -> Result<f64> {
        let root = (1.0 - value.powi(2)).max(0.0).sqrt();
        if root <= f64::EPSILON.sqrt() {
            Ok(0.5)
        } else {
            Ok(modified_bessel_i1(parameter * root)? / (parameter * root))
        }
    };
    let mut sum = integrand(0.0)? + integrand(endpoint)?;
    for index in 1..intervals {
        let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * integrand(index as f64 * step)?;
    }
    Ok(sum * step / 3.0)
}

fn taper_junction(left: &Network, right: &Network) -> Result<Network> {
    if left.frequency != right.frequency || left.ports() != 2 || right.ports() != 2 {
        return Err(Error::Unsupported(
            "taper sections must be compatible two-port networks".to_owned(),
        ));
    }
    let points = left.frequency_points();
    let mut scattering = Array3::zeros((points, 2, 2));
    let mut z0 = Array2::zeros((points, 2));
    for point in 0..points {
        let left_reference = left.z0[(point, 1)];
        let right_reference = right.z0[(point, 0)];
        if left_reference.im != 0.0
            || right_reference.im != 0.0
            || left_reference.re <= 0.0
            || right_reference.re <= 0.0
        {
            return Err(Error::Unsupported(
                "taper junctions currently require positive real reference impedances".to_owned(),
            ));
        }
        let denominator = left_reference.re + right_reference.re;
        let reflection = (right_reference.re - left_reference.re) / denominator;
        let transmission = 2.0 * (left_reference.re * right_reference.re).sqrt() / denominator;
        scattering[(point, 0, 0)] = Complex64::new(reflection, 0.0);
        scattering[(point, 1, 1)] = Complex64::new(-reflection, 0.0);
        scattering[(point, 0, 1)] = Complex64::new(transmission, 0.0);
        scattering[(point, 1, 0)] = Complex64::new(transmission, 0.0);
        z0[(point, 0)] = left_reference;
        z0[(point, 1)] = right_reference;
    }
    let mut junction = Network::new(left.frequency.clone(), scattering, z0)?;
    junction.s_definition = SParameterDefinition::Power;
    Ok(junction)
}
