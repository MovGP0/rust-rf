//! One-dimensional stepped transmission-line tapers.
//!
//! [`Taper1D`] samples a profile between two endpoint values, creates one
//! medium and line section per sample, inserts impedance junctions, and
//! cascades the result.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use num_traits::ToPrimitive;

use crate::math::modified_bessel_i1;
use crate::media::{LengthUnit, Media};
use crate::{Error, Network, Result, SParameterDefinition};

/// A one-dimensional taper built from discrete uniform-media sections.
pub struct Taper1D<M>
where
    M: Media,
{
    /// Factory that constructs a medium from a sampled profile value.
    pub media_at: Box<dyn Fn(f64) -> Result<M> + Send + Sync>,
    /// Profile value at the beginning of the taper.
    pub start: f64,
    /// Profile value at the end of the taper.
    pub stop: f64,
    /// Total taper length.
    pub length: f64,
    /// Unit of the total and per-section lengths.
    pub length_unit: LengthUnit,
    /// Number of uniform sections used to approximate the taper.
    pub section_count: usize,
    /// Profile used to interpolate between endpoint values.
    pub profile: TaperProfile,
    /// Maximum reflection coefficient for a Klopfenstein taper.
    pub maximum_reflection: f64,
    custom_normalized_profile: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Shape used to sample a one-dimensional taper.
pub enum TaperProfile {
    /// Linear interpolation between endpoints.
    Linear,
    /// Exponential interpolation between positive endpoints.
    Exponential,
    /// Cubic smooth-step interpolation $3x^2-2x^3$.
    SmoothStep,
    /// Minimum-length Klopfenstein impedance taper.
    Klopfenstein,
    /// User-supplied normalized profile for $0 \le x \le 1$.
    CustomNormalized,
}

impl<M> Taper1D<M>
where
    M: Media,
{
    /// Creates a taper with the selected built-in profile.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the section count, length, endpoint
    /// values, or selected profile parameters are invalid.
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
    /// Creates a taper with explicit reflection and custom-profile options.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the section count, length, endpoint
    /// values, reflection bound, or custom-profile configuration is invalid.
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

    /// Creates a taper from a user-supplied normalized profile function.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the section count, length, or
    /// endpoint values are invalid.
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

    /// Creates a Klopfenstein taper for a specified maximum reflection.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the section count or length is
    /// invalid, an endpoint is not positive and finite, or
    /// `maximum_reflection` is outside the open interval `(0, 1)`.
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

    /// Returns the uniform length assigned to each section.
    #[must_use]
    pub fn section_length(&self) -> f64 {
        self.length / self.section_count.to_f64().unwrap_or(f64::INFINITY)
    }

    /// Returns the sampled profile values from start to stop.
    ///
    /// # Errors
    ///
    /// Returns an error when a custom profile is missing or returns a
    /// non-finite value, or when Klopfenstein profile evaluation fails.
    pub fn value_vector(&self) -> Result<Array1<f64>> {
        let denominator = self
            .section_count
            .saturating_sub(1)
            .max(1)
            .to_f64()
            .unwrap_or(f64::INFINITY);
        let mut values = Array1::zeros(self.section_count);
        for index in 0..self.section_count {
            let normalized = index.to_f64().unwrap_or(f64::INFINITY) / denominator;
            values[index] = match self.profile {
                TaperProfile::Linear => self.start + normalized * (self.stop - self.start),
                TaperProfile::Exponential => {
                    self.start * (normalized * (self.stop / self.start).ln()).exp()
                }
                TaperProfile::SmoothStep => {
                    let shaped = 2.0f64.mul_add(-normalized.powi(3), 3.0 * normalized.powi(2));
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

    /// Constructs a medium for one profile value.
    ///
    /// # Errors
    ///
    /// Returns any error produced by the configured medium factory.
    pub fn medium_at(&self, value: f64) -> Result<M> {
        (self.media_at)(value)
    }

    /// Constructs one uniform line section for a profile value.
    ///
    /// # Errors
    ///
    /// Returns an error when medium construction or line-section construction
    /// fails.
    pub fn section_at(&self, value: f64) -> Result<Network> {
        self.medium_at(value)?
            .line(self.section_length(), self.length_unit)
    }

    /// Constructs the media for all sampled values.
    ///
    /// # Errors
    ///
    /// Returns an error when profile sampling or construction of any medium
    /// fails.
    pub fn media(&self) -> Result<Vec<M>> {
        self.value_vector()?
            .iter()
            .map(|value| self.medium_at(*value))
            .collect()
    }

    /// Constructs all uniform line-section networks.
    ///
    /// # Errors
    ///
    /// Returns an error when profile sampling, medium construction, or
    /// line-section construction fails.
    pub fn sections(&self) -> Result<Vec<Network>> {
        self.value_vector()?
            .iter()
            .map(|value| self.section_at(*value))
            .collect()
    }

    /// Cascades the sections and their impedance-step junctions into one network.
    ///
    /// # Errors
    ///
    /// Returns an error when section construction, junction construction, or
    /// network cascading fails.
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
        let log_value = (log_ratio / parameter.cosh()).mul_add(
            parameter.powi(2).mul_add(phi, endpoint_step),
            (self.start * self.stop).ln() / 2.0,
        );
        Ok(log_value.exp())
    }
}

fn klopfenstein_phi(endpoint: f64, parameter: f64) -> Result<f64> {
    let intervals = 512_usize;
    let step = endpoint / intervals.to_f64().unwrap_or(f64::INFINITY);
    let integrand = |value: f64| -> Result<f64> {
        let root = value.mul_add(-value, 1.0).max(0.0).sqrt();
        if root <= f64::EPSILON.sqrt() {
            Ok(0.5)
        } else {
            Ok(modified_bessel_i1(parameter * root)? / (parameter * root))
        }
    };
    let mut sum = integrand(0.0)? + integrand(endpoint)?;
    for index in 1..intervals {
        let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * integrand(index.to_f64().unwrap_or(f64::INFINITY) * step)?;
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
