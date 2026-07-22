//! Calibration sets and calibration uncertainty combinatorics.
//!
//! This module provides the [`crate::calibration_set::DotCalibrationSet`] type and functions for
//! constructing calibrations from corresponding or Cartesian combinations of
//! measured standards.

use crate::calibration::Calibration;
use crate::{Error, Network, NetworkSet, Result};

/// Constructs calibrations from corresponding observations in each measured set.
///
/// All measured sets must have the same non-zero length. Observation `n` from
/// each set is combined with `ideals` to create calibration `n`.
///
/// # Errors
///
/// Returns an error if the ideals and measured sets are misaligned or the factory rejects an
/// observation.
pub fn dot_product<C, F>(
    ideals: &[Network],
    measured_sets: &[NetworkSet],
    factory: &F,
) -> Result<Vec<C>>
where
    C: Calibration,
    F: Fn(Vec<Network>, Vec<Network>) -> Result<C>,
{
    validate_measured_sets(ideals, measured_sets)?;
    let observations = measured_sets[0].len();
    (0..observations)
        .map(|observation| {
            let measured = measured_sets
                .iter()
                .map(|set| set.networks[observation].clone())
                .collect();
            factory(measured, ideals.to_vec())
        })
        .collect()
}

/// Constructs calibrations for the Cartesian product of the measured sets.
///
/// Each combination contains one measured network for every ideal standard.
///
/// # Errors
///
/// Returns an error if the ideals and measured sets are misaligned, a measured set is empty, or
/// the factory rejects a combination.
pub fn cartesian_product<C, F>(
    ideals: &[Network],
    measured_sets: &[NetworkSet],
    factory: &F,
) -> Result<Vec<C>>
where
    C: Calibration,
    F: Fn(Vec<Network>, Vec<Network>) -> Result<C>,
{
    if measured_sets.len() != ideals.len() || measured_sets.is_empty() {
        return Err(Error::IncompatibleShape(format!(
            "{} ideals require the same number of non-empty measured sets, got {}",
            ideals.len(),
            measured_sets.len()
        )));
    }
    if measured_sets.iter().any(NetworkSet::is_empty) {
        return Err(Error::IncompatibleShape(
            "cartesian calibration sets cannot contain an empty measured set".to_owned(),
        ));
    }
    let mut combinations = vec![Vec::new()];
    for set in measured_sets {
        let mut expanded = Vec::with_capacity(combinations.len() * set.len());
        for prefix in combinations {
            for network in &set.networks {
                let mut measured = prefix.clone();
                measured.push(network.clone());
                expanded.push(measured);
            }
        }
        combinations = expanded;
    }
    combinations
        .into_iter()
        .map(|measured| factory(measured, ideals.to_vec()))
        .collect()
}

/// A set of calibrations built from corresponding measured observations.
///
/// This supports experimental calibration-uncertainty analysis by applying a
/// collection of calibrations to the same network.
///
/// # References
///
/// A. Arsenovic, L. Chen, M. F. Bauwens, H. Li, N. S. Barker, and R. M.
/// Weikle, "An Experimental Technique for Calibration Uncertainty Analysis,"
/// *IEEE Transactions on Microwave Theory and Techniques*, vol. 61, no. 1,
/// pp. 263–269, 2013.
#[derive(Clone, Debug)]
pub struct DotCalibrationSet<C> {
    /// Ideal calibration standards shared by every calibration.
    pub ideals: Vec<Network>,
    /// Repeated measurements corresponding to the ideal standards.
    ///
    /// Every set must contain the same number of observations.
    pub measured_sets: Vec<NetworkSet>,
    /// Calibrations produced from corresponding observations.
    pub calibrations: Vec<C>,
    /// Optional name assigned to network sets produced by [`Self::apply`].
    pub name: Option<String>,
}

impl<C> DotCalibrationSet<C>
where
    C: Calibration,
{
    /// Creates a calibration set using `factory` as the calibration class.
    ///
    /// `measured_sets[i]` contains repeated measurements corresponding to
    /// `ideals[i]`. Each generated calibration uses one observation from every
    /// measured set.
    ///
    /// # Errors
    ///
    /// Returns an error if the ideals and measured sets are misaligned or the factory rejects an
    /// observation.
    pub fn new<F>(
        ideals: Vec<Network>,
        measured_sets: Vec<NetworkSet>,
        name: Option<String>,
        factory: F,
    ) -> Result<Self>
    where
        F: Fn(Vec<Network>, Vec<Network>) -> Result<C>,
    {
        let calibrations = dot_product(&ideals, &measured_sets, &factory)?;
        Ok(Self {
            ideals,
            measured_sets,
            calibrations,
            name,
        })
    }

    /// Returns the number of calibrations in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.calibrations.len()
    }

    /// Returns `true` when the set contains no calibrations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.calibrations.is_empty()
    }

    /// Applies every calibration to `raw_network` and returns the results as a set.
    ///
    /// # Errors
    ///
    /// Returns an error if a calibration cannot correct the network or the corrected networks are
    /// incompatible.
    pub fn apply(&self, raw_network: &Network) -> Result<NetworkSet> {
        NetworkSet::new(
            self.calibrations
                .iter()
                .map(|calibration| calibration.apply(raw_network))
                .collect::<Result<Vec<_>>>()?,
            self.name.clone(),
        )
    }

    /// Applies every calibration to `raw_network`.
    ///
    /// This is an alias for [`Self::apply`], corresponding to scikit-rf's
    /// `CalibrationSet.apply_cal` method.
    ///
    /// # Errors
    ///
    /// Returns an error under the conditions described by [`Self::apply`].
    pub fn apply_cal(&self, raw_network: &Network) -> Result<NetworkSet> {
        self.apply(raw_network)
    }

    /// Returns the corrected networks grouped by calibration standard.
    ///
    /// Each returned set contains one corrected network from every calibration
    /// for the corresponding element of [`Self::ideals`].
    ///
    /// # Errors
    ///
    /// Returns an error if corrected standards cannot be obtained, calibrations return different
    /// standard counts, or grouped networks are incompatible.
    pub fn corrected_sets(&self) -> Result<Vec<NetworkSet>> {
        let calibrated = self
            .calibrations
            .iter()
            .map(Calibration::calibrated_standards)
            .collect::<Result<Vec<_>>>()?;
        let standards = self.ideals.len();
        if calibrated
            .iter()
            .any(|networks| networks.len() != standards)
        {
            return Err(Error::IncompatibleShape(
                "calibrations returned different standard counts".to_owned(),
            ));
        }
        (0..standards)
            .map(|standard| {
                NetworkSet::new(
                    calibrated
                        .iter()
                        .map(|networks| networks[standard].clone())
                        .collect(),
                    self.ideals[standard].name.clone(),
                )
            })
            .collect()
    }
}

/// Validates the dimensions required by [`dot_product`].
fn validate_measured_sets(ideals: &[Network], measured_sets: &[NetworkSet]) -> Result<()> {
    if measured_sets.len() != ideals.len() || measured_sets.is_empty() {
        return Err(Error::IncompatibleShape(format!(
            "{} ideals require the same number of measured sets, got {}",
            ideals.len(),
            measured_sets.len()
        )));
    }
    let observations = measured_sets[0].len();
    if observations == 0 || measured_sets.iter().any(|set| set.len() != observations) {
        return Err(Error::IncompatibleShape(
            "all measured NetworkSets must have the same non-zero length".to_owned(),
        ));
    }
    Ok(())
}
