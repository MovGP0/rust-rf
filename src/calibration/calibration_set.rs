//! Calibration uncertainty combinatorics.
//!
//! Origin: `skrf/calibration/calibrationSet.py`.

use crate::calibration::Calibration;
use crate::{Error, Network, NetworkSet, Result};

/// Port of `skrf.calibration.calibrationSet.dot_product`.
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

/// Port of `skrf.calibration.calibrationSet.cartesian_product`.
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

/// Origin: `skrf/calibration/calibrationSet.py::Dot`.
#[derive(Clone, Debug)]
pub struct DotCalibrationSet<C> {
    pub ideals: Vec<Network>,
    pub measured_sets: Vec<NetworkSet>,
    pub calibrations: Vec<C>,
    pub name: Option<String>,
}

impl<C> DotCalibrationSet<C>
where
    C: Calibration,
{
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

    pub fn len(&self) -> usize {
        self.calibrations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.calibrations.is_empty()
    }

    /// Port of `CalibrationSet.apply_cal`.
    pub fn apply(&self, raw_network: &Network) -> Result<NetworkSet> {
        NetworkSet::new(
            self.calibrations
                .iter()
                .map(|calibration| calibration.apply(raw_network))
                .collect::<Result<Vec<_>>>()?,
            self.name.clone(),
        )
    }

    /// Port of `CalibrationSet.apply_cal`.
    pub fn apply_cal(&self, raw_network: &Network) -> Result<NetworkSet> {
        self.apply(raw_network)
    }

    /// Port of `CalibrationSet.corrected_sets`.
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
