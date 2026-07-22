//! Sets of aligned networks, set-wise statistics, parameter interpolation, and uncertainty.
//!
//! A [`NetworkSet`] applies calculations such as mean and standard deviation
//! across multiple N-port networks and returns results as ordinary [`Network`]
//! values so they can be plotted or serialized normally.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::NaiveDateTime;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use num_traits::ToPrimitive;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

use crate::io::{Citi, Mdif, StoredObject};
use crate::{Error, Frequency, Network, Result};

/// Network component selected for a set-wise statistic.
///
/// Origin: the dynamic `attribute` argument of
/// `skrf.networkSet.NetworkSet.uncertainty_ntwk_triplet`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NetworkSetAttribute {
    /// Complex scattering parameters.
    Scattering,
    /// Scattering magnitude.
    Magnitude,
    /// Scattering phase in degrees.
    PhaseDegrees,
    /// Scattering magnitude in $20\log_{10}$ decibels.
    Decibel,
    /// Scattering magnitude in $10\log_{10}$ decibels.
    Decibel10,
    /// Real scattering component.
    Real,
    /// Imaginary scattering component.
    Imaginary,
    /// Voltage standing-wave ratio.
    Vswr,
}

/// Scalar network view used by the legacy attribute `DataFrame` adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NetworkScalarAttribute {
    /// Scattering magnitude.
    Magnitude,
    /// $20\log_{10}$ magnitude.
    Decibel,
    /// $10\log_{10}$ magnitude.
    Decibel10,
    /// Phase in degrees.
    PhaseDegrees,
    /// Real component.
    Real,
    /// Imaginary component.
    Imaginary,
    /// Voltage standing-wave ratio.
    Vswr,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Network parameter representation used for set-wise statistics.
pub enum NetworkParameter {
    /// Scattering parameters.
    #[default]
    Scattering,
    /// Impedance parameters.
    Impedance,
    /// Admittance parameters.
    Admittance,
    /// Two-port ABCD parameters.
    Abcd,
    /// Inverse-hybrid $G$ parameters.
    InverseHybrid,
    /// Hybrid $H$ parameters.
    Hybrid,
    /// Scattering-transfer $T$ parameters.
    ScatteringTransfer,
}

/// Named result of [`tuner_constellation`].
#[derive(Clone, Debug, PartialEq)]
pub struct TunerConstellation {
    /// Networks corresponding to the tuner points.
    pub networks: NetworkSet,
    /// Real parts of the reflection coefficients.
    pub real: Array1<f64>,
    /// Imaginary parts of the reflection coefficients.
    pub imaginary: Array1<f64>,
    /// Complex reflection coefficients.
    pub reflection: Array1<Complex64>,
}

/// An unordered collection of aligned N-port networks.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NetworkSet {
    /// Networks in the set.
    pub networks: Vec<Network>,
    /// Optional set name.
    pub name: Option<String>,
    /// Numeric parameter values aligned with the networks.
    pub parameters: BTreeMap<String, Vec<f64>>,
    /// Text parameter values aligned with the networks.
    pub text_parameters: BTreeMap<String, Vec<String>>,
}

impl NetworkSet {
    /// Creates a set whose members share frequency axis and port count.
    ///
    /// # Errors
    ///
    /// Returns an error if the networks do not share the same port count and frequency axis.
    pub fn new(networks: Vec<Network>, name: Option<String>) -> Result<Self> {
        if let Some(first) = networks.first() {
            for network in networks.iter().skip(1) {
                if network.ports() != first.ports() {
                    return Err(Error::IncompatibleShape(
                        "all networks in a set must have the same number of ports".to_owned(),
                    ));
                }
                if network.frequency != first.frequency {
                    return Err(Error::InvalidFrequency(
                        "all networks in a set must share the same frequency axis".to_owned(),
                    ));
                }
            }
        }
        Ok(Self {
            networks,
            name,
            parameters: BTreeMap::new(),
            text_parameters: BTreeMap::new(),
        })
    }

    /// Constructs a set from every Touchstone member of a ZIP archive.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.from_zip`.
    ///
    /// # Errors
    ///
    /// Returns an error if the archive cannot be read or its networks are incompatible.
    pub fn from_zip(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let networks = crate::io::read_zipped_touchstones(path)?
            .into_values()
            .collect();
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned());
        Self::new(networks, name)
    }

    /// Creates a set from supported network files in one directory.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.from_dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read or its networks are incompatible.
    pub fn from_directory(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let networks = crate::io::read_all_networks(path, None, false)?
            .into_values()
            .collect();
        Self::new(
            networks,
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned()),
        )
    }

    /// Creates named networks from scattering matrices and a shared frequency axis.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.from_s_dict`.
    ///
    /// # Errors
    ///
    /// Returns an error if a matrix is non-square, its point count differs from the frequency
    /// axis, or the constructed networks are incompatible.
    pub fn from_s_map(
        values: BTreeMap<String, Array3<Complex64>>,
        frequency: impl std::borrow::Borrow<Frequency>,
        reference_impedance: Complex64,
        name: Option<String>,
    ) -> Result<Self> {
        let frequency = frequency.borrow();
        let mut networks = Vec::with_capacity(values.len());
        for (network_name, scattering) in values {
            let (points, rows, columns) = scattering.dim();
            if rows != columns {
                return Err(Error::IncompatibleShape(format!(
                    "network '{network_name}' has a non-square scattering matrix"
                )));
            }
            if points != frequency.points() {
                return Err(Error::IncompatibleShape(format!(
                    "network '{network_name}' has {points} points for a {}-point frequency axis",
                    frequency.points()
                )));
            }
            let z0 = Array2::from_elem((points, rows), reference_impedance);
            let mut network = Network::new(frequency.clone(), scattering, z0)?;
            network.name = Some(network_name);
            networks.push(network);
        }
        Self::new(networks, name)
    }

    /// Reads a Generalized MDIF file into a parameterized set.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.from_mdif`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, parsed, or converted to a network set.
    pub fn from_mdif(path: impl AsRef<Path>) -> Result<Self> {
        Mdif::from_path(path)?.to_network_set()
    }

    /// Reads a CITI file into a parameterized set.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.from_citi`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, parsed, or converted to a network set.
    pub fn from_citi(path: impl AsRef<Path>) -> Result<Self> {
        Citi::from_path(path)?.to_network_set()
    }

    /// Returns the number of networks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.networks.len()
    }

    /// Returns whether the set contains no networks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.networks.is_empty()
    }

    /// Returns cloned networks keyed by their names.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.to_dict`.
    ///
    /// # Errors
    ///
    /// Returns an error if any network is unnamed.
    pub fn to_network_map(&self) -> Result<BTreeMap<String, Network>> {
        self.networks
            .iter()
            .map(|network| {
                network
                    .name
                    .clone()
                    .ok_or_else(|| {
                        Error::Unsupported(
                            "all networks must be named for dictionary conversion".to_owned(),
                        )
                    })
                    .map(|name| (name, network.clone()))
            })
            .collect()
    }

    /// Returns cloned scattering arrays keyed by network name.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.to_s_dict`.
    ///
    /// # Errors
    ///
    /// Returns an error if any network is unnamed.
    pub fn to_s_map(&self) -> Result<BTreeMap<String, Array3<Complex64>>> {
        self.to_network_map().map(|networks| {
            networks
                .into_iter()
                .map(|(name, network)| (name, network.s))
                .collect()
        })
    }

    /// Rust representation of `NetworkSet.params`.
    pub fn parameter_names(&self) -> Vec<&str> {
        self.parameters
            .keys()
            .chain(self.text_parameters.keys())
            .map(String::as_str)
            .collect()
    }

    /// Port of `NetworkSet.has_params` for the typed parameter stores.
    #[must_use]
    pub fn has_parameters(&self) -> bool {
        (!self.parameters.is_empty() || !self.text_parameters.is_empty())
            && self.validate_parameters().is_ok()
    }

    /// Validate and assign one parameter coordinate to every network.
    ///
    /// # Errors
    ///
    /// Returns an error if the value count differs from the network count or a value is not
    /// finite.
    pub fn set_parameter(&mut self, name: impl Into<String>, values: Vec<f64>) -> Result<()> {
        if values.len() != self.networks.len() {
            return Err(Error::IncompatibleShape(format!(
                "parameter contains {} values for {} networks",
                values.len(),
                self.networks.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Unsupported(
                "NetworkSet parameters must be finite".to_owned(),
            ));
        }
        self.parameters.insert(name.into(), values);
        Ok(())
    }

    /// Assign a string-valued coordinate to every network.
    ///
    /// # Errors
    ///
    /// Returns an error if the value count differs from the network count.
    pub fn set_text_parameter(
        &mut self,
        name: impl Into<String>,
        values: Vec<String>,
    ) -> Result<()> {
        if values.len() != self.networks.len() {
            return Err(Error::IncompatibleShape(format!(
                "text parameter contains {} values for {} networks",
                values.len(),
                self.networks.len()
            )));
        }
        self.text_parameters.insert(name.into(), values);
        Ok(())
    }

    /// Port of `skrf.networkSet.NetworkSet.sel` for numeric parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if stored parameter coordinates are inconsistent with the network count.
    pub fn select(&self, indexers: &BTreeMap<String, Vec<f64>>) -> Result<Self> {
        self.validate_parameters()?;
        if indexers.is_empty() {
            return Ok(self.clone());
        }
        if indexers
            .keys()
            .any(|name| !self.parameters.contains_key(name))
        {
            return Self::new(Vec::new(), self.name.clone());
        }
        let indices = (0..self.networks.len())
            .filter(|index| {
                indexers
                    .iter()
                    .all(|(name, accepted)| accepted.contains(&self.parameters[name][*index]))
            })
            .collect::<Vec<_>>();
        self.select_indices(&indices)
    }

    /// Port of `skrf.networkSet.NetworkSet.interpolate_from_params` for a
    /// numeric interpolation axis and optional exact numeric filters.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter is absent, coordinates are invalid, or interpolation
    /// cannot be performed at the target.
    pub fn interpolate_from_parameter(
        &self,
        parameter: &str,
        target: f64,
        filters: &BTreeMap<String, Vec<f64>>,
    ) -> Result<Network> {
        if !self.parameters.contains_key(parameter) {
            return Err(Error::Unsupported(format!(
                "parameter '{parameter}' is not present"
            )));
        }
        let selected = self.select(filters)?;
        let values = selected
            .parameters
            .get(parameter)
            .ok_or_else(|| Error::Unsupported(format!("parameter '{parameter}' is not present")))?;
        selected.interpolate_from_values(values, target)
    }

    /// Port of `skrf.networkSet.NetworkSet.interpolate_frequency`.
    ///
    /// # Errors
    ///
    /// Returns an error if any member cannot be interpolated to the requested frequency axis.
    pub fn interpolate_frequency(&self, frequency: &crate::Frequency) -> Result<Self> {
        let networks = self
            .networks
            .iter()
            .map(|network| network.interpolate(frequency))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            networks,
            name: self.name.clone(),
            parameters: self.parameters.clone(),
            text_parameters: self.text_parameters.clone(),
        })
    }

    /// Port of `skrf.networkSet.NetworkSet.inv`.
    ///
    /// # Errors
    ///
    /// Returns an error if any member network cannot be inverted.
    pub fn inverse(&self) -> Result<Self> {
        let networks = self
            .networks
            .iter()
            .map(Network::inverse)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            networks,
            name: self.name.clone(),
            parameters: self.parameters.clone(),
            text_parameters: self.text_parameters.clone(),
        })
    }

    /// Port of `skrf.networkSet.NetworkSet.filter`.
    ///
    /// # Errors
    ///
    /// Returns an error if the selected parameter coordinates are inconsistent.
    pub fn filter_names(&self, needle: &str) -> Result<Self> {
        let indices = self
            .networks
            .iter()
            .enumerate()
            .filter_map(|(index, network)| {
                network
                    .name
                    .as_deref()
                    .is_some_and(|name| name.contains(needle))
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        self.select_indices(&indices)
    }

    /// Port of the default name-keyed `NetworkSet.sort` behavior.
    pub fn sort_by_name(&mut self) {
        let mut indices = (0..self.networks.len()).collect::<Vec<_>>();
        indices.sort_by(|left, right| {
            self.networks[*left]
                .name
                .as_deref()
                .cmp(&self.networks[*right].name.as_deref())
        });
        self.networks = indices
            .iter()
            .map(|index| self.networks[*index].clone())
            .collect();
        for values in self.parameters.values_mut() {
            *values = indices.iter().map(|index| values[*index]).collect();
        }
        for values in self.text_parameters.values_mut() {
            *values = indices.iter().map(|index| values[*index].clone()).collect();
        }
    }

    /// Returns a name-sorted clone while retaining all parameter coordinates.
    #[must_use]
    pub fn sorted_by_name(&self) -> Self {
        let mut sorted = self.clone();
        sorted.sort_by_name();
        sorted
    }

    /// Returns `count` random samples with replacement using a caller-owned RNG.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.rand`.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty.
    pub fn random_networks_with_rng<R: Rng + ?Sized>(
        &self,
        count: usize,
        rng: &mut R,
    ) -> Result<Vec<Network>> {
        if self.networks.is_empty() {
            return Err(Error::IncompatibleShape(
                "cannot sample an empty NetworkSet".to_owned(),
            ));
        }
        Ok((0..count)
            .map(|_| self.networks[rng.random_range(0..self.networks.len())].clone())
            .collect())
    }

    /// Applies one typed network operation to every member and preserves coordinates.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.element_wise_method`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails or produces mutually incompatible networks.
    pub fn map_networks(
        &self,
        mut operation: impl FnMut(&Network) -> Result<Network>,
    ) -> Result<Self> {
        let networks = self
            .networks
            .iter()
            .map(&mut operation)
            .collect::<Result<Vec<_>>>()?;
        let mut mapped = Self::new(networks, self.name.clone())?;
        mapped.parameters = self.parameters.clone();
        mapped.text_parameters = self.text_parameters.clone();
        Ok(mapped)
    }

    /// Applies a typed pairwise operation to two compatible sets.
    ///
    /// # Errors
    ///
    /// Returns an error if the sets differ in length, the operation fails, or its results are
    /// mutually incompatible.
    pub fn zip_networks(
        &self,
        other: &Self,
        mut operation: impl FnMut(&Network, &Network) -> Result<Network>,
    ) -> Result<Self> {
        if self.networks.len() != other.networks.len() {
            return Err(Error::IncompatibleShape(format!(
                "NetworkSet lengths differ: {} and {}",
                self.networks.len(),
                other.networks.len()
            )));
        }
        let networks = self
            .networks
            .iter()
            .zip(&other.networks)
            .map(|(left, right)| operation(left, right))
            .collect::<Result<Vec<_>>>()?;
        let mut mapped = Self::new(networks, self.name.clone())?;
        mapped.parameters = self.parameters.clone();
        mapped.text_parameters = self.text_parameters.clone();
        Ok(mapped)
    }

    /// Adds paired networks element by element.
    ///
    /// # Errors
    ///
    /// Returns an error if the sets differ in length or paired networks are incompatible.
    pub fn add_set(&self, other: &Self) -> Result<Self> {
        self.zip_networks(other, Network::add_elementwise)
    }

    /// Subtracts paired networks element by element.
    ///
    /// # Errors
    ///
    /// Returns an error if the sets differ in length or paired networks are incompatible.
    pub fn subtract_set(&self, other: &Self) -> Result<Self> {
        self.zip_networks(other, Network::subtract_elementwise)
    }

    /// Multiplies paired networks element by element.
    ///
    /// # Errors
    ///
    /// Returns an error if the sets differ in length or paired networks are incompatible.
    pub fn multiply_set(&self, other: &Self) -> Result<Self> {
        self.zip_networks(other, Network::multiply_elementwise)
    }

    /// Divides paired networks element by element.
    ///
    /// # Errors
    ///
    /// Returns an error if the sets differ in length or paired networks are incompatible.
    pub fn divide_set(&self, other: &Self) -> Result<Self> {
        self.zip_networks(other, Network::divide_elementwise)
    }

    /// Cascades paired networks.
    ///
    /// # Errors
    ///
    /// Returns an error if the sets differ in length or paired networks cannot be cascaded.
    pub fn cascade_set(&self, other: &Self) -> Result<Self> {
        self.zip_networks(other, Network::cascade)
    }

    /// De-embeds each network using its paired fixture.
    ///
    /// # Errors
    ///
    /// Returns an error if set lengths differ or a network cannot be de-embedded by its fixture.
    pub fn deembed_set(&self, fixtures: &Self) -> Result<Self> {
        self.zip_networks(fixtures, |network, fixture| network.deembed(fixture, None))
    }

    /// Adds one network elementwise to every member.
    ///
    /// # Errors
    ///
    /// Returns an error if the supplied network is incompatible with any member.
    pub fn add_network(&self, other: &Network) -> Result<Self> {
        self.map_networks(|network| network.add_elementwise(other))
    }

    /// Subtracts one network elementwise from every member.
    ///
    /// # Errors
    ///
    /// Returns an error if the supplied network is incompatible with any member.
    pub fn subtract_network(&self, other: &Network) -> Result<Self> {
        self.map_networks(|network| network.subtract_elementwise(other))
    }

    /// Multiplies every member elementwise by one network.
    ///
    /// # Errors
    ///
    /// Returns an error if the supplied network is incompatible with any member.
    pub fn multiply_network(&self, other: &Network) -> Result<Self> {
        self.map_networks(|network| network.multiply_elementwise(other))
    }

    /// Divides every member elementwise by one network.
    ///
    /// # Errors
    ///
    /// Returns an error if the supplied network is incompatible with any member.
    pub fn divide_network(&self, other: &Network) -> Result<Self> {
        self.map_networks(|network| network.divide_elementwise(other))
    }

    /// Cascades every member with one network.
    ///
    /// # Errors
    ///
    /// Returns an error if the supplied network cannot be cascaded with any member.
    pub fn cascade_network(&self, other: &Network) -> Result<Self> {
        self.map_networks(|network| network.cascade(other))
    }

    /// De-embeds the same fixture from every member.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixture cannot de-embed any member.
    pub fn deembed_network(&self, fixture: &Network) -> Result<Self> {
        self.map_networks(|network| network.deembed(fixture, None))
    }

    /// Returns the complex mean scattering matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty or the derived network cannot be constructed.
    pub fn mean_s(&self) -> Result<Network> {
        let first = self.first()?;
        let mut s = Array3::zeros(first.s.dim());
        for network in &self.networks {
            s += &network.s;
        }
        s.mapv_inplace(|value| value / self.networks.len().to_f64().unwrap_or(f64::INFINITY));
        self.derived_network(s, "mean")
    }

    /// Returns scattering sample standard deviation stored in a network.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty or the derived network cannot be constructed.
    pub fn std_s(&self) -> Result<Network> {
        let mean = self.mean_s()?;
        let mut variances = Array3::<f64>::zeros(mean.s.dim());
        for network in &self.networks {
            for (variance, (value, mean_value)) in variances
                .iter_mut()
                .zip(network.s.iter().zip(mean.s.iter()))
            {
                *variance += (*value - *mean_value).norm_sqr();
            }
        }
        let count = self.networks.len().to_f64().unwrap_or(f64::INFINITY);
        let standard_deviation =
            variances.mapv(|variance| Complex64::new((variance / count).sqrt(), 0.0));
        self.derived_network(standard_deviation, "std")
    }

    /// Returns the complex mean of a selected network-parameter representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty, parameter conversion fails, or the result cannot be
    /// represented as a network.
    pub fn mean_parameter(&self, parameter: NetworkParameter) -> Result<Network> {
        let first = self.first()?;
        let mut mean = Array3::zeros(first.s.dim());
        for network in &self.networks {
            mean += &parameter_values(network, parameter)?;
        }
        mean.mapv_inplace(|value| value / self.networks.len().to_f64().unwrap_or(f64::INFINITY));
        self.derived_network(mean, &format!("mean-{parameter:?}"))
    }

    /// Returns sample standard deviation of a selected parameter representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty, parameter conversion fails, or the result cannot be
    /// represented as a network.
    pub fn std_parameter(&self, parameter: NetworkParameter) -> Result<Network> {
        let mean = self.mean_parameter(parameter)?;
        let mut variance = Array3::<f64>::zeros(mean.s.dim());
        for network in &self.networks {
            let values = parameter_values(network, parameter)?;
            for (variance, (value, mean_value)) in
                variance.iter_mut().zip(values.iter().zip(mean.s.iter()))
            {
                *variance += (*value - *mean_value).norm_sqr();
            }
        }
        let count = self.networks.len().to_f64().unwrap_or(f64::INFINITY);
        self.derived_network(
            variance.mapv(|value| Complex64::new((value / count).sqrt(), 0.0)),
            &format!("std-{parameter:?}"),
        )
    }

    /// Returns the mean of a real-valued component of a parameter representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty, parameter conversion fails, or the result cannot be
    /// represented as a network.
    pub fn mean_parameter_component(
        &self,
        parameter: NetworkParameter,
        component: NetworkScalarAttribute,
    ) -> Result<Network> {
        self.scalar_parameter_statistic(parameter, component, false)
    }

    /// Returns standard deviation of a real-valued parameter component.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty, parameter conversion fails, or the result cannot be
    /// represented as a network.
    pub fn std_parameter_component(
        &self,
        parameter: NetworkParameter,
        component: NetworkScalarAttribute,
    ) -> Result<Network> {
        self.scalar_parameter_statistic(parameter, component, true)
    }

    /// Mean scattering magnitude, stored as the real component of `Network.s`.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty or the derived network cannot be constructed.
    pub fn mean_s_magnitude(&self) -> Result<Network> {
        let first = self.first()?;
        let mut mean = Array3::<f64>::zeros(first.s.dim());
        for network in &self.networks {
            mean += &network.s.mapv(Complex64::norm);
        }
        mean.mapv_inplace(|value| value / self.networks.len().to_f64().unwrap_or(f64::INFINITY));
        self.derived_network(mean.mapv(|value| Complex64::new(value, 0.0)), "mean-s-mag")
    }

    /// Population standard deviation of scattering magnitude.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty or the derived network cannot be constructed.
    pub fn std_s_magnitude(&self) -> Result<Network> {
        let mean = self.mean_s_magnitude()?;
        let mut variance = Array3::<f64>::zeros(mean.s.dim());
        for network in &self.networks {
            for (variance, (value, mean_value)) in
                variance.iter_mut().zip(network.s.iter().zip(mean.s.iter()))
            {
                *variance += (value.norm() - mean_value.re).powi(2);
            }
        }
        let count = self.networks.len().to_f64().unwrap_or(f64::INFINITY);
        self.derived_network(
            variance.mapv(|value| Complex64::new((value / count).sqrt(), 0.0)),
            "std-s-mag",
        )
    }

    /// Mean scattering phase in degrees, stored in the real component.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty or the derived network cannot be constructed.
    pub fn mean_s_phase_degrees(&self) -> Result<Network> {
        let first = self.first()?;
        let mut mean = Array3::<f64>::zeros(first.s.dim());
        for network in &self.networks {
            mean += &network.s.mapv(|value| value.arg().to_degrees());
        }
        mean.mapv_inplace(|value| value / self.networks.len().to_f64().unwrap_or(f64::INFINITY));
        self.derived_network(mean.mapv(|value| Complex64::new(value, 0.0)), "mean-s-deg")
    }

    /// Population standard deviation of scattering phase in degrees.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty or the derived network cannot be constructed.
    pub fn std_s_phase_degrees(&self) -> Result<Network> {
        let mean = self.mean_s_phase_degrees()?;
        let mut variance = Array3::<f64>::zeros(mean.s.dim());
        for network in &self.networks {
            for (variance, (value, mean_value)) in
                variance.iter_mut().zip(network.s.iter().zip(mean.s.iter()))
            {
                *variance += (value.arg().to_degrees() - mean_value.re).powi(2);
            }
        }
        let count = self.networks.len().to_f64().unwrap_or(f64::INFINITY);
        self.derived_network(
            variance.mapv(|value| Complex64::new((value / count).sqrt(), 0.0)),
            "std-s-deg",
        )
    }

    /// Mean magnitude converted to decibels after aggregation.
    ///
    /// # Errors
    ///
    /// Returns an error if the magnitude mean cannot be computed.
    pub fn mean_s_db(&self) -> Result<Network> {
        let mut network = self.mean_s_magnitude()?;
        network.s.mapv_inplace(|value| {
            Complex64::new(crate::math::magnitude_to_db(value.re, true), 0.0)
        });
        Ok(network)
    }

    /// Magnitude standard deviation converted to decibels after aggregation.
    ///
    /// # Errors
    ///
    /// Returns an error if the magnitude standard deviation cannot be computed.
    pub fn std_s_db(&self) -> Result<Network> {
        let mut network = self.std_s_magnitude()?;
        network.s.mapv_inplace(|value| {
            Complex64::new(crate::math::magnitude_to_db(value.re, true), 0.0)
        });
        Ok(network)
    }

    /// Returns the mean, lower bound, and upper bound for a selected attribute.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.uncertainty_ntwk_triplet`.
    ///
    /// # Errors
    ///
    /// Returns an error if `deviations` is not finite or the selected statistic cannot be
    /// computed.
    pub fn uncertainty_network_triplet(
        &self,
        attribute: NetworkSetAttribute,
        deviations: f64,
    ) -> Result<(Network, Network, Network)> {
        if !deviations.is_finite() {
            return Err(Error::Unsupported(
                "uncertainty deviations must be finite".to_owned(),
            ));
        }
        let (mean, standard_deviation) = match attribute {
            NetworkSetAttribute::Scattering => (self.mean_s()?, self.std_s()?),
            NetworkSetAttribute::Magnitude => (self.mean_s_magnitude()?, self.std_s_magnitude()?),
            NetworkSetAttribute::PhaseDegrees => {
                (self.mean_s_phase_degrees()?, self.std_s_phase_degrees()?)
            }
            NetworkSetAttribute::Decibel => (
                self.mean_parameter_component(
                    NetworkParameter::Scattering,
                    NetworkScalarAttribute::Decibel,
                )?,
                self.std_parameter_component(
                    NetworkParameter::Scattering,
                    NetworkScalarAttribute::Decibel,
                )?,
            ),
            NetworkSetAttribute::Decibel10 => (
                self.mean_parameter_component(
                    NetworkParameter::Scattering,
                    NetworkScalarAttribute::Decibel10,
                )?,
                self.std_parameter_component(
                    NetworkParameter::Scattering,
                    NetworkScalarAttribute::Decibel10,
                )?,
            ),
            NetworkSetAttribute::Real => (
                self.mean_parameter_component(
                    NetworkParameter::Scattering,
                    NetworkScalarAttribute::Real,
                )?,
                self.std_parameter_component(
                    NetworkParameter::Scattering,
                    NetworkScalarAttribute::Real,
                )?,
            ),
            NetworkSetAttribute::Imaginary => (
                self.mean_parameter_component(
                    NetworkParameter::Scattering,
                    NetworkScalarAttribute::Imaginary,
                )?,
                self.std_parameter_component(
                    NetworkParameter::Scattering,
                    NetworkScalarAttribute::Imaginary,
                )?,
            ),
            NetworkSetAttribute::Vswr => (
                self.mean_parameter_component(
                    NetworkParameter::Scattering,
                    NetworkScalarAttribute::Vswr,
                )?,
                self.std_parameter_component(
                    NetworkParameter::Scattering,
                    NetworkScalarAttribute::Vswr,
                )?,
            ),
        };
        let deviation = standard_deviation.s.mapv(|value| value * deviations);
        let mut lower = mean.clone();
        lower.s = &mean.s - &deviation;
        lower.name = self.name.as_ref().map(|name| format!("{name}-lower-bound"));
        let mut upper = mean.clone();
        upper.s = &mean.s + &deviation;
        upper.name = self.name.as_ref().map(|name| format!("{name}-upper-bound"));
        Ok((mean, lower, upper))
    }

    /// Adds independent Gaussian magnitude and phase noise characterized by this set.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.add_polar_noise`.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty, the target shape differs, or noise generation fails.
    pub fn add_polar_noise(&self, network: &Network) -> Result<Network> {
        let first = self.first()?;
        if network.frequency != first.frequency || network.s.dim() != first.s.dim() {
            return Err(Error::IncompatibleShape(
                "noise target must match the NetworkSet frequency and port shape".to_owned(),
            ));
        }
        let magnitude_deviation = self.std_s_magnitude()?.s.mapv(|value| value.re);
        let phase_deviation = self.std_s_phase_degrees()?.s.mapv(|value| value.re);
        let magnitude_noise = crate::math::random_normal_like(&magnitude_deviation)?;
        let phase_noise = crate::math::random_normal_like(&phase_deviation)?;
        let mut noisy = network.clone();
        for (value, (magnitude_noise, phase_noise)) in noisy
            .s
            .iter_mut()
            .zip(magnitude_noise.iter().zip(phase_noise.iter()))
        {
            *value = Complex64::from_polar(
                value.norm() + magnitude_noise,
                (value.arg().to_degrees() + phase_noise).to_radians(),
            );
        }
        Ok(noisy)
    }

    /// Parses every network name as a sortable scikit-rf timestamp.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.datetime_index`.
    ///
    /// # Errors
    ///
    /// Returns an error if any network is unnamed or its name is not a supported timestamp.
    pub fn datetime_index(&self) -> Result<Vec<NaiveDateTime>> {
        self.networks
            .iter()
            .map(|network| {
                let name = network.name.as_deref().ok_or_else(|| {
                    Error::Unsupported(
                        "all networks must be named to build a datetime index".to_owned(),
                    )
                })?;
                crate::util::parse_now_string(name)
            })
            .collect()
    }

    /// Writes this set using the safe Rust object format.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.write`.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination exists without overwrite permission, serialization
    /// fails, or the file cannot be written.
    pub fn write_to_path(&self, path: impl AsRef<Path>, overwrite: bool) -> Result<PathBuf> {
        crate::io::write_object(path, &StoredObject::NetworkSet(self.clone()), overwrite)
    }

    /// Writes this set using its name as the file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is unnamed or writing the object fails.
    pub fn write_named(&self, overwrite: bool) -> Result<PathBuf> {
        let name = self.name.as_deref().ok_or_else(|| {
            Error::Unsupported("an unnamed NetworkSet needs an explicit output path".to_owned())
        })?;
        self.write_to_path(name, overwrite)
    }

    /// Writes a Generalized MDIF representation.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.write_mdif`.
    ///
    /// # Errors
    ///
    /// Returns an error if the set cannot be represented as MDIF or the file cannot be written.
    pub fn write_mdif(&self, path: impl AsRef<Path>, comments: &[String]) -> Result<()> {
        Mdif::write_to_path(self, path, comments)
    }

    /// Writes one worksheet per network.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.write_spreadsheet`.
    ///
    /// # Errors
    ///
    /// Returns an error if network data cannot be converted or the workbook cannot be written.
    #[cfg(feature = "xlsx")]
    pub fn write_spreadsheet(
        &self,
        path: impl AsRef<Path>,
        format: crate::io::NetworkDataFormat,
    ) -> Result<()> {
        crate::io::general::write_network_set_xlsx(self, path, format)
    }

    /// Port of `NetworkSet.scalar_mat` for scattering parameters.
    ///
    /// Axes are frequency, observation, and column-major port/re-imaginary
    /// component index.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty.
    pub fn scalar_s_matrix(&self) -> Result<Array3<f64>> {
        let first = self.first()?;
        let components = 2 * first.ports() * first.ports();
        let mut scalar = Array3::zeros((first.frequency_points(), self.networks.len(), components));
        for point in 0..first.frequency_points() {
            for (observation, network) in self.networks.iter().enumerate() {
                let mut component = 0;
                for column in 0..first.ports() {
                    for row in 0..first.ports() {
                        let value = network.s[(point, row, column)];
                        scalar[(point, observation, component)] = value.re;
                        scalar[(point, observation, component + 1)] = value.im;
                        component += 2;
                    }
                }
            }
        }
        Ok(scalar)
    }

    /// Port of `NetworkSet.cov` using `NumPy`'s sample-covariance convention.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than two networks are available or scalar projection fails.
    pub fn covariance_s(&self) -> Result<Array3<f64>> {
        if self.networks.len() < 2 {
            return Err(Error::IncompatibleShape(
                "sample covariance requires at least two networks".to_owned(),
            ));
        }
        let scalar = self.scalar_s_matrix()?;
        let (points, observations, components) = scalar.dim();
        let mut covariance = Array3::zeros((points, components, components));
        for point in 0..points {
            let means = (0..components)
                .map(|component| {
                    (0..observations)
                        .map(|observation| scalar[(point, observation, component)])
                        .sum::<f64>()
                        / observations.to_f64().unwrap_or(f64::INFINITY)
                })
                .collect::<Vec<_>>();
            for row in 0..components {
                for column in 0..components {
                    covariance[(point, row, column)] = (0..observations)
                        .map(|observation| {
                            (scalar[(point, observation, row)] - means[row])
                                * (scalar[(point, observation, column)] - means[column])
                        })
                        .sum::<f64>()
                        / (observations - 1).to_f64().unwrap_or(f64::INFINITY);
                }
            }
        }
        Ok(covariance)
    }

    /// Interpolates a network using the first numeric set parameter.
    ///
    /// # Errors
    ///
    /// Returns an error unless exactly one parameter axis exists or interpolation fails.
    pub fn interpolate_from_network(&self, parameter: f64) -> Result<Network> {
        if self.parameters.len() != 1 {
            return Err(Error::Unsupported(
                "parameter interpolation requires exactly one NetworkSet parameter axis".to_owned(),
            ));
        }
        let values = self.parameters.values().next().ok_or_else(|| {
            Error::Unsupported(
                "parameter interpolation requires exactly one NetworkSet parameter axis".to_owned(),
            )
        })?;
        self.interpolate_from_values(values, parameter)
    }

    /// Port of `skrf.networkSet.NetworkSet.interpolate_from_network` with the
    /// upstream `ntw_param` argument represented explicitly.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty set, mismatched or non-finite values, duplicate coordinates,
    /// an out-of-range target, or a network construction failure.
    pub fn interpolate_from_values(&self, values: &[f64], target: f64) -> Result<Network> {
        self.first()?;
        if values.len() != self.networks.len() || values.len() < 2 {
            return Err(Error::IncompatibleShape(format!(
                "{} interpolation values were supplied for {} networks",
                values.len(),
                self.networks.len()
            )));
        }
        if !target.is_finite() || values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Unsupported(
                "network interpolation requires finite parameter values".to_owned(),
            ));
        }
        let mut ordered = values
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (value, index))
            .collect::<Vec<_>>();
        ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
        if ordered.windows(2).any(|pair| {
            (pair[0].0 - pair[1].0).abs()
                <= f64::EPSILON * pair[0].0.abs().max(pair[1].0.abs()).max(1.0)
        }) {
            return Err(Error::Unsupported(
                "network interpolation parameter values must be unique".to_owned(),
            ));
        }
        if target < ordered[0].0 || target > ordered[ordered.len() - 1].0 {
            return Err(Error::Unsupported(
                "network interpolation target lies outside the parameter range".to_owned(),
            ));
        }
        if let Some((_, index)) = ordered.iter().find(|(value, _)| {
            (*value - target).abs() <= f64::EPSILON * value.abs().max(target.abs()).max(1.0)
        }) {
            return Ok(self.networks[*index].clone());
        }
        let upper = ordered
            .iter()
            .position(|(value, _)| *value > target)
            .ok_or_else(|| {
                Error::Unsupported(
                    "network interpolation target lies outside the parameter range".to_owned(),
                )
            })?;
        let (lower_value, lower_index) = ordered[upper - 1];
        let (upper_value, upper_index) = ordered[upper];
        let fraction = (target - lower_value) / (upper_value - lower_value);
        let lower = &self.networks[lower_index];
        let upper = &self.networks[upper_index];
        let s = &lower.s + &((&upper.s - &lower.s) * fraction);
        self.derived_network(s, "interpolated")
    }

    /// Converts the parameterized set into a row-oriented `DataFrame`.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty, has no parameters, parameter lengths differ from the
    /// network count, or `DataFrame` construction fails.
    #[cfg(feature = "dataframe")]
    pub fn to_dataframe(&self) -> Result<polars::frame::DataFrame> {
        use polars::prelude::Column;

        let first = self.first()?;
        if self.parameters.is_empty() {
            return Err(Error::Unsupported(
                "a NetworkSet must have parameters before conversion to a DataFrame".to_owned(),
            ));
        }
        for (name, values) in &self.parameters {
            if values.len() != self.networks.len() {
                return Err(Error::IncompatibleShape(format!(
                    "parameter {name} contains {} values for {} networks",
                    values.len(),
                    self.networks.len()
                )));
            }
        }
        let rows_per_network = first.frequency_points() * first.ports() * first.ports();
        let row_count = rows_per_network * self.networks.len();
        let mut network_index = Vec::with_capacity(row_count);
        let mut frequency_hz = Vec::with_capacity(row_count);
        let mut output_port = Vec::with_capacity(row_count);
        let mut input_port = Vec::with_capacity(row_count);
        let mut s_real = Vec::with_capacity(row_count);
        let mut s_imag = Vec::with_capacity(row_count);
        for (network_number, network) in self.networks.iter().enumerate() {
            for point in 0..network.frequency_points() {
                for output in 0..network.ports() {
                    for input in 0..network.ports() {
                        network_index.push(network_number as u64);
                        frequency_hz.push(network.frequency.values_hz()[point]);
                        output_port.push(output as u64);
                        input_port.push(input as u64);
                        s_real.push(network.s[(point, output, input)].re);
                        s_imag.push(network.s[(point, output, input)].im);
                    }
                }
            }
        }
        let mut columns =
            Vec::with_capacity(self.parameters.len() + self.text_parameters.len() + 6);
        for (name, values) in &self.parameters {
            let repeated = values
                .iter()
                .flat_map(|value| std::iter::repeat_n(*value, rows_per_network))
                .collect::<Vec<_>>();
            columns.push(Column::new(name.as_str().into(), repeated));
        }
        for (name, values) in &self.text_parameters {
            if values.len() != self.networks.len() {
                return Err(Error::IncompatibleShape(format!(
                    "text parameter {name} contains {} values for {} networks",
                    values.len(),
                    self.networks.len()
                )));
            }
            let repeated = values
                .iter()
                .flat_map(|value| std::iter::repeat_n(value.clone(), rows_per_network))
                .collect::<Vec<_>>();
            columns.push(Column::new(name.as_str().into(), repeated));
        }
        columns.extend([
            Column::new("network_index".into(), network_index),
            Column::new("frequency_hz".into(), frequency_hz),
            Column::new("output_port".into(), output_port),
            Column::new("input_port".into(), input_port),
            Column::new("s_real".into(), s_real),
            Column::new("s_imag".into(), s_imag),
        ]);
        polars::frame::DataFrame::new(row_count, columns)
            .map_err(|error| Error::Unsupported(format!("DataFrame construction failed: {error}")))
    }

    /// Exports one scalar port component per named network.
    ///
    /// Origin: `skrf.networkSet.NetworkSet.ntwk_attr_2_df`.
    ///
    /// # Errors
    ///
    /// Returns an error if the set is empty, either port is invalid, or `DataFrame` construction
    /// fails.
    #[cfg(feature = "dataframe")]
    pub fn network_attribute_dataframe(
        &self,
        attribute: NetworkScalarAttribute,
        output_port: usize,
        input_port: usize,
    ) -> Result<polars::frame::DataFrame> {
        use polars::prelude::Column;

        let first = self.first()?;
        if output_port >= first.ports() || input_port >= first.ports() {
            return Err(Error::InvalidPort {
                port: output_port.max(input_port),
                ports: first.ports(),
            });
        }
        let mut columns = vec![Column::new(
            format!("Freq({})", first.frequency.unit().symbol()).into(),
            first.frequency.scaled().to_vec(),
        )];
        for (index, network) in self.networks.iter().enumerate() {
            let name = network
                .name
                .clone()
                .unwrap_or_else(|| format!("Network{index}"));
            let values = (0..network.frequency_points())
                .map(|point| {
                    let value = network.s[(point, output_port, input_port)];
                    scalar_component(value, attribute)
                })
                .collect::<Vec<_>>();
            columns.push(Column::new(name.into(), values));
        }
        polars::frame::DataFrame::new(first.frequency_points(), columns)
            .map_err(|error| Error::Unsupported(format!("DataFrame construction failed: {error}")))
    }

    fn scalar_parameter_statistic(
        &self,
        parameter: NetworkParameter,
        component: NetworkScalarAttribute,
        standard_deviation: bool,
    ) -> Result<Network> {
        let first = self.first()?;
        let projected = self
            .networks
            .iter()
            .map(|network| {
                Ok(parameter_values(network, parameter)?
                    .mapv(|value| scalar_component(value, component)))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut mean = Array3::<f64>::zeros(first.s.dim());
        for values in &projected {
            mean += values;
        }
        mean.mapv_inplace(|value| value / projected.len().to_f64().unwrap_or(f64::INFINITY));
        let values = if standard_deviation {
            let mut variance = Array3::<f64>::zeros(first.s.dim());
            for values in &projected {
                for (variance, (value, mean)) in
                    variance.iter_mut().zip(values.iter().zip(mean.iter()))
                {
                    *variance += (*value - *mean).powi(2);
                }
            }
            variance
                .mapv(|value| (value / projected.len().to_f64().unwrap_or(f64::INFINITY)).sqrt())
        } else {
            mean
        };
        self.derived_network(
            values.mapv(|value| Complex64::new(value, 0.0)),
            &format!(
                "{}-{parameter:?}-{component:?}",
                if standard_deviation { "std" } else { "mean" }
            ),
        )
    }

    fn first(&self) -> Result<&Network> {
        self.networks.first().ok_or_else(|| {
            Error::IncompatibleShape("an empty NetworkSet has no aggregate".to_owned())
        })
    }

    fn validate_parameters(&self) -> Result<()> {
        for (name, values) in &self.parameters {
            if values.len() != self.networks.len() {
                return Err(Error::IncompatibleShape(format!(
                    "parameter {name} contains {} values for {} networks",
                    values.len(),
                    self.networks.len()
                )));
            }
        }
        for (name, values) in &self.text_parameters {
            if values.len() != self.networks.len() {
                return Err(Error::IncompatibleShape(format!(
                    "text parameter {name} contains {} values for {} networks",
                    values.len(),
                    self.networks.len()
                )));
            }
        }
        Ok(())
    }

    fn select_indices(&self, indices: &[usize]) -> Result<Self> {
        let networks = indices
            .iter()
            .map(|index| self.networks[*index].clone())
            .collect::<Vec<_>>();
        let mut selected = Self::new(networks, self.name.clone())?;
        selected.parameters = self
            .parameters
            .iter()
            .map(|(name, values)| {
                (
                    name.clone(),
                    indices.iter().map(|index| values[*index]).collect(),
                )
            })
            .collect();
        selected.text_parameters = self
            .text_parameters
            .iter()
            .map(|(name, values)| {
                (
                    name.clone(),
                    indices.iter().map(|index| values[*index].clone()).collect(),
                )
            })
            .collect();
        Ok(selected)
    }

    fn derived_network(&self, s: Array3<Complex64>, operation: &str) -> Result<Network> {
        let first = self.first()?;
        let mut network = Network::new(first.frequency.clone(), s, first.z0.clone())?;
        network.name = self.name.as_ref().map(|name| format!("{name}-{operation}"));
        network.comments.clone_from(&first.comments);
        network.port_names.clone_from(&first.port_names);
        network.variables = first.variables.clone();
        network.s_definition = first.s_definition;
        Ok(network)
    }
}

fn parameter_values(network: &Network, parameter: NetworkParameter) -> Result<Array3<Complex64>> {
    match parameter {
        NetworkParameter::Scattering => Ok(network.s.clone()),
        NetworkParameter::Impedance => network.impedance(),
        NetworkParameter::Admittance => network.admittance(),
        NetworkParameter::Abcd => network.abcd(),
        NetworkParameter::InverseHybrid => network.inverse_hybrid(),
        NetworkParameter::Hybrid => network.hybrid(),
        NetworkParameter::ScatteringTransfer => network.scattering_transfer(),
    }
}

fn scalar_component(value: Complex64, component: NetworkScalarAttribute) -> f64 {
    match component {
        NetworkScalarAttribute::Magnitude => value.norm(),
        NetworkScalarAttribute::Decibel => 20.0 * value.norm().log10(),
        NetworkScalarAttribute::Decibel10 => 10.0 * value.norm().log10(),
        NetworkScalarAttribute::PhaseDegrees => value.arg().to_degrees(),
        NetworkScalarAttribute::Real => value.re,
        NetworkScalarAttribute::Imaginary => value.im,
        NetworkScalarAttribute::Vswr => (1.0 + value.norm()) / (1.0 - value.norm()),
    }
}

/// Applies a typed aggregate to the scattering matrices of compatible networks.
///
/// Origin: `skrf.networkSet.func_on_networks` (`fon`).
///
/// # Errors
///
/// Returns an error if the input is empty or incompatible, the aggregate fails, or its output
/// shape differs from the input shape.
pub fn function_on_networks(
    networks: &[Network],
    name: Option<String>,
    function: impl FnOnce(&[Array3<Complex64>]) -> Result<Array3<Complex64>>,
) -> Result<Network> {
    let set = NetworkSet::new(networks.to_vec(), None)?;
    let first = set.first()?;
    let matrices = networks
        .iter()
        .map(|network| network.s.clone())
        .collect::<Vec<_>>();
    let scattering = function(&matrices)?;
    if scattering.dim() != first.s.dim() {
        return Err(Error::IncompatibleShape(format!(
            "aggregate returned {:?}, expected {:?}",
            scattering.dim(),
            first.s.dim()
        )));
    }
    let mut network = first.clone();
    network.s = scattering;
    if name.is_some() {
        network.name = name;
    }
    Ok(network)
}

/// Selects dictionary entries whose key contains a substring.
///
/// Origin: `skrf.networkSet.getset`.
///
/// # Errors
///
/// Returns an error if the selected networks do not share a port count and frequency axis.
pub fn get_set(
    networks: &BTreeMap<String, Network>,
    needle: &str,
    name: Option<String>,
) -> Result<Option<NetworkSet>> {
    let selected = networks
        .iter()
        .filter(|(key, _)| key.contains(needle))
        .map(|(_, network)| network.clone())
        .collect::<Vec<_>>();
    if selected.is_empty() {
        Ok(None)
    } else {
        NetworkSet::new(selected, name).map(Some)
    }
}

/// Builds a one-port tuner constellation over radial and angular grids.
///
/// The tuple contains the network set, real coordinates, imaginary coordinates,
/// and complex reflection coefficients. Origin: `skrf.networkSet.tuner_constellation`.
///
/// # Errors
///
/// Returns an error for zero grid dimensions, invalid frequency or impedance, or a failure to
/// construct the frequency axis or networks.
pub fn tuner_constellation(
    name: &str,
    frequency_hz: f64,
    reference_impedance: f64,
    radial_points: usize,
    angular_points: usize,
) -> Result<TunerConstellation> {
    if radial_points == 0 || angular_points == 0 {
        return Err(Error::IncompatibleShape(
            "a tuner constellation needs radial and angular points".to_owned(),
        ));
    }
    if !frequency_hz.is_finite()
        || frequency_hz <= 0.0
        || !reference_impedance.is_finite()
        || reference_impedance <= 0.0
    {
        return Err(Error::Unsupported(
            "tuner frequency and reference impedance must be positive and finite".to_owned(),
        ));
    }
    let radial_denominator = radial_points
        .saturating_sub(1)
        .max(1)
        .to_f64()
        .unwrap_or(f64::INFINITY);
    let angular_denominator = angular_points
        .saturating_sub(1)
        .max(1)
        .to_f64()
        .unwrap_or(f64::INFINITY);
    let mut gamma = Vec::with_capacity(radial_points * angular_points);
    for angle_index in 0..angular_points {
        let angle = std::f64::consts::TAU * angle_index.to_f64().unwrap_or(f64::INFINITY)
            / angular_denominator;
        for radial_index in 0..radial_points {
            let radius = if radial_points == 1 {
                0.1
            } else {
                0.1 + 0.8 * radial_index.to_f64().unwrap_or(f64::INFINITY) / radial_denominator
            };
            gamma.push(Complex64::from_polar(radius, angle));
        }
    }
    let frequency = Frequency::from_hz(Array1::from_vec(vec![frequency_hz]))?;
    let z0 = Array2::from_elem((1, 1), Complex64::new(reference_impedance, 0.0));
    let networks = gamma
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let mut network = Network::new(
                frequency.clone(),
                Array3::from_elem((1, 1, 1), *value),
                z0.clone(),
            )?;
            network.name = Some(format!("{name}_{index}"));
            Ok(network)
        })
        .collect::<Result<Vec<_>>>()?;
    let gamma = Array1::from_vec(gamma);
    let real = gamma.mapv(|value| value.re);
    let imaginary = gamma.mapv(|value| value.im);
    Ok(TunerConstellation {
        networks: NetworkSet::new(networks, Some(name.to_owned()))?,
        real,
        imaginary,
        reflection: gamma,
    })
}
