//! Common Instrumentation Transfer and Interchange (CITI) file support.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

use crate::network::z_to_s;
use crate::{Error, Frequency, Network, NetworkSet, Result, SParameterDefinition};

/// Numeric representation used by a CITI `DATA` block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CitiFormat {
    /// Real and imaginary components (`RI`).
    RealImaginary,
    /// Linear magnitude and phase in degrees (`MAGANGLE`).
    MagnitudeAngle,
    /// Decibel magnitude and phase in degrees (`DBANGLE`).
    DecibelAngle,
}

impl CitiFormat {
    fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_uppercase().as_str() {
            "RI" => Ok(Self::RealImaginary),
            "MAGANGLE" => Ok(Self::MagnitudeAngle),
            "DBANGLE" => Ok(Self::DecibelAngle),
            other => Err(Error::Unsupported(format!(
                "CITI data format '{other}' is not implemented"
            ))),
        }
    }

    fn decode(self, first: f64, second: f64) -> Complex64 {
        match self {
            Self::RealImaginary => Complex64::new(first, second),
            Self::MagnitudeAngle => Complex64::from_polar(first, second.to_radians()),
            Self::DecibelAngle => {
                Complex64::from_polar(10.0_f64.powf(first / 20.0), second.to_radians())
            }
        }
    }
}

/// One CITI `VAR` declaration and its `VAR_LIST_BEGIN` values.
#[derive(Clone, Debug, PartialEq)]
pub struct CitiVariable {
    /// Variable name; the mandatory frequency variable is normalized to `freq`.
    pub name: String,
    /// Format token from the `VAR` declaration.
    pub format: String,
    /// Variable values in declaration order.
    pub values: Vec<f64>,
}

/// One CITI `DATA` declaration and its decoded complex values.
#[derive(Clone, Debug, PartialEq)]
pub struct CitiData {
    /// Data name, such as `S[1,1]` or `PortZ[1]`.
    pub name: String,
    /// Numeric representation used by the data block.
    pub format: CitiFormat,
    /// Complex values decoded from the data block.
    pub values: Vec<Complex64>,
}

type CitiMatrixEntry<'a> = (usize, usize, &'a CitiData);
type CitiNetworkParameterEntries<'a> = (char, Vec<CitiMatrixEntry<'a>>, usize);

/// Reader for CITI N-port files.
///
/// CITI (`.cti`) is a standardized format for exchanging data between
/// computers and instruments. A document may describe one network or a family
/// of networks indexed by named variables.
///
/// # Example
///
/// ```no_run
/// use rust_rf::io::Citi;
///
/// # fn main() -> rust_rf::Result<()> {
/// let citi = Citi::from_path("network.cti")?;
/// let networks = citi.networks()?;
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// - [Keysight CITIfile format](https://na.support.keysight.com/plts/help/WebHelp/FilePrint/CITIfile_Format.htm)
/// - J. P. Dunsmore, *Handbook of Microwave Component Measurements: With
///   Advanced VNA Techniques*, 2nd ed., section 6.1.6.1, 2020.
#[derive(Clone, Debug, Default)]
pub struct Citi {
    /// Source path when the document was read with [`Self::from_path`].
    pub filename: Option<PathBuf>,
    /// Comment lines beginning with `#` or `!`.
    pub comments: Vec<String>,
    /// Name of the CITI package.
    pub name: String,
    /// Declared variables in file order.
    pub variables: Vec<CitiVariable>,
    /// Declared and decoded data blocks in file order.
    pub data: Vec<CitiData>,
}

impl Citi {
    /// Reads and parses a CITI document from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read as UTF-8 text or the CITI
    /// document is malformed or unsupported.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)?;
        let mut citi = Self::parse(&text)?;
        citi.filename = Some(path.to_path_buf());
        Ok(citi)
    }

    /// Reads and parses a CITI document from a byte reader.
    ///
    /// # Errors
    ///
    /// Returns an error when the reader cannot supply UTF-8 text or the CITI
    /// document is malformed or unsupported.
    pub fn from_reader(mut reader: impl Read) -> Result<Self> {
        let mut text = String::new();
        reader.read_to_string(&mut text)?;
        Self::parse(&text)
    }

    /// Parses a CITI document already held in memory.
    ///
    /// `RI`, `MAGANGLE`, and `DBANGLE` data formats are supported. Unsupported
    /// numeric formats and malformed declarations return an error.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed declarations, unmatched `VAR_LIST_BEGIN`
    /// or `BEGIN` blocks, unsupported data formats, invalid numeric values, or
    /// overflowing block sizes.
    pub fn parse(text: &str) -> Result<Self> {
        let lines = text.lines().collect::<Vec<_>>();
        let mut citi = Self::default();
        let mut pending_variables = Vec::new();
        let mut pending_data = Vec::new();
        let mut index = 0;

        while index < lines.len() {
            let line = lines[index];
            let trimmed = line.trim();
            let upper = trimmed.to_ascii_uppercase();

            if trimmed.starts_with(['#', '!']) {
                citi.comments.push(line.to_owned());
            } else if upper.starts_with("NAME ") {
                trimmed[5..].clone_into(&mut citi.name);
            } else if upper.starts_with("VAR ") {
                let fields = trimmed.split_whitespace().collect::<Vec<_>>();
                if fields.len() != 4 {
                    return Err(Error::Parse(format!(
                        "invalid CITI variable declaration '{trimmed}'"
                    )));
                }
                let occurrences = parse_usize(fields[3], "variable occurrence count")?;
                let name = if fields[1].eq_ignore_ascii_case("freq") {
                    "freq".to_owned()
                } else {
                    fields[1].to_owned()
                };
                citi.variables.push(CitiVariable {
                    name,
                    format: fields[2].to_owned(),
                    values: Vec::with_capacity(occurrences),
                });
                pending_variables.push((citi.variables.len() - 1, occurrences));
            } else if upper.starts_with("DATA ") {
                let fields = trimmed.split_whitespace().collect::<Vec<_>>();
                if fields.len() != 3 {
                    return Err(Error::Parse(format!(
                        "invalid CITI data declaration '{trimmed}'"
                    )));
                }
                citi.data.push(CitiData {
                    name: fields[1].to_owned(),
                    format: CitiFormat::parse(fields[2])?,
                    values: Vec::new(),
                });
                pending_data.push(citi.data.len() - 1);
            } else if upper == "VAR_LIST_BEGIN" {
                let (variable_index, occurrences) =
                    pending_variables.first().copied().ok_or_else(|| {
                        Error::Parse(
                            "CITI variable list has no matching VAR declaration".to_owned(),
                        )
                    })?;
                pending_variables.remove(0);
                let end = index
                    .checked_add(occurrences)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| Error::Parse("CITI variable list is too large".to_owned()))?;
                if end > lines.len() {
                    return Err(Error::Parse(format!(
                        "CITI variable '{}' expected {occurrences} values",
                        citi.variables[variable_index].name
                    )));
                }
                for value in &lines[index + 1..end] {
                    citi.variables[variable_index]
                        .values
                        .push(parse_float(value.trim())?);
                }
                index = end - 1;
            } else if upper == "BEGIN" {
                index = parse_citi_data_block(&lines, index, &mut citi, &mut pending_data)?;
            }
            index += 1;
        }

        if !pending_variables.is_empty() {
            return Err(Error::Parse(
                "one or more CITI variables have no value list".to_owned(),
            ));
        }
        if !pending_data.is_empty() {
            return Err(Error::Parse(
                "one or more CITI data declarations have no data block".to_owned(),
            ));
        }
        Ok(citi)
    }

    /// Returns named variables excluding the mandatory `freq` variable.
    #[must_use]
    pub fn parameters(&self) -> Vec<&str> {
        self.variables
            .iter()
            .filter(|variable| !variable.name.eq_ignore_ascii_case("freq"))
            .map(|variable| variable.name.as_str())
            .collect()
    }

    /// Returns all networks described by the CITI document.
    ///
    /// Named-variable combinations produce separate networks. S-parameter
    /// data is used directly; Z-parameter data is converted to S-parameters.
    /// Per-port `PortZ` data supplies reference impedances, otherwise 50 Ω is
    /// used.
    ///
    /// # Errors
    ///
    /// Returns an error when frequency or network data is missing, incomplete,
    /// inconsistent in shape, or cannot be converted into a valid network.
    pub fn networks(&self) -> Result<Vec<Network>> {
        let frequency_variable = self
            .variables
            .iter()
            .find(|variable| variable.name.eq_ignore_ascii_case("freq"))
            .ok_or_else(|| Error::Parse("CITI frequency points were not found".to_owned()))?;
        if frequency_variable.values.is_empty() {
            return Err(Error::InvalidFrequency(
                "a CITI frequency variable must contain at least one point".to_owned(),
            ));
        }
        let frequency = Frequency::from_hz(Array1::from_vec(frequency_variable.values.clone()))?;
        let parameter_variables = self
            .variables
            .iter()
            .filter(|variable| !variable.name.eq_ignore_ascii_case("freq"))
            .collect::<Vec<_>>();
        let parameter_sets = cartesian_parameter_sets(&parameter_variables);
        let set_count = parameter_sets.len();

        let (parameter_prefix, parameter_entries, rank) = self.network_parameter_entries()?;

        let points = frequency.points();
        let expected_values = set_count.checked_mul(points).ok_or_else(|| {
            Error::IncompatibleShape("CITI network data size overflowed".to_owned())
        })?;
        let mut parameter_values = Array3::zeros((expected_values, rank, rank));
        if rank == 1 && parameter_entries.is_empty() {
            let entry = self
                .data
                .iter()
                .find(|entry| entry.name == "S")
                .ok_or_else(|| Error::Parse("CITI scalar S data is missing".to_owned()))?;
            validate_value_count(entry, expected_values)?;
            for (index, value) in entry.values.iter().copied().enumerate() {
                parameter_values[(index, 0, 0)] = value;
            }
        } else {
            for (row, column, entry) in parameter_entries {
                validate_value_count(entry, expected_values)?;
                for (index, value) in entry.values.iter().copied().enumerate() {
                    parameter_values[(index, row - 1, column - 1)] = value;
                }
            }
        }

        let mut reference_impedance =
            Array2::from_elem((expected_values, rank), Complex64::new(50.0, 0.0));
        for port in 1..=rank {
            if let Some(entry) = self.data.iter().find(|entry| {
                port_coordinates(&entry.name).is_some_and(|entry_port| entry_port == port)
            }) {
                validate_value_count(entry, expected_values)?;
                for (index, value) in entry.values.iter().copied().enumerate() {
                    reference_impedance[(index, port - 1)] = value;
                }
            }
        }

        let mut networks = Vec::with_capacity(set_count);
        for (set_index, parameter_set) in parameter_sets.iter().enumerate() {
            let start = set_index * points;
            let mut raw = Array3::zeros((points, rank, rank));
            let mut z0 = Array2::zeros((points, rank));
            for point in 0..points {
                for row in 0..rank {
                    z0[(point, row)] = reference_impedance[(start + point, row)];
                    for column in 0..rank {
                        raw[(point, row, column)] = parameter_values[(start + point, row, column)];
                    }
                }
            }
            let s = if parameter_prefix == 'S' {
                raw
            } else {
                z_to_s(&raw, &z0, SParameterDefinition::Power)?
            };
            let mut network = Network::new(frequency.clone(), s, z0)?;
            network.variables = parameter_variables
                .iter()
                .zip(parameter_set)
                .map(|(variable, value)| (variable.name.clone(), value.to_string()))
                .collect::<BTreeMap<_, _>>();
            networks.push(network);
        }
        Ok(networks)
    }

    /// Converts the CITI data and named variables into a [`NetworkSet`].
    ///
    /// # Errors
    ///
    /// Returns an error when network conversion fails, the generated networks
    /// cannot form a set, or a named CITI variable has an incompatible value
    /// count.
    pub fn to_network_set(&self) -> Result<NetworkSet> {
        let parameter_variables = self
            .variables
            .iter()
            .filter(|variable| !variable.name.eq_ignore_ascii_case("freq"))
            .collect::<Vec<_>>();
        let parameter_sets = cartesian_parameter_sets(&parameter_variables);
        let mut set = NetworkSet::new(self.networks()?, Some(self.name.clone()))?;
        for (parameter_index, variable) in parameter_variables.iter().enumerate() {
            set.set_parameter(
                variable.name.clone(),
                parameter_sets
                    .iter()
                    .map(|values| values[parameter_index])
                    .collect(),
            )?;
        }
        Ok(set)
    }

    fn matrix_entries(&self, prefix: char) -> Vec<CitiMatrixEntry<'_>> {
        self.data
            .iter()
            .filter_map(|entry| {
                matrix_coordinates(&entry.name, prefix).map(|(row, column)| (row, column, entry))
            })
            .collect()
    }

    fn network_parameter_entries(&self) -> Result<CitiNetworkParameterEntries<'_>> {
        let (prefix, entries) = if self
            .data
            .iter()
            .any(|entry| matrix_coordinates(&entry.name, 'S').is_some() || entry.name == "S")
        {
            ('S', self.matrix_entries('S'))
        } else if self
            .data
            .iter()
            .any(|entry| matrix_coordinates(&entry.name, 'Z').is_some())
        {
            ('Z', self.matrix_entries('Z'))
        } else {
            return Err(Error::Unsupported(
                "no S or Z network parameters were found in the CITI file".to_owned(),
            ));
        };
        let rank = if entries.is_empty() && prefix == 'S' {
            1
        } else {
            let maximum_port = entries
                .iter()
                .flat_map(|(row, column, _)| [*row, *column])
                .max()
                .ok_or_else(|| {
                    Error::Parse("CITI network rank could not be determined".to_owned())
                })?;
            if entries.len() != maximum_port * maximum_port {
                return Err(Error::Parse(format!(
                    "CITI {prefix}-parameter matrix is incomplete"
                )));
            }
            maximum_port
        };
        Ok((prefix, entries, rank))
    }
}

fn parse_citi_data_block(
    lines: &[&str],
    index: usize,
    citi: &mut Citi,
    pending_data: &mut Vec<usize>,
) -> Result<usize> {
    let data_index = pending_data.first().copied().ok_or_else(|| {
        Error::Parse("CITI data block has no matching DATA declaration".to_owned())
    })?;
    pending_data.remove(0);
    let value_count = citi.variables.iter().try_fold(1_usize, |count, variable| {
        count
            .checked_mul(variable.values.len())
            .ok_or_else(|| Error::Parse("CITI data block size overflowed".to_owned()))
    })?;
    let end = index
        .checked_add(value_count)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| Error::Parse("CITI data block is too large".to_owned()))?;
    if end > lines.len() {
        return Err(Error::Parse(format!(
            "CITI data '{}' expected {value_count} values",
            citi.data[data_index].name
        )));
    }
    let format = citi.data[data_index].format;
    for value in &lines[index + 1..end] {
        let pair = value.split(',').map(str::trim).collect::<Vec<_>>();
        if pair.len() != 2 {
            return Err(Error::Parse(format!(
                "invalid CITI data pair '{}'",
                value.trim()
            )));
        }
        citi.data[data_index]
            .values
            .push(format.decode(parse_float(pair[0])?, parse_float(pair[1])?));
    }
    Ok(end - 1)
}

impl FromStr for Citi {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        Self::parse(text)
    }
}

fn cartesian_parameter_sets(variables: &[&CitiVariable]) -> Vec<Vec<f64>> {
    let mut sets = vec![Vec::new()];
    for variable in variables {
        let mut expanded = Vec::with_capacity(sets.len() * variable.values.len());
        for existing in &sets {
            for value in &variable.values {
                let mut next = existing.clone();
                next.push(*value);
                expanded.push(next);
            }
        }
        sets = expanded;
    }
    sets
}

fn matrix_coordinates(name: &str, prefix: char) -> Option<(usize, usize)> {
    let uppercase = name.to_ascii_uppercase();
    let body = uppercase
        .strip_prefix(prefix)?
        .strip_prefix('[')?
        .strip_suffix(']')?;
    let (row, column) = body.split_once(',')?;
    Some((row.trim().parse().ok()?, column.trim().parse().ok()?))
}

fn port_coordinates(name: &str) -> Option<usize> {
    let uppercase = name.to_ascii_uppercase();
    let body = uppercase.strip_prefix("PORTZ[")?.strip_suffix(']')?;
    body.trim().parse().ok()
}

fn validate_value_count(data: &CitiData, expected: usize) -> Result<()> {
    if data.values.len() == expected {
        Ok(())
    } else {
        Err(Error::IncompatibleShape(format!(
            "CITI data '{}' contains {} values, expected {expected}",
            data.name,
            data.values.len()
        )))
    }
}

fn parse_float(value: &str) -> Result<f64> {
    value
        .replace(['d', 'D'], "E")
        .parse::<f64>()
        .map_err(|error| Error::Parse(format!("invalid CITI number '{value}': {error}")))
}

fn parse_usize(value: &str, description: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|error| Error::Parse(format!("invalid CITI {description} '{value}': {error}")))
}
