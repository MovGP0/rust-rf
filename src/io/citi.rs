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
///
/// Origin: `skrf/io/citi.py::Citi._parse_citi`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CitiFormat {
    RealImaginary,
    MagnitudeAngle,
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
///
/// Origin: `skrf/io/citi.py::Citi._params`.
#[derive(Clone, Debug, PartialEq)]
pub struct CitiVariable {
    pub name: String,
    pub format: String,
    pub values: Vec<f64>,
}

/// One CITI `DATA` declaration and its decoded complex values.
///
/// Origin: `skrf/io/citi.py::Citi._data`.
#[derive(Clone, Debug, PartialEq)]
pub struct CitiData {
    pub name: String,
    pub format: CitiFormat,
    pub values: Vec<Complex64>,
}

/// Reader for Common Instrumentation Transfer and Interchange files.
///
/// Origin: `skrf/io/citi.py::Citi`.
#[derive(Clone, Debug, Default)]
pub struct Citi {
    pub filename: Option<PathBuf>,
    pub comments: Vec<String>,
    pub name: String,
    pub variables: Vec<CitiVariable>,
    pub data: Vec<CitiData>,
}

impl Citi {
    /// Port of `Citi.__init__` for a filesystem path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)?;
        let mut citi = Self::parse(&text)?;
        citi.filename = Some(path.to_path_buf());
        Ok(citi)
    }

    /// Port of `Citi.__init__` for a file-like object.
    pub fn from_reader(mut reader: impl Read) -> Result<Self> {
        let mut text = String::new();
        reader.read_to_string(&mut text)?;
        Self::parse(&text)
    }

    /// Parse a CITI document already held in memory.
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
                citi.name = trimmed[5..].to_owned();
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
                index = end - 1;
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

    /// Named variables excluding the mandatory frequency variable.
    pub fn parameters(&self) -> Vec<&str> {
        self.variables
            .iter()
            .filter(|variable| !variable.name.eq_ignore_ascii_case("freq"))
            .map(|variable| variable.name.as_str())
            .collect()
    }

    /// Port of `Citi.networks`.
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

        let (parameter_prefix, parameter_entries) = if self
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

        let rank = if parameter_entries.is_empty() && parameter_prefix == 'S' {
            1
        } else {
            let maximum_port = parameter_entries
                .iter()
                .flat_map(|(row, column, _)| [*row, *column])
                .max()
                .ok_or_else(|| {
                    Error::Parse("CITI network rank could not be determined".to_owned())
                })?;
            if parameter_entries.len() != maximum_port * maximum_port {
                return Err(Error::Parse(format!(
                    "CITI {parameter_prefix}-parameter matrix is incomplete"
                )));
            }
            maximum_port
        };

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

    /// Port of `Citi.to_networkset`.
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

    fn matrix_entries(&self, prefix: char) -> Vec<(usize, usize, &CitiData)> {
        self.data
            .iter()
            .filter_map(|entry| {
                matrix_coordinates(&entry.name, prefix).map(|(row, column)| (row, column, entry))
            })
            .collect()
    }
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
