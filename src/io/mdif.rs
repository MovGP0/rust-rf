//! Read and write Generalized MDIF N-port data.
//!
//! MDIF files store networks that vary with frequency and with one or more
//! named parameters. [`Mdif`] preserves those parameter coordinates and can
//! expose the collection as a [`crate::NetworkSet`].

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

use crate::network::{y_to_s, z_to_s};
use crate::{Error, Frequency, FrequencyUnit, Network, NetworkSet, Result, SParameterDefinition};

/// Value assigned by an MDIF `VAR` declaration.
///
/// Origin: `skrf/io/mdif.py::Mdif._parse_mdif`.
#[derive(Clone, Debug, PartialEq)]
pub enum MdifValue {
    /// A numeric parameter coordinate.
    Number(f64),
    /// A textual parameter coordinate.
    Text(String),
}

impl MdifValue {
    fn parse(value: &str) -> Self {
        value.trim().parse::<f64>().map_or_else(
            |_| Self::Text(value.trim().trim_matches('"').to_owned()),
            Self::Number,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MdifFormat {
    RealImaginary,
    MagnitudeAngle,
    DecibelAngle,
}

impl MdifFormat {
    fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "ri" => Ok(Self::RealImaginary),
            "ma" => Ok(Self::MagnitudeAngle),
            "db" => Ok(Self::DecibelAngle),
            other => Err(Error::Unsupported(format!(
                "MDIF data format '{other}' is not implemented"
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

/// Reader and writer for Generalized MDIF N-port files.
///
/// MDIF files store network parameters that vary with frequency and with one
/// or more named variables. Parsed data can be converted to a [`NetworkSet`]
/// for selection and interpolation by those variables.
///
/// The supported syntax follows the [AWR Generalized MDIF format].
///
/// [AWR Generalized MDIF format]: https://awrcorp.com/download/faq/english/docs/Users_Guide/data_file_formats.html#generalized_mdif
#[derive(Clone, Debug, Default)]
pub struct Mdif {
    /// Source file path, or `None` for input parsed from a reader or string.
    pub filename: Option<PathBuf>,
    /// File-level comments found before the first data block.
    pub comments: Vec<String>,
    /// Named variables declared by `VAR` records.
    pub parameters: Vec<String>,
    /// Parameter values associated with each network.
    pub parameter_values: Vec<BTreeMap<String, MdifValue>>,
    /// Networks stored in the MDIF data blocks.
    pub networks: Vec<Network>,
}

impl Mdif {
    /// Load an MDIF document from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or contains invalid or
    /// unsupported MDIF data.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)?;
        let mut mdif = Self::parse(&text)?;
        mdif.filename = Some(path.to_path_buf());
        Ok(mdif)
    }

    /// Load an MDIF document from a reader.
    ///
    /// # Errors
    ///
    /// Returns an error if the reader fails or supplies invalid or unsupported
    /// MDIF data.
    pub fn from_reader(mut reader: impl Read) -> Result<Self> {
        let mut text = String::new();
        reader.read_to_string(&mut text)?;
        Self::parse(&text)
    }

    /// Parse an MDIF document from text.
    ///
    /// `ACDATA` blocks are decoded as S, Z, or Y parameters in RI, MA, or DB
    /// form. Z and Y data are converted to scattering parameters. Associated
    /// `NDATA` blocks populate the network noise parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if declarations or data blocks are malformed,
    /// unsupported, incomplete, or inconsistent with the network dimensions.
    pub fn parse(text: &str) -> Result<Self> {
        let mut mdif = Self::default();
        let mut current_parameters = BTreeMap::new();
        let mut block = Vec::new();
        let mut in_data = false;
        let mut in_noise = false;
        let mut before_first_block = true;

        for line in text.lines() {
            let trimmed = line.trim();
            let lower = trimmed.to_ascii_lowercase();

            if before_first_block && trimmed.starts_with('!') {
                let comment = trimmed.trim_start_matches('!').trim();
                if !comment.is_empty() && !comment.chars().all(|character| character == '-') {
                    mdif.comments.push(comment.to_owned());
                }
            }

            if !in_data && !in_noise && lower.starts_with("var ") {
                let declaration = trimmed[3..].trim();
                let (raw_name, raw_value) = declaration.split_once('=').ok_or_else(|| {
                    Error::Parse(format!("invalid MDIF variable declaration '{trimmed}'"))
                })?;
                let name = raw_name
                    .split_once('(')
                    .map_or(raw_name, |(name, _)| name)
                    .trim()
                    .to_owned();
                if !mdif.parameters.contains(&name) {
                    mdif.parameters.push(name.clone());
                }
                current_parameters.insert(name, MdifValue::parse(raw_value));
                continue;
            }

            if lower.starts_with("begin ndata") {
                before_first_block = false;
                in_noise = true;
                block.clear();
                continue;
            }
            if lower.starts_with("begin") {
                before_first_block = false;
                in_data = true;
                block.clear();
                continue;
            }
            if lower.starts_with("end") {
                if in_data {
                    let mut network = parse_data_block(&block)?;
                    network.variables = current_parameters
                        .iter()
                        .map(|(name, value)| {
                            let value = match value {
                                MdifValue::Number(value) => value.to_string(),
                                MdifValue::Text(value) => value.clone(),
                            };
                            (name.clone(), value)
                        })
                        .collect();
                    mdif.networks.push(network);
                    mdif.parameter_values.push(current_parameters.clone());
                    current_parameters.clear();
                    in_data = false;
                } else if in_noise {
                    let network = mdif.networks.last_mut().ok_or_else(|| {
                        Error::Parse("MDIF NDATA block appears before ACDATA".to_owned())
                    })?;
                    parse_noise_block(&block, network)?;
                    in_noise = false;
                }
                block.clear();
                continue;
            }
            if in_data || in_noise {
                block.push(line.to_owned());
            }
        }

        if in_data || in_noise {
            return Err(Error::Parse(
                "unterminated MDIF data block at end of file".to_owned(),
            ));
        }
        Ok(mdif)
    }

    /// Return the parsed MDIF data as a network set.
    ///
    /// Named variables are copied to the network set as numeric or textual
    /// parameters. A parameter that mixes both kinds is rejected.
    ///
    /// # Errors
    ///
    /// Returns an error if the networks cannot form a set, a parameter value
    /// is missing, or a parameter mixes numeric and textual values.
    pub fn to_network_set(&self) -> Result<NetworkSet> {
        let mut set = NetworkSet::new(self.networks.clone(), None)?;
        for name in &self.parameters {
            let values = self
                .parameter_values
                .iter()
                .map(|parameters| {
                    parameters.get(name).cloned().ok_or_else(|| {
                        Error::Parse(format!("MDIF network is missing parameter '{name}'"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if values
                .iter()
                .all(|value| matches!(value, MdifValue::Number(_)))
            {
                set.set_parameter(
                    name.clone(),
                    values
                        .into_iter()
                        .map(|value| match value {
                            MdifValue::Number(value) => value,
                            MdifValue::Text(_) => unreachable!("numeric values were validated"),
                        })
                        .collect(),
                )?;
            } else if values
                .iter()
                .all(|value| matches!(value, MdifValue::Text(_)))
            {
                set.set_text_parameter(
                    name.clone(),
                    values
                        .into_iter()
                        .map(|value| match value {
                            MdifValue::Text(value) => value,
                            MdifValue::Number(_) => unreachable!("text values were validated"),
                        })
                        .collect(),
                )?;
            } else {
                return Err(Error::Unsupported(format!(
                    "MDIF parameter '{name}' mixes numeric and string values"
                )));
            }
        }
        Ok(set)
    }

    /// Write a network set to an MDIF file.
    ///
    /// Numeric and textual parameters come from the [`NetworkSet`]. Each item
    /// in `comments` becomes a separate file-level comment. Network data uses
    /// an explicit, round-trippable real/imaginary layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the output file cannot be created or written, or
    /// the network-set parameters have inconsistent lengths.
    pub fn write_to_path(
        network_set: &NetworkSet,
        path: impl AsRef<Path>,
        comments: &[String],
    ) -> Result<()> {
        let file = fs::File::create(path)?;
        Self::write(network_set, file, comments)
    }

    /// Write a network set as MDIF to an arbitrary writer.
    ///
    /// Each network receives its numeric and textual `VAR` declarations, an
    /// `ACDATA` block, and an optional `NDATA` noise block. Network data is
    /// written in hertz with real and imaginary components.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails or a network-set parameter does not
    /// contain exactly one value per network.
    pub fn write(
        network_set: &NetworkSet,
        mut writer: impl Write,
        comments: &[String],
    ) -> Result<()> {
        for comment in comments {
            writeln!(writer, "! {comment}")?;
        }
        for (network_index, network) in network_set.networks.iter().enumerate() {
            writeln!(writer, "!{}", "-".repeat(79))?;
            for (name, values) in &network_set.parameters {
                if values.len() != network_set.networks.len() {
                    return Err(Error::IncompatibleShape(format!(
                        "parameter {name} contains {} values for {} networks",
                        values.len(),
                        network_set.networks.len()
                    )));
                }
                writeln!(writer, "VAR {name} = {}", values[network_index])?;
            }
            for (name, values) in &network_set.text_parameters {
                if values.len() != network_set.networks.len() {
                    return Err(Error::IncompatibleShape(format!(
                        "text parameter {name} contains {} values for {} networks",
                        values.len(),
                        network_set.networks.len()
                    )));
                }
                writeln!(writer, "VAR {name} = \"{}\"", values[network_index])?;
            }
            writeln!(writer, "\nBEGIN ACDATA")?;
            let reference = network.z0[(0, 0)].re;
            writeln!(writer, "# Hz S RI R {reference}")?;
            if let Some(name) = &network.name {
                writeln!(writer, "! network name: {name}")?;
            }
            write!(writer, "% F")?;
            for row in 0..network.ports() {
                for column in 0..network.ports() {
                    write!(writer, " S[{},{}](complex)", row + 1, column + 1)?;
                }
            }
            writeln!(writer)?;
            for point in 0..network.frequency_points() {
                write!(writer, "{:.17e}", network.frequency.values_hz()[point])?;
                for row in 0..network.ports() {
                    for column in 0..network.ports() {
                        let value = network.s[(point, row, column)];
                        write!(writer, " {:.17e} {:.17e}", value.re, value.im)?;
                    }
                }
                writeln!(writer)?;
            }
            if let Some(noise) = &network.noise {
                writeln!(writer, "END\n\nBEGIN NDATA")?;
                writeln!(writer, "%F nfmin n11x n11y rn")?;
                writeln!(writer, "# Hz S MA R {reference}")?;
                let frequencies = noise.frequency.values_hz();
                for point in 0..noise.frequency.points() {
                    let gamma = noise.optimal_reflection[point];
                    writeln!(
                        writer,
                        "{:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                        frequencies[point],
                        noise.minimum_noise_figure_db[point],
                        gamma.norm(),
                        gamma.arg().to_degrees(),
                        noise.equivalent_noise_resistance[point] / reference
                    )?;
                }
            }
            writeln!(writer, "END\n")?;
        }
        Ok(())
    }

    /// Create the MDIF field-description string for `ports` ports.
    ///
    /// Two-port fields use the conventional $S_{11}, S_{21}, S_{12}, S_{22}$
    /// ordering. Ports above nine are separated with an underscore, and long
    /// records are wrapped after the field counts permitted by Touchstone.
    #[must_use]
    pub fn option_string(ports: usize) -> String {
        let mut output = "%F ".to_owned();
        let coordinates = if ports == 2 {
            vec![(1, 1), (2, 1), (1, 2), (2, 2)]
        } else {
            (1..=ports)
                .flat_map(|row| (1..=ports).map(move |column| (row, column)))
                .collect()
        };
        for (index, (row, column)) in coordinates.into_iter().enumerate() {
            if ports > 9 {
                let _ = write!(&mut output, "n{row}_{column}x n{row}_{column}y");
            } else {
                let _ = write!(&mut output, "n{row}{column}x n{row}{column}y");
            }
            if index + 1 < ports * ports {
                output.push(' ');
            }
            if (ports == 3 && column == 3) || (ports >= 4 && column % 4 == 0) {
                output.push('\n');
            }
        }
        output
    }
}

impl FromStr for Mdif {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        Self::parse(text)
    }
}

impl PartialEq for Mdif {
    fn eq(&self, other: &Self) -> bool {
        self.parameters == other.parameters
            && self.parameter_values == other.parameter_values
            && self.networks.len() == other.networks.len()
            && self
                .networks
                .iter()
                .zip(&other.networks)
                .all(|(left, right)| {
                    left.frequency == right.frequency
                        && left.s == right.s
                        && left.z0 == right.z0
                        && left.name == right.name
                        && left.comments == right.comments
                        && left.noise == right.noise
                })
    }
}

fn parse_data_block(lines: &[String]) -> Result<Network> {
    let mut unit = FrequencyUnit::Hz;
    let mut parameter = None;
    let mut format = MdifFormat::RealImaginary;
    let mut reference = 50.0;
    let mut kind_lines = Vec::new();
    let mut data_lines = Vec::new();
    let mut comments = Vec::new();
    let mut network_name = None;
    let mut saw_option = false;

    for line in lines {
        let trimmed = line.trim();
        if let Some(option) = trimmed.strip_prefix('#') {
            saw_option = true;
            let mut fields = option.split_whitespace().collect::<Vec<_>>();
            let defaults = ["GHz", "S", "MA", "R", "50"];
            while fields.len() < defaults.len() {
                fields.push(defaults[fields.len()]);
            }
            unit = parse_frequency_unit(fields[0])?;
            parameter = Some(fields[1].to_ascii_lowercase());
            format = MdifFormat::parse(fields[2])?;
            if !fields[3].eq_ignore_ascii_case("r") {
                return Err(Error::Parse(format!(
                    "invalid MDIF option line '{trimmed}'"
                )));
            }
            reference = parse_float(fields[4])?;
        } else if let Some(comment) = trimmed.strip_prefix('!') {
            let comment = comment.trim();
            if comment.to_ascii_lowercase().starts_with("network name:") {
                network_name = comment
                    .split_once(':')
                    .map(|(_, name)| name.trim().to_owned());
            }
            if saw_option {
                comments.push(comment.to_owned());
            }
        } else if let Some(kinds) = trimmed.strip_prefix('%') {
            kind_lines.push(kinds.split_whitespace().map(clean_kind).collect::<Vec<_>>());
        } else if !trimmed.is_empty() {
            data_lines.push(
                trimmed
                    .split_whitespace()
                    .map(parse_float)
                    .collect::<Result<Vec<_>>>()?,
            );
        }
    }

    if kind_lines.is_empty() {
        return Err(Error::Parse(
            "MDIF data block has no % field list".to_owned(),
        ));
    }
    if data_lines.is_empty() {
        return Err(Error::Parse(
            "MDIF data block has no numeric data".to_owned(),
        ));
    }
    let lines_per_point = kind_lines.len();
    if data_lines.len() % lines_per_point != 0 {
        return Err(Error::Parse(
            "MDIF continuation lines do not form complete frequency records".to_owned(),
        ));
    }
    let kinds = kind_lines.into_iter().flatten().collect::<Vec<_>>();
    let records = data_lines
        .chunks(lines_per_point)
        .map(|chunk| chunk.iter().flatten().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let frequencies = records.iter().map(|record| record[0]).collect::<Vec<_>>();
    let frequency = Frequency::from_values(Array1::from_vec(frequencies), unit)?;
    let reference_impedance = Array2::from_elem(
        (frequency.points(), determine_rank(&kinds)?),
        Complex64::new(reference, 0.0),
    );

    let raw = if kinds.iter().any(|kind| kind == "s[1,1]") {
        parse_explicit_matrix(&records, &kinds, 's', format)?
    } else if kinds.iter().any(|kind| kind == "z[1,1]") {
        parse_explicit_matrix(&records, &kinds, 'z', format)?
    } else if kinds.iter().any(|kind| kind == "y[1,1]") {
        parse_explicit_matrix(&records, &kinds, 'y', format)?
    } else if parameter.as_deref() == Some("s") && kinds.iter().any(|kind| kind == "n11x") {
        parse_component_matrix(&records, &kinds, format)?
    } else {
        return Err(Error::Unsupported(
            "MDIF data block does not contain recognized S, Z, or Y parameters".to_owned(),
        ));
    };
    let s = if kinds.iter().any(|kind| kind == "z[1,1]") {
        z_to_s(&raw, &reference_impedance, SParameterDefinition::Power)?
    } else if kinds.iter().any(|kind| kind == "y[1,1]") {
        y_to_s(&raw, &reference_impedance, SParameterDefinition::Power)?
    } else {
        raw
    };
    let mut network = Network::new(frequency, s, reference_impedance)?;
    network.name = network_name;
    network.comments = comments.join("\n");
    Ok(network)
}

fn parse_noise_block(lines: &[String], network: &mut Network) -> Result<()> {
    let mut rows = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('!')
            || trimmed.starts_with('#')
            || trimmed.starts_with('%')
        {
            continue;
        }
        let row = trimmed
            .split_whitespace()
            .map(parse_float)
            .collect::<Result<Vec<_>>>()?;
        if row.len() != 5 {
            return Err(Error::Parse(format!(
                "MDIF noise row contains {} values, expected 5",
                row.len()
            )));
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(Error::Parse("MDIF NDATA block is empty".to_owned()));
    }
    let frequency = Frequency::from_values(
        Array1::from_iter(rows.iter().map(|row| row[0])),
        network.frequency.unit(),
    )?;
    let minimum_noise_figure_db = Array1::from_iter(rows.iter().map(|row| row[1]));
    let optimal_reflection = Array1::from_iter(
        rows.iter()
            .map(|row| Complex64::from_polar(row[2], row[3].to_radians())),
    );
    let reference = network.z0[(0, 0)].re;
    let equivalent_noise_resistance = Array1::from_iter(rows.iter().map(|row| row[4] * reference));
    network.set_noise_parameters(
        frequency,
        minimum_noise_figure_db,
        optimal_reflection,
        equivalent_noise_resistance,
    )
}

fn parse_explicit_matrix(
    records: &[Vec<f64>],
    kinds: &[String],
    prefix: char,
    format: MdifFormat,
) -> Result<Array3<Complex64>> {
    let entries = kinds
        .iter()
        .enumerate()
        .filter_map(|(kind_index, kind)| {
            matrix_coordinates(kind, prefix).map(|coordinates| (kind_index, coordinates))
        })
        .collect::<Vec<_>>();
    let rank = complete_rank(&entries.iter().map(|(_, value)| *value).collect::<Vec<_>>())?;
    let expected = 1 + 2 * (kinds.len() - 1);
    let mut matrix = Array3::zeros((records.len(), rank, rank));
    for (point, record) in records.iter().enumerate() {
        if record.len() != expected {
            return Err(Error::IncompatibleShape(format!(
                "MDIF record contains {} values, expected {expected}",
                record.len()
            )));
        }
        for (kind_index, (row, column)) in &entries {
            let value_index = 1 + 2 * (kind_index - 1);
            matrix[(point, row - 1, column - 1)] =
                format.decode(record[value_index], record[value_index + 1]);
        }
    }
    Ok(matrix)
}

fn parse_component_matrix(
    records: &[Vec<f64>],
    kinds: &[String],
    format: MdifFormat,
) -> Result<Array3<Complex64>> {
    let entries = kinds
        .iter()
        .enumerate()
        .filter_map(|(index, kind)| component_coordinates(kind).map(|value| (index, value)))
        .collect::<Vec<_>>();
    let coordinates = entries
        .iter()
        .map(|(_, (row, column, _))| (*row, *column))
        .collect::<Vec<_>>();
    let rank = complete_rank(&coordinates)?;
    let mut matrix = Array3::zeros((records.len(), rank, rank));
    for (point, record) in records.iter().enumerate() {
        if record.len() != kinds.len() {
            return Err(Error::IncompatibleShape(format!(
                "MDIF component record contains {} values, expected {}",
                record.len(),
                kinds.len()
            )));
        }
        for (kind_index, (row, column, component)) in &entries {
            if *component != 'x' {
                continue;
            }
            let y_index = kinds
                .iter()
                .position(|kind| {
                    component_coordinates(kind).is_some_and(|value| value == (*row, *column, 'y'))
                })
                .ok_or_else(|| Error::Parse(format!("MDIF N{row}{column}X has no Y component")))?;
            matrix[(point, row - 1, column - 1)] =
                format.decode(record[*kind_index], record[y_index]);
        }
    }
    Ok(matrix)
}

fn determine_rank(kinds: &[String]) -> Result<usize> {
    let coordinates = kinds
        .iter()
        .filter_map(|kind| {
            matrix_coordinates(kind, 's')
                .or_else(|| matrix_coordinates(kind, 'z'))
                .or_else(|| matrix_coordinates(kind, 'y'))
                .or_else(|| component_coordinates(kind).map(|(row, column, _)| (row, column)))
        })
        .collect::<Vec<_>>();
    complete_rank(&coordinates)
}

fn complete_rank(coordinates: &[(usize, usize)]) -> Result<usize> {
    let rank = coordinates
        .iter()
        .flat_map(|(row, column)| [*row, *column])
        .max()
        .ok_or_else(|| Error::Parse("MDIF network rank could not be determined".to_owned()))?;
    let unique = coordinates
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if unique.len() != rank * rank {
        return Err(Error::Parse(
            "MDIF parameter matrix is incomplete".to_owned(),
        ));
    }
    Ok(rank)
}

fn matrix_coordinates(kind: &str, prefix: char) -> Option<(usize, usize)> {
    let body = kind
        .strip_prefix(prefix)?
        .strip_prefix('[')?
        .strip_suffix(']')?;
    let (row, column) = body.split_once(',')?;
    Some((row.parse().ok()?, column.parse().ok()?))
}

fn component_coordinates(kind: &str) -> Option<(usize, usize, char)> {
    let body = kind.strip_prefix('n')?;
    let component = body.chars().last()?;
    if component != 'x' && component != 'y' {
        return None;
    }
    let coordinates = &body[..body.len() - 1];
    if let Some((row, column)) = coordinates.split_once('_') {
        Some((row.parse().ok()?, column.parse().ok()?, component))
    } else if coordinates.len() == 2 {
        let mut chars = coordinates.chars();
        Some((
            chars.next()?.to_digit(10)? as usize,
            chars.next()?.to_digit(10)? as usize,
            component,
        ))
    } else {
        None
    }
}

fn clean_kind(kind: &str) -> String {
    kind.split_once('(')
        .map_or(kind, |(kind, _)| kind)
        .to_ascii_lowercase()
}

fn parse_frequency_unit(value: &str) -> Result<FrequencyUnit> {
    match value.to_ascii_lowercase().as_str() {
        "hz" => Ok(FrequencyUnit::Hz),
        "khz" => Ok(FrequencyUnit::KHz),
        "mhz" => Ok(FrequencyUnit::MHz),
        "ghz" => Ok(FrequencyUnit::GHz),
        other => Err(Error::Unsupported(format!(
            "MDIF frequency unit '{other}' is not implemented"
        ))),
    }
}

fn parse_float(value: &str) -> Result<f64> {
    value
        .replace(['d', 'D'], "E")
        .parse::<f64>()
        .map_err(|error| Error::Parse(format!("invalid MDIF number '{value}': {error}")))
}
