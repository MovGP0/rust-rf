use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use encoding_rs::WINDOWS_1252;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

use crate::media::DefinedGammaZ0;
use crate::network::{h_to_z, s_to_g, s_to_h, s_to_y, s_to_z, y_to_s, z_to_s};
use crate::{Error, Frequency, FrequencyUnit, Network, PortMode, Result, SParameterDefinition};
/// Extracts the frequency, propagation constants, and port impedances from an HFSS file.
///
/// Origin: `skrf.io.touchstone.hfss_touchstone_2_gamma_z0`.
pub fn hfss_touchstone_2_gamma_z0(
    path: impl AsRef<Path>,
) -> Result<(Frequency, Array2<Complex64>, Array2<Complex64>)> {
    let touchstone = Touchstone::from_path(path)?;
    let frequency = Frequency::from_hz(touchstone.frequencies_hz.clone())?;
    let gamma = touchstone.propagation_constants.ok_or_else(|| {
        Error::Parse("Touchstone file does not contain HFSS Gamma comments".to_owned())
    })?;
    let z0 = touchstone.port_impedances.ok_or_else(|| {
        Error::Parse("Touchstone file does not contain HFSS Port Impedance comments".to_owned())
    })?;
    Ok((frequency, gamma, z0))
}

/// Builds one defined medium per HFSS port.
///
/// Origin: `skrf.io.touchstone.hfss_touchstone_2_media`.
pub fn hfss_touchstone_2_media(path: impl AsRef<Path>) -> Result<Vec<DefinedGammaZ0>> {
    let (frequency, gamma, z0) = hfss_touchstone_2_gamma_z0(path)?;
    (0..gamma.ncols())
        .map(|port| {
            DefinedGammaZ0::new(
                frequency.clone(),
                gamma.column(port).to_owned(),
                z0.column(port).to_owned(),
                None,
            )
        })
        .collect()
}

/// Reads an HFSS Touchstone file into a Network with per-frequency port metadata.
///
/// Origin: `skrf.io.touchstone.hfss_touchstone_2_network`.
pub fn hfss_touchstone_2_network(path: impl AsRef<Path>) -> Result<Network> {
    Network::read_touchstone(path)
}

/// Reads every Touchstone member of a ZIP archive.
///
/// Origin: `skrf.io.touchstone.read_zipped_touchstones`.
pub fn read_zipped_touchstones(path: impl AsRef<Path>) -> Result<BTreeMap<String, Network>> {
    let file = File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|error| Error::Parse(format!("invalid ZIP archive: {error}")))?;
    let mut networks = BTreeMap::new();
    for index in 0..archive.len() {
        let mut member = archive
            .by_index(index)
            .map_err(|error| Error::Parse(format!("invalid ZIP member: {error}")))?;
        if member.is_dir() {
            continue;
        }
        let member_path = PathBuf::from(member.name());
        let Ok(rank) = rank_from_path(&member_path) else {
            continue;
        };
        let mut bytes = Vec::new();
        member.read_to_end(&mut bytes)?;
        let mut network = Touchstone::from_reader(bytes.as_slice(), rank)?.network()?;
        let name = member_path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .ok_or_else(|| Error::Parse("ZIP Touchstone member has no filename".to_owned()))?;
        network.name = Some(name.clone());
        networks.insert(name, network);
    }
    Ok(networks)
}

/// Writes a Touchstone 2.0 file using the selected network parameter and value format.
///
/// Origin: `skrf.network.Network.write_touchstone`.
pub fn write_touchstone(
    network: &Network,
    path: impl AsRef<Path>,
    parameter: TouchstoneParameter,
    format: TouchstoneFormat,
) -> Result<()> {
    let text = touchstone_string(network, parameter, format)?;
    fs::write(path, text)?;
    Ok(())
}

/// Renders a complete Touchstone 2.0 document without requiring a filesystem.
pub fn touchstone_string(
    network: &Network,
    parameter: TouchstoneParameter,
    format: TouchstoneFormat,
) -> Result<String> {
    let points = network.frequency_points();
    let ports = network.ports();
    let reference = (0..ports)
        .map(|port| network.z0[(0, port)])
        .collect::<Vec<_>>();
    if network.z0.rows().into_iter().any(|row| {
        row.iter()
            .zip(&reference)
            .any(|(value, expected)| value != expected)
    }) {
        return Err(Error::Unsupported(
            "Touchstone reference impedances that vary with frequency cannot be represented"
                .to_owned(),
        ));
    }
    if reference.iter().any(|value| value.im != 0.0) {
        return Err(Error::Unsupported(
            "complex Touchstone 2.0 reference impedances are not supported by the text writer"
                .to_owned(),
        ));
    }
    let values = match parameter {
        TouchstoneParameter::Scattering => network.s.clone(),
        TouchstoneParameter::Impedance => s_to_z(&network.s, &network.z0, network.s_definition)?,
        TouchstoneParameter::Admittance => s_to_y(&network.s, &network.z0, network.s_definition)?,
        TouchstoneParameter::Hybrid => s_to_h(&network.s, &network.z0, network.s_definition)?,
        TouchstoneParameter::InverseHybrid => {
            s_to_g(&network.s, &network.z0, network.s_definition)?
        }
    };
    let parameter_name = match parameter {
        TouchstoneParameter::Scattering => "S",
        TouchstoneParameter::Impedance => "Z",
        TouchstoneParameter::Admittance => "Y",
        TouchstoneParameter::Hybrid => "H",
        TouchstoneParameter::InverseHybrid => "G",
    };
    let format_name = match format {
        TouchstoneFormat::MagnitudeAngle => "MA",
        TouchstoneFormat::DecibelAngle => "DB",
        TouchstoneFormat::RealImaginary => "RI",
    };
    let mut text = String::new();
    for line in network.comments.lines() {
        text.push_str("! ");
        text.push_str(line);
        text.push('\n');
    }
    text.push_str("[Version] 2.0\n");
    text.push_str(&format!("# Hz {parameter_name} {format_name} R 50\n"));
    text.push_str(&format!("[Number of Ports] {ports}\n"));
    text.push_str(&format!("[Number of Frequencies] {points}\n"));
    text.push_str("[Reference]");
    for value in &reference {
        text.push_str(&format!(" {:.17e}", value.re));
    }
    text.push_str("\n[Network Data]\n");
    for point in 0..points {
        text.push_str(&format!("{:.17e}", network.frequency.values_hz()[point]));
        for row in 0..ports {
            for column in 0..ports {
                let value = values[(point, row, column)];
                let (first, second) = match format {
                    TouchstoneFormat::RealImaginary => (value.re, value.im),
                    TouchstoneFormat::MagnitudeAngle => (value.norm(), value.arg().to_degrees()),
                    TouchstoneFormat::DecibelAngle => {
                        (20.0 * value.norm().log10(), value.arg().to_degrees())
                    }
                };
                text.push_str(&format!(" {:.17e} {:.17e}", first, second));
            }
        }
        text.push('\n');
    }
    text.push_str("[End]\n");
    Ok(text)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TouchstoneFormat {
    #[default]
    MagnitudeAngle,
    DecibelAngle,
    RealImaginary,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TouchstoneParameter {
    #[default]
    Scattering,
    Impedance,
    Admittance,
    Hybrid,
    InverseHybrid,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum MatrixFormat {
    #[default]
    Full,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParseSection {
    NetworkData,
    NoiseData,
    Reference,
    Other,
}

/// Origin: `skrf/io/touchstone.py::Touchstone`.
#[derive(Clone, Debug)]
pub struct Touchstone {
    pub filename: Option<String>,
    pub comments: Vec<String>,
    pub comments_after_option_line: Vec<String>,
    pub rank: usize,
    pub version: String,
    pub frequency_unit: FrequencyUnit,
    pub parameter: TouchstoneParameter,
    pub format: TouchstoneFormat,
    pub resistance: Complex64,
    pub port_names: Vec<String>,
    pub reference_impedances: Vec<Complex64>,
    pub port_impedances: Option<Array2<Complex64>>,
    pub propagation_constants: Option<Array2<Complex64>>,
    pub noise: Option<Array2<f64>>,
    pub port_modes: Vec<PortMode>,
    frequencies_hz: Array1<f64>,
    s: Array3<Complex64>,
}

impl Touchstone {
    /// Port of `skrf.io.touchstone.Touchstone.__init__` for filesystem input.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        let text = decode_touchstone(&bytes);
        let rank = rank_from_path(path)?;
        let mut touchstone = Self::parse(&text, rank)?;
        touchstone.filename = Some(path.to_string_lossy().into_owned());
        Ok(touchstone)
    }

    /// Rust reader equivalent of constructing `Touchstone` from a Python file
    /// object. The rank is explicit because a stream has no required suffix.
    pub fn from_reader(mut reader: impl Read, rank: usize) -> Result<Self> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::parse(&decode_touchstone(&bytes), rank)
    }

    pub fn frequencies_hz(&self) -> &Array1<f64> {
        &self.frequencies_hz
    }

    pub fn s_parameters(&self) -> &Array3<Complex64> {
        &self.s
    }

    /// Whether the file supplies HFSS per-frequency port impedances.
    pub fn has_hfss_port_impedances(&self) -> bool {
        self.port_impedances.is_some()
    }

    /// Port of `skrf.io.touchstone.Touchstone.get_sparameter_arrays`.
    pub fn s_parameter_arrays(&self) -> (Array1<f64>, Array3<Complex64>) {
        (self.frequencies_hz.clone(), self.s.clone())
    }

    /// Port of `skrf.io.touchstone.Touchstone.get_sparameter_data`.
    pub fn s_parameter_data(&self, format: TouchstoneFormat) -> BTreeMap<String, Array1<f64>> {
        let mut data = BTreeMap::new();
        data.insert("frequency".to_owned(), self.frequencies_hz.clone());
        for column in 0..self.rank {
            for row in 0..self.rank {
                let values = Array1::from_iter(
                    (0..self.frequencies_hz.len()).map(|point| self.s[(point, row, column)]),
                );
                let name = format!("S{}{}", row + 1, column + 1);
                match format {
                    TouchstoneFormat::RealImaginary => {
                        data.insert(format!("{name}R"), values.mapv(|value| value.re));
                        data.insert(format!("{name}I"), values.mapv(|value| value.im));
                    }
                    TouchstoneFormat::MagnitudeAngle => {
                        data.insert(format!("{name}M"), values.mapv(|value| value.norm()));
                        data.insert(
                            format!("{name}A"),
                            values.mapv(|value| value.arg().to_degrees()),
                        );
                    }
                    TouchstoneFormat::DecibelAngle => {
                        data.insert(
                            format!("{name}DB"),
                            values.mapv(|value| 20.0 * value.norm().log10()),
                        );
                        data.insert(
                            format!("{name}A"),
                            values.mapv(|value| value.arg().to_degrees()),
                        );
                    }
                }
            }
        }
        data
    }

    /// Port of `Touchstone.get_comments` without Python's mutable fallback state.
    pub fn comments_excluding(&self, ignored_comments: &[&str]) -> String {
        self.comments
            .iter()
            .filter(|comment| {
                !ignored_comments
                    .iter()
                    .any(|ignored| comment.contains(ignored))
            })
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Extracts HFSS-style `name = value unit` comment variables.
    pub fn comment_variables(&self) -> BTreeMap<String, (String, String)> {
        let Ok(assignment) =
            regex::Regex::new(r"^\s*([A-Za-z0-9_]*)\s*=\s*([0-9]*\.?[0-9]+)\s*(\w*)")
        else {
            return BTreeMap::new();
        };
        self.comments
            .iter()
            .filter_map(|comment| {
                let captures = assignment.captures(comment)?;
                Some((
                    captures.get(1)?.as_str().to_owned(),
                    (
                        captures.get(2)?.as_str().to_owned(),
                        captures.get(3)?.as_str().to_owned(),
                    ),
                ))
            })
            .collect()
    }

    /// Port of `Touchstone.get_format`. `None` reports the original file format.
    pub fn format_description(&self, format: Option<TouchstoneFormat>) -> String {
        let unit = if format.is_some() {
            "Hz".to_owned()
        } else {
            self.frequency_unit.symbol().to_owned()
        };
        let parameter = match self.parameter {
            TouchstoneParameter::Scattering => "S",
            TouchstoneParameter::Impedance => "Z",
            TouchstoneParameter::Admittance => "Y",
            TouchstoneParameter::Hybrid => "H",
            TouchstoneParameter::InverseHybrid => "G",
        };
        let format = match format.unwrap_or(self.format) {
            TouchstoneFormat::RealImaginary => "RI",
            TouchstoneFormat::MagnitudeAngle => "MA",
            TouchstoneFormat::DecibelAngle => "DB",
        };
        format!("{unit} {parameter} {format} R {}", self.resistance)
    }

    pub fn s_parameter_names(&self, format: TouchstoneFormat) -> Vec<String> {
        self.s_parameter_data(format).into_keys().collect()
    }

    /// Port of `Touchstone.get_gamma_z0`.
    pub fn gamma_z0(&self) -> (Option<Array2<Complex64>>, Option<Array2<Complex64>>) {
        (
            self.propagation_constants.clone(),
            self.port_impedances.clone(),
        )
    }

    pub fn network(&self) -> Result<Network> {
        let frequency = Frequency::from_hz(self.frequencies_hz.clone())?;
        let z0 = self.port_impedances.clone().unwrap_or_else(|| {
            Array2::from_shape_fn((frequency.points(), self.rank), |(_, port)| {
                self.reference_impedances
                    .get(port)
                    .copied()
                    .unwrap_or(self.resistance)
            })
        });
        let mut network = Network::new(frequency, self.s.clone(), z0)?;
        network.comments = self.comments.join("\n");
        network.name = self.filename.as_ref().and_then(|filename| {
            PathBuf::from(filename)
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        });
        network.port_names.clone_from(&self.port_names);
        network.port_modes.clone_from(&self.port_modes);
        network
            .propagation_constants
            .clone_from(&self.propagation_constants);
        if let Some(noise) = &self.noise {
            let noise_frequency = Frequency::from_hz(noise.column(0).to_owned())?;
            let minimum_noise_figure_db = noise.column(1).to_owned();
            let optimal_reflection = Array1::from_iter(
                noise
                    .rows()
                    .into_iter()
                    .map(|row| Complex64::from_polar(row[2], row[3].to_radians())),
            );
            let resistance_scale = if self.version == "1.0" {
                network.z0[(0, 0)].re
            } else {
                1.0
            };
            let equivalent_noise_resistance =
                noise.column(4).mapv(|value| value * resistance_scale);
            network.set_noise_parameters(
                noise_frequency,
                minimum_noise_figure_db,
                optimal_reflection,
                equivalent_noise_resistance,
            )?;
        }
        Ok(network)
    }

    fn parse(text: &str, rank_hint: usize) -> Result<Self> {
        let mut comments = Vec::new();
        let mut comments_after_option_line = Vec::new();
        let mut version = "1.0".to_owned();
        let is_version_two = text.lines().any(|line| {
            line.trim_start_matches('\u{feff}')
                .trim()
                .to_ascii_lowercase()
                .starts_with("[version]")
        });
        let mut section = if is_version_two {
            ParseSection::Other
        } else {
            ParseSection::NetworkData
        };
        let mut rank = rank_hint;
        let mut expected_points = None;
        let mut expected_noise_points = None;
        let mut matrix_format = MatrixFormat::Full;
        let mut legacy_two_port_order = !is_version_two;
        let mut frequency_unit = FrequencyUnit::GHz;
        let mut parameter = TouchstoneParameter::Scattering;
        let mut format = TouchstoneFormat::MagnitudeAngle;
        let mut resistance = Complex64::new(50.0, 0.0);
        let mut port_names = Vec::new();
        let mut reference_impedances = Vec::new();
        let mut hfss_port_impedances = Vec::new();
        let mut hfss_propagation_constants = Vec::new();
        let mut noise_rows = Vec::new();
        let mut option_seen = false;
        let mut numeric_values = Vec::new();
        let mut mixed_mode_order = None;
        let mut hfss_continuation = None;

        for raw_line in text.lines() {
            let line = raw_line.trim_start_matches('\u{feff}').trim();
            if line.is_empty() {
                continue;
            }
            if let Some(comment) = line.strip_prefix('!') {
                let comment = comment.trim().to_owned();
                parse_port_name(&comment, &mut port_names);
                if let Some(values) = parse_hfss_complex_comment(&comment, "Port Impedance")? {
                    hfss_port_impedances.push(values);
                    hfss_continuation = Some(HfssContinuation::PortImpedance);
                } else if let Some(values) = parse_hfss_complex_comment(&comment, "Gamma")? {
                    hfss_propagation_constants.push(values);
                    hfss_continuation = Some(HfssContinuation::Gamma);
                } else if let Some(values) = parse_hfss_complex_continuation(&comment)? {
                    match hfss_continuation {
                        Some(HfssContinuation::PortImpedance) => {
                            let row = hfss_port_impedances.last_mut().ok_or_else(|| {
                                Error::Parse(
                                    "HFSS port-impedance continuation has no preceding row"
                                        .to_owned(),
                                )
                            })?;
                            row.extend(values);
                        }
                        Some(HfssContinuation::Gamma) => {
                            let row = hfss_propagation_constants.last_mut().ok_or_else(|| {
                                Error::Parse(
                                    "HFSS propagation continuation has no preceding row".to_owned(),
                                )
                            })?;
                            row.extend(values);
                        }
                        None => {}
                    }
                } else {
                    hfss_continuation = None;
                }
                let uppercase_comment = comment.to_ascii_uppercase();
                if uppercase_comment.contains("NOISE PARAMETERS") {
                    section = ParseSection::NoiseData;
                } else if uppercase_comment.contains("NETWORK PARAMETERS") {
                    section = ParseSection::NetworkData;
                }
                if option_seen {
                    comments_after_option_line.push(comment.clone());
                } else {
                    comments.push(comment.clone());
                }
                continue;
            }
            hfss_continuation = None;
            if line.starts_with('[') {
                let closing_bracket = line.find(']').ok_or_else(|| {
                    Error::Parse(format!("invalid Touchstone keyword line '{line}'"))
                })?;
                let keyword = line[1..closing_bracket].trim().to_ascii_lowercase();
                let value = line[closing_bracket + 1..].trim();
                match keyword.as_str() {
                    "version" => version = value.to_owned(),
                    "number of ports" => {
                        let declared_rank = value.parse::<usize>().map_err(|error| {
                            Error::Parse(format!("invalid Touchstone port count: {error}"))
                        })?;
                        if rank != 0 && rank != declared_rank {
                            return Err(Error::Parse(format!(
                                "file extension declares {rank} ports but Touchstone data declares {declared_rank}"
                            )));
                        }
                        rank = declared_rank;
                    }
                    "number of frequencies" => {
                        expected_points = Some(value.parse::<usize>().map_err(|error| {
                            Error::Parse(format!("invalid Touchstone frequency count: {error}"))
                        })?);
                    }
                    "number of noise frequencies" => {
                        expected_noise_points = Some(value.parse::<usize>().map_err(|error| {
                            Error::Parse(format!(
                                "invalid Touchstone noise frequency count: {error}"
                            ))
                        })?);
                    }
                    "matrix format" => {
                        matrix_format = match value.to_ascii_lowercase().as_str() {
                            "full" => MatrixFormat::Full,
                            "lower" => MatrixFormat::Lower,
                            "upper" => MatrixFormat::Upper,
                            other => {
                                return Err(Error::Unsupported(format!(
                                    "Touchstone matrix format '{other}' is not supported"
                                )));
                            }
                        };
                    }
                    "reference" => {
                        section = ParseSection::Reference;
                        for token in value.split_whitespace() {
                            reference_impedances.push(Complex64::new(parse_float(token)?, 0.0));
                        }
                    }
                    "network data" => section = ParseSection::NetworkData,
                    "noise data" => section = ParseSection::NoiseData,
                    "two-port data order" => {
                        legacy_two_port_order = value.eq_ignore_ascii_case("21_12");
                    }
                    "mixed-mode order" => mixed_mode_order = Some(value.to_owned()),
                    "end" => break,
                    _ => section = ParseSection::Other,
                }
                continue;
            }
            if let Some(option_line) = line.strip_prefix('#') {
                if !option_seen {
                    let parsed = parse_option_line(option_line)?;
                    frequency_unit = parsed.0;
                    parameter = parsed.1;
                    format = parsed.2;
                    resistance = parsed.3;
                    option_seen = true;
                }
                section = ParseSection::NetworkData;
                continue;
            }

            let data_before_comment = line.split_once('!').map_or(line, |(data, _)| data);
            match section {
                ParseSection::NetworkData => {
                    for token in data_before_comment.split_whitespace() {
                        numeric_values.push(parse_float(token)?);
                    }
                }
                ParseSection::Reference => {
                    for token in data_before_comment.split_whitespace() {
                        reference_impedances.push(Complex64::new(parse_float(token)?, 0.0));
                    }
                }
                ParseSection::NoiseData => {
                    let row = data_before_comment
                        .split_whitespace()
                        .map(parse_float)
                        .collect::<Result<Vec<_>>>()?;
                    if !row.is_empty() {
                        noise_rows.push(row);
                    }
                }
                ParseSection::Other => {}
            }
        }

        if rank == 0 {
            return Err(Error::Parse(
                "Touchstone port count is missing from the extension and file keywords".to_owned(),
            ));
        }
        let parameter_positions = parameter_positions(rank, matrix_format);
        let values_per_frequency = 1 + 2 * parameter_positions.len();
        if numeric_values.len() % values_per_frequency != 0 {
            return Err(Error::Parse(format!(
                "Touchstone data contains {} values, which is not divisible into {values_per_frequency}-value records",
                numeric_values.len()
            )));
        }
        let points = numeric_values.len() / values_per_frequency;
        if let Some(expected) = expected_points {
            if expected != points {
                return Err(Error::Parse(format!(
                    "Touchstone declares {expected} frequency points but contains {points}"
                )));
            }
        }
        let mut frequencies_hz = Array1::zeros(points);
        let mut parameters = Array3::zeros((points, rank, rank));
        for point in 0..points {
            let record_offset = point * values_per_frequency;
            frequencies_hz[point] = numeric_values[record_offset] * frequency_unit.multiplier();
            for (parameter, &(mut row, mut column)) in parameter_positions.iter().enumerate() {
                let pair_offset = record_offset + 1 + parameter * 2;
                let value = decode_parameter_pair(
                    numeric_values[pair_offset],
                    numeric_values[pair_offset + 1],
                    format,
                );
                if legacy_two_port_order && rank == 2 {
                    std::mem::swap(&mut row, &mut column);
                }
                parameters[(point, row, column)] = value;
                if matrix_format != MatrixFormat::Full && row != column {
                    parameters[(point, column, row)] = value;
                }
            }
        }

        if reference_impedances.is_empty() {
            reference_impedances = vec![resistance; rank];
        } else if reference_impedances.len() != rank {
            return Err(Error::Parse(format!(
                "Touchstone reference section has {} values for {rank} ports",
                reference_impedances.len()
            )));
        }
        port_names.resize(rank, String::new());
        let port_impedances =
            complex_rows_to_array(hfss_port_impedances, points, rank, "HFSS port impedance")?;
        let conversion_reference = port_impedances.clone().unwrap_or_else(|| {
            Array2::from_shape_fn((points, rank), |(_, port)| reference_impedances[port])
        });
        let mut port_modes = vec![PortMode::SingleEnded; rank];
        if let Some(order) = mixed_mode_order {
            let mixed = parse_mixed_mode_order(&order, rank, &reference_impedances)?;
            parameters = Array3::from_shape_fn((points, rank, rank), |(point, row, column)| {
                parameters[(point, mixed[row].source_index, mixed[column].source_index)]
            });
            reference_impedances = mixed.iter().map(|entry| entry.reference).collect();
            port_modes = mixed.iter().map(|entry| entry.mode).collect();
        }
        let mut conversion_parameters = parameters;
        if version == "1.0" && parameter != TouchstoneParameter::Scattering {
            for point in 0..points {
                for row in 0..rank {
                    for column in 0..rank {
                        conversion_parameters[(point, row, column)] *=
                            conversion_reference[(point, row)];
                    }
                }
            }
        }
        let s = match parameter {
            TouchstoneParameter::Scattering => conversion_parameters,
            TouchstoneParameter::Impedance => z_to_s(
                &conversion_parameters,
                &conversion_reference,
                SParameterDefinition::Power,
            )?,
            TouchstoneParameter::Admittance => y_to_s(
                &conversion_parameters,
                &conversion_reference,
                SParameterDefinition::Power,
            )?,
            TouchstoneParameter::Hybrid => z_to_s(
                &h_to_z(&conversion_parameters)?,
                &conversion_reference,
                SParameterDefinition::Power,
            )?,
            TouchstoneParameter::InverseHybrid => y_to_s(
                &h_to_z(&conversion_parameters)?,
                &conversion_reference,
                SParameterDefinition::Power,
            )?,
        };
        let propagation_constants = complex_rows_to_array(
            hfss_propagation_constants,
            points,
            rank,
            "HFSS propagation constant",
        )?;
        let noise = noise_rows_to_array(
            noise_rows,
            expected_noise_points,
            frequency_unit.multiplier(),
        )?;

        Ok(Self {
            filename: None,
            comments,
            comments_after_option_line,
            rank,
            version,
            frequency_unit,
            parameter,
            format,
            resistance,
            port_names,
            reference_impedances,
            port_impedances,
            propagation_constants,
            noise,
            port_modes,
            frequencies_hz,
            s,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct MixedModeEntry {
    source_index: usize,
    physical_port: usize,
    mode: PortMode,
    reference: Complex64,
}

fn parse_mixed_mode_order(
    value: &str,
    rank: usize,
    references: &[Complex64],
) -> Result<Vec<MixedModeEntry>> {
    let tokens = value.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != rank || references.len() != rank {
        return Err(Error::Parse(format!(
            "mixed-mode order contains {} entries for {rank} ports",
            tokens.len()
        )));
    }
    let mut entries = Vec::with_capacity(rank);
    for (source_index, token) in tokens.iter().enumerate() {
        let mode = match token.chars().next().map(|value| value.to_ascii_uppercase()) {
            Some('S') => PortMode::SingleEnded,
            Some('D') => PortMode::Differential,
            Some('C') => PortMode::Common,
            _ => {
                return Err(Error::Parse(format!(
                    "invalid mixed-mode order entry '{token}'"
                )));
            }
        };
        let ports = token[1..]
            .split(',')
            .map(|port| {
                port.parse::<usize>()
                    .map_err(|error| Error::Parse(format!("invalid mixed-mode port: {error}")))
            })
            .collect::<Result<Vec<_>>>()?;
        if ports.is_empty() || ports.iter().any(|port| *port == 0 || *port > rank) {
            return Err(Error::Parse(format!(
                "mixed-mode entry '{token}' references an invalid port"
            )));
        }
        let reference = match mode {
            PortMode::SingleEnded if ports.len() == 1 => references[ports[0] - 1],
            PortMode::Differential | PortMode::Common if ports.len() == 2 => {
                let first = references[ports[0] - 1];
                let second = references[ports[1] - 1];
                if mode == PortMode::Differential {
                    first + second
                } else if first + second == Complex64::new(0.0, 0.0) {
                    return Err(Error::Parse(
                        "common-mode reference impedance has a zero denominator".to_owned(),
                    ));
                } else {
                    first * second / (first + second)
                }
            }
            _ => {
                return Err(Error::Parse(format!(
                    "mixed-mode entry '{token}' has the wrong number of ports"
                )));
            }
        };
        let physical_port = ports
            .iter()
            .min()
            .copied()
            .ok_or_else(|| Error::Parse(format!("mixed-mode entry '{token}' has no ports")))?;
        entries.push(MixedModeEntry {
            source_index,
            physical_port,
            mode,
            reference,
        });
    }
    entries.sort_by_key(|entry| {
        let mode_order = match entry.mode {
            PortMode::SingleEnded => 0,
            PortMode::Differential => 1,
            PortMode::Common => 2,
        };
        (entry.physical_port, mode_order)
    });
    Ok(entries)
}

fn decode_touchstone(bytes: &[u8]) -> String {
    if let Ok(text) = std::str::from_utf8(bytes) {
        text.trim_start_matches('\u{feff}').to_owned()
    } else {
        let (text, _, _) = WINDOWS_1252.decode(bytes);
        text.into_owned()
    }
}

fn parse_port_name(comment: &str, port_names: &mut Vec<String>) {
    let Some(port) = comment.strip_prefix("Port[") else {
        return;
    };
    let Some((index, name)) = port.split_once(']') else {
        return;
    };
    let Ok(index) = index.parse::<usize>() else {
        return;
    };
    if index == 0 {
        return;
    }
    let name = name.trim().strip_prefix('=').map(str::trim).unwrap_or("");
    if port_names.len() < index {
        port_names.resize(index, String::new());
    }
    port_names[index - 1] = name.to_owned();
}

fn parse_hfss_complex_comment(comment: &str, prefix: &str) -> Result<Option<Vec<Complex64>>> {
    let Some(values) = comment.strip_prefix(prefix) else {
        return Ok(None);
    };
    let numbers = values
        .split_whitespace()
        .filter(|token| *token != "!")
        .map(parse_float)
        .collect::<Result<Vec<_>>>()?;
    if numbers.len() % 2 != 0 {
        return Err(Error::Parse(format!(
            "{prefix} comment contains an odd number of real/imaginary values"
        )));
    }
    Ok(Some(
        numbers
            .chunks_exact(2)
            .map(|pair| Complex64::new(pair[0], pair[1]))
            .collect(),
    ))
}

fn parse_hfss_complex_continuation(comment: &str) -> Result<Option<Vec<Complex64>>> {
    let tokens = comment
        .split_whitespace()
        .filter(|token| *token != "!")
        .collect::<Vec<_>>();
    if tokens.is_empty() || tokens.iter().any(|token| parse_float(token).is_err()) {
        return Ok(None);
    }
    let numbers = tokens
        .iter()
        .map(|token| parse_float(token))
        .collect::<Result<Vec<_>>>()?;
    if numbers.len() % 2 != 0 {
        return Err(Error::Parse(
            "HFSS continuation contains an odd number of real/imaginary values".to_owned(),
        ));
    }
    Ok(Some(
        numbers
            .chunks_exact(2)
            .map(|pair| Complex64::new(pair[0], pair[1]))
            .collect(),
    ))
}

#[derive(Clone, Copy, Debug)]
enum HfssContinuation {
    PortImpedance,
    Gamma,
}

fn complex_rows_to_array(
    rows: Vec<Vec<Complex64>>,
    points: usize,
    rank: usize,
    description: &str,
) -> Result<Option<Array2<Complex64>>> {
    if rows.is_empty() {
        return Ok(None);
    }
    if rows.len() != points || rows.iter().any(|row| row.len() != rank) {
        return Err(Error::Parse(format!(
            "{description} data has incompatible dimensions for {points} points and {rank} ports"
        )));
    }
    Ok(Some(Array2::from_shape_fn(
        (points, rank),
        |(point, port)| rows[point][port],
    )))
}

fn noise_rows_to_array(
    rows: Vec<Vec<f64>>,
    expected_points: Option<usize>,
    frequency_multiplier: f64,
) -> Result<Option<Array2<f64>>> {
    if rows.is_empty() {
        return Ok(None);
    }
    if rows.iter().any(|row| row.len() != 5) {
        return Err(Error::Parse(
            "Touchstone noise rows must contain five values".to_owned(),
        ));
    }
    if let Some(expected) = expected_points {
        if expected != rows.len() {
            return Err(Error::Parse(format!(
                "Touchstone declares {expected} noise frequencies but contains {}",
                rows.len()
            )));
        }
    }
    Ok(Some(Array2::from_shape_fn(
        (rows.len(), 5),
        |(row, column)| {
            if column == 0 {
                rows[row][column] * frequency_multiplier
            } else {
                rows[row][column]
            }
        },
    )))
}

fn parameter_positions(rank: usize, matrix_format: MatrixFormat) -> Vec<(usize, usize)> {
    let mut positions = Vec::new();
    for row in 0..rank {
        for column in 0..rank {
            if matrix_format == MatrixFormat::Full
                || (matrix_format == MatrixFormat::Lower && row >= column)
                || (matrix_format == MatrixFormat::Upper && row <= column)
            {
                positions.push((row, column));
            }
        }
    }
    positions
}

fn rank_from_path(path: &Path) -> Result<usize> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .ok_or_else(|| Error::Parse("Touchstone file has no extension".to_owned()))?
        .to_ascii_lowercase();
    if extension == "ts" || extension == "sp" {
        return Ok(0);
    }
    let rank_text = extension
        .strip_prefix('s')
        .and_then(|value| value.strip_suffix('p'))
        .ok_or_else(|| {
            Error::Parse(format!(
                "Touchstone extension must look like .s2p, got .{extension}"
            ))
        })?;
    rank_text
        .parse::<usize>()
        .map_err(|error| Error::Parse(format!("invalid Touchstone port count: {error}")))
}

fn parse_option_line(
    line: &str,
) -> Result<(
    FrequencyUnit,
    TouchstoneParameter,
    TouchstoneFormat,
    Complex64,
)> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return Ok((
            FrequencyUnit::GHz,
            TouchstoneParameter::Scattering,
            TouchstoneFormat::MagnitudeAngle,
            Complex64::new(50.0, 0.0),
        ));
    }
    let frequency_unit = tokens
        .first()
        .ok_or_else(|| Error::Parse("empty Touchstone option line".to_owned()))?;
    let frequency_unit = match frequency_unit.to_ascii_lowercase().as_str() {
        "hz" => FrequencyUnit::Hz,
        "khz" => FrequencyUnit::KHz,
        "mhz" => FrequencyUnit::MHz,
        "ghz" => FrequencyUnit::GHz,
        "thz" => FrequencyUnit::THz,
        other => return Err(Error::Parse(format!("unknown frequency unit '{other}'"))),
    };

    let parameter = match tokens
        .get(1)
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("s") => TouchstoneParameter::Scattering,
        Some("z") => TouchstoneParameter::Impedance,
        Some("y") => TouchstoneParameter::Admittance,
        Some("h") => TouchstoneParameter::Hybrid,
        Some("g") => TouchstoneParameter::InverseHybrid,
        Some(other) => {
            return Err(Error::Parse(format!(
                "unknown Touchstone parameter type '{other}'"
            )));
        }
        None => {
            return Err(Error::Parse(
                "Touchstone parameter type is missing".to_owned(),
            ));
        }
    };
    let format = match tokens
        .get(2)
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("ri") => TouchstoneFormat::RealImaginary,
        Some("ma") => TouchstoneFormat::MagnitudeAngle,
        Some("db") => TouchstoneFormat::DecibelAngle,
        Some(other) => return Err(Error::Parse(format!("unknown Touchstone format '{other}'"))),
        None => return Err(Error::Parse("Touchstone format is missing".to_owned())),
    };
    let resistance = if let Some(reference_index) = tokens
        .iter()
        .position(|token| token.eq_ignore_ascii_case("r"))
    {
        parse_complex_reference(
            tokens
                .get(reference_index + 1)
                .ok_or_else(|| Error::Parse("reference resistance is missing".to_owned()))?,
        )?
    } else {
        Complex64::new(50.0, 0.0)
    };
    Ok((frequency_unit, parameter, format, resistance))
}

fn parse_complex_reference(token: &str) -> Result<Complex64> {
    let value = token.trim_matches(['(', ')']).trim_end_matches(['j', 'J']);
    let split_index = value
        .char_indices()
        .skip(1)
        .filter(|(_, character)| *character == '+' || *character == '-')
        .map(|(index, _)| index)
        .last();
    if let Some(index) = split_index {
        Ok(Complex64::new(
            parse_float(&value[..index])?,
            parse_float(&value[index..])?,
        ))
    } else {
        Ok(Complex64::new(parse_float(value)?, 0.0))
    }
}

fn parse_float(token: &str) -> Result<f64> {
    token
        .replace(['d', 'D'], "E")
        .parse::<f64>()
        .map_err(|error| Error::Parse(format!("invalid numeric value '{token}': {error}")))
}

fn decode_parameter_pair(first: f64, second: f64, format: TouchstoneFormat) -> Complex64 {
    match format {
        TouchstoneFormat::RealImaginary => Complex64::new(first, second),
        TouchstoneFormat::MagnitudeAngle => Complex64::from_polar(first, second.to_radians()),
        TouchstoneFormat::DecibelAngle => {
            Complex64::from_polar(10.0_f64.powf(first / 20.0), second.to_radians())
        }
    }
}
