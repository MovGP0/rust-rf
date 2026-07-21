use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{Error, Frequency, Network, NetworkSet, Result};

/// Rust-native, data-only counterpart to the Python pickle object surface.
///
/// Origin: `skrf/io/general.py::read` and `write`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum StoredObject {
    Frequency(Frequency),
    Network(Box<Network>),
    NetworkSet(NetworkSet),
}

impl StoredObject {
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Frequency(_) => "freq",
            Self::Network(_) => "ntwk",
            Self::NetworkSet(_) => "ns",
        }
    }
}

pub fn write_object(
    path: impl AsRef<Path>,
    object: &StoredObject,
    overwrite: bool,
) -> Result<PathBuf> {
    let mut path = path.as_ref().to_path_buf();
    if path.extension().is_none() {
        path.set_extension(object.extension());
    }
    if path.exists() && !overwrite {
        return Err(Error::Unsupported(format!(
            "file '{}' already exists",
            path.display()
        )));
    }
    let bytes = serde_json::to_vec(object)
        .map_err(|error| Error::Parse(format!("object serialization failed: {error}")))?;
    fs::write(&path, bytes)?;
    Ok(path)
}

pub fn read_object(path: impl AsRef<Path>) -> Result<StoredObject> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| Error::Parse(format!("object deserialization failed: {error}")))
}

/// Safe JSON serialization counterpart to `to_json_string`.
pub fn to_json_string(network: &Network) -> Result<String> {
    serde_json::to_string(network)
        .map_err(|error| Error::Parse(format!("Network JSON serialization failed: {error}")))
}

/// Safe JSON deserialization counterpart to `from_json_string`.
pub fn from_json_string(text: &str) -> Result<Network> {
    serde_json::from_str(text)
        .map_err(|error| Error::Parse(format!("Network JSON deserialization failed: {error}")))
}

/// Port of `read_all_networks` for Touchstone and Rust JSON Network files.
pub fn read_all_networks(
    directory: impl AsRef<Path>,
    contains: Option<&str>,
    recursive: bool,
) -> Result<BTreeMap<String, Network>> {
    let mut files = Vec::new();
    collect_files(directory.as_ref(), recursive, &mut files)?;
    files.sort();
    let mut networks = BTreeMap::new();
    for path in files {
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if contains.is_some_and(|needle| !filename.contains(needle)) {
            continue;
        }
        let network = if is_touchstone_path(&path) {
            Network::read_touchstone(&path).ok()
        } else if path.extension().and_then(|value| value.to_str()) == Some("ntwk") {
            match read_object(&path) {
                Ok(StoredObject::Network(network)) => Some(*network),
                _ => None,
            }
        } else {
            None
        };
        if let Some(network) = network {
            if let Some(stem) = path.file_stem() {
                networks.insert(stem.to_string_lossy().into_owned(), network);
            }
        }
    }
    Ok(networks)
}

/// Typed counterpart to `read_all`, retaining mixed supported RF objects.
pub fn read_all_objects(
    directory: impl AsRef<Path>,
    contains: Option<&str>,
    recursive: bool,
) -> Result<BTreeMap<String, StoredObject>> {
    let mut files = Vec::new();
    collect_files(directory.as_ref(), recursive, &mut files)?;
    files.sort();
    let mut objects = BTreeMap::new();
    for path in files {
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if contains.is_some_and(|needle| !filename.contains(needle)) {
            continue;
        }
        let object = if is_touchstone_path(&path) {
            Network::read_touchstone(&path)
                .ok()
                .map(|network| StoredObject::Network(Box::new(network)))
        } else if matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("freq" | "ntwk" | "ns")
        ) {
            read_object(&path).ok()
        } else {
            None
        };
        if let Some(object) = object {
            if let Some(stem) = path.file_stem() {
                objects.insert(stem.to_string_lossy().into_owned(), object);
            }
        }
    }
    Ok(objects)
}

pub fn write_all_networks(
    networks: &BTreeMap<String, Network>,
    directory: impl AsRef<Path>,
) -> Result<Vec<PathBuf>> {
    let directory = directory.as_ref();
    if !directory.is_dir() {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("directory '{}' does not exist", directory.display()),
        )));
    }
    let mut paths = Vec::with_capacity(networks.len());
    for (name, network) in networks {
        let path = directory.join(format!("{name}.ntwk"));
        write_object(
            &path,
            &StoredObject::Network(Box::new(network.clone())),
            true,
        )?;
        paths.push(path);
    }
    Ok(paths)
}

pub fn write_all_objects(
    objects: &BTreeMap<String, StoredObject>,
    directory: impl AsRef<Path>,
    overwrite: bool,
) -> Result<Vec<PathBuf>> {
    let directory = directory.as_ref();
    if !directory.is_dir() {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("directory '{}' does not exist", directory.display()),
        )));
    }
    objects
        .iter()
        .map(|(name, object)| write_object(directory.join(name), object, overwrite))
        .collect()
}

pub fn statistical_to_touchstone(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    header: Option<&str>,
) -> Result<()> {
    let source_text = fs::read_to_string(source)?;
    let mut writer = fs::File::create(destination)?;
    writeln!(writer, "{}", header.unwrap_or("# GHz S RI R 50.0"))?;
    writer.write_all(source_text.as_bytes())?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NetworkDataFormat {
    DecibelAngle,
    MagnitudeAngle,
    RealImaginary,
}

/// Build the numeric table used by spreadsheet writers.
pub fn network_table(network: &Network, format: NetworkDataFormat) -> (Vec<String>, Array2<f64>) {
    let columns = 1 + 2 * network.ports() * network.ports();
    let mut names = Vec::with_capacity(columns);
    names.push(format!("Freq({})", network.frequency.unit().symbol()));
    let mut data = Array2::zeros((network.frequency_points(), columns));
    data.column_mut(0).assign(&network.frequency.scaled());
    let mut column_index = 1;
    for input in 0..network.ports() {
        for output in 0..network.ports() {
            let trace = format!("S{}{}", output + 1, input + 1);
            match format {
                NetworkDataFormat::DecibelAngle => {
                    names.push(format!("{trace} Log Mag(dB)"));
                    names.push(format!("{trace} Phase(deg)"));
                }
                NetworkDataFormat::MagnitudeAngle => {
                    names.push(format!("{trace} Mag(lin)"));
                    names.push(format!("{trace} Phase(deg)"));
                }
                NetworkDataFormat::RealImaginary => {
                    names.push(format!("{trace} Real"));
                    names.push(format!("{trace} Imag"));
                }
            }
            for point in 0..network.frequency_points() {
                let value = network.s[(point, output, input)];
                let (first, second) = match format {
                    NetworkDataFormat::DecibelAngle => {
                        (20.0 * value.norm().log10(), value.arg().to_degrees())
                    }
                    NetworkDataFormat::MagnitudeAngle => (value.norm(), value.arg().to_degrees()),
                    NetworkDataFormat::RealImaginary => (value.re, value.im),
                };
                data[(point, column_index)] = first;
                data[(point, column_index + 1)] = second;
            }
            column_index += 2;
        }
    }
    (names, data)
}

pub fn write_network_csv(
    network: &Network,
    path: impl AsRef<Path>,
    format: NetworkDataFormat,
) -> Result<()> {
    let (names, data) = network_table(network, format);
    let mut writer =
        csv::Writer::from_path(path).map_err(|error| Error::Io(std::io::Error::other(error)))?;
    writer
        .write_record(&names)
        .map_err(|error| Error::Io(std::io::Error::other(error)))?;
    for row in data.rows() {
        writer
            .write_record(row.iter().map(ToString::to_string))
            .map_err(|error| Error::Io(std::io::Error::other(error)))?;
    }
    writer
        .flush()
        .map_err(|error| Error::Io(std::io::Error::other(error)))?;
    Ok(())
}

pub fn write_network_html(
    network: &Network,
    path: impl AsRef<Path>,
    format: NetworkDataFormat,
) -> Result<()> {
    let (names, data) = network_table(network, format);
    let mut writer = fs::File::create(path)?;
    writeln!(writer, "<!doctype html><html><body><table>")?;
    write!(writer, "<thead><tr>")?;
    for name in names {
        write!(writer, "<th>{}</th>", escape_html(&name))?;
    }
    writeln!(writer, "</tr></thead><tbody>")?;
    for row in data.rows() {
        write!(writer, "<tr>")?;
        for value in row {
            write!(writer, "<td>{value}</td>")?;
        }
        writeln!(writer, "</tr>")?;
    }
    writeln!(writer, "</tbody></table></body></html>")?;
    Ok(())
}

#[cfg(feature = "dataframe")]
pub fn network_to_dataframe(
    network: &Network,
    attributes: &[&str],
    ports: Option<&[(usize, usize)]>,
    port_separator: Option<&str>,
) -> Result<polars::frame::DataFrame> {
    use polars::prelude::Column;

    let default_ports = (0..network.ports())
        .flat_map(|input| (0..network.ports()).map(move |output| (output, input)))
        .collect::<Vec<_>>();
    let ports = ports.unwrap_or(&default_ports);
    let separator = port_separator.unwrap_or(if network.ports() > 10 { "_" } else { "" });
    let mut columns = Vec::with_capacity(attributes.len() * ports.len());
    for attribute in attributes {
        for (output, input) in ports {
            if *output >= network.ports() || *input >= network.ports() {
                return Err(Error::InvalidPort {
                    port: (*output).max(*input),
                    ports: network.ports(),
                });
            }
            let values = (0..network.frequency_points())
                .map(|point| {
                    let value = network.s[(point, *output, *input)];
                    match *attribute {
                        "s_db" => Ok(20.0 * value.norm().log10()),
                        "s_deg" => Ok(value.arg().to_degrees()),
                        "s_mag" => Ok(value.norm()),
                        "s_re" => Ok(value.re),
                        "s_im" => Ok(value.im),
                        other => Err(Error::Unsupported(format!(
                            "Network DataFrame attribute '{other}' is not implemented"
                        ))),
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            columns.push(Column::new(
                format!("{attribute} {}{separator}{}", output + 1, input + 1).into(),
                values,
            ));
        }
    }
    polars::frame::DataFrame::new(network.frequency_points(), columns)
        .map_err(|error| Error::Unsupported(format!("DataFrame construction failed: {error}")))
}

#[cfg(feature = "xlsx")]
pub fn write_network_xlsx(
    network: &Network,
    path: impl AsRef<Path>,
    format: NetworkDataFormat,
) -> Result<()> {
    let (names, data) = network_table(network, format);
    let mut workbook = rust_xlsxwriter::Workbook::new();
    let worksheet = workbook.add_worksheet();
    for (column, name) in names.iter().enumerate() {
        worksheet
            .write_string(0, column as u16, name)
            .map_err(|error| Error::Unsupported(format!("XLSX write failed: {error}")))?;
    }
    for (row, values) in data.rows().into_iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            worksheet
                .write_number((row + 1) as u32, column as u16, *value)
                .map_err(|error| Error::Unsupported(format!("XLSX write failed: {error}")))?;
        }
    }
    workbook
        .save(path)
        .map_err(|error| Error::Unsupported(format!("XLSX save failed: {error}")))
}

#[cfg(feature = "xlsx")]
pub fn write_network_set_xlsx(
    network_set: &NetworkSet,
    path: impl AsRef<Path>,
    format: NetworkDataFormat,
) -> Result<()> {
    if network_set.networks.is_empty() {
        return Err(Error::IncompatibleShape(
            "cannot write an empty NetworkSet workbook".to_owned(),
        ));
    }
    let mut workbook = rust_xlsxwriter::Workbook::new();
    for (index, network) in network_set.networks.iter().enumerate() {
        let (names, data) = network_table(network, format);
        let worksheet = workbook.add_worksheet();
        let fallback_name = format!("Network{}", index + 1);
        let sheet_name = network.name.as_deref().unwrap_or(&fallback_name);
        worksheet
            .set_name(sheet_name)
            .map_err(|error| Error::Unsupported(format!("XLSX sheet name failed: {error}")))?;
        for (column, name) in names.iter().enumerate() {
            worksheet
                .write_string(0, column as u16, name)
                .map_err(|error| Error::Unsupported(format!("XLSX write failed: {error}")))?;
        }
        for (row, values) in data.rows().into_iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                worksheet
                    .write_number((row + 1) as u32, column as u16, *value)
                    .map_err(|error| Error::Unsupported(format!("XLSX write failed: {error}")))?;
            }
        }
    }
    workbook
        .save(path)
        .map_err(|error| Error::Unsupported(format!("XLSX save failed: {error}")))
}

fn collect_files(directory: &Path, recursive: bool, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && recursive {
            collect_files(&path, true, output)?;
        } else if path.is_file() {
            output.push(path);
        }
    }
    Ok(())
}

fn is_touchstone_path(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
        return false;
    };
    let lower = extension.to_ascii_lowercase();
    lower == "ts"
        || lower
            .strip_prefix('s')
            .and_then(|value| value.strip_suffix('p'))
            .is_some_and(|ports| {
                !ports.is_empty() && ports.chars().all(|value| value.is_ascii_digit())
            })
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
