use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

use crate::{Error, Frequency, FrequencyUnit, Network, Result};

/// Header, comments, and numeric rows extracted from an instrument CSV block.
///
/// Origin: `skrf/io/csv.py::read_pna_csv` and `AgilentCSV.read`.
#[derive(Clone, Debug, PartialEq)]
pub struct CsvTable {
    pub header: String,
    pub comments: String,
    pub data: Array2<f64>,
}

/// Read the first `BEGIN`/`END` PNA data block and normalize its frequency
/// column to hertz.
pub fn read_pna_csv(path: impl AsRef<Path>) -> Result<CsvTable> {
    let text = fs::read_to_string(path)?;
    let mut table = parse_begin_end_table(&text, true)?;
    let columns = split_header(&table.header, table.data.ncols(), None);
    let unit = frequency_unit_from_column(&columns[0]).ok_or_else(|| {
        Error::Parse(format!(
            "could not parse frequency unit from '{}'",
            columns[0]
        ))
    })?;
    table
        .data
        .column_mut(0)
        .mapv_inplace(|value| value * unit.multiplier());
    Ok(table)
}

/// Parse a PNA CSV into one-port traces, combining real/imaginary or
/// dB/degree column pairs where possible.
pub fn pna_csv_to_networks(path: impl AsRef<Path>) -> Result<Vec<Network>> {
    AgilentCsv::from_path(path)?.networks()
}

/// Parse a four-trace PNA dB/degree export as one two-port network.
pub fn pna_csv_to_two_port(path: impl AsRef<Path>) -> Result<Network> {
    let path = path.as_ref();
    let table = read_pna_csv(path)?;
    let columns = split_header(&table.header, table.data.ncols(), Some(path));
    let points = table.data.nrows();
    let mut s = Array3::zeros((points, 2, 2));
    let mut found = [[false; 2]; 2];
    for (column, heading) in columns.iter().enumerate().skip(1) {
        let lower = heading.to_ascii_lowercase();
        let Some((row, input)) = scattering_coordinates(&lower) else {
            continue;
        };
        if lower.contains("db") {
            let angle_column = columns
                .iter()
                .enumerate()
                .find(|(_, candidate)| {
                    scattering_coordinates(&candidate.to_ascii_lowercase()) == Some((row, input))
                        && candidate.to_ascii_lowercase().contains("deg")
                })
                .map(|(index, _)| index)
                .ok_or_else(|| Error::Parse(format!("'{heading}' has no phase column")))?;
            for point in 0..points {
                s[(point, row, input)] = Complex64::from_polar(
                    10.0_f64.powf(table.data[(point, column)] / 20.0),
                    table.data[(point, angle_column)].to_radians(),
                );
            }
            found[row][input] = true;
        }
    }
    if found.iter().flatten().any(|value| !value) {
        return Err(Error::Unsupported(
            "PNA two-port conversion requires dB/degree columns for S11, S12, S21, and S22"
                .to_owned(),
        ));
    }
    let frequency = Frequency::from_hz(table.data.column(0).to_owned())?;
    let z0 = Array2::from_elem((points, 2), Complex64::new(50.0, 0.0));
    let mut network = Network::new(frequency, s, z0)?;
    network.name = file_stem(path);
    network.comments = table.comments;
    Ok(network)
}

pub fn read_all_csv(
    directory: impl AsRef<Path>,
    contains: Option<&str>,
) -> Result<BTreeMap<String, Network>> {
    let mut networks = BTreeMap::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if path.extension().and_then(|extension| extension.to_str()) != Some("csv")
            || contains.is_some_and(|needle| !filename.contains(needle))
        {
            continue;
        }
        let Some(stem) = path.file_stem() else {
            continue;
        };
        if let Ok(network) = pna_csv_to_two_port(&path) {
            networks.insert(stem.to_string_lossy().into_owned(), network);
        }
    }
    Ok(networks)
}

/// Agilent-style scalar or complex traces versus frequency.
///
/// Origin: `skrf/io/csv.py::AgilentCSV`.
#[derive(Clone, Debug)]
pub struct AgilentCsv {
    pub filename: Option<PathBuf>,
    pub header: String,
    pub comments: String,
    pub data: Array2<f64>,
}

impl AgilentCsv {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)?;
        let table = parse_begin_end_table(&text, false)?;
        Ok(Self {
            filename: Some(path.to_path_buf()),
            header: table.header,
            comments: table.comments,
            data: table.data,
        })
    }

    pub fn parse(text: &str) -> Result<Self> {
        let table = parse_begin_end_table(text, false)?;
        Ok(Self {
            filename: None,
            header: table.header,
            comments: table.comments,
            data: table.data,
        })
    }

    pub fn columns(&self) -> Vec<String> {
        split_header(&self.header, self.data.ncols(), self.filename.as_deref())
    }

    pub fn frequency(&self) -> Result<Frequency> {
        let columns = self.columns();
        let unit = frequency_unit_from_column(&columns[0]).unwrap_or(FrequencyUnit::Hz);
        Frequency::from_values(self.data.column(0).to_owned(), unit)
    }

    pub fn trace_count(&self) -> usize {
        self.data.ncols().saturating_sub(1)
    }

    pub fn scalar_networks(&self) -> Result<Vec<Network>> {
        let frequency = self.frequency()?;
        let columns = self.columns();
        (1..self.data.ncols())
            .map(|column| {
                let values = self.data.column(column).mapv(|value| {
                    if columns[column].to_ascii_lowercase().contains("db") {
                        Complex64::new(10.0_f64.powf(value / 20.0), 0.0)
                    } else {
                        Complex64::new(value, 0.0)
                    }
                });
                self.one_port_network(frequency.clone(), values, columns[column].clone())
            })
            .collect()
    }

    pub fn networks(&self) -> Result<Vec<Network>> {
        if self.trace_count() < 2 {
            return self.scalar_networks();
        }
        let frequency = self.frequency()?;
        let columns = self.columns();
        let pair_count = self.trace_count() / 2;
        let mut networks = Vec::with_capacity(pair_count);
        for pair in 0..pair_count {
            let first = pair * 2 + 1;
            let second = first + 1;
            let first_name = columns[first].to_ascii_lowercase();
            let second_name = columns[second].to_ascii_lowercase();
            let values = Array1::from_iter((0..self.data.nrows()).map(|point| {
                let left = self.data[(point, first)];
                let right = self.data[(point, second)];
                if first_name.contains("db") && second_name.contains("deg") {
                    Complex64::from_polar(10.0_f64.powf(left / 20.0), right.to_radians())
                } else {
                    Complex64::new(left, right)
                }
            }));
            networks.push(self.one_port_network(
                frequency.clone(),
                values,
                columns[first].clone(),
            )?);
        }
        Ok(networks)
    }

    pub fn as_columns(&self) -> BTreeMap<String, Array1<f64>> {
        self.columns()
            .into_iter()
            .enumerate()
            .map(|(index, name)| (name, self.data.column(index).to_owned()))
            .collect()
    }

    #[cfg(feature = "dataframe")]
    pub fn to_dataframe(&self) -> Result<polars::frame::DataFrame> {
        use polars::prelude::Column;

        let columns = self
            .columns()
            .into_iter()
            .enumerate()
            .map(|(index, name)| Column::new(name.into(), self.data.column(index).to_vec()))
            .collect();
        polars::frame::DataFrame::new(self.data.nrows(), columns)
            .map_err(|error| Error::Unsupported(format!("DataFrame construction failed: {error}")))
    }

    fn one_port_network(
        &self,
        frequency: Frequency,
        values: Array1<Complex64>,
        name: String,
    ) -> Result<Network> {
        let points = values.len();
        let s = Array3::from_shape_vec((points, 1, 1), values.to_vec()).map_err(|error| {
            Error::IncompatibleShape(format!("could not shape CSV trace: {error}"))
        })?;
        let z0 = Array2::from_elem((points, 1), Complex64::new(50.0, 0.0));
        let mut network = Network::new(frequency, s, z0)?;
        network.name = Some(name);
        network.comments.clone_from(&self.comments);
        Ok(network)
    }
}

/// Read a Rohde & Schwarz ZVA comma-delimited data file.
pub fn read_zva_dat(path: impl AsRef<Path>) -> Result<CsvTable> {
    let text = fs::read_to_string(path)?;
    let mut header = None;
    let mut comments = String::new();
    let mut rows = Vec::new();
    let mut after_header = false;
    for line in text.lines() {
        if line.starts_with('%') {
            comments.push_str(line.trim_start_matches('%'));
            comments.push('\n');
            header = Some(line.to_owned());
            after_header = true;
        } else if after_header && !line.trim().is_empty() {
            rows.push(parse_numeric_csv_line(line)?);
        }
    }
    rows_to_table(
        header.ok_or_else(|| Error::Parse("ZVA header was not found".to_owned()))?,
        comments,
        rows,
    )
}

pub fn zva_dat_to_network(path: impl AsRef<Path>) -> Result<Network> {
    let path = path.as_ref();
    let table = read_zva_dat(path)?;
    let columns = table.header.split(',').map(str::trim).collect::<Vec<_>>();
    let points = table.data.nrows();
    let frequency = Frequency::from_hz(table.data.column(0).to_owned())?;
    let mut s = Array3::zeros((points, 2, 2));
    let mut found = [[false; 2]; 2];
    for (column, heading) in columns.iter().enumerate() {
        let lower = heading.to_ascii_lowercase();
        let Some((row, input)) = scattering_coordinates(&lower) else {
            continue;
        };
        if lower.contains("re") && column + 1 < columns.len() {
            for point in 0..points {
                s[(point, row, input)] =
                    Complex64::new(table.data[(point, column)], table.data[(point, column + 1)]);
            }
            found[row][input] = true;
        } else if lower.contains("db") {
            if let Some(angle) = columns.iter().enumerate().find_map(|(index, candidate)| {
                (scattering_coordinates(&candidate.to_ascii_lowercase()) == Some((row, input))
                    && candidate.to_ascii_lowercase().contains("deg"))
                .then_some(index)
            }) {
                for point in 0..points {
                    s[(point, row, input)] = Complex64::from_polar(
                        10.0_f64.powf(table.data[(point, column)] / 20.0),
                        table.data[(point, angle)].to_radians(),
                    );
                }
                found[row][input] = true;
            }
        }
    }
    if found.iter().flatten().any(|value| !value) {
        return Err(Error::Unsupported(
            "ZVA data does not contain a complete two-port matrix".to_owned(),
        ));
    }
    let z0 = Array2::from_elem((points, 2), Complex64::new(50.0, 0.0));
    let mut network = Network::new(frequency, s, z0)?;
    network.name = file_stem(path);
    network.comments = table.comments;
    Ok(network)
}

/// Read an Anritsu VectorStar CSV into one-port traces.
pub fn vectorstar_csv_to_networks(path: impl AsRef<Path>) -> Result<Vec<Network>> {
    let text = fs::read_to_string(path)?;
    let comments = text
        .lines()
        .filter_map(|line| line.strip_prefix('!'))
        .collect::<Vec<_>>()
        .join("\n");
    let names = comments
        .lines()
        .find(|line| line.starts_with("PARAMETER"))
        .map(|line| line.split(',').skip(1).map(str::trim).collect::<Vec<_>>())
        .ok_or_else(|| Error::Parse("VectorStar PARAMETER comment was not found".to_owned()))?;
    let mut after_header = false;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.starts_with("PNT") {
            after_header = true;
            continue;
        }
        if after_header && !line.starts_with('!') && !line.trim().is_empty() {
            rows.push(parse_numeric_csv_line(line)?);
        }
    }
    let width = rows
        .first()
        .map(Vec::len)
        .ok_or_else(|| Error::Parse("VectorStar file contains no numeric rows".to_owned()))?;
    if width % 3 != 0 || rows.iter().any(|row| row.len() != width) {
        return Err(Error::IncompatibleShape(
            "VectorStar rows must contain frequency/real/imaginary triples".to_owned(),
        ));
    }
    let traces = width / 3;
    if names.len() < traces {
        return Err(Error::Parse(
            "VectorStar PARAMETER list is shorter than its trace data".to_owned(),
        ));
    }
    (0..traces)
        .map(|trace| {
            let frequency =
                Frequency::from_hz(Array1::from_iter(rows.iter().map(|row| row[trace * 3])))?;
            let values = Array1::from_iter(
                rows.iter()
                    .map(|row| Complex64::new(row[trace * 3 + 1], row[trace * 3 + 2])),
            );
            let helper = AgilentCsv {
                filename: None,
                header: String::new(),
                comments: comments.clone(),
                data: Array2::zeros((0, 0)),
            };
            helper.one_port_network(frequency, values, names[trace].to_owned())
        })
        .collect()
}

fn parse_begin_end_table(text: &str, first_block: bool) -> Result<CsvTable> {
    let lines = text.lines().collect::<Vec<_>>();
    let mut comments = String::new();
    for line in &lines {
        if let Some(comment) = line.strip_prefix('!') {
            comments.push_str(comment);
            comments.push('\n');
        }
    }
    let begin = if first_block {
        lines.iter().position(|line| line.starts_with("BEGIN"))
    } else {
        lines.iter().rposition(|line| line.starts_with("BEGIN"))
    }
    .ok_or_else(|| Error::Parse("CSV BEGIN marker was not found".to_owned()))?;
    let end = lines
        .iter()
        .enumerate()
        .skip(begin + 1)
        .find(|(_, line)| line.starts_with("END"))
        .map(|(index, _)| index)
        .ok_or_else(|| Error::Parse("CSV END marker was not found".to_owned()))?;
    let header = lines
        .get(begin + 1)
        .ok_or_else(|| Error::Parse("CSV header was not found".to_owned()))?
        .replace('°', "deg");
    let rows = lines[begin + 2..end]
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| parse_numeric_csv_line(line))
        .collect::<Result<Vec<_>>>()?;
    rows_to_table(header, comments, rows)
}

fn rows_to_table(header: String, comments: String, rows: Vec<Vec<f64>>) -> Result<CsvTable> {
    let width = rows
        .first()
        .map(Vec::len)
        .ok_or_else(|| Error::Parse("CSV data block is empty".to_owned()))?;
    if rows.iter().any(|row| row.len() != width) {
        return Err(Error::IncompatibleShape(
            "CSV rows do not have a consistent number of columns".to_owned(),
        ));
    }
    let height = rows.len();
    let data = Array2::from_shape_vec((height, width), rows.into_iter().flatten().collect())
        .map_err(|error| Error::IncompatibleShape(format!("could not shape CSV data: {error}")))?;
    Ok(CsvTable {
        header,
        comments,
        data,
    })
}

fn parse_numeric_csv_line(line: &str) -> Result<Vec<f64>> {
    line.split(',')
        .map(|value| {
            value.trim().parse::<f64>().map_err(|error| {
                Error::Parse(format!("invalid CSV number '{}': {error}", value.trim()))
            })
        })
        .collect()
}

fn split_header(header: &str, column_count: usize, path: Option<&Path>) -> Vec<String> {
    let traces = column_count.saturating_sub(1);
    if header.matches(',').count() == traces {
        return header.split(',').map(str::to_owned).collect();
    }
    if header.matches("),").count() == traces {
        let split = header.split("),").collect::<Vec<_>>();
        return split
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if index + 1 < split.len() {
                    format!("{value})")
                } else {
                    (*value).to_owned()
                }
            })
            .collect();
    }
    let stem = path
        .and_then(file_stem)
        .unwrap_or_else(|| "trace".to_owned());
    std::iter::once("Freq(?)".to_owned())
        .chain((0..traces).map(|index| format!("{stem}-{index}")))
        .collect()
}

fn frequency_unit_from_column(column: &str) -> Option<FrequencyUnit> {
    let unit = column.split_once('(')?.1.split_once(')')?.0;
    match unit.to_ascii_lowercase().as_str() {
        "hz" => Some(FrequencyUnit::Hz),
        "khz" => Some(FrequencyUnit::KHz),
        "mhz" => Some(FrequencyUnit::MHz),
        "ghz" => Some(FrequencyUnit::GHz),
        "thz" => Some(FrequencyUnit::THz),
        _ => None,
    }
}

fn scattering_coordinates(heading: &str) -> Option<(usize, usize)> {
    for row in 1..=2 {
        for column in 1..=2 {
            if heading.contains(&format!("s{row}{column}")) {
                return Some((row - 1, column - 1));
            }
        }
    }
    None
}

fn file_stem(path: &Path) -> Option<String> {
    path.file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
}
