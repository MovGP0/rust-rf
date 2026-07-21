use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::ops::{Index, IndexMut, RangeInclusive};
use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::{Datelike, Local, NaiveDate, NaiveDateTime, Timelike};
use ndarray::Array1;

use crate::{Error, Result};

/// Current local timestamp in the sortable format used by `skrf.util.now_string`.
pub fn now_string() -> String {
    let now = Local::now().naive_local();
    format!(
        "{:04}.{:02}.{:02}.{:02}.{:02}.{:02}.{:06}",
        now.year(),
        now.month(),
        now.day(),
        now.hour(),
        now.minute(),
        now.second(),
        now.and_utc().timestamp_subsec_micros()
    )
}

/// Parses a timestamp produced by [`now_string`].
///
/// Origin: `skrf.util.now_string_2_dt`.
pub fn parse_now_string(value: &str) -> Result<NaiveDateTime> {
    let fields = value
        .split('.')
        .map(|field| {
            field
                .parse::<u32>()
                .map_err(|error| Error::Parse(format!("invalid timestamp component: {error}")))
        })
        .collect::<Result<Vec<_>>>()?;
    if fields.len() != 6 && fields.len() != 7 {
        return Err(Error::Parse(
            "timestamp must contain year through seconds and optional microseconds".to_owned(),
        ));
    }
    let date = NaiveDate::from_ymd_opt(fields[0] as i32, fields[1], fields[2])
        .ok_or_else(|| Error::Parse("timestamp contains an invalid date".to_owned()))?;
    date.and_hms_micro_opt(
        fields[3],
        fields[4],
        fields[5],
        fields.get(6).copied().unwrap_or(0),
    )
    .ok_or_else(|| Error::Parse("timestamp contains an invalid time".to_owned()))
}

/// Port of `skrf.util.find_nearest_index`.
pub fn find_nearest_index(values: &Array1<f64>, target: f64) -> Result<usize> {
    values
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (*left - target).abs().total_cmp(&(*right - target).abs()))
        .map(|(index, _)| index)
        .ok_or_else(|| Error::IncompatibleShape("cannot search an empty array".to_owned()))
}

/// Port of `skrf.util.find_nearest`.
pub fn find_nearest(values: &Array1<f64>, target: f64) -> Result<f64> {
    Ok(values[find_nearest_index(values, target)?])
}

/// Port of `skrf.util.slice_domain` using an inclusive Rust range.
pub fn slice_domain(values: &Array1<f64>, domain: (f64, f64)) -> Result<RangeInclusive<usize>> {
    Ok(find_nearest_index(values, domain.0)?..=find_nearest_index(values, domain.1)?)
}

/// Port of `skrf.util.get_extn`.
pub fn extension(path: impl AsRef<Path>) -> Option<String> {
    path.as_ref()
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.is_empty())
        .map(str::to_owned)
}

/// Port of `skrf.util.basename_noext`.
pub fn basename_without_extension(path: impl AsRef<Path>) -> Option<String> {
    path.as_ref()
        .file_stem()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
}

/// Runs `git describe` in a repository, returning `None` when no description exists.
///
/// Origin: the deprecated `skrf.util.git_version`.
pub fn git_version(repository: impl AsRef<Path>) -> Result<Option<String>> {
    let output = Command::new("git")
        .arg("describe")
        .current_dir(repository)
        .output()?;
    if !output.status.success() {
        return Ok(None);
    }
    let description = String::from_utf8(output.stdout)
        .map_err(|error| Error::Parse(format!("git describe returned invalid UTF-8: {error}")))?
        .trim()
        .to_owned();
    Ok((!description.is_empty()).then_some(description))
}

/// Typed record corresponding to one structured dictionary key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredRecord<T> {
    pub fields: Vec<String>,
    pub value: T,
}

/// Splits structured keys while preserving values in typed Rust records.
///
/// Origin: the deprecated `skrf.util.dict_2_recarray`.
pub fn dictionary_to_records<T: Clone>(
    values: &BTreeMap<String, T>,
    delimiter: &str,
) -> Result<Vec<StructuredRecord<T>>> {
    if delimiter.is_empty() {
        return Err(Error::Unsupported(
            "structured-record delimiter must not be empty".to_owned(),
        ));
    }
    Ok(values
        .iter()
        .map(|(key, value)| StructuredRecord {
            fields: key.split(delimiter).map(str::to_owned).collect(),
            value: value.clone(),
        })
        .collect())
}

/// Recursively replaces text in UTF-8 files matching `*`, `*.extension`, or an exact name.
///
/// Origin: the deprecated `skrf.util.findReplace`.
pub fn replace_in_files(
    directory: impl AsRef<Path>,
    find: &str,
    replacement: &str,
    file_pattern: &str,
) -> Result<Vec<PathBuf>> {
    if find.is_empty() {
        return Err(Error::Unsupported("find text must not be empty".to_owned()));
    }
    let directory = directory.as_ref();
    if !directory.is_dir() {
        return Err(Error::Unsupported(format!(
            "replacement root '{}' is not a directory",
            directory.display()
        )));
    }
    let mut changed = Vec::new();
    replace_in_directory(directory, find, replacement, file_pattern, &mut changed)?;
    Ok(changed)
}

fn replace_in_directory(
    directory: &Path,
    find: &str,
    replacement: &str,
    file_pattern: &str,
    changed: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            replace_in_directory(&path, find, replacement, file_pattern, changed)?;
        } else if matches_file_pattern(&path, file_pattern) {
            let text = fs::read_to_string(&path)?;
            let replaced = text.replace(find, replacement);
            if replaced != text {
                fs::write(&path, replaced)?;
                changed.push(path);
            }
        }
    }
    Ok(())
}

fn matches_file_pattern(path: &Path, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(extension) = pattern.strip_prefix("*.") {
        return path.extension().and_then(|value| value.to_str()) == Some(extension);
    }
    path.file_name().and_then(|value| value.to_str()) == Some(pattern)
}

/// Port of `skrf.util.has_duplicate_value`.
pub fn duplicate_index<T: PartialEq>(
    value: &T,
    values: &[T],
    exclude: Option<usize>,
) -> Option<usize> {
    values.iter().enumerate().find_map(|(index, candidate)| {
        (Some(index) != exclude && candidate == value).then_some(index)
    })
}

/// Port of `skrf.util.unique_name`.
pub fn unique_name(name: &str, names: &[String], exclude: Option<usize>) -> String {
    let candidate = name.to_owned();
    if duplicate_index(&candidate, names, exclude).is_none() {
        return candidate;
    }
    let bytes = name.as_bytes();
    let (base, start) = if bytes.len() >= 3
        && bytes[bytes.len() - 3] == b'_'
        && bytes[bytes.len() - 2..].iter().all(u8::is_ascii_digit)
    {
        (
            &name[..name.len() - 3],
            name[name.len() - 2..].parse::<usize>().unwrap_or(1),
        )
    } else {
        (name, 1)
    };
    for suffix in start..100 {
        let candidate = format!("{base}_{suffix:02}");
        if duplicate_index(&candidate, names, exclude).is_none() {
            return candidate;
        }
    }
    format!("{base}_99")
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SmoothingWindow {
    #[default]
    Flat,
    Hanning,
    Hamming,
    Bartlett,
    Blackman,
}

/// Port of `skrf.util.smooth`.
pub fn smooth(
    values: &Array1<f64>,
    window_length: usize,
    window: SmoothingWindow,
) -> Result<Array1<f64>> {
    if values.len() < window_length {
        return Err(Error::IncompatibleShape(
            "input vector must be longer than the smoothing window".to_owned(),
        ));
    }
    if window_length < 3 {
        return Ok(values.clone());
    }
    let mut reflected = Vec::with_capacity(values.len() + 2 * (window_length - 1));
    reflected.extend((1..window_length).rev().map(|index| values[index]));
    reflected.extend(values.iter().copied());
    reflected.extend(
        (values.len() - window_length..values.len() - 1)
            .rev()
            .map(|index| values[index]),
    );
    let mut weights = (0..window_length)
        .map(|index| window_weight(window, index, window_length))
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    weights.iter_mut().for_each(|weight| *weight /= total);

    let mut full = vec![0.0; reflected.len() + window_length - 1];
    for (input, value) in reflected.iter().copied().enumerate() {
        for (weight, coefficient) in weights.iter().copied().enumerate() {
            full[input + weight] += value * coefficient;
        }
    }
    let same_start = (window_length - 1) / 2;
    let same = &full[same_start..same_start + reflected.len()];
    Ok(Array1::from_vec(
        same[window_length - 1..same.len() - (window_length - 1)].to_vec(),
    ))
}

fn window_weight(window: SmoothingWindow, index: usize, length: usize) -> f64 {
    let phase = std::f64::consts::TAU * index as f64 / (length - 1) as f64;
    match window {
        SmoothingWindow::Flat => 1.0,
        SmoothingWindow::Hanning => 0.5 - 0.5 * phase.cos(),
        SmoothingWindow::Hamming => 0.54 - 0.46 * phase.cos(),
        SmoothingWindow::Bartlett => {
            let middle = (length - 1) as f64 / 2.0;
            2.0 / (length - 1) as f64 * (middle - (index as f64 - middle).abs())
        }
        SmoothingWindow::Blackman => 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos(),
    }
}

/// Typed Rust counterpart of the deprecated dynamic `skrf.util.HomoList`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HomoList<T> {
    pub values: Vec<T>,
}

impl<T> HomoList<T> {
    pub fn new(values: impl IntoIterator<Item = T>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn select(&self, indexes: &[usize]) -> Result<Self>
    where
        T: Clone,
    {
        indexes
            .iter()
            .map(|index| {
                self.values.get(*index).cloned().ok_or(Error::InvalidPort {
                    port: *index,
                    ports: self.values.len(),
                })
            })
            .collect::<Result<Vec<_>>>()
            .map(Self::new)
    }

    pub fn matching_indexes(&self, predicate: impl Fn(&T) -> bool) -> Vec<usize> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| predicate(value).then_some(index))
            .collect()
    }

    pub fn map<U>(&self, operation: impl Fn(&T) -> U) -> HomoList<U> {
        HomoList::new(self.values.iter().map(operation))
    }
}

impl<T> Index<usize> for HomoList<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<T> IndexMut<usize> for HomoList<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

/// Typed Rust counterpart of the deprecated dynamic `skrf.util.HomoDict`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HomoDict<K, V> {
    pub values: BTreeMap<K, V>,
}

impl<K: Ord, V> HomoDict<K, V> {
    pub fn new(values: impl IntoIterator<Item = (K, V)>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn matching_keys(&self, predicate: impl Fn(&V) -> bool) -> Vec<&K> {
        self.values
            .iter()
            .filter_map(|(key, value)| predicate(value).then_some(key))
            .collect()
    }

    pub fn select(&self, keys: &[K]) -> Result<Self>
    where
        K: Clone,
        V: Clone,
    {
        keys.iter()
            .map(|key| {
                self.values
                    .get(key)
                    .cloned()
                    .map(|value| (key.clone(), value))
                    .ok_or_else(|| Error::Unsupported("HomoDict key was not found".to_owned()))
            })
            .collect::<Result<Vec<_>>>()
            .map(Self::new)
    }

    pub fn map_values<U>(&self, operation: impl Fn(&V) -> U) -> HomoDict<K, U>
    where
        K: Clone,
    {
        HomoDict::new(
            self.values
                .iter()
                .map(|(key, value)| (key.clone(), operation(value))),
        )
    }
}

impl<K: Ord, V> Index<&K> for HomoDict<K, V> {
    type Output = V;

    fn index(&self, index: &K) -> &Self::Output {
        &self.values[index]
    }
}

/// Deterministic text progress bar replacing the deprecated printing helper.
///
/// Origin: `skrf.util.ProgressBar`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgressBar {
    iterations: usize,
    label: String,
    width: usize,
    elapsed: usize,
}

impl ProgressBar {
    pub fn new(iterations: usize, label: impl Into<String>) -> Result<Self> {
        if iterations == 0 {
            return Err(Error::Unsupported(
                "progress bar requires at least one iteration".to_owned(),
            ));
        }
        Ok(Self {
            iterations,
            label: label.into(),
            width: 50,
            elapsed: 0,
        })
    }

    pub fn update(&mut self, elapsed: usize) {
        self.elapsed = elapsed.min(self.iterations);
    }

    pub fn advance(&mut self) {
        self.update(self.elapsed + 1);
    }

    pub fn render(&self) -> String {
        let interior = self.width - 2;
        let percent = (100.0 * self.elapsed as f64 / self.iterations as f64).round() as usize;
        let filled = (interior as f64 * percent as f64 / 100.0).round() as usize;
        let mut bar = format!("[{}{}]", "*".repeat(filled), " ".repeat(interior - filled));
        let percent_text = format!("{percent}%");
        let start = bar.len() / 2 - percent_text.len();
        bar.replace_range(start..start + percent_text.len(), &percent_text);
        format!(
            "{bar}  {} of {} {} complete",
            self.elapsed, self.iterations, self.label
        )
    }
}

impl fmt::Display for ProgressBar {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.render())
    }
}
