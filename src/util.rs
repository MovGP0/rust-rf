//! General-purpose utilities.
//!
//! This module contains sortable timestamps, numeric search and smoothing,
//! filename helpers, structured records, homogeneous collections, recursive
//! text replacement, unique-name generation, and a deterministic progress bar.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::ops::{Index, IndexMut, RangeInclusive};
use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::{Datelike, Local, NaiveDate, NaiveDateTime, Timelike};
use ndarray::Array1;
use num_traits::ToPrimitive;

use crate::{Error, Result};

/// Returns the current local time as a sortable timestamp string.
///
/// The format is `YYYY.MM.DD.hh.mm.ss.ffffff`, which is convenient for
/// date-stamped filenames. Use [`parse_now_string`] to convert it back to a
/// date-time value.
#[must_use]
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
/// Both timestamps with microseconds (`YYYY.MM.DD.hh.mm.ss.ffffff`) and the
/// shorter second-resolution form are accepted.
///
/// # Errors
///
/// Returns an error if the timestamp contains non-numeric fields, has the wrong
/// number of fields, or does not represent a valid date and time.
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
    let year = i32::try_from(fields[0]).map_err(|error| {
        Error::Parse(format!("timestamp year is outside the i32 range: {error}"))
    })?;
    let date = NaiveDate::from_ymd_opt(year, fields[1], fields[2])
        .ok_or_else(|| Error::Parse("timestamp contains an invalid date".to_owned()))?;
    date.and_hms_micro_opt(
        fields[3],
        fields[4],
        fields[5],
        fields.get(6).copied().unwrap_or(0),
    )
    .ok_or_else(|| Error::Parse("timestamp contains an invalid time".to_owned()))
}

/// Finds the index of the value numerically closest to `target`.
///
/// # Errors
///
/// Returns an error when `values` is empty.
pub fn find_nearest_index(values: &Array1<f64>, target: f64) -> Result<usize> {
    values
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (*left - target).abs().total_cmp(&(*right - target).abs()))
        .map(|(index, _)| index)
        .ok_or_else(|| Error::IncompatibleShape("cannot search an empty array".to_owned()))
}

/// Finds the value numerically closest to `target`.
///
/// # Errors
///
/// Returns an error when `values` is empty.
pub fn find_nearest(values: &Array1<f64>, target: f64) -> Result<f64> {
    Ok(values[find_nearest_index(values, target)?])
}

/// Returns the inclusive index range closest to the requested domain endpoints.
///
/// # Errors
///
/// Returns an error when `values` is empty.
pub fn slice_domain(values: &Array1<f64>, domain: (f64, f64)) -> Result<RangeInclusive<usize>> {
    Ok(find_nearest_index(values, domain.0)?..=find_nearest_index(values, domain.1)?)
}

/// Returns the final extension of a path, without the leading period.
///
/// Returns `None` when the path has no non-empty UTF-8 extension.
pub fn extension(path: impl AsRef<Path>) -> Option<String> {
    path.as_ref()
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.is_empty())
        .map(str::to_owned)
}

/// Returns a path's basename without its final extension.
///
/// Returns `None` when the file stem is not valid UTF-8.
pub fn basename_without_extension(path: impl AsRef<Path>) -> Option<String> {
    path.as_ref()
        .file_stem()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
}

/// Runs `git describe` in a repository, returning `None` when no description exists.
///
/// # Errors
///
/// Returns an error if Git cannot be started or its successful output is not
/// valid UTF-8.
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

/// A value and the fields parsed from its structured dictionary key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredRecord<T> {
    /// Fields obtained by splitting the original key.
    pub fields: Vec<String>,
    /// Value associated with the original key.
    pub value: T,
}

/// Converts dictionary entries with delimited keys into typed records.
///
/// This supports file naming conventions used as a lightweight database: each
/// key is split by `delimiter`, while its associated value is retained in the
/// resulting [`StructuredRecord`].
///
/// # Errors
///
/// Returns an error when `delimiter` is empty.
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
/// Returns the paths of files whose contents changed.
///
/// # Errors
///
/// Returns an error when `find` is empty, `directory` is not a directory, a
/// matching file is not UTF-8, or a filesystem operation fails.
///
/// # Reference
///
/// Adapted from [this recursive find-and-replace approach](https://stackoverflow.com/questions/4205854/python-way-to-recursively-find-and-replace-string-in-text-files).
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

/// Finds another occurrence of `value` in a slice.
///
/// `exclude` identifies an index that must not be considered, allowing callers
/// to search for duplicates of an item already present in the slice. Returns the
/// first matching index or `None`.
pub fn duplicate_index<T: PartialEq>(
    value: &T,
    values: &[T],
    exclude: Option<usize>,
) -> Option<usize> {
    values.iter().enumerate().find_map(|(index, candidate)| {
        (Some(index) != exclude && candidate == value).then_some(index)
    })
}

/// Adds or increments a `_NN` suffix until `name` is unique in `names`.
///
/// The optional `exclude` index is ignored when checking for duplicates. This
/// is useful when renaming an existing item in place.
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

/// Window used to smooth a one-dimensional signal.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SmoothingWindow {
    /// A flat window, producing a moving average.
    #[default]
    Flat,
    /// A Hanning window.
    Hanning,
    /// A Hamming window.
    Hamming,
    /// A Bartlett window.
    Bartlett,
    /// A Blackman window.
    Blackman,
}

/// Smooths a one-dimensional signal with a window of the requested length.
///
/// The normalized window is convolved with reflected copies of the signal at
/// both ends, reducing boundary transients. A [`SmoothingWindow::Flat`] window
/// produces moving-average smoothing.
///
/// # Errors
///
/// Returns an error when the smoothing window is longer than the input signal.
/// Window lengths below three return an unchanged copy.
///
/// # Reference
///
/// Based on the [SciPy Cookbook signal-smoothing recipe](https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html).
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
        .collect::<Result<Vec<_>>>()?;
    let total = weights.iter().sum::<f64>();
    for weight in &mut weights {
        *weight /= total;
    }

    let mut full = vec![0.0; reflected.len() + window_length - 1];
    for (input, value) in reflected.iter().copied().enumerate() {
        for (weight, coefficient) in weights.iter().copied().enumerate() {
            full[input + weight] = value.mul_add(coefficient, full[input + weight]);
        }
    }
    let same_start = (window_length - 1) / 2;
    let same = &full[same_start..same_start + reflected.len()];
    Ok(Array1::from_vec(
        same[window_length - 1..same.len() - (window_length - 1)].to_vec(),
    ))
}

fn window_weight(window: SmoothingWindow, index: usize, length: usize) -> Result<f64> {
    let index = index
        .to_f64()
        .ok_or_else(|| Error::Unsupported("smoothing index cannot be represented as f64".into()))?;
    let length_minus_one = (length - 1).to_f64().ok_or_else(|| {
        Error::Unsupported("smoothing window length cannot be represented as f64".into())
    })?;
    let phase = std::f64::consts::TAU * index / length_minus_one;
    Ok(match window {
        SmoothingWindow::Flat => 1.0,
        SmoothingWindow::Hanning => 0.5f64.mul_add(-phase.cos(), 0.5),
        SmoothingWindow::Hamming => 0.46f64.mul_add(-phase.cos(), 0.54),
        SmoothingWindow::Bartlett => {
            let middle = length_minus_one / 2.0;
            2.0 / length_minus_one * (middle - (index - middle).abs())
        }
        SmoothingWindow::Blackman => {
            let base = 0.5f64.mul_add(-phase.cos(), 0.42);
            0.08f64.mul_add((2.0 * phase).cos(), base)
        }
    })
}

/// A homogeneous sequence with bulk mapping and predicate-based selection.
///
/// Unlike the dynamic Python `HomoList`, Rust operations are expressed through
/// typed closures: [`map`](Self::map) applies an operation to every value,
/// [`matching_indexes`](Self::matching_indexes) searches values, and
/// [`select`](Self::select) builds a subset.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HomoList<T> {
    /// Values in the homogeneous sequence.
    pub values: Vec<T>,
}

impl<T> HomoList<T> {
    /// Creates a homogeneous list from an iterator of values.
    #[must_use]
    pub fn new(values: impl IntoIterator<Item = T>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }

    /// Returns the number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` when the list contains no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Clones values at the requested indexes into a new homogeneous list.
    ///
    /// # Errors
    ///
    /// Returns an error when an index is outside the list.
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

    /// Returns the indexes of values for which `predicate` is `true`.
    pub fn matching_indexes(&self, predicate: impl Fn(&T) -> bool) -> Vec<usize> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| predicate(value).then_some(index))
            .collect()
    }

    /// Applies `operation` to every value and returns the mapped homogeneous list.
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

/// A homogeneous ordered mapping with bulk mapping and predicate-based selection.
///
/// Unlike the dynamic Python `HomoDict`, Rust operations are expressed through
/// typed closures: [`map_values`](Self::map_values) applies an operation to each
/// value, [`matching_keys`](Self::matching_keys) searches values, and
/// [`select`](Self::select) builds a subset.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HomoDict<K, V> {
    /// Ordered key-value storage.
    pub values: BTreeMap<K, V>,
}

impl<K: Ord, V> HomoDict<K, V> {
    /// Creates a homogeneous dictionary from key-value pairs.
    #[must_use]
    pub fn new(values: impl IntoIterator<Item = (K, V)>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }

    /// Returns the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` when the dictionary contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns keys whose values satisfy `predicate`.
    pub fn matching_keys(&self, predicate: impl Fn(&V) -> bool) -> Vec<&K> {
        self.values
            .iter()
            .filter_map(|(key, value)| predicate(value).then_some(key))
            .collect()
    }

    /// Clones the requested entries into a new homogeneous dictionary.
    ///
    /// # Errors
    ///
    /// Returns an error when a requested key does not exist.
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

    /// Applies `operation` to every value while preserving the keys.
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

/// A deterministic text progress bar for long-running operations.
///
/// This is useful for operations such as collecting many VNA measurements. It
/// renders progress without writing directly to a terminal, so the caller can
/// choose where and when to display it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgressBar {
    iterations: usize,
    label: String,
    width: usize,
    elapsed: usize,
}

impl ProgressBar {
    /// Creates a progress bar for `iterations` expected items and a label.
    ///
    /// # Errors
    ///
    /// Returns an error when `iterations` is zero.
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

    /// Sets the completed item count, clamped to the configured total.
    pub fn update(&mut self, elapsed: usize) {
        self.elapsed = elapsed.min(self.iterations);
    }

    /// Advances the completed item count by one.
    pub fn advance(&mut self) {
        self.update(self.elapsed + 1);
    }

    /// Renders the current progress bar and completion summary.
    #[must_use]
    pub fn render(&self) -> String {
        let interior = self.width - 2;
        let percent = self
            .elapsed
            .saturating_mul(100)
            .saturating_add(self.iterations / 2)
            / self.iterations;
        let filled = interior.saturating_mul(percent).saturating_add(50) / 100;
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
