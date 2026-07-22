//! Time-domain signal-processing functions.
//!
//! This module provides peak detection, window generation, and time-domain
//! gating for one-port network data.

use ndarray::Array1;
use num_complex::Complex64;
use realfft::RealFftPlanner;
use rustfft::FftPlanner;

use crate::{Error, Network, Result};

/// A sampled window used for time-domain or frequency-domain weighting.
#[derive(Clone, Copy, Debug)]
pub enum Window {
    /// A constant, unweighted window.
    Rectangular,
    /// A Hann window.
    Hann,
    /// A Hamming window.
    Hamming,
    /// A Blackman window.
    Blackman,
    /// A cosine window.
    Cosine,
    /// A Kaiser window with the contained beta parameter.
    Kaiser(f64),
}

/// Selects whether the time gate retains or rejects the gated interval.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GateMode {
    /// Retain the response inside the gate.
    #[default]
    BandPass,
    /// Reject the response inside the gate.
    BandStop,
}

/// Selects the transform used to apply a time-domain gate.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GateMethod {
    /// Transform the gate and convolve it with the frequency-domain data.
    Convolution,
    /// Apply the gate to a complex-valued inverse-FFT response.
    #[default]
    Fft,
    /// Apply the gate to a real-valued inverse-FFT response constructed from a
    /// Hermitian spectrum.
    RealFft,
}

/// Unit used for time-gate coordinates.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TimeUnit {
    /// Seconds.
    #[default]
    Seconds,
    /// Milliseconds.
    Milliseconds,
    /// Microseconds.
    Microseconds,
    /// Nanoseconds.
    Nanoseconds,
    /// Picoseconds.
    Picoseconds,
}

impl TimeUnit {
    const fn multiplier(self) -> f64 {
        match self {
            Self::Seconds => 1.0,
            Self::Milliseconds => 1.0e-3,
            Self::Microseconds => 1.0e-6,
            Self::Nanoseconds => 1.0e-9,
            Self::Picoseconds => 1.0e-12,
        }
    }
}

/// Options for [`time_gate_with_options`].
#[derive(Clone, Debug)]
pub struct TimeGateOptions {
    /// Start of the gate in [`time_unit`](Self::time_unit).
    pub start: Option<f64>,
    /// End of the gate in [`time_unit`](Self::time_unit).
    pub stop: Option<f64>,
    /// Center of the gate in [`time_unit`](Self::time_unit).
    ///
    /// When omitted and a span is supplied, the gate is centered on the
    /// strongest time-domain peak.
    pub center: Option<f64>,
    /// Width of the gate in [`time_unit`](Self::time_unit).
    ///
    /// When omitted, the width is half the distance to the second-strongest
    /// time-domain peak.
    pub span: Option<f64>,
    /// Whether to retain or reject the gated interval.
    pub mode: GateMode,
    /// Window applied across the gated interval.
    pub gate_window: Window,
    /// Transform method used to apply the gate.
    pub method: GateMethod,
    /// Optional frequency-domain window applied before the inverse transform
    /// and removed after the forward transform.
    pub fft_window: Option<Window>,
    /// Unit used by `start`, `stop`, `center`, and `span`.
    pub time_unit: TimeUnit,
}

impl Default for TimeGateOptions {
    fn default() -> Self {
        Self {
            start: None,
            stop: None,
            center: None,
            span: None,
            mode: GateMode::BandPass,
            gate_window: Window::Kaiser(6.0),
            method: GateMethod::Fft,
            fft_window: Some(Window::Cosine),
            time_unit: TimeUnit::Seconds,
        }
    }
}

/// Finds the indexes of peaks in a signed one-dimensional signal.
///
/// Peaks are located from the first-order difference. `threshold` is normalized
/// to the signal range and must be in the inclusive range `0.0..=1.0`.
/// `minimum_distance` suppresses nearby peaks in favor of the one with the
/// greatest amplitude.
///
/// # Errors
///
/// Returns an error when the threshold is outside its valid range or any signal
/// sample is not finite.
///
/// # Notes
///
/// The algorithm is derived from
/// [PeakUtils 1.1.0](http://pythonhosted.org/PeakUtils/index.html).
pub fn peak_indexes(values: &[f64], threshold: f64, minimum_distance: usize) -> Result<Vec<usize>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(Error::Unsupported(
            "peak threshold must be between zero and one".to_owned(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Unsupported(
            "peak detection requires finite samples".to_owned(),
        ));
    }
    if values.len() < 3 {
        return Ok(Vec::new());
    }

    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum.total_cmp(&minimum).is_eq() {
        return Ok(Vec::new());
    }
    let absolute_threshold = threshold.mul_add(maximum - minimum, minimum);
    let mut differences = values
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    fill_plateau_differences(&mut differences);

    let mut peaks = (1..values.len() - 1)
        .filter(|index| {
            differences[*index - 1] > 0.0
                && differences[*index] < 0.0
                && values[*index] > absolute_threshold
        })
        .collect::<Vec<_>>();
    if peaks.len() <= 1 || minimum_distance <= 1 {
        return Ok(peaks);
    }

    peaks.sort_by(|left, right| values[*right].total_cmp(&values[*left]));
    let mut retained = Vec::new();
    for peak in peaks {
        if retained
            .iter()
            .all(|other: &usize| peak.abs_diff(*other) >= minimum_distance)
        {
            retained.push(peak);
        }
    }
    retained.sort_unstable();
    Ok(retained)
}

/// Finds a requested number of peaks in a signal.
///
/// The search starts at `threshold` and progressively lowers it until at least
/// `count` peaks are found. The returned indexes identify the `count` largest
/// detected peaks.
///
/// # Errors
///
/// Returns an error if [`peak_indexes`] rejects the input or if the requested
/// number of peaks cannot be found.
pub fn find_n_peaks(
    values: &[f64],
    count: usize,
    threshold: f64,
    minimum_distance: usize,
) -> Result<Vec<usize>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut threshold = threshold;
    for _ in 0..10 {
        let indexes = peak_indexes(values, threshold.clamp(0.0, 1.0), minimum_distance)?;
        if indexes.len() >= count {
            let mut indexes = indexes;
            indexes.sort_by(|left, right| values[*right].total_cmp(&values[*left]));
            indexes.truncate(count);
            return Ok(indexes);
        }
        threshold *= 0.5;
    }
    Err(Error::Unsupported(format!(
        "could not find {count} peaks in the time-domain signal"
    )))
}

/// Generates `length` samples of a built-in [`Window`].
///
/// # Errors
///
/// Returns an error when a Kaiser window has a non-finite beta parameter.
pub fn window_samples(window: &Window, length: usize) -> Result<Array1<f64>> {
    if length == 0 {
        return Ok(Array1::zeros(0));
    }
    if let Window::Kaiser(beta) = window
        && !beta.is_finite()
    {
        return Err(Error::Unsupported("Kaiser beta must be finite".to_owned()));
    }
    if length == 1 {
        return Ok(Array1::from_vec(vec![1.0]));
    }
    let denominator = f64::from(
        u32::try_from(length)
            .map_err(|_| Error::Unsupported("window length exceeds u32::MAX".to_owned()))?,
    );
    let samples = (0..length)
        .map(|index| -> Result<f64> {
            let index_as_float = f64::from(u32::try_from(index).map_err(|_| {
                Error::Unsupported("window sample index exceeds u32::MAX".to_owned())
            })?);
            let angle = std::f64::consts::TAU * index_as_float / denominator;
            Ok(match window {
                Window::Rectangular => 1.0,
                Window::Hann => 0.5f64.mul_add(-angle.cos(), 0.5),
                Window::Hamming => 0.46f64.mul_add(-angle.cos(), 0.54),
                Window::Blackman => {
                    0.08f64.mul_add((2.0 * angle).cos(), 0.5f64.mul_add(-angle.cos(), 0.42))
                }
                Window::Cosine => {
                    (std::f64::consts::PI * (index_as_float + 0.5) / denominator).sin()
                }
                Window::Kaiser(beta) => {
                    let normalized = 2.0 * index_as_float / denominator - 1.0;
                    modified_bessel_i0(*beta * (1.0 - normalized * normalized).sqrt())
                        / modified_bessel_i0(*beta)
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Array1::from_vec(samples))
}

/// Detects the time span between the two largest response peaks.
///
/// The returned span is expressed in seconds.
///
/// # Errors
///
/// Returns an error when the network is unsuitable for time gating or two
/// peaks cannot be found.
pub fn detect_span(network: &Network) -> Result<f64> {
    validate_time_network(network)?;
    let time_response = inverse_complex_fft(&network_spectrum(network))?;
    let shifted = fft_shift(&time_response);
    let decibels = shifted
        .iter()
        .map(|value| 20.0 * value.norm().max(f64::MIN_POSITIVE).log10())
        .collect::<Vec<_>>();
    let peaks = find_n_peaks(&decibels, 2, 0.9, 1)?;
    let times = network.frequency.time()?;
    Ok((times[peaks[0]] - times[peaks[1]]).abs())
}

/// Applies an FFT-based time-domain gate to a one-port network.
///
/// Gate coordinates are expressed in seconds. Start/stop take precedence over
/// center/span. With no coordinates, the strongest impulse is selected and the
/// gate width is half the distance to the second strongest peak.
///
/// # Notes
///
/// Strong reflections can obscure responses behind them because of multiple
/// reflections. To gate an N-port network, gate each S-parameter independently.
///
/// # Warning
///
/// Sharp gates can make the band edges inaccurate because of FFT properties.
/// The result is not renormalized.
///
/// # Errors
///
/// Returns an error when the network, coordinates, or window are invalid.
pub fn time_gate(
    network: &Network,
    start: Option<f64>,
    stop: Option<f64>,
    center: Option<f64>,
    span: Option<f64>,
    window: Window,
) -> Result<Network> {
    time_gate_with_options(
        network,
        &TimeGateOptions {
            start,
            stop,
            center,
            span,
            gate_window: window,
            fft_window: None,
            ..TimeGateOptions::default()
        },
    )
}

/// Applies a configurable time-domain gate to a one-port network.
///
/// With [`GateMethod::Convolution`], the gate is transformed into the frequency
/// domain and convolved with the network data. [`GateMethod::Fft`] applies the
/// gate to a complex time-domain signal with the same sample count as the
/// positive-frequency input. [`GateMethod::RealFft`] constructs a Hermitian
/// spectrum and applies the gate to a real signal with improved time resolution;
/// this method requires a DC sample.
///
/// If neither endpoint nor center is supplied, the gate is centered on the
/// strongest impulse. If its span is also omitted, the span is derived from the
/// two strongest peaks.
///
/// # Notes
///
/// Strong reflections can obscure responses behind them because of multiple
/// reflections. To gate an N-port network, gate each S-parameter independently.
///
/// # Warning
///
/// Sharp gates can make the band edges inaccurate because of FFT properties.
/// The result is not renormalized.
///
/// # Errors
///
/// Returns an error when the network is not a uniformly sampled one-port, the
/// coordinate combination is invalid, peak detection fails, or the selected
/// window cannot be removed safely.
pub fn time_gate_with_options(network: &Network, options: &TimeGateOptions) -> Result<Network> {
    validate_time_network(network)?;
    let coordinates = scaled_gate_coordinates(options)?;
    let spectrum = network_spectrum(network);
    let frequency_window = frequency_window(options, spectrum.len())?;
    let windowed_spectrum = spectrum
        .iter()
        .zip(frequency_window.iter())
        .map(|(value, weight)| *value * weight)
        .collect::<Vec<_>>();
    let time_length = match options.method {
        GateMethod::RealFft => 2 * spectrum.len() - 1,
        GateMethod::Convolution | GateMethod::Fft => spectrum.len(),
    };
    let frequency_step = network.frequency.step().ok_or_else(|| {
        Error::InvalidFrequency("time gating requires at least two points".to_owned())
    })?;
    let time = centered_time_axis(time_length, frequency_step)?;
    let time_response = match options.method {
        GateMethod::RealFft => fft_shift(
            &irfft(&Array1::from_vec(windowed_spectrum.clone()), time_length)?
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect::<Vec<_>>(),
        ),
        GateMethod::Convolution | GateMethod::Fft => {
            fft_shift(&inverse_complex_fft(&windowed_spectrum)?)
        }
    };
    let (gate_start, gate_stop) = resolve_gate_bounds(network, &time, &time_response, coordinates)?;

    let start_index = nearest_index(&time, gate_start).ok_or_else(|| {
        Error::InvalidFrequency("time gating requires a non-empty time axis".to_owned())
    })?;
    let stop_index = nearest_index(&time, gate_stop).ok_or_else(|| {
        Error::InvalidFrequency("time gating requires a non-empty time axis".to_owned())
    })?;
    let (start_index, stop_index) = if start_index <= stop_index {
        (start_index, stop_index)
    } else {
        (stop_index, start_index)
    };
    let samples = window_samples(&options.gate_window, stop_index - start_index + 1)?;
    let mut gate = vec![Complex64::new(0.0, 0.0); time_response.len()];
    let mut gated_time = vec![Complex64::new(0.0, 0.0); time_response.len()];
    for (offset, sample) in samples.iter().enumerate() {
        gate[start_index + offset] = Complex64::new(*sample, 0.0);
        gated_time[start_index + offset] = time_response[start_index + offset] * sample;
    }

    let mut gated_spectrum = match options.method {
        GateMethod::Convolution => {
            let unshifted_gate = ifft_shift(&gate);
            let mut kernel = forward_complex_fft(&unshifted_gate);
            let scale = f64::from(u32::try_from(kernel.len()).map_err(|_| {
                Error::Unsupported("convolution kernel length exceeds u32::MAX".to_owned())
            })?);
            for value in &mut kernel {
                *value /= scale;
            }
            let kernel = fft_shift(&kernel);
            circular_convolution(&windowed_spectrum, &kernel)
        }
        GateMethod::Fft => forward_complex_fft(&ifft_shift(&gated_time)),
        GateMethod::RealFft => {
            let real_time = ifft_shift(&gated_time)
                .into_iter()
                .map(|value| value.re)
                .collect::<Vec<_>>();
            real_fft(&real_time)?
        }
    };
    for (value, weight) in gated_spectrum.iter_mut().zip(frequency_window.iter()) {
        if weight.abs() <= f64::EPSILON {
            return Err(Error::Unsupported(
                "the selected frequency window contains a zero sample".to_owned(),
            ));
        }
        *value /= weight;
    }
    let mut gated = network.clone();
    for (point, value) in gated_spectrum.into_iter().enumerate() {
        gated.s[(point, 0, 0)] = match options.mode {
            GateMode::BandPass => value,
            GateMode::BandStop => network.s[(point, 0, 0)] - value,
        };
    }
    Ok(gated)
}

fn scaled_gate_coordinates(options: &TimeGateOptions) -> Result<[Option<f64>; 4]> {
    let coordinates = [options.start, options.stop, options.center, options.span];
    if coordinates
        .into_iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Error::Unsupported(
            "time gate coordinates must be finite".to_owned(),
        ));
    }
    let multiplier = options.time_unit.multiplier();
    Ok(coordinates.map(|coordinate| coordinate.map(|value| value * multiplier)))
}

fn resolve_gate_bounds(
    network: &Network,
    time: &Array1<f64>,
    time_response: &[Complex64],
    coordinates: [Option<f64>; 4],
) -> Result<(f64, f64)> {
    let [start, stop, center, span] = coordinates;
    match (start, stop) {
        (Some(start), Some(stop)) => Ok((start.min(stop), start.max(stop))),
        (Some(_), None) | (None, Some(_)) => Err(Error::Unsupported(
            "time gate start and stop must be supplied together".to_owned(),
        )),
        (None, None) => {
            let center = if let Some(center) = center {
                center
            } else {
                let peak = time_response
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.norm_sqr().total_cmp(&right.norm_sqr()))
                    .ok_or_else(|| {
                        Error::InvalidFrequency(
                            "time gating requires a non-empty frequency axis".to_owned(),
                        )
                    })?
                    .0;
                time[peak]
            };
            let span = match span {
                Some(span) if span >= 0.0 => span,
                Some(_) => {
                    return Err(Error::Unsupported(
                        "time gate span must not be negative".to_owned(),
                    ));
                }
                None => detect_span(network)? / 2.0,
            };
            Ok((center - span / 2.0, center + span / 2.0))
        }
    }
}

/// NumPy-compatible inverse real FFT, including `1/N` normalization.
///
/// # Errors
///
/// Returns an error when the spectrum shape does not match `output_length` or
/// the inverse FFT implementation cannot process the supplied buffers.
pub fn irfft(spectrum: &Array1<Complex64>, output_length: usize) -> Result<Array1<f64>> {
    if output_length == 0 || spectrum.len() != output_length / 2 + 1 {
        return Err(Error::IncompatibleShape(format!(
            "an inverse real FFT of length {output_length} requires {} complex bins, got {}",
            output_length / 2 + 1,
            spectrum.len()
        )));
    }
    let mut planner = RealFftPlanner::<f64>::new();
    let transform = planner.plan_fft_inverse(output_length);
    let mut input = spectrum.to_vec();
    let mut output = transform.make_output_vec();
    transform
        .process(&mut input, &mut output)
        .map_err(|error| Error::Unsupported(format!("inverse real FFT failed: {error}")))?;
    let scale = f64::from(
        u32::try_from(output_length)
            .map_err(|_| Error::Unsupported("inverse FFT length exceeds u32::MAX".to_owned()))?,
    );
    for value in &mut output {
        *value /= scale;
    }
    Ok(Array1::from_vec(output))
}

fn frequency_window(options: &TimeGateOptions, frequency_points: usize) -> Result<Vec<f64>> {
    if options.method == GateMethod::Convolution {
        return Ok(vec![1.0; frequency_points]);
    }
    let Some(window) = options.fft_window.as_ref() else {
        return Ok(vec![1.0; frequency_points]);
    };
    if options.method == GateMethod::RealFft {
        let samples = window_samples(window, 2 * frequency_points)?;
        Ok(samples.iter().skip(frequency_points).copied().collect())
    } else {
        Ok(window_samples(window, frequency_points)?.to_vec())
    }
}

fn centered_time_axis(length: usize, frequency_step_hz: f64) -> Result<Array1<f64>> {
    let length_as_float = f64::from(
        u32::try_from(length)
            .map_err(|_| Error::Unsupported("time-axis length exceeds u32::MAX".to_owned()))?,
    );
    let step = 1.0 / (length_as_float * frequency_step_hz);
    let start = -(length_as_float / 2.0).floor() * step;
    let values = (0..length)
        .map(|index| {
            let index_as_float = f64::from(u32::try_from(index).map_err(|_| {
                Error::Unsupported("time-axis sample index exceeds u32::MAX".to_owned())
            })?);
            Ok(index_as_float.mul_add(step, start))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Array1::from_vec(values))
}

fn real_fft(time_values: &[f64]) -> Result<Vec<Complex64>> {
    let mut planner = RealFftPlanner::<f64>::new();
    let transform = planner.plan_fft_forward(time_values.len());
    let mut input = time_values.to_vec();
    let mut output = transform.make_output_vec();
    transform
        .process(&mut input, &mut output)
        .map_err(|error| Error::Unsupported(format!("real FFT failed: {error}")))?;
    Ok(output)
}

fn circular_convolution(values: &[Complex64], kernel: &[Complex64]) -> Vec<Complex64> {
    let length = values.len();
    (0..length)
        .map(|output| {
            kernel
                .iter()
                .enumerate()
                .map(|(offset, coefficient)| {
                    values[(output + length - offset) % length] * coefficient
                })
                .sum()
        })
        .collect()
}

fn validate_time_network(network: &Network) -> Result<()> {
    if network.ports() != 1 {
        return Err(Error::IncompatibleShape(
            "time gating requires a one-port Network".to_owned(),
        ));
    }
    if network.frequency_points() < 2 {
        return Err(Error::InvalidFrequency(
            "time gating requires at least two frequency points".to_owned(),
        ));
    }
    let step = network.frequency.step().unwrap_or(0.0);
    if !network.frequency.is_monotonic_increasing() || step <= 0.0 {
        return Err(Error::InvalidFrequency(
            "time gating requires an increasing, uniformly spaced frequency axis".to_owned(),
        ));
    }
    let tolerance = step.abs() * 1.0e-9;
    if network
        .frequency
        .values_hz()
        .windows(2)
        .into_iter()
        .any(|pair| ((pair[1] - pair[0]) - step).abs() > tolerance)
    {
        return Err(Error::InvalidFrequency(
            "time gating requires a uniformly spaced frequency axis".to_owned(),
        ));
    }
    Ok(())
}

fn network_spectrum(network: &Network) -> Vec<Complex64> {
    network
        .s
        .outer_iter()
        .map(|matrix| matrix[(0, 0)])
        .collect()
}

fn inverse_complex_fft(spectrum: &[Complex64]) -> Result<Vec<Complex64>> {
    let mut values = spectrum.to_vec();
    let mut planner = FftPlanner::new();
    planner.plan_fft_inverse(values.len()).process(&mut values);
    let scale = f64::from(
        u32::try_from(values.len())
            .map_err(|_| Error::Unsupported("inverse FFT length exceeds u32::MAX".to_owned()))?,
    );
    for value in &mut values {
        *value /= scale;
    }
    Ok(values)
}

fn forward_complex_fft(time_values: &[Complex64]) -> Vec<Complex64> {
    let mut values = time_values.to_vec();
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(values.len()).process(&mut values);
    values
}

fn fft_shift(values: &[Complex64]) -> Vec<Complex64> {
    let split = values.len().div_ceil(2);
    values[split..]
        .iter()
        .chain(values[..split].iter())
        .copied()
        .collect()
}

fn ifft_shift(values: &[Complex64]) -> Vec<Complex64> {
    let split = values.len() / 2;
    values[split..]
        .iter()
        .chain(values[..split].iter())
        .copied()
        .collect()
}

fn nearest_index(values: &Array1<f64>, target: f64) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            (**left - target).abs().total_cmp(&(**right - target).abs())
        })
        .map(|(index, _)| index)
}

fn fill_plateau_differences(differences: &mut [f64]) {
    if differences.iter().all(|value| *value == 0.0) {
        return;
    }
    loop {
        let previous = differences.to_vec();
        let mut changed = false;
        for (index, difference) in differences.iter_mut().enumerate() {
            if *difference != 0.0 {
                continue;
            }
            let right = previous.get(index + 1).copied().unwrap_or(0.0);
            let left = index
                .checked_sub(1)
                .and_then(|left| previous.get(left))
                .copied()
                .unwrap_or(0.0);
            let replacement = if right == 0.0 { left } else { right };
            if replacement != 0.0 {
                *difference = replacement;
                changed = true;
            }
        }
        if !changed || differences.iter().all(|value| *value != 0.0) {
            break;
        }
    }
}

fn modified_bessel_i0(value: f64) -> f64 {
    let quarter_square = value * value / 4.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    for order in 1..=100 {
        term *= quarter_square / f64::from(order * order);
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            break;
        }
    }
    sum
}
