//! Backend-neutral plotting data for networks and network sets.
//!
//! Functions return simple plot, heatmap, violin, and band structures. Optional
//! renderers can consume these structures without coupling RF calculations to
//! a particular plotting library.

#[cfg(feature = "plot")]
use std::path::Path;

use ndarray::Array3;
use num_complex::Complex64;
use num_traits::ToPrimitive;

use crate::math::inverse_fft_centered;
use crate::network::{passivity, reciprocity, s_to_y, s_to_z};
use crate::{Error, Network, NetworkSet, Result};

/// Network parameter selected for plotting.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Parameter {
    /// Scattering parameters.
    #[default]
    Scattering,
    /// Impedance parameters.
    Impedance,
    /// Admittance parameters.
    Admittance,
}

/// Scalar projection of a complex network parameter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Component {
    /// Magnitude as $20\log_{10}|x|$.
    #[default]
    Decibels,
    /// Magnitude as $10\log_{10}|x|$.
    Decibels10,
    /// Linear magnitude $|x|$.
    Magnitude,
    /// Phase in degrees.
    PhaseDegrees,
    /// Real component.
    Real,
    /// Imaginary component.
    Imaginary,
    /// Voltage standing-wave ratio.
    Vswr,
}

#[derive(Clone, Debug, PartialEq)]
/// One labeled XY trace.
pub struct PlotSeries {
    /// Legend label.
    pub label: String,
    /// Horizontal coordinates.
    pub x: Vec<f64>,
    /// Vertical coordinates.
    pub y: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
/// Backend-neutral collection of XY traces and axis labels.
pub struct Plot {
    /// Plot title.
    pub title: String,
    /// Horizontal-axis label.
    pub x_label: String,
    /// Vertical-axis label.
    pub y_label: String,
    /// Traces to render.
    pub series: Vec<PlotSeries>,
}

#[derive(Clone, Debug, PartialEq)]
/// Rectangular scalar field for contour or image rendering.
pub struct Heatmap {
    /// Heatmap title.
    pub title: String,
    /// Horizontal coordinates.
    pub x: Vec<f64>,
    /// Vertical coordinates.
    pub y: Vec<f64>,
    /// Values indexed by vertical then horizontal coordinate.
    pub values: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
/// Samples forming one slice of a violin plot.
pub struct ViolinSlice {
    /// Position of the slice.
    pub x: f64,
    /// Sample values at that position.
    pub samples: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Colored rectangular band spanning an X interval and Y range.
pub struct ShadedBand {
    /// Start of the horizontal interval.
    pub x_start: f64,
    /// End of the horizontal interval.
    pub x_stop: f64,
    /// Lower vertical bound.
    pub y_minimum: f64,
    /// Upper vertical bound.
    pub y_maximum: f64,
    /// Normalized position used to select the band color.
    pub color_fraction: f64,
}

/// Returns the embedded scikit-rf style when `SKRF_PLOT_ENV=pylab-skrf-style`.
///
/// This is the Rust equivalent of `skrf.setup_plotting` without process-global
/// plotting side effects.
#[must_use]
pub fn configured_style() -> Option<&'static str> {
    std::env::var("SKRF_PLOT_ENV")
        .ok()
        .filter(|value| value.eq_ignore_ascii_case("pylab-skrf-style"))
        .map(|_| crate::data::SKRF_MATPLOTLIB_STYLE)
}

/// Builds rectangular plot data for every port pair or one selected pair.
///
/// Origin: `skrf.plotting.plot_rectangular`, `subplot_params`, and generated
/// `Network.plot_*` methods.
///
/// # Errors
///
/// Returns an error if parameter conversion or port selection fails.
pub fn network_plot(
    network: &Network,
    parameter: Parameter,
    component: Component,
    ports: Option<(usize, usize)>,
) -> Result<Plot> {
    let values = parameter_values(network, parameter)?;
    let pairs = selected_ports(network.ports(), ports)?;
    let series = pairs
        .into_iter()
        .map(|(output, input)| PlotSeries {
            label: format!("{}{}{}", parameter_symbol(parameter), output + 1, input + 1),
            x: network.frequency.values_hz().to_vec(),
            y: (0..network.frequency_points())
                .map(|point| project(values[(point, output, input)], component))
                .collect(),
        })
        .collect();
    Ok(Plot {
        title: network.name.clone().unwrap_or_else(|| "Network".to_owned()),
        x_label: "Frequency (Hz)".to_owned(),
        y_label: component_label(component).to_owned(),
        series,
    })
}

/// Builds complex-plane data. With `polar`, x is phase in radians and y is magnitude.
///
/// Origin: `plot_complex_rectangular`, `plot_complex_polar`, and `plot_polar`.
///
/// # Errors
///
/// Returns an error if parameter conversion or port selection fails.
pub fn complex_plot(
    network: &Network,
    parameter: Parameter,
    ports: Option<(usize, usize)>,
    polar: bool,
) -> Result<Plot> {
    let values = parameter_values(network, parameter)?;
    let series = selected_ports(network.ports(), ports)?
        .into_iter()
        .map(|(output, input)| {
            let points =
                (0..network.frequency_points()).map(|point| values[(point, output, input)]);
            let (x, y) = if polar {
                points.map(|value| (value.arg(), value.norm())).unzip()
            } else {
                points.map(|value| (value.re, value.im)).unzip()
            };
            PlotSeries {
                label: format!("{}{}{}", parameter_symbol(parameter), output + 1, input + 1),
                x,
                y,
            }
        })
        .collect();
    Ok(Plot {
        title: network.name.clone().unwrap_or_else(|| "Network".to_owned()),
        x_label: if polar { "Phase (rad)" } else { "Real" }.to_owned(),
        y_label: if polar { "Magnitude" } else { "Imaginary" }.to_owned(),
        series,
    })
}

/// Smith-chart data uses normalized complex scattering values.
///
/// # Errors
///
/// Returns an error if complex scattering plot construction fails.
pub fn smith_plot(network: &Network, ports: Option<(usize, usize)>) -> Result<Plot> {
    let mut plot = complex_plot(network, Parameter::Scattering, ports, false)?;
    plot.title = format!("{} - Smith chart", plot.title);
    "Normalized resistance".clone_into(&mut plot.x_label);
    "Normalized reactance".clone_into(&mut plot.y_label);
    Ok(plot)
}

/// Builds diagonal passivity-metric traces for one or every port.
///
/// # Errors
///
/// Returns an error if passivity calculation fails or `port` is invalid.
pub fn passivity_plot(network: &Network, port: Option<usize>) -> Result<Plot> {
    let values = passivity(&network.s)?;
    let ports = match port {
        Some(port) if port < network.ports() => vec![port],
        Some(port) => {
            return Err(Error::InvalidPort {
                port,
                ports: network.ports(),
            });
        }
        None => (0..network.ports()).collect(),
    };
    Ok(Plot {
        title: "Passivity".to_owned(),
        x_label: "Frequency (Hz)".to_owned(),
        y_label: "Passivity".to_owned(),
        series: ports
            .into_iter()
            .map(|port| PlotSeries {
                label: format!("port {}", port + 1),
                x: network.frequency.values_hz().to_vec(),
                y: (0..network.frequency_points())
                    .map(|point| values[(point, port, port)].re)
                    .collect(),
            })
            .collect(),
    })
}

/// Builds traces of reciprocity error $|S_{ij}-S_{ji}|$.
///
/// # Errors
///
/// Returns an error if reciprocity calculation fails.
pub fn reciprocity_plot(network: &Network, decibels: bool) -> Result<Plot> {
    let values = reciprocity(&network.s)?;
    let mut series = Vec::new();
    for output in 0..network.ports() {
        for input in 0..output {
            series.push(PlotSeries {
                label: format!("{}{} - {}{}", output + 1, input + 1, input + 1, output + 1),
                x: network.frequency.values_hz().to_vec(),
                y: (0..network.frequency_points())
                    .map(|point| {
                        let value = values[(point, output, input)];
                        if decibels {
                            20.0 * value.log10()
                        } else {
                            value
                        }
                    })
                    .collect(),
            });
        }
    }
    Ok(Plot {
        title: "Reciprocity error".to_owned(),
        x_label: "Frequency (Hz)".to_owned(),
        y_label: if decibels {
            "Magnitude (dB)"
        } else {
            "Magnitude"
        }
        .to_owned(),
        series,
    })
}

/// Distance of `Sij / Sji` from unity (upstream reciprocity metric #2).
///
/// # Errors
///
/// Returns an error if a reciprocal ratio cannot be represented.
pub fn reciprocity2_plot(network: &Network, decibels: bool) -> Result<Plot> {
    let mut series = Vec::new();
    for output in 0..network.ports() {
        for input in 0..output {
            series.push(PlotSeries {
                label: format!("ports {}{}", output + 1, input + 1),
                x: network.frequency.values_hz().to_vec(),
                y: (0..network.frequency_points())
                    .map(|point| {
                        let reverse = network.s[(point, input, output)];
                        let value = if reverse == Complex64::new(0.0, 0.0) {
                            f64::INFINITY
                        } else {
                            (Complex64::new(1.0, 0.0) - network.s[(point, output, input)] / reverse)
                                .norm()
                        };
                        if decibels {
                            20.0 * value.log10()
                        } else {
                            value
                        }
                    })
                    .collect(),
            });
        }
    }
    Ok(Plot {
        title: "Reciprocity metric #2".to_owned(),
        x_label: "Frequency (Hz)".to_owned(),
        y_label: if decibels {
            "Distance (dB)"
        } else {
            "Distance"
        }
        .to_owned(),
        series,
    })
}

/// Centered inverse-FFT S-parameter plot data.
///
/// # Errors
///
/// Returns an error if port selection or inverse-FFT calculation fails.
pub fn time_domain_plot(
    network: &Network,
    component: Component,
    ports: Option<(usize, usize)>,
) -> Result<Plot> {
    let pairs = selected_ports(network.ports(), ports)?;
    let time = network.frequency.time()?;
    let series = pairs
        .into_iter()
        .map(|(output, input)| {
            let spectrum = ndarray::Array1::from_iter(
                (0..network.frequency_points()).map(|point| network.s[(point, output, input)]),
            );
            let values = inverse_fft_centered(&spectrum);
            PlotSeries {
                label: format!("S{}{} time", output + 1, input + 1),
                x: time.to_vec(),
                y: values
                    .iter()
                    .map(|value| project(*value, component))
                    .collect(),
            }
        })
        .collect();
    Ok(Plot {
        title: "Time-domain scattering".to_owned(),
        x_label: "Time (s)".to_owned(),
        y_label: component_label(component).to_owned(),
        series,
    })
}

/// Mean with one-standard-deviation bounds for a `NetworkSet` component.
///
/// # Errors
///
/// Returns an error if the set is empty or the selected ports are invalid.
pub fn uncertainty_plot(
    set: &NetworkSet,
    component: Component,
    output: usize,
    input: usize,
) -> Result<Plot> {
    let first = set.networks.first().ok_or_else(|| {
        Error::IncompatibleShape("uncertainty plotting requires a non-empty NetworkSet".to_owned())
    })?;
    selected_ports(first.ports(), Some((output, input)))?;
    let points = first.frequency_points();
    let projected = set
        .networks
        .iter()
        .map(|network| {
            (0..points)
                .map(|point| project(network.s[(point, output, input)], component))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let sample_count = projected.len().to_f64().ok_or_else(|| {
        Error::Unsupported("sample count cannot be represented as f64".to_owned())
    })?;
    let mean = (0..points)
        .map(|point| projected.iter().map(|values| values[point]).sum::<f64>() / sample_count)
        .collect::<Vec<_>>();
    let deviation = (0..points)
        .map(|point| {
            (projected
                .iter()
                .map(|values| (values[point] - mean[point]).powi(2))
                .sum::<f64>()
                / sample_count)
                .sqrt()
        })
        .collect::<Vec<_>>();
    let x = first.frequency.values_hz().to_vec();
    Ok(Plot {
        title: set
            .name
            .clone()
            .unwrap_or_else(|| "NetworkSet uncertainty".to_owned()),
        x_label: "Frequency (Hz)".to_owned(),
        y_label: component_label(component).to_owned(),
        series: vec![
            PlotSeries {
                label: "mean".to_owned(),
                x: x.clone(),
                y: mean.clone(),
            },
            PlotSeries {
                label: "mean - sigma".to_owned(),
                x: x.clone(),
                y: mean
                    .iter()
                    .zip(&deviation)
                    .map(|(mean, deviation)| mean - deviation)
                    .collect(),
            },
            PlotSeries {
                label: "mean + sigma".to_owned(),
                x,
                y: mean
                    .iter()
                    .zip(deviation)
                    .map(|(mean, deviation)| mean + deviation)
                    .collect(),
            },
        ],
    })
}

/// Builds mean traces with symmetric standard-deviation bounds.
///
/// # Errors
///
/// Returns an error if `deviations` is invalid or uncertainty calculation fails.
pub fn uncertainty_bounds_plot(
    set: &NetworkSet,
    component: Component,
    output: usize,
    input: usize,
    deviations: f64,
) -> Result<Plot> {
    if !deviations.is_finite() || deviations < 0.0 {
        return Err(Error::Unsupported(
            "uncertainty deviations must be finite and non-negative".to_owned(),
        ));
    }
    let mut plot = uncertainty_plot(set, component, output, input)?;
    let mean = plot.series[0].y.clone();
    for series in &mut plot.series[1..] {
        for (point, value) in series.y.iter_mut().enumerate() {
            *value = deviations.mul_add(*value - mean[point], mean[point]);
        }
    }
    Ok(plot)
}

/// Builds mean traces bounded by the minimum and maximum set members.
///
/// # Errors
///
/// Returns an error if set projection or port selection fails.
pub fn minmax_bounds_plot(
    set: &NetworkSet,
    component: Component,
    output: usize,
    input: usize,
) -> Result<Plot> {
    let (first, projected) = projected_set(set, component, output, input)?;
    let points = first.frequency_points();
    let x = first.frequency.values_hz().to_vec();
    let aggregate = |operation: fn(f64, f64) -> f64, initial: f64| {
        (0..points)
            .map(|point| {
                projected
                    .iter()
                    .map(|values| values[point])
                    .fold(initial, operation)
            })
            .collect::<Vec<_>>()
    };
    Ok(Plot {
        title: "NetworkSet min/max bounds".to_owned(),
        x_label: "Frequency (Hz)".to_owned(),
        y_label: component_label(component).to_owned(),
        series: vec![
            PlotSeries {
                label: "minimum".to_owned(),
                x: x.clone(),
                y: aggregate(f64::min, f64::INFINITY),
            },
            PlotSeries {
                label: "maximum".to_owned(),
                x,
                y: aggregate(f64::max, f64::NEG_INFINITY),
            },
        ],
    })
}

/// Builds total, within-group, and between-group uncertainty traces.
///
/// # Errors
///
/// Returns an error if uncertainty or port-selection calculation fails.
pub fn uncertainty_decomposition_plot(
    set: &NetworkSet,
    output: usize,
    input: usize,
) -> Result<Plot> {
    let magnitude = uncertainty_plot(set, Component::Magnitude, output, input)?;
    let phase = uncertainty_plot(set, Component::PhaseDegrees, output, input)?;
    Ok(Plot {
        title: "Uncertainty decomposition".to_owned(),
        x_label: "Frequency (Hz)".to_owned(),
        y_label: "Standard deviation".to_owned(),
        series: vec![
            PlotSeries {
                label: "magnitude sigma".to_owned(),
                x: magnitude.series[0].x.clone(),
                y: magnitude.series[2]
                    .y
                    .iter()
                    .zip(&magnitude.series[0].y)
                    .map(|(upper, mean)| upper - mean)
                    .collect(),
            },
            PlotSeries {
                label: "phase sigma (degrees)".to_owned(),
                x: phase.series[0].x.clone(),
                y: phase.series[2]
                    .y
                    .iter()
                    .zip(&phase.series[0].y)
                    .map(|(upper, mean)| upper - mean)
                    .collect(),
            },
        ],
    })
}

/// Builds $\log_{10}(\sigma)$ versus mean magnitude for a selected S-parameter.
///
/// # Errors
///
/// Returns an error if uncertainty calculation or port selection fails.
pub fn log_sigma_plot(set: &NetworkSet, output: usize, input: usize) -> Result<Plot> {
    let mut plot = uncertainty_plot(set, Component::Magnitude, output, input)?;
    "Log sigma".clone_into(&mut plot.title);
    "log10(sigma)".clone_into(&mut plot.y_label);
    plot.series = vec![PlotSeries {
        label: "log sigma".to_owned(),
        x: plot.series[0].x.clone(),
        y: plot.series[2]
            .y
            .iter()
            .zip(&plot.series[0].y)
            .map(|(upper, mean)| (upper - mean).log10())
            .collect(),
    }];
    Ok(plot)
}

/// Builds a network-set signature heatmap over member and frequency indices.
///
/// # Errors
///
/// Returns an error if set projection or port selection fails.
pub fn signature(
    set: &NetworkSet,
    component: Component,
    output: usize,
    input: usize,
) -> Result<Heatmap> {
    let (first, projected) = projected_set(set, component, output, input)?;
    let member_indices = (0..projected.len())
        .map(|index| {
            index.to_f64().ok_or_else(|| {
                Error::Unsupported("network index cannot be represented as f64".to_owned())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Heatmap {
        title: "NetworkSet signature".to_owned(),
        x: first.frequency.values_hz().to_vec(),
        y: member_indices,
        values: projected,
    })
}

/// Returns per-frequency sample slices for a violin plot.
///
/// # Errors
///
/// Returns an error if set projection or port selection fails.
pub fn violin_data(
    set: &NetworkSet,
    component: Component,
    output: usize,
    input: usize,
) -> Result<Vec<ViolinSlice>> {
    let (first, projected) = projected_set(set, component, output, input)?;
    Ok((0..first.frequency_points())
        .map(|point| ViolinSlice {
            x: first.frequency.values_hz()[point],
            samples: projected.iter().map(|values| values[point]).collect(),
        })
        .collect())
}

/// Builds one plot frame per network-set member.
///
/// # Errors
///
/// Returns an error if any member plot cannot be constructed.
pub fn animation_frames(
    set: &NetworkSet,
    component: Component,
    ports: Option<(usize, usize)>,
) -> Result<Vec<Plot>> {
    set.networks
        .iter()
        .map(|network| network_plot(network, Parameter::Scattering, component, ports))
        .collect()
}

/// Validates and packages gridded values as a [`Heatmap`].
///
/// # Errors
///
/// Returns an error if the value grid dimensions do not match the axes.
pub fn contour_data(
    x: &[f64],
    y: &[f64],
    values: &[Vec<f64>],
    title: impl Into<String>,
) -> Result<Heatmap> {
    if values.len() != y.len() || values.iter().any(|row| row.len() != x.len()) {
        return Err(Error::IncompatibleShape(
            "contour values must have one row per y and one column per x".to_owned(),
        ));
    }
    Ok(Heatmap {
        title: title.into(),
        x: x.to_vec(),
        y: y.to_vec(),
        values: values.to_vec(),
    })
}

/// Creates alternating normalized-color bands between consecutive X edges.
///
/// # Errors
///
/// Returns an error if edges are not increasing or the Y range is invalid.
pub fn shaded_bands(edges: &[f64], y_range: (f64, f64)) -> Result<Vec<ShadedBand>> {
    if edges.len() < 2
        || edges.windows(2).any(|pair| pair[0] >= pair[1])
        || !y_range.0.is_finite()
        || !y_range.1.is_finite()
        || y_range.0 >= y_range.1
    {
        return Err(Error::Unsupported(
            "shaded bands require increasing edges and a finite y range".to_owned(),
        ));
    }
    let color_steps = edges
        .len()
        .to_f64()
        .ok_or_else(|| Error::Unsupported("edge count cannot be represented as f64".to_owned()))?;
    edges
        .windows(2)
        .enumerate()
        .map(|(index, pair)| {
            let color_index = index.to_f64().ok_or_else(|| {
                Error::Unsupported("band index cannot be represented as f64".to_owned())
            })?;
            Ok(ShadedBand {
                x_start: pair[0],
                x_stop: pair[1],
                y_minimum: y_range.0,
                y_maximum: y_range.1,
                color_fraction: color_index / color_steps,
            })
        })
        .collect()
}

/// Builds a two-point complex vector trace beginning at `offset`.
#[must_use]
pub fn vector_plot(value: Complex64, offset: Complex64) -> Plot {
    Plot {
        title: "Complex vector".to_owned(),
        x_label: "Real".to_owned(),
        y_label: "Imaginary".to_owned(),
        series: vec![PlotSeries {
            label: "vector".to_owned(),
            x: vec![offset.re, offset.re + value.re],
            y: vec![offset.im, offset.im + value.im],
        }],
    }
}

#[cfg(feature = "plot")]
/// Renders a plot to an SVG file.
///
/// # Errors
///
/// Returns an error if bounds calculation, file creation, or rendering fails.
pub fn render_svg(plot: &Plot, path: impl AsRef<Path>, size: (u32, u32)) -> Result<()> {
    use plotters::prelude::*;

    let (x_min, x_max) = bounds(
        plot.series
            .iter()
            .flat_map(|series| series.x.iter().copied()),
    )?;
    let (y_min, y_max) = bounds(
        plot.series
            .iter()
            .flat_map(|series| series.y.iter().copied()),
    )?;
    let root = SVGBackend::new(path.as_ref(), size).into_drawing_area();
    root.fill(&WHITE).map_err(plot_error)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(&plot.title, ("sans-serif", 24))
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(plot_error)?;
    chart
        .configure_mesh()
        .x_desc(&plot.x_label)
        .y_desc(&plot.y_label)
        .draw()
        .map_err(plot_error)?;
    let colors = [&BLUE, &RED, &MAGENTA, &GREEN, &CYAN, &BLACK];
    for (index, series) in plot.series.iter().enumerate() {
        chart
            .draw_series(LineSeries::new(
                series.x.iter().copied().zip(series.y.iter().copied()),
                colors[index % colors.len()],
            ))
            .map_err(plot_error)?
            .label(&series.label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], colors[index % colors.len()])
            });
    }
    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .map_err(plot_error)?;
    root.present().map_err(plot_error)
}

fn projected_set(
    set: &NetworkSet,
    component: Component,
    output: usize,
    input: usize,
) -> Result<(&Network, Vec<Vec<f64>>)> {
    let first = set.networks.first().ok_or_else(|| {
        Error::IncompatibleShape("plotting requires a non-empty NetworkSet".to_owned())
    })?;
    selected_ports(first.ports(), Some((output, input)))?;
    let projected = set
        .networks
        .iter()
        .map(|network| {
            if network.frequency != first.frequency || network.ports() != first.ports() {
                return Err(Error::IncompatibleShape(
                    "NetworkSet plot members must share frequency and port dimensions".to_owned(),
                ));
            }
            Ok((0..network.frequency_points())
                .map(|point| project(network.s[(point, output, input)], component))
                .collect())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((first, projected))
}

fn parameter_values(network: &Network, parameter: Parameter) -> Result<Array3<Complex64>> {
    match parameter {
        Parameter::Scattering => Ok(network.s.clone()),
        Parameter::Impedance => s_to_z(&network.s, &network.z0, network.s_definition),
        Parameter::Admittance => s_to_y(&network.s, &network.z0, network.s_definition),
    }
}

fn selected_ports(ports: usize, selected: Option<(usize, usize)>) -> Result<Vec<(usize, usize)>> {
    if let Some((output, input)) = selected {
        if output >= ports {
            return Err(Error::InvalidPort {
                port: output,
                ports,
            });
        }
        if input >= ports {
            return Err(Error::InvalidPort { port: input, ports });
        }
        Ok(vec![(output, input)])
    } else {
        Ok((0..ports)
            .flat_map(|input| (0..ports).map(move |output| (output, input)))
            .collect())
    }
}

fn project(value: Complex64, component: Component) -> f64 {
    match component {
        Component::Decibels => 20.0 * value.norm().log10(),
        Component::Decibels10 => 10.0 * value.norm().log10(),
        Component::Magnitude => value.norm(),
        Component::PhaseDegrees => value.arg().to_degrees(),
        Component::Real => value.re,
        Component::Imaginary => value.im,
        Component::Vswr => (1.0 + value.norm()) / (1.0 - value.norm()),
    }
}

const fn parameter_symbol(parameter: Parameter) -> &'static str {
    match parameter {
        Parameter::Scattering => "S",
        Parameter::Impedance => "Z",
        Parameter::Admittance => "Y",
    }
}

const fn component_label(component: Component) -> &'static str {
    match component {
        Component::Decibels => "Magnitude (dB)",
        Component::Decibels10 => "Magnitude (dB10)",
        Component::Magnitude => "Magnitude",
        Component::PhaseDegrees => "Phase (degrees)",
        Component::Real => "Real",
        Component::Imaginary => "Imaginary",
        Component::Vswr => "VSWR",
    }
}

#[cfg(feature = "plot")]
fn bounds(values: impl Iterator<Item = f64>) -> Result<(f64, f64)> {
    let finite = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if finite.is_empty() {
        return Err(Error::Unsupported(
            "plot contains no finite data".to_owned(),
        ));
    }
    let mut minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let mut maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scale = minimum.abs().max(maximum.abs()).max(1.0);
    if (minimum - maximum).abs() <= f64::EPSILON * scale {
        minimum -= 1.0;
        maximum += 1.0;
    }
    Ok((minimum, maximum))
}

#[cfg(feature = "plot")]
fn plot_error(error: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("plot rendering failed: {error}"))
}
