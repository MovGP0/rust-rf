//! Backend-neutral port of `skrf/notebook/bokeh_.py`.
//!
//! Python dynamically installs Bokeh methods on `Network`. Rust exposes the
//! same plot construction explicitly and leaves rendering to the selected
//! plotting backend.

use crate::plotting::{Component, Parameter, Plot, complex_plot, network_plot};
use crate::{Network, Result};

/// Rust representation of the meaningful `default_kwargs` fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BokehPlotOptions {
    /// Network parameter representation to plot.
    pub parameter: Parameter,
    /// Component such as magnitude, decibels, phase, real, or imaginary.
    pub component: Component,
    /// Optional `(output, input)` port pair; `None` plots every trace.
    pub ports: Option<(usize, usize)>,
    /// Whether an integrating frontend should display the result immediately.
    pub show: bool,
}

impl Default for BokehPlotOptions {
    fn default() -> Self {
        Self {
            parameter: Parameter::Scattering,
            component: Component::Decibels,
            ports: None,
            show: true,
        }
    }
}

/// Builds rectangular plot data for a network.
///
/// This is the backend-neutral Rust analogue of the Python function returning
/// a Bokeh `Figure`.
///
/// # Errors
///
/// Returns an error if the requested network plot cannot be constructed.
pub fn plot_rectangular(network: &Network, options: BokehPlotOptions) -> Result<Plot> {
    network_plot(network, options.parameter, options.component, options.ports)
}

/// Compatibility wrapper for the initial Rust port's public name.
///
/// # Errors
///
/// Returns an error if the rectangular plot cannot be constructed.
pub fn rectangular_plot(network: &Network, component: Component) -> Result<Plot> {
    plot_rectangular(
        network,
        BokehPlotOptions {
            component,
            ..BokehPlotOptions::default()
        },
    )
}

/// Backend-neutral polar plot data corresponding to the upstream placeholder.
///
/// # Errors
///
/// Returns an error if the polar plot cannot be constructed.
pub fn plot_polar(
    network: &Network,
    parameter: Parameter,
    ports: Option<(usize, usize)>,
) -> Result<Plot> {
    complex_plot(network, parameter, ports, true)
}

/// Names of plot methods that Python's `use_bokeh` installs dynamically.
pub const BOKEH_NETWORK_METHODS: [&str; 7] = [
    "plot_s_db",
    "plot_s_db10",
    "plot_s_mag",
    "plot_s_deg",
    "plot_s_re",
    "plot_s_im",
    "plot_s_vswr",
];

/// Rust has no runtime monkey-patching; this exposes the methods available to adapters.
#[must_use]
pub const fn use_bokeh() -> &'static [&'static str] {
    &BOKEH_NETWORK_METHODS
}
