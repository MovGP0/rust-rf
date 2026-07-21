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
    pub parameter: Parameter,
    pub component: Component,
    pub ports: Option<(usize, usize)>,
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

/// Port of `plot_rectangular`.
pub fn plot_rectangular(network: &Network, options: BokehPlotOptions) -> Result<Plot> {
    network_plot(network, options.parameter, options.component, options.ports)
}

/// Compatibility wrapper for the initial Rust port's public name.
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
pub const fn use_bokeh() -> &'static [&'static str] {
    &BOKEH_NETWORK_METHODS
}
