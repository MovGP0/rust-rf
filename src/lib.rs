#![forbid(unsafe_code)]
#![doc = "Rust port of scikit-rf for RF and microwave engineering."]

pub mod calibration;
pub mod circuit;
pub mod constants;
pub mod data;
pub mod docs;
pub mod error;
pub mod frequency;
pub mod instances;
pub mod io;
pub mod math;
pub mod media;
pub mod network;
pub mod network_set;
pub mod notebook;
pub mod plotting;
pub mod qfactor;
pub mod taper;
pub mod time;
pub mod transmission_line;
pub mod util;
pub mod vector_fitting;

#[cfg(feature = "visa")]
pub mod vi;

pub use calibration::calibration_set;
pub use error::{Error, Result};
pub use frequency::{Frequency, FrequencyUnit, SweepType};
pub use network::{InterpolationMode, Network, NoiseParameters, PortMode, SParameterDefinition};
pub use network_set::{
    NetworkParameter, NetworkScalarAttribute, NetworkSet, NetworkSetAttribute, TunerConstellation,
    function_on_networks, get_set, tuner_constellation,
};
pub use num_complex::Complex64;

/// Crate version, corresponding to Python's `skrf.__version__`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Resolves the plotting environment configured for this process.
pub fn setup_plotting() -> Option<&'static str> {
    plotting::configured_style()
}
