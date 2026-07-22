//! Virtual-instrument interfaces for RF measurement equipment.
//!
//! Drivers implement the common [`Vna`] interface while translating operations
//! to the command syntax required by a particular device and manufacturer. A new
//! driver normally consists of an instrument type implementing that interface
//! and SCPI commands taken from the instrument programming manual.
//!
//! ## Naming conventions
//!
//! | Name | Meaning |
//! | --- | --- |
//! | `frequency` | Frequency-axis settings |
//! | `freq_start` | Start frequency in hertz |
//! | `freq_stop` | Stop frequency in hertz |
//! | `freq_step` | Frequency step in hertz |
//! | `freq_center` | Center frequency in hertz |
//! | `freq_span` | Frequency span in hertz |
//! | `npoints` | Number of frequency points |
//! | `sweep_time` | Sweep duration in seconds |
//! | `sweep_type` | Frequency-point distribution, such as linear or logarithmic |
//! | `sweep_mode` | Trigger behavior, such as continuous or single |
//! | `averaging_on` | Whether averaging is enabled |
//! | `averaging_count` | Measurements combined into an average |
//! | `averaging_mode` | How measurements are averaged |
//! | `if_bandwidth` | Intermediate-frequency bandwidth in hertz |

pub mod scpi_errors;
pub mod validators;
#[cfg(all(
    rust_rf_native_visa,
    windows,
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub mod visa;
pub mod vna;
#[cfg(all(
    rust_rf_native_visa,
    windows,
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub use visa::VisaSession;
pub use vna::hp::{Hp8510C, Hp8720B};
pub use vna::keysight::{FieldFox, Pna};
pub use vna::nanovna::NanoVnaV2;
pub use vna::rohde_schwarz::{RohdeSchwarzVna, Zna, Zva};
pub use vna::{InstrumentSession, Vna};
