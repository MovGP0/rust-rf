//! Interfaces for vector network analyzers from multiple manufacturers.
//!
//! [`Vna`] provides the shared session, channel, SCPI, and value-transfer
//! behavior used by the concrete HP, Keysight, `NanoVNA`, and Rohde & Schwarz
//! drivers.
//!
//! ## Writing a driver
//!
//! A driver should first establish whether the instrument uses SCPI and whether
//! it supports multiple independent channels. Instrument-wide settings belong on
//! the driver type; per-measurement settings belong on its channel representation.
//! Drivers use [`format_command`] to substitute values in command templates and
//! the types in [`crate::vi::validators`] to convert between Rust values and the
//! command or response format expected by the instrument.
//!
//! Command templates use angle-bracket placeholders. For example,
//! `SENS<channel>:FREQ:STAR <value>` becomes `SENS1:FREQ:STAR 100000` when the
//! parameter map contains `channel = 1` and `value = 100000`.

pub mod hp;
pub mod keysight;
pub mod nanovna;
pub mod rohde_schwarz;
#[allow(
    clippy::module_inception,
    reason = "the nested module preserves the upstream skrf/vi/vna/vna.py path"
)]
pub mod vna;

pub use hp::hp8510c_sweep_plan;
pub use vna::*;
