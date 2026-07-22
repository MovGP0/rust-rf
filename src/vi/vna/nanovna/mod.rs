//! `NanoVNA` drivers.
//!
//! The module exposes [`nanovna::NanoVnaV2`], the Rust counterpart of the upstream
//! `NanoVNA` V2 driver.

#[allow(
    clippy::module_inception,
    reason = "the nested module preserves the upstream skrf/vi/vna/nanovna/nanovna.py path"
)]
/// NanoVNA V2 driver implementation.
pub mod nanovna;

pub use nanovna::*;
