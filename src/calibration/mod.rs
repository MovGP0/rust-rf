//! Calibration and de-embedding algorithms.
//!
//! Package origin: `skrf/calibration/__init__.py`.

use num_complex::Complex64;

use crate::{Error, Result};

#[allow(
    clippy::module_inception,
    reason = "the nested module preserves the upstream skrf/calibration/calibration.py path"
)]
pub mod calibration;
pub mod calibration_set;
pub mod deembedding;

pub use calibration::*;
pub use deembedding::*;

fn ensure_nonzero(value: Complex64, message: &str) -> Result<()> {
    if value.norm_sqr() <= f64::EPSILON {
        Err(Error::Unsupported(message.to_owned()))
    } else {
        Ok(())
    }
}
