//! Calibration and de-embedding algorithms.
//!
//! Package origin: `skrf/calibration/__init__.py`.

use num_complex::Complex64;

use crate::{Error, Result};

#[allow(
    clippy::module_inception,
    reason = "the nested module preserves the upstream skrf/calibration/calibration.py path"
)]
/// Calibration algorithms, standards, coefficients, and error models.
pub mod calibration;
/// Collections of related calibration results.
pub mod calibration_set;
/// Fixture de-embedding algorithms.
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
