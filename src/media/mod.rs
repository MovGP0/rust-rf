use std::fmt;
use std::path::Path;

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

use crate::constants::{FREE_SPACE_PERMEABILITY, FREE_SPACE_PERMITTIVITY, SPEED_OF_LIGHT};
use crate::math::{
    bessel_j_zero, complete_elliptic_integral_first_kind, db_to_nepers, random_complex,
    random_gaussian_polar,
};
use crate::{Error, Frequency, Network, Result, SParameterDefinition};

pub mod circular_waveguide;
pub mod coaxial;
pub mod cpw;
pub mod defined_a_ep_tand_z0;
pub mod device;
pub mod distributed_circuit;
pub mod freespace;
#[allow(
    clippy::module_inception,
    reason = "the nested module preserves the upstream skrf/media/media.py path"
)]
pub mod media;
pub mod mline;
pub mod rectangular_waveguide;

pub use circular_waveguide::*;
pub use coaxial::*;
pub use cpw::*;
pub use defined_a_ep_tand_z0::*;
pub use device::*;
pub use distributed_circuit::*;
pub use freespace::*;
pub use media::*;
pub use mline::*;
pub use rectangular_waveguide::*;

/// Upstream-compatible public type name for `skrf.media.CPW`.
pub type CPW = Cpw;

/// Upstream-compatible public type name for `skrf.media.MLine`.
pub type MLine = MicrostripLine;
