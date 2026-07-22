//! Transmission-line media and network constructors.
//!
//! [`Media`] defines propagation constant $\gamma$ and characteristic
//! impedance $`Z_0`$, then supplies generic constructors such as [`Media::line`]
//! and [`Media::delay_short`]. Concrete media—including [`Freespace`] and
//! [`RectangularWaveguide`]—provide the physical quantities while reusing
//! those network-building operations.

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

/// Circular-waveguide transmission media.
pub mod circular_waveguide;
/// Coaxial transmission media.
pub mod coaxial;
/// Coplanar-waveguide transmission media.
pub mod cpw;
/// Media defined by attenuation, permittivity, loss tangent, and impedance.
pub mod defined_a_ep_tand_z0;
/// Idealized couplers and multiport devices.
pub mod device;
/// Distributed-circuit transmission media.
pub mod distributed_circuit;
/// Free-space transmission media.
pub mod freespace;
#[allow(
    clippy::module_inception,
    reason = "the nested module preserves the upstream skrf/media/media.py path"
)]
/// Base media types and shared network constructors.
pub mod media;
/// Microstrip transmission media.
pub mod mline;
/// Rectangular-waveguide transmission media.
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

/// Conventional public alias for [`Cpw`].
pub type CPW = Cpw;

/// Conventional public alias for [`MicrostripLine`].
pub type MLine = MicrostripLine;
