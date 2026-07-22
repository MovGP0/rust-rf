//! Embedded material properties and example/test networks.
//!
//! [`MATERIALS`] contains common conductor and dielectric properties, while
//! [`DATA`] provides embedded Touchstone networks without requiring a runtime
//! data directory.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::calibration::Calibration;
use crate::io::Touchstone;
use crate::{Error, Network, Result};

/// Electrical properties of a conductor or dielectric material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialProperties {
    /// Electrical resistivity in ohm-meters, when the material is a conductor.
    pub resistivity_ohm_meter: Option<f64>,
    /// Relative dielectric permittivity, when the material is a dielectric.
    pub relative_permittivity: Option<f64>,
    /// Dielectric loss tangent, when the material is a dielectric.
    pub loss_tangent: Option<f64>,
}

impl MaterialProperties {
    const fn conductor(resistivity_ohm_meter: f64) -> Self {
        Self {
            resistivity_ohm_meter: Some(resistivity_ohm_meter),
            relative_permittivity: None,
            loss_tangent: None,
        }
    }

    const fn dielectric(relative_permittivity: f64, loss_tangent: f64) -> Self {
        Self {
            resistivity_ohm_meter: None,
            relative_permittivity: Some(relative_permittivity),
            loss_tangent: Some(loss_tangent),
        }
    }
}

/// Material catalog and aliases ported from `skrf/data/__init__.py`.
pub static MATERIALS: LazyLock<BTreeMap<&'static str, MaterialProperties>> = LazyLock::new(|| {
    let copper = MaterialProperties::conductor(1.68e-8);
    let aluminum = MaterialProperties::conductor(2.82e-8);
    let gold = MaterialProperties::conductor(2.44e-8);
    BTreeMap::from([
        ("copper", copper),
        ("cu", copper),
        ("aluminum", aluminum),
        ("al", aluminum),
        ("gold", gold),
        ("au", gold),
        ("lead", MaterialProperties::conductor(1.0 / 4.56e6)),
        (
            "steel(stainless)",
            MaterialProperties::conductor(1.0 / 1.1e6),
        ),
        ("mylar", MaterialProperties::dielectric(3.1, 500e-4)),
        ("quartz", MaterialProperties::dielectric(3.8, 1.5e-4)),
        ("silicon", MaterialProperties::dielectric(11.68, 8e-4)),
        ("teflon", MaterialProperties::dielectric(2.1, 5e-4)),
        ("duroid 5880", MaterialProperties::dielectric(2.25, 40e-4)),
    ])
});

/// Lazy, embedded example and test networks from `skrf.data.StaticData`.
///
/// The Touchstone files are embedded in the crate so consumers do not need a
/// Python installation or a runtime data directory.
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticData;

impl StaticData {
    /// Loads the embedded `ntwk1.s2p` two-port example network.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn ntwk1(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/ntwk1.s2p"), 2, "ntwk1")
    }

    /// Loads the two-port `line` network spanning 75–110 GHz with 201 points.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn line(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/line.s2p"), 2, "line")
    }

    /// Loads the two-port `open` network spanning 1–10 GHz with 10 points.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn open_2p(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/open.s2p"), 2, "open")
    }

    /// Loads the two-port `short` network spanning 1–10 GHz with 10 points.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn short_2p(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/short.s2p"), 2, "short")
    }

    /// Loads the measured 1 nH series inductor network, including parasitics.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn ind(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/ind.s2p"), 2, "ind")
    }

    /// Loads the two-port `ring slot` network spanning 75–110 GHz with 201 points.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn ring_slot(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/ring slot.s2p"), 2, "ring slot")
    }

    /// Loads the three-port `tee` network spanning 330–500 GHz with 201 points.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn tee(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/tee.s3p"), 3, "tee")
    }

    /// Loads the measured one-port `ring slot` network spanning 75–110 GHz with 101 points.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn ring_slot_meas(&self) -> Result<Network> {
        embedded_network(
            include_bytes!("../data/ring slot measured.s1p"),
            1,
            "ring slot measured",
        )
    }

    /// Loads the WR-2.2 two-port line network.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn wr2p2_line(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/wr2p2,line.s2p"), 2, "wr2p2,line")
    }

    /// Loads the second WR-2.2 two-port line network.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn wr2p2_line1(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/wr2p2,line1.s2p"), 2, "wr2p2,line1")
    }

    /// Loads the WR-2.2 one-port delay-short network.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn wr2p2_delayshort(&self) -> Result<Network> {
        embedded_network(
            include_bytes!("../data/wr2p2,delayshort.s1p"),
            1,
            "wr2p2,delayshort",
        )
    }

    /// Loads the WR-2.2 one-port short network.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn wr2p2_short(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/wr2p2,short.s1p"), 1, "wr2p2,short")
    }

    /// Loads the WR-1.5 two-port line network.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn wr1p5_line(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/wr1p5,line.s2p"), 2, "wr1p5,line")
    }

    /// Loads the WR-1.5 one-port short network.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn wr1p5_short(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/wr1p5,short.s1p"), 1, "wr1p5,short")
    }

    /// Loads the first one-port `ro` example network.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn ro_1(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/ro,1.s1p"), 1, "ro,1")
    }

    /// Loads the second one-port `ro` example network.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn ro_2(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/ro,2.s1p"), 1, "ro,2")
    }

    /// Loads the third one-port `ro` example network.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded Touchstone data cannot be parsed or converted.
    pub fn ro_3(&self) -> Result<Network> {
        embedded_network(include_bytes!("../data/ro,3.s1p"), 1, "ro,3")
    }

    /// The upstream object is a Python pickle and is deliberately not decoded.
    /// Rust calibration objects use the crate's typed serialization instead.
    ///
    /// # Errors
    ///
    /// Always returns an error because the upstream Python pickle is unsafe to decode.
    pub fn one_port_calibration(&self) -> Result<Box<dyn Calibration>> {
        Err(Error::Unsupported(
            "Python pickle data/one_port.cal is unsafe and is not portable to Rust".to_owned(),
        ))
    }
}

/// Global accessor for the embedded example and test networks.
pub const DATA: StaticData = StaticData;

/// Upstream plotting style, embedded for callers that integrate with a plotting backend.
pub const SKRF_MATPLOTLIB_STYLE: &str = include_str!("../data/skrf.mplstyle");

fn embedded_network(bytes: &[u8], ports: usize, name: &str) -> Result<Network> {
    let mut network = Touchstone::from_reader(bytes, ports)?.network()?;
    network.name = Some(name.to_owned());
    Ok(network)
}
