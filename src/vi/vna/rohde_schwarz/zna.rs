//! Origin: `skrf/vi/vna/rohde_schwarz/zna.py`.
//!
//! Rohde & Schwarz ZNA family driver.

use std::ops::{Deref, DerefMut};

use crate::Result;

use super::super::InstrumentSession;
use super::rs_vna::{RohdeSchwarzVna, RsFamily};
/// Rohde & Schwarz ZNA vector network analyzer.
///
/// Supported two- and four-port models include ZNA26, ZNA43, ZNA50, and ZNA67.
pub struct Zna<S: InstrumentSession>(
    /// Shared Rohde & Schwarz driver implementation.
    pub RohdeSchwarzVna<S>,
);

impl<S: InstrumentSession> Zna<S> {
    /// Connects to a ZNA and initializes the shared driver.
    ///
    /// # Errors
    ///
    /// Returns an error when the instrument identification query or initial channel setup fails.
    pub fn new(address: impl Into<String>, session: S) -> Result<Self> {
        RohdeSchwarzVna::new(address, session, RsFamily::Zna).map(Self)
    }
}

impl<S: InstrumentSession> Deref for Zna<S> {
    type Target = RohdeSchwarzVna<S>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: InstrumentSession> DerefMut for Zna<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
