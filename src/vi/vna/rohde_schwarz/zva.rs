//! Origin: `skrf/vi/vna/rohde_schwarz/zva.py`.
//!
//! Rohde & Schwarz ZVA family driver.

use std::ops::{Deref, DerefMut};

use crate::Result;

use super::super::InstrumentSession;
use super::rs_vna::{RohdeSchwarzVna, RsFamily};
/// Rohde & Schwarz ZVA vector network analyzer.
///
/// This driver covers the ZVA family, including the ZVA40.
pub struct Zva<S: InstrumentSession>(
    /// Shared Rohde & Schwarz driver implementation.
    pub RohdeSchwarzVna<S>,
);

impl<S: InstrumentSession> Zva<S> {
    /// Connects to a ZVA and initializes the shared driver.
    ///
    /// # Errors
    ///
    /// Returns an error if instrument identification or initial channel setup fails.
    pub fn new(address: impl Into<String>, session: S) -> Result<Self> {
        RohdeSchwarzVna::new(address, session, RsFamily::Zva).map(Self)
    }
}

impl<S: InstrumentSession> Deref for Zva<S> {
    type Target = RohdeSchwarzVna<S>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: InstrumentSession> DerefMut for Zva<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
