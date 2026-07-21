//! Origin: `skrf/vi/vna/rohde_schwarz/zna.py`.

use std::ops::{Deref, DerefMut};

use crate::Result;

use super::super::InstrumentSession;
use super::rs_vna::{RohdeSchwarzVna, RsFamily};
pub struct Zna<S: InstrumentSession>(pub RohdeSchwarzVna<S>);

impl<S: InstrumentSession> Zna<S> {
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
