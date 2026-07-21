//! Origin: `skrf/vi/vna/rohde_schwarz/zva.py`.

use std::ops::{Deref, DerefMut};

use crate::Result;

use super::super::InstrumentSession;
use super::rs_vna::{RohdeSchwarzVna, RsFamily};
pub struct Zva<S: InstrumentSession>(pub RohdeSchwarzVna<S>);

impl<S: InstrumentSession> Zva<S> {
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
