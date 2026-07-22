use std::io::{Read, Write};

use crate::{Network, Result};

/// Input and output formats for RF networks and related data.
///
/// [`general`] provides generic object serialization, while [`touchstone`] provides
/// Touchstone parsing and writing used by [`Network`].
/// CITI file support.
pub mod citi;
/// CSV table readers and writers.
pub mod csv;
/// General network and object serialization helpers.
pub mod general;
/// MDIF file parsing.
pub mod mdif;
/// Meta-data conversion helpers.
pub mod metas;
/// Touchstone parsing, writing, and HFSS conversion helpers.
pub mod touchstone;

pub use citi::{Citi, CitiData, CitiFormat, CitiVariable};
pub use csv::{AgilentCsv, CsvTable};
pub use general::{
    NetworkDataFormat, StoredObject, from_json_string, network_table, read_all_networks,
    read_all_objects, read_object, statistical_to_touchstone, to_json_string, write_all_networks,
    write_all_objects, write_network_csv, write_network_html, write_object,
};
pub use mdif::{Mdif, MdifValue};
pub use metas::{write_sdatcv, write_sdatcv_to_path};
pub use touchstone::{
    Touchstone, TouchstoneFormat, TouchstoneParameter, hfss_touchstone_2_gamma_z0,
    hfss_touchstone_2_media, hfss_touchstone_2_network, read_zipped_touchstones, touchstone_string,
    write_touchstone,
};

/// Reads a [`Network`] from a byte-oriented source.
pub trait NetworkReader {
    /// Parses one network from `reader`.
    ///
    /// # Errors
    ///
    /// Returns an error when the input cannot be read or does not contain a valid network in the
    /// reader's format.
    fn read<R: Read>(&self, reader: R) -> Result<Network>;
}

/// Writes a [`Network`] to a byte-oriented destination.
pub trait NetworkWriter {
    /// Serializes `network` to `writer`.
    ///
    /// # Errors
    ///
    /// Returns an error when the network cannot be represented in the writer's format or the
    /// destination cannot be written.
    fn write<W: Write>(&self, network: &Network, writer: W) -> Result<()>;
}
