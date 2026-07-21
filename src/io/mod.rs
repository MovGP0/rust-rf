use std::io::{Read, Write};

use crate::{Network, Result};

pub mod citi;
pub mod csv;
pub mod general;
pub mod mdif;
pub mod metas;
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

pub trait NetworkReader {
    fn read<R: Read>(&self, reader: R) -> Result<Network>;
}

pub trait NetworkWriter {
    fn write<W: Write>(&self, network: &Network, writer: W) -> Result<()>;
}
