use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::{Error, NetworkSet, Result};

/// Write a METAS `sdatcv` file containing the complex mean and sample
/// covariance of a compatible network set.
///
/// Origin: `skrf/io/metas.py::ns_2_sdatcv`.
pub fn write_sdatcv_to_path(network_set: &NetworkSet, path: impl AsRef<Path>) -> Result<()> {
    let writer = BufWriter::new(fs::File::create(path)?);
    write_sdatcv(network_set, writer)
}

pub fn write_sdatcv(network_set: &NetworkSet, mut writer: impl Write) -> Result<()> {
    let first = network_set.networks.first().ok_or_else(|| {
        Error::IncompatibleShape("cannot write an empty NetworkSet as SDATCV".to_owned())
    })?;
    let ports = first.ports();
    let covariance = network_set.covariance_s()?;
    let mean = network_set.mean_s()?;

    writeln!(writer, "SDATCV")?;
    writeln!(writer, "Ports")?;
    for port in 0..ports {
        if port > 0 {
            write!(writer, "\t")?;
        }
        write!(writer, "{}\t", port + 1)?;
    }
    writeln!(writer)?;

    for port in 0..ports {
        if port > 0 {
            write!(writer, "\t")?;
        }
        write!(writer, "Zr[{}]re\tZr[{}]im", port + 1, port + 1)?;
    }
    writeln!(writer)?;
    for port in 0..ports {
        if port > 0 {
            write!(writer, "\t")?;
        }
        let value = first.z0[(0, port)];
        write!(writer, "{}\t{}", value.re, value.im)?;
    }
    writeln!(writer)?;

    write!(writer, "Freq")?;
    for column in 0..ports {
        for row in 0..ports {
            write!(
                writer,
                "\tS[{},{}]re\tS[{},{}]im",
                row + 1,
                column + 1,
                row + 1,
                column + 1
            )?;
        }
    }
    let components = 2 * ports * ports;
    for column in 0..components {
        for row in 0..components {
            write!(writer, "\tCV[{},{}]", column + 1, row + 1)?;
        }
    }
    writeln!(writer)?;

    for point in 0..first.frequency_points() {
        write!(writer, "{:.17e}", first.frequency.values_hz()[point])?;
        for column in 0..ports {
            for row in 0..ports {
                let value = mean.s[(point, row, column)];
                write!(writer, "\t{:.17e}\t{:.17e}", value.re, value.im)?;
            }
        }
        for column in 0..components {
            for row in 0..components {
                write!(writer, "\t{:.17e}", covariance[(point, row, column)])?;
            }
        }
        writeln!(writer)?;
    }
    Ok(())
}
