//! General I/O regressions.
//!
//! These tests cover object read/write equivalence, directory discovery,
//! network JSON round trips, statistical conversion, and spreadsheet output.

use std::collections::BTreeMap;

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rust_rf::io::general::{
    NetworkDataFormat, StoredObject, from_json_string, network_table, read_all_networks,
    read_all_objects, read_object, statistical_to_touchstone, to_json_string, write_all_networks,
    write_all_objects, write_network_csv, write_network_html, write_object,
};
use rust_rf::{Frequency, Network, NetworkSet};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Verifies that JSON serialization preserves the complete network state.
#[test]
fn json_round_trips_complete_network_state() -> TestResult {
    let mut network = one_port()?;
    network.name = Some("json".to_owned());
    network.comments = "comment".to_owned();
    network.port_names = vec!["input".to_owned()];
    network
        .variables
        .insert("temperature".to_owned(), "290 K".to_owned());
    let restored = from_json_string(&to_json_string(&network).unwrap()).unwrap();
    assert_eq!(restored.frequency, network.frequency);
    assert_eq!(restored.s, network.s);
    assert_eq!(restored.z0, network.z0);
    assert_eq!(restored.name, network.name);
    assert_eq!(restored.comments, network.comments);
    assert_eq!(restored.port_names, network.port_names);
    assert_eq!(restored.variables, network.variables);
    Ok(())
}

/// Writes, reads, and discovers supported network and mixed-object files.
#[test]
fn stores_reads_and_discovers_network_objects() -> TestResult {
    let directory = temporary_directory("objects")?;
    let network = one_port()?;
    let path = write_object(
        directory.join("sample"),
        &StoredObject::Network(Box::new(network.clone())),
        true,
    )
    .unwrap();
    assert_eq!(path.extension().unwrap(), "ntwk");
    match read_object(&path).unwrap() {
        StoredObject::Network(restored) => assert_eq!(restored.s, network.s),
        _ => panic!("stored object type changed"),
    }
    let set = NetworkSet::new(vec![network.clone()], None).unwrap();
    let set_path =
        write_object(directory.join("set"), &StoredObject::NetworkSet(set), true).unwrap();
    assert_eq!(set_path.extension().unwrap(), "ns");

    let nested = directory.join("nested");
    std::fs::create_dir(&nested).unwrap();
    std::fs::write(
        nested.join("touchstone.s1p"),
        "# Hz S RI R 50\n1 0.5 -0.25\n",
    )
    .unwrap();
    let found = read_all_networks(&directory, None, true).unwrap();
    assert!(found.contains_key("sample"));
    assert!(found.contains_key("touchstone"));

    let all = BTreeMap::from([("copy".to_owned(), network)]);
    assert_eq!(write_all_networks(&all, &directory).unwrap().len(), 1);
    let mixed = BTreeMap::from([(
        "frequency".to_owned(),
        StoredObject::Frequency(Frequency::from_hz(ndarray::arr1(&[3.0])).unwrap()),
    )]);
    assert_eq!(
        write_all_objects(&mixed, &directory, true).unwrap().len(),
        1
    );
    assert!(matches!(
        read_all_objects(&directory, Some("frequency"), false).unwrap()["frequency"],
        StoredObject::Frequency(_)
    ));
    std::fs::remove_dir_all(directory).unwrap();
    Ok(())
}

/// Converts statistical data and writes network tables in supported formats.
#[test]
fn writes_statistical_touchstone_and_spreadsheet_tables() -> TestResult {
    let directory = temporary_directory("tables")?;
    let source = directory.join("statistical.txt");
    let touchstone = directory.join("converted.s1p");
    std::fs::write(&source, "1 0 0\n").unwrap();
    statistical_to_touchstone(&source, &touchstone, None).unwrap();
    assert_eq!(
        std::fs::read_to_string(&touchstone).unwrap(),
        "# GHz S RI R 50.0\n1 0 0\n"
    );

    let network = one_port()?;
    let (columns, table) = network_table(&network, NetworkDataFormat::RealImaginary);
    assert_eq!(columns, vec!["Freq(Hz)", "S11 Real", "S11 Imag"]);
    assert_eq!(table.row(0).to_vec(), vec![1.0, 0.5, -0.25]);
    let csv = directory.join("network.csv");
    write_network_csv(&network, &csv, NetworkDataFormat::MagnitudeAngle).unwrap();
    assert!(
        std::fs::read_to_string(csv)
            .unwrap()
            .starts_with("Freq(Hz),S11 Mag(lin),S11 Phase(deg)")
    );
    let html = directory.join("network.html");
    write_network_html(&network, &html, NetworkDataFormat::DecibelAngle).unwrap();
    assert!(
        std::fs::read_to_string(html)
            .unwrap()
            .contains("<th>S11 Log Mag(dB)</th>")
    );
    std::fs::remove_dir_all(directory).unwrap();
    Ok(())
}

#[cfg(feature = "dataframe")]
/// Builds data frames with unambiguous scattering-parameter port names.
#[test]
fn builds_network_dataframes_with_unambiguous_port_names() -> TestResult {
    let frame =
        rust_rf::io::general::network_to_dataframe(&one_port()?, &["s_db", "s_deg"], None, None)
            .unwrap();
    assert_eq!(frame.height(), 2);
    assert_eq!(
        frame
            .get_column_names()
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>(),
        vec!["s_db 11", "s_deg 11"]
    );
    Ok(())
}

#[cfg(feature = "xlsx")]
/// Writes individual networks and network sets to Excel workbooks.
#[test]
fn writes_network_and_network_set_xlsx_workbooks() -> TestResult {
    let directory = temporary_directory("xlsx")?;
    let mut network = one_port()?;
    network.name = Some("First".to_owned());
    let network_path = directory.join("network.xlsx");
    rust_rf::io::general::write_network_xlsx(
        &network,
        &network_path,
        NetworkDataFormat::RealImaginary,
    )
    .unwrap();
    assert_eq!(&std::fs::read(&network_path).unwrap()[..2], b"PK");
    let set_path = directory.join("set.xlsx");
    let set = NetworkSet::new(vec![network], None).unwrap();
    rust_rf::io::general::write_network_set_xlsx(&set, &set_path, NetworkDataFormat::DecibelAngle)
        .unwrap();
    assert_eq!(&std::fs::read(&set_path).unwrap()[..2], b"PK");
    std::fs::remove_dir_all(directory).unwrap();
    Ok(())
}

fn temporary_directory(label: &str) -> std::io::Result<std::path::PathBuf> {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".temp")
        .join(format!("general-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    Ok(directory)
}

fn one_port() -> rust_rf::Result<Network> {
    let frequency = Frequency::from_hz(ndarray::arr1(&[1.0, 2.0]))?;
    let s = Array3::from_shape_vec(
        (2, 1, 1),
        vec![Complex64::new(0.5, -0.25), Complex64::new(0.25, 0.5)],
    )
    .map_err(|error| {
        rust_rf::Error::IncompatibleShape(format!(
            "one-port fixture S-parameter shape is invalid: {error}"
        ))
    })?;
    let z0 = Array2::from_elem((2, 1), Complex64::new(50.0, 0.0));
    Network::new(frequency, s, z0)
}
