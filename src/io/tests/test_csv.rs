//! Agilent-style CSV input/output regressions.
//!
//! The fixtures exercise column parsing, comments, numeric data, frequency
//! units, complex and scalar network conversion, and instrument-specific
//! readers.

use approx::assert_relative_eq;
use num_complex::Complex64;
use rust_rf::io::csv::{
    AgilentCsv, pna_csv_to_two_port, read_all_csv, read_pna_csv, vectorstar_csv_to_networks,
    zva_dat_to_network,
};

const PNA_RI: &str = "!this is a comment\n\
!line\n\
\n\
BEGIN CH1_DATA\n\
Freq(Hz),\"A,1\"(REAL),\"A,1\"(IMAG),\"R1,1\"(REAL),\"R1,1\"(IMAG)\n\
750000000000,1,2,3,4\n\
1100000000000,5,6,7,8\n\
END\n";

/// Reads column names, comment lines, data, and frequency from an Agilent CSV.
#[test]
fn parses_agilent_columns_comments_data_and_frequency() {
    let csv = AgilentCsv::parse(PNA_RI).unwrap();
    assert_eq!(
        csv.columns(),
        vec![
            "Freq(Hz)",
            "\"A,1\"(REAL)",
            "\"A,1\"(IMAG)",
            "\"R1,1\"(REAL)",
            "\"R1,1\"(IMAG)"
        ]
    );
    assert_eq!(csv.comments, "this is a comment\nline\n");
    assert_relative_eq!(csv.data[(0, 0)], 750_000_000_000.0);
    assert_relative_eq!(csv.data[(1, 4)], 8.0);
    assert_eq!(
        csv.frequency().unwrap().values_hz().to_vec(),
        vec![750_000_000_000.0, 1_100_000_000_000.0]
    );
}

/// Builds complex and scalar one-port networks from CSV trace columns.
#[test]
fn builds_complex_and_scalar_one_port_networks() {
    let csv = AgilentCsv::parse(PNA_RI).unwrap();
    let networks = csv.networks().unwrap();
    assert_eq!(networks.len(), 2);
    assert_eq!(networks[0].s[(0, 0, 0)], Complex64::new(1.0, 2.0));
    assert_eq!(networks[1].s[(1, 0, 0)], Complex64::new(7.0, 8.0));
    assert_eq!(csv.scalar_networks().unwrap().len(), 4);
    assert_relative_eq!(csv.as_columns()["Freq(Hz)"][0], 750_000_000_000.0);
}

/// Decodes dB/degree pairs and scales the frequency unit to hertz.
#[test]
fn decodes_db_degree_pairs_and_frequency_units() {
    let text = "BEGIN DATA\nFreq(GHz),S11 Log Mag(dB),S11 Phase(deg)\n1,-20,90\n2,0,180\nEND\n";
    let csv = AgilentCsv::parse(text).unwrap();
    let network = &csv.networks().unwrap()[0];
    assert_eq!(network.frequency.values_hz().to_vec(), vec![1.0e9, 2.0e9]);
    assert_relative_eq!(network.s[(0, 0, 0)].re, 0.0, epsilon = 1.0e-12);
    assert_relative_eq!(network.s[(0, 0, 0)].im, 0.1, epsilon = 1.0e-12);
}

/// Rejects CSV input with missing block markers or inconsistent row widths.
#[test]
fn rejects_missing_markers_and_ragged_rows() {
    assert!(AgilentCsv::parse("Freq(Hz),S\n1,2\n").is_err());
    assert!(AgilentCsv::parse("BEGIN X\nFreq(Hz),A,B\n1,2,3\n2,4\nEND\n").is_err());
}

/// Reads a PNA CSV file and normalizes its frequency column to hertz.
#[test]
fn path_reader_normalizes_frequency_to_hertz() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".temp")
        .join(format!("csv-pna-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join(format!("rust-rf-pna-{}.csv", std::process::id()));
    std::fs::write(&path, "BEGIN DATA\nFreq(MHz),value\n1,2\n2,3\nEND\n").unwrap();
    let table = read_pna_csv(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(directory).unwrap();
    assert_eq!(table.data.column(0).to_vec(), vec![1.0e6, 2.0e6]);
}

/// Converts representative PNA, ZVA, and `VectorStar` exports to networks.
#[test]
fn converts_two_port_pna_zva_and_vectorstar_files() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".temp")
        .join(format!("csv-instruments-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();

    let pna_path = directory.join("pna.csv");
    std::fs::write(
        &pna_path,
        "BEGIN DATA\nFreq(Hz),S11(dB),S11(deg),S12(dB),S12(deg),S21(dB),S21(deg),S22(dB),S22(deg)\n1,-20,0,-40,10,-30,20,-10,30\nEND\n",
    )
    .unwrap();
    let pna = pna_csv_to_two_port(&pna_path).unwrap();
    assert_relative_eq!(
        pna.s[(0, 1, 0)].norm(),
        10.0_f64.powf(-1.5),
        epsilon = 1.0e-12
    );
    assert!(
        read_all_csv(&directory, Some("pna"))
            .unwrap()
            .contains_key("pna")
    );

    let zva_path = directory.join("zva.dat");
    std::fs::write(
        &zva_path,
        "%Freq,S11 Re,S11 Im,S12 Re,S12 Im,S21 Re,S21 Im,S22 Re,S22 Im\n1,1,2,3,4,5,6,7,8\n",
    )
    .unwrap();
    let zva = zva_dat_to_network(&zva_path).unwrap();
    assert_eq!(zva.s[(0, 0, 1)], Complex64::new(3.0, 4.0));

    let vectorstar_path = directory.join("vectorstar.csv");
    std::fs::write(
        &vectorstar_path,
        "!PARAMETER,S11,S22\nPNT,F1,R1,I1,F2,R2,I2\n0,1,2,0,1,3\n1,4,5,1,6,7\n",
    )
    .unwrap();
    let vectorstar = vectorstar_csv_to_networks(&vectorstar_path).unwrap();
    assert_eq!(vectorstar.len(), 2);
    assert_eq!(vectorstar[1].s[(1, 0, 0)], Complex64::new(6.0, 7.0));

    std::fs::remove_dir_all(directory).unwrap();
}
