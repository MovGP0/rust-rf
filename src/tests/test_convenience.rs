use std::path::PathBuf;

use approx::assert_relative_eq;
use ndarray::Array2;
use num_complex::Complex64;
use rust_rf::NetworkSet;
use rust_rf::io::{Touchstone, hfss_touchstone_2_media};
use rust_rf::{Network, SParameterDefinition};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("convenience")
        .join(name)
}

#[test]
fn reads_hfss_gamma_and_impedance_for_high_port_counts() {
    for (name, ports) in [
        ("hfss_18.2.s3p", 3),
        ("hfss_19.2.s8p", 8),
        ("hfss_19.2.s10p", 10),
    ] {
        let touchstone = Touchstone::from_path(fixture(name)).expect("HFSS file should parse");
        assert!(touchstone.has_hfss_port_impedances());
        assert_eq!(touchstone.rank, ports);
        assert_eq!(
            touchstone.port_impedances.as_ref().expect("z0").ncols(),
            ports
        );
        assert_eq!(
            touchstone
                .propagation_constants
                .as_ref()
                .expect("gamma")
                .ncols(),
            ports
        );
    }
    assert!(
        !Touchstone::from_path(fixture("ntwk1.s2p"))
            .expect("ordinary Touchstone should parse")
            .has_hfss_port_impedances()
    );
}

#[test]
fn creates_media_for_each_hfss_port() {
    let one_port =
        hfss_touchstone_2_media(fixture("hfss_oneport.s1p")).expect("one-port media should load");
    let two_port =
        hfss_touchstone_2_media(fixture("hfss_twoport.s2p")).expect("two-port media should load");
    assert_eq!(one_port.len(), 1);
    assert_eq!(two_port.len(), 2);
    assert_eq!(one_port[0].gamma.len(), one_port[0].frequency.points());
}

#[test]
fn renormalizes_hfss_networks_to_their_fifty_ohm_exports() {
    for (source, expected) in [
        ("hfss_threeport_DB.s3p", "hfss_threeport_DB_50Ohm.s3p"),
        ("hfss_threeport_MA.s3p", "hfss_threeport_MA_50Ohm.s3p"),
    ] {
        let mut actual = Network::read_touchstone(fixture(source)).expect("source should parse");
        let expected = Network::read_touchstone(fixture(expected)).expect("target should parse");
        let reference = Array2::from_elem(actual.z0.dim(), Complex64::new(50.0, 0.0));
        actual
            .renormalize(reference, SParameterDefinition::Power)
            .expect("renormalization should succeed");
        for (actual, expected) in actual.s.iter().zip(&expected.s) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1e-6);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1e-6);
        }
    }

    let without_metadata =
        Network::read_touchstone(fixture("hfss_threeport_MA_without_gamma_z0_50Ohm.s3p"))
            .expect("metadata-free export should parse");
    let with_metadata = Network::read_touchstone(fixture("hfss_threeport_MA_50Ohm.s3p"))
        .expect("metadata export should parse");
    for (actual, expected) in without_metadata.s.iter().zip(&with_metadata.s) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1e-12);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1e-12);
    }
}

#[test]
fn reads_cst_agilent_rohde_schwarz_and_helic_exports() {
    for (name, ports) in [
        ("cst_example_4ports.s4p", 4),
        ("cst_example_6ports.s6p", 6),
        ("cst_example_6ports_V2.s6p", 6),
        ("cst_example_6ports_V2.ts", 6),
        ("Agilent_E5071B.s4p", 4),
        ("RS_ZNB8.s4p", 4),
        ("RS_ZVR_1.20_beta_f.s2p", 2),
    ] {
        let network = Network::read_touchstone(fixture(name))
            .unwrap_or_else(|error| panic!("{name} should parse: {error}"));
        assert_eq!(network.ports(), ports, "{name}");
    }

    let agilent =
        Network::read_touchstone(fixture("Agilent_E5071B.s4p")).expect("Agilent file should parse");
    assert!(agilent.z0.iter().all(|value| value.re == 75.0));
    let expected_db = [-52.52684, -0.2278388, -44.35702, -82.35984];
    for (column, expected) in expected_db.into_iter().enumerate() {
        assert_relative_eq!(
            20.0 * agilent.s[(0, 1, column)].norm().log10(),
            expected,
            epsilon = 1e-5
        );
    }

    let helic_ts = Network::read_touchstone(fixture("helic_example_6ports_V2.ts"))
        .expect("Helic .ts should parse");
    let helic_sp = Network::read_touchstone(fixture("helic_example_6ports_V2.sp"))
        .expect("Helic .sp should parse");
    assert_eq!(
        helic_ts.frequency.values_hz(),
        helic_sp.frequency.values_hz()
    );
    assert_eq!(helic_ts.z0, helic_sp.z0);
    assert_eq!(helic_ts.s, helic_sp.s);
}

#[test]
fn loads_touchstone_archives_as_network_sets() {
    let set = NetworkSet::from_zip(fixture("ntwks.zip")).expect("archive should load");
    assert_eq!(set.len(), 3);
    assert_eq!(set.name.as_deref(), Some("ntwks"));
    assert!(set.networks.iter().all(|network| network.ports() == 2));
}

#[test]
fn creates_the_shared_dc_extrapolated_network_fixture() {
    let network = Network::read_touchstone(fixture("ntwk1.s2p")).expect("fixture should load");
    let extrapolated = network
        .extrapolate_to_dc(None, None)
        .expect("network should extrapolate to DC");
    assert_eq!(extrapolated.frequency.values_hz()[0], 0.0);
    assert!(
        extrapolated
            .s
            .index_axis(ndarray::Axis(0), 0)
            .iter()
            .all(|value| value.im == 0.0)
    );
    let steps = extrapolated
        .frequency
        .values_hz()
        .windows(2)
        .into_iter()
        .map(|window| window[1] - window[0])
        .collect::<Vec<_>>();
    assert!(
        steps
            .windows(2)
            .all(|pair| (pair[0] - pair[1]).abs() < 1e-6)
    );
}
