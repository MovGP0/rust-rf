//! Ported from `skrf/tests/test_static_data.py`.

mod data {
    use rust_rf::data::{DATA, MATERIALS, SKRF_MATPLOTLIB_STYLE};

    /// Checks every embedded network named by the upstream static-data suite.
    #[test]
    fn loads_every_embedded_static_network() {
        let networks = [
            DATA.ntwk1().expect("ntwk1 should parse"),
            DATA.line().expect("line should parse"),
            DATA.open_2p().expect("open should parse"),
            DATA.short_2p().expect("short should parse"),
            DATA.ind().expect("inductor should parse"),
            DATA.ring_slot().expect("ring slot should parse"),
            DATA.tee().expect("tee should parse"),
            DATA.ring_slot_meas()
                .expect("measured ring slot should parse"),
            DATA.wr2p2_line().expect("WR2.2 line should parse"),
            DATA.wr2p2_line1().expect("WR2.2 line 1 should parse"),
            DATA.wr2p2_delayshort()
                .expect("WR2.2 delay short should parse"),
            DATA.wr2p2_short().expect("WR2.2 short should parse"),
            DATA.wr1p5_line().expect("WR1.5 line should parse"),
            DATA.wr1p5_short().expect("WR1.5 short should parse"),
            DATA.ro_1().expect("radiating open 1 should parse"),
            DATA.ro_2().expect("radiating open 2 should parse"),
            DATA.ro_3().expect("radiating open 3 should parse"),
        ];

        let expected_ports = [2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1];
        for (network, ports) in networks.iter().zip(expected_ports) {
            assert_eq!(network.ports(), ports);
            assert!(network.frequency_points() > 0);
            assert!(network.name.is_some());
        }
    }

    /// Checks material aliases, representative properties, and the embedded plot style.
    #[test]
    fn exposes_material_properties_and_upstream_aliases() {
        assert_eq!(MATERIALS["cu"], MATERIALS["copper"]);
        assert_eq!(MATERIALS["al"], MATERIALS["aluminum"]);
        assert_eq!(MATERIALS["au"], MATERIALS["gold"]);
        assert_eq!(MATERIALS["silicon"].relative_permittivity, Some(11.68));
        assert_eq!(MATERIALS["teflon"].loss_tangent, Some(5e-4));
        assert!(SKRF_MATPLOTLIB_STYLE.contains("axes.grid"));
    }

    /// Checks that the unsafe Python pickle calibration fixture is not decoded.
    #[test]
    fn rejects_the_unsafe_python_pickle_calibration_fixture() {
        let Err(error) = DATA.one_port_calibration() else {
            panic!("Python pickle should not be decoded");
        };
        assert!(error.to_string().contains("pickle"));
    }
}

mod instances {
    use approx::assert_relative_eq;
    use num_complex::Complex64;
    use rust_rf::instances::{INSTANCES, WaveguideBand};
    use rust_rf::media::Media;

    /// Checks the default free-space and 50-ohm free-space instances.
    #[test]
    fn creates_default_air_and_fifty_ohm_air() {
        let air = INSTANCES.air().unwrap();
        assert_eq!(air.frequency.points(), 101);
        assert_eq!(air.frequency.start(), Some(1.0e9));
        let air50 = INSTANCES.air50().unwrap();
        assert_eq!(
            air50.characteristic_impedance().unwrap()[0],
            Complex64::new(50.0, 0.0)
        );
    }

    /// Checks every standard WR and WM frequency/waveguide pair.
    #[test]
    fn constructs_every_standard_waveguide_band() {
        for band in WaveguideBand::ALL {
            let frequency = INSTANCES.frequency(band).unwrap();
            let waveguide = INSTANCES.waveguide(band).unwrap();
            assert_eq!(frequency.points(), 1001);
            assert!(frequency.start().unwrap() < frequency.stop().unwrap());
            assert_eq!(waveguide.frequency, frequency);
            assert!(waveguide.width > 0.0);
            assert!(waveguide.height > 0.0);
            assert_eq!(
                waveguide
                    .characteristic_impedance_override
                    .as_ref()
                    .unwrap()[0],
                Complex64::new(50.0, 0.0)
            );
        }
    }

    /// Checks representative named WR and WM accessors against upstream values.
    #[test]
    fn exposes_named_wr_and_wm_accessors_with_upstream_values() {
        let wr10 = INSTANCES.wr10().unwrap();
        assert_eq!(INSTANCES.f_wr10().unwrap().start(), Some(75.0e9));
        assert_eq!(INSTANCES.f_wr10().unwrap().stop(), Some(110.0e9));
        assert_relative_eq!(wr10.width, 100.0 * 25.4e-6, epsilon = 1.0e-15);
        let wm106 = INSTANCES.wm106().unwrap();
        assert_eq!(INSTANCES.f_wm106().unwrap().start(), Some(1.7e12));
        assert_relative_eq!(wm106.width, 106.0e-6, epsilon = 1.0e-18);
    }
}
