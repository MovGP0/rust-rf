#![allow(unused_imports)]

use approx::assert_relative_eq;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use rust_rf::constants::{FREE_SPACE_PERMEABILITY, FREE_SPACE_PERMITTIVITY, SPEED_OF_LIGHT};
use rust_rf::math::db_to_nepers;
use rust_rf::math::set_random_seed;
use rust_rf::media::{
    AttenuationUnit, CircularWaveguide, Coaxial, Cpw, CpwCompatibilityMode, DefinedAEpTandZ0,
    DefinedCharacteristicImpedance, DefinedGammaZ0, DielectricDispersionModel, DistributedCircuit,
    Freespace, LengthUnit, Media, MicrostripDispersionModel, MicrostripLine,
    MicrostripQuasiStaticModel, RectangularWaveguide, WaveguideMode,
};
use rust_rf::{Frequency, FrequencyUnit, Network, SweepType};

#[test]
fn calculates_cpw_quasi_static_characteristics() {
    let frequency = Frequency::new(1.0, 20.0, 21, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let quartz = Cpw::lossless(frequency.clone(), 40.0e-6, 20.0e-6, 100.0e-3, 3.78)
        .expect("quartz CPW should be valid");
    let (impedance, effective) = quartz
        .quasi_static_characteristics()
        .expect("CPW quasi-static characteristics should be defined");
    assert_relative_eq!(impedance[0].re, 77.93, epsilon = 0.02);
    assert_relative_eq!(effective[0].re, 2.39, epsilon = 0.02);

    let gaas = Cpw::lossless(frequency, 75.0e-6, 50.0e-6, 100.0e-3, 12.9)
        .expect("GaAs CPW should be valid");
    let (_, effective) = gaas
        .quasi_static_characteristics()
        .expect("GaAs CPW characteristics should be defined");
    assert_relative_eq!(effective[0].re, 6.94, epsilon = 0.02);
}

#[test]
fn handles_cpw_zero_thickness_and_losses() {
    let frequency = Frequency::new(1.0, 20.0, 5, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let points = frequency.points();
    let zero_thickness = Cpw::new(
        frequency.clone(),
        3.0e-3,
        0.3e-3,
        1.55e-3,
        Some(0.0),
        4.5,
        0.0,
        None,
        DielectricDispersionModel::FrequencyInvariant,
        false,
        CpwCompatibilityMode::Qucs,
        Some(Array1::from_elem(points, Complex64::new(50.0, 0.0))),
        None,
    )
    .expect("zero-thickness CPW should be valid");
    assert!(
        zero_thickness
            .attenuation()
            .expect("zero-thickness attenuation should be defined")
            .0
            .iter()
            .all(|value| *value == 0.0)
    );

    let lossy = Cpw::new(
        frequency,
        1.6e-3,
        0.3e-3,
        1.55e-3,
        Some(35.0e-6),
        4.5,
        0.018,
        Some(1.7e-8),
        DielectricDispersionModel::FrequencyInvariant,
        true,
        CpwCompatibilityMode::Qucs,
        Some(Array1::from_elem(points, Complex64::new(50.0, 0.0))),
        None,
    )
    .expect("lossy CPW should be valid");
    let (conductor, dielectric) = lossy
        .attenuation()
        .expect("lossy CPW attenuation should be defined");
    assert!(conductor.iter().all(|value| *value > 0.0));
    assert!(dielectric.iter().all(|value| *value > 0.0));
    assert!(
        lossy
            .line(25.0, LengthUnit::Millimeter)
            .expect("lossy CPW line should be constructed")
            .s[(0, 1, 0)]
            .norm()
            < 1.0
    );
}

#[test]
fn matches_cpw_qucs_and_ads_fixtures() {
    struct Fixture<'a> {
        simulator: &'a str,
        name: &'a str,
        width: f64,
        thickness: f64,
        height: f64,
        metal_backside: bool,
        tolerance: f64,
    }
    let fixtures = [
        Fixture {
            simulator: "qucs",
            name: "cpw,t=35um,w=1.6mm,s=0.3mm,l=25mm,backside=metal.s2p",
            width: 1.6e-3,
            thickness: 35.0e-6,
            height: 1.55e-3,
            metal_backside: true,
            tolerance: 2.0e-3,
        },
        Fixture {
            simulator: "qucs",
            name: "cpw,t=35um,w=3mm,s=0.3mm,l=25mm,backside=air.s2p",
            width: 3.0e-3,
            thickness: 35.0e-6,
            height: 1.55e-3,
            metal_backside: false,
            tolerance: 2.0e-3,
        },
        Fixture {
            simulator: "qucs",
            name: "cpw,t=0,h=100mm,w=3mm,s=0.3mm,l=25mm,backside=air.s2p",
            width: 3.0e-3,
            thickness: 0.0,
            height: 100.0e-3,
            metal_backside: false,
            tolerance: 2.0e-3,
        },
        Fixture {
            simulator: "ads",
            name: "cpw,t=0um.s2p",
            width: 3.0e-3,
            thickness: 0.0,
            height: 1.55e-3,
            metal_backside: false,
            tolerance: 1.0e-3,
        },
        Fixture {
            simulator: "ads",
            name: "cpwg,t=0um.s2p",
            width: 1.6e-3,
            thickness: 0.0,
            height: 1.55e-3,
            metal_backside: true,
            tolerance: 1.0e-3,
        },
    ];
    for fixture in fixtures {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/media/cpw")
            .join(fixture.simulator)
            .join(fixture.name);
        let reference = Network::read_touchstone(path).expect("CPW fixture should load");
        let points = reference.frequency_points();
        let (dispersion, compatibility) = if fixture.simulator == "qucs" {
            (
                DielectricDispersionModel::FrequencyInvariant,
                CpwCompatibilityMode::Qucs,
            )
        } else {
            (
                DielectricDispersionModel::DjordjevicSvensson {
                    low_frequency_hz: 1.0e3,
                    high_frequency_hz: 1.0e12,
                    specification_frequency_hz: 1.0e9,
                },
                CpwCompatibilityMode::Ads,
            )
        };
        let mut cpw = Cpw::new(
            reference.frequency.clone(),
            fixture.width,
            0.3e-3,
            fixture.height,
            Some(fixture.thickness),
            4.5,
            0.018,
            Some(1.7e-8),
            dispersion,
            fixture.metal_backside,
            compatibility,
            Some(Array1::from_elem(points, Complex64::new(50.0, 0.0))),
            None,
        )
        .expect("CPW should be valid");
        let actual = cpw
            .line(25.0e-3, LengthUnit::Meter)
            .expect("CPW line should be constructed");
        let maximum_error = actual
            .s
            .iter()
            .zip(reference.s.iter())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            maximum_error < fixture.tolerance,
            "{} maximum error was {maximum_error}",
            fixture.name
        );
        cpw.set_resistivity_material("copper")
            .expect("copper should resolve");
        assert!(cpw.to_string().contains("Coplanar Waveguide Media"));
    }
}

#[test]
fn matches_cpw_qucs_impedance_curve() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/media/cpw/qucs/cpw_qucs_ep_r9dot5.csv");
    let text = std::fs::read_to_string(path).expect("CPW curve fixture should load");
    let frequency = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let values = line
            .split(';')
            .map(|value| {
                value
                    .trim()
                    .parse::<f64>()
                    .expect("curve value should parse")
            })
            .collect::<Vec<_>>();
        let cpw = Cpw::lossless(frequency.clone(), 1.0, 1.0 / values[0], 1.0e9, 9.5)
            .expect("CPW should be valid");
        let impedance = cpw
            .characteristic_impedance()
            .expect("CPW impedance should be defined")[0]
            .re;
        assert!(
            ((values[1] - impedance) / values[1]).abs() < 0.03,
            "w/s={} expected {} but got {impedance}",
            values[0],
            values[1]
        );
    }
}
