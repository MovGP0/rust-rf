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

const TOLERANCE: f64 = 1.0e-10;

#[test]
fn constructs_defined_media_and_matched_line() {
    let media = defined_media(None);
    let line = media
        .line(0.25, LengthUnit::Meter)
        .expect("line should be constructed");
    let expected = (-Complex64::new(0.1, 2.0) * 0.25).exp();
    for point in 0..line.frequency_points() {
        assert_complex_close(line.s[(point, 0, 0)], Complex64::new(0.0, 0.0));
        assert_complex_close(line.s[(point, 1, 1)], Complex64::new(0.0, 0.0));
        assert_complex_close(line.s[(point, 1, 0)], expected);
        assert_complex_close(line.s[(point, 0, 1)], expected);
    }
}

#[test]
fn constructs_thru_open_short_and_load() {
    let media = defined_media(None);
    let thru = media.thru().expect("thru should be constructed");
    assert_complex_close(thru.s[(0, 1, 0)], Complex64::new(1.0, 0.0));
    assert_complex_close(thru.s[(0, 0, 0)], Complex64::new(0.0, 0.0));

    let open = media.open().expect("open should be constructed");
    assert_complex_close(open.s[(0, 0, 0)], Complex64::new(1.0, 0.0));
    let short = media.short().expect("short should be constructed");
    assert_complex_close(short.s[(0, 0, 0)], Complex64::new(-1.0, 0.0));
    let load = media
        .load(Complex64::new(0.25, -0.5))
        .expect("load should be constructed");
    assert_complex_close(load.s[(0, 0, 0)], Complex64::new(0.25, -0.5));
}

#[test]
fn converts_physical_and_electrical_length_units() {
    let media = defined_media(None);
    assert_relative_eq!(
        media
            .physical_length(10.0, LengthUnit::Centimeter)
            .expect("centimeters should convert"),
        0.1,
        epsilon = TOLERANCE
    );
    assert_relative_eq!(
        media
            .physical_length(180.0, LengthUnit::Degree)
            .expect("degrees should convert"),
        std::f64::consts::PI / 2.0,
        epsilon = TOLERANCE
    );
}

#[test]
fn exposes_shared_media_properties_time_units_and_plot_data() {
    let frequency = Frequency::new(1.0, 3.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let media = Freespace::vacuum(frequency).expect("vacuum should be valid");
    assert_eq!(media.points(), 3);
    assert!(
        media
            .attenuation_constant()
            .expect("attenuation should be defined")
            .iter()
            .all(|value| value.abs() < 1.0e-12)
    );
    assert!(
        media
            .phase_constant()
            .expect("phase should be defined")
            .iter()
            .all(|value| *value > 0.0)
    );
    let distance = media
        .physical_length(1.0, LengthUnit::Nanosecond)
        .expect("time should convert through group velocity");
    assert_relative_eq!(distance, SPEED_OF_LIGHT * 1.0e-9, max_relative = 1.0e-12);
    let angle = media
        .electrical_length(distance, false)
        .expect("electrical length should be defined");
    let recovered = media
        .distance_from_electrical_length(angle[1].im, false)
        .expect("distance should be recovered");
    assert_relative_eq!(recovered[1], distance, max_relative = 1.0e-12);
    assert_relative_eq!(
        media
            .center_distance_from_electrical_length(angle[1].im.to_degrees(), true)
            .expect("center distance should be recovered"),
        distance,
        max_relative = 1.0e-12
    );
    let plot = media.plot_frequency();
    assert_eq!(plot.series.len(), 1);
    assert_eq!(plot.series[0].y.len(), 3);
}

#[test]
fn renormalizes_lines_to_port_reference_impedance() {
    let port_reference = Array1::from_elem(3, Complex64::new(75.0, 0.0));
    let media = defined_media(Some(port_reference));
    let line = media
        .thru()
        .expect("renormalized thru should be constructed");
    assert_eq!(line.z0[(0, 0)], Complex64::new(75.0, 0.0));
    assert_eq!(line.z0[(0, 1)], Complex64::new(75.0, 0.0));
}

#[test]
fn supports_complex_impedance_mismatches_and_wave_definitions() {
    let media = defined_media(None);
    let left = Array1::from_elem(3, Complex64::new(10.0, 10.0));
    let right = Array1::from_elem(3, Complex64::new(50.0, -20.0));
    for definition in [
        rust_rf::SParameterDefinition::Traveling,
        rust_rf::SParameterDefinition::Pseudo,
        rust_rf::SParameterDefinition::Power,
    ] {
        let mismatch = media
            .impedance_mismatch_complex(&left, &right, definition)
            .expect("complex mismatch should be constructed");
        assert_eq!(mismatch.s_definition, definition);
        assert!(mismatch.s.iter().all(|value| value.is_finite()));
    }

    let complex_media = DefinedGammaZ0::new(
        Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear)
            .expect("frequency should be valid"),
        Array1::from_elem(1, Complex64::new(0.0, 1.0)),
        Array1::from_elem(1, Complex64::new(10.0, 20.0)),
        None,
    )
    .expect("complex media should be valid");
    let short = complex_media.short().expect("short should be constructed");
    assert_ne!(short.s[(0, 0, 0)], Complex64::new(-1.0, 0.0));
    let mut traveling = short.clone();
    traveling
        .renormalize(
            traveling.z0.clone(),
            rust_rf::SParameterDefinition::Traveling,
        )
        .expect("short should convert to traveling waves");
    assert_relative_eq!(traveling.s[(0, 0, 0)].re, -1.0, epsilon = 1.0e-12);
    assert_relative_eq!(traveling.s[(0, 0, 0)].im, 0.0, epsilon = 1.0e-12);
}

#[test]
fn calculates_vacuum_wave_quantities() {
    let frequency = Frequency::new(75.0, 110.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let media = Freespace::vacuum(frequency.clone()).expect("vacuum should be valid");
    let impedance = media
        .characteristic_impedance()
        .expect("vacuum impedance should be defined");
    let gamma = media
        .propagation_constant()
        .expect("vacuum propagation constant should be defined");
    let expected_impedance = (FREE_SPACE_PERMEABILITY / FREE_SPACE_PERMITTIVITY).sqrt();
    for point in 0..frequency.points() {
        assert_relative_eq!(impedance[point].re, expected_impedance, epsilon = 1.0e-9);
        assert_relative_eq!(impedance[point].im, 0.0, epsilon = TOLERANCE);
        assert_relative_eq!(gamma[point].re, 0.0, epsilon = TOLERANCE);
        assert_relative_eq!(
            gamma[point].im,
            frequency.angular()[point] / SPEED_OF_LIGHT,
            max_relative = 1.0e-9
        );
    }
    assert_eq!(impedance[0].re.round(), 377.0);

    let line = media
        .line(1.0, LengthUnit::Millimeter)
        .expect("vacuum line should be constructed");
    assert_eq!(line.ports(), 2);
    assert_eq!(line.frequency_points(), 3);
}

#[test]
fn applies_freespace_loss_tangents_and_resistivity() {
    let frequency = Frequency::new(1.0, 3.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let media = Freespace::new(
        frequency,
        Array1::from_elem(3, Complex64::new(4.0, -99.0)),
        Array1::from_elem(3, Complex64::new(2.0, -99.0)),
        Some(Array1::from_elem(3, 0.02)),
        Some(Array1::from_elem(3, 0.01)),
        Some(Array1::from_elem(3, 1.0e6)),
        None,
        None,
    )
    .expect("lossy freespace should be valid");
    let permittivity = media.permittivity();
    let permeability = media.permeability();
    assert_complex_close(
        permittivity[0],
        Complex64::new(4.0, -0.08) * FREE_SPACE_PERMITTIVITY,
    );
    assert_complex_close(
        permeability[0],
        Complex64::new(2.0, -0.02) * FREE_SPACE_PERMEABILITY,
    );
    assert!(
        media
            .propagation_constant()
            .expect("lossy propagation constant should be defined")
            .iter()
            .all(|value| value.re > 0.0)
    );
}

#[test]
fn resolves_freespace_materials_formats_and_builds_plot_data() {
    let frequency = Frequency::new(75.0, 110.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let mut media = Freespace::new(
        frequency,
        Array1::from_vec(vec![
            Complex64::new(2.0, -0.1),
            Complex64::new(2.1, -0.2),
            Complex64::new(2.2, -0.3),
        ]),
        Array1::from_vec(vec![
            Complex64::new(1.0, -0.01),
            Complex64::new(1.1, -0.02),
            Complex64::new(1.2, -0.03),
        ]),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("freespace should be valid");
    media
        .set_resistivity_material("al")
        .expect("material alias should resolve");
    assert_relative_eq!(
        media
            .resistivity
            .as_ref()
            .expect("resistivity should exist")[0],
        2.82e-8,
        epsilon = 1.0e-20
    );
    assert!(media.set_resistivity_material("teflon").is_err());
    assert!(media.set_resistivity_material("unobtainium").is_err());
    assert_eq!(media.plot_permittivity().series.len(), 2);
    assert_eq!(media.plot_permeability().series.len(), 2);
    let combined = media.plot_permittivity_and_permeability();
    assert_eq!(combined.series.len(), 4);
    assert_eq!(combined.series[0].y, vec![2.0, 2.1, 2.2]);
    assert_eq!(combined.series[3].y, vec![-0.01, -0.02, -0.03]);
    assert!(media.to_string().contains("75-110 GHz"));
}

#[test]
fn converts_distributed_circuit_to_freespace() {
    let frequency = Frequency::new(1.0, 3.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let circuit = DistributedCircuit::from_scalars(
        frequency,
        0.0,
        0.0,
        FREE_SPACE_PERMEABILITY,
        FREE_SPACE_PERMITTIVITY,
    )
    .expect("distributed circuit should be valid");
    let media = Freespace::from_distributed_circuit(&circuit)
        .expect("distributed circuit should convert to freespace");
    for value in &media.relative_permittivity {
        assert_complex_close(*value, Complex64::new(1.0, 0.0));
    }
    for value in &media.relative_permeability {
        assert_complex_close(*value, Complex64::new(1.0, 0.0));
    }
}

#[test]
fn calculates_circular_waveguide_cutoff_and_propagation() {
    let radius = 0.5 * 2.39e-3;
    let frequency = Frequency::new(60.0, 110.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let waveguide = CircularWaveguide::dominant_mode(frequency, radius)
        .expect("dominant circular waveguide should be valid");
    let root = waveguide.modal_root().expect("TE11 root should be defined");
    assert_relative_eq!(root, 1.841_183_781_340_659_3, epsilon = 1.0e-12);
    let cutoff = waveguide
        .cutoff_frequency()
        .expect("circular cutoff should be defined");
    assert_relative_eq!(
        cutoff[0],
        SPEED_OF_LIGHT * root / (std::f64::consts::TAU * radius),
        max_relative = 1.0e-9
    );
    let gamma = waveguide
        .propagation_constant()
        .expect("circular propagation should be defined");
    assert!(gamma[0].re > 0.0 && gamma[0].im == 0.0);
    assert!(gamma[1].re == 0.0 && gamma[1].im > 0.0);
    assert!(gamma[2].re == 0.0 && gamma[2].im > gamma[1].im);
}

#[test]
fn derives_circular_waveguide_radius_from_impedance() {
    let frequency = Frequency::new(90.0, 90.0, 1, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let waveguide =
        CircularWaveguide::from_characteristic_impedance(frequency, 500.0, 90.0e9, 1.0, 1.0)
            .expect("impedance-defined circular waveguide should be valid");
    let impedance = waveguide
        .characteristic_impedance()
        .expect("circular impedance should be defined");
    assert_relative_eq!(impedance[0].re, 500.0, max_relative = 1.0e-12);
    assert_relative_eq!(impedance[0].im, 0.0, epsilon = TOLERANCE);
}

#[test]
fn applies_circular_waveguide_conductor_loss() {
    let frequency = Frequency::new(88.0, 110.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let points = frequency.points();
    let waveguide = CircularWaveguide::new(
        frequency,
        Array1::from_elem(points, 0.5 * 2.39e-3),
        WaveguideMode::TransverseElectric,
        1,
        1,
        Array1::ones(points),
        Array1::ones(points),
        Some(Array1::from_elem(points, 1.0 / 3.8e7)),
        None,
        Some(Array1::from_elem(points, Complex64::new(50.0, 0.0))),
    )
    .expect("lossy circular waveguide should be valid");
    assert!(
        waveguide
            .conductor_attenuation()
            .expect("circular conductor loss should be defined")
            .iter()
            .all(|value| *value > 0.0)
    );
    assert!(
        waveguide
            .line(1.0, LengthUnit::Inch)
            .expect("circular line should be constructed")
            .s[(0, 1, 0)]
            .norm()
            < 1.0
    );
}

#[test]
fn resolves_circular_waveguide_material_and_formats() {
    let frequency = Frequency::new(88.0, 110.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let mut waveguide = CircularWaveguide::dominant_mode(frequency, 0.5 * 2.39e-3)
        .expect("dominant circular waveguide should be valid");
    waveguide
        .set_resistivity_material("cu")
        .expect("copper alias should resolve");
    assert_relative_eq!(
        waveguide
            .resistivity
            .as_ref()
            .expect("resistivity should exist")[0],
        1.68e-8,
        epsilon = 1.0e-20
    );
    assert!(waveguide.set_resistivity_material("quartz").is_err());
    assert!(waveguide.to_string().contains("Circular Waveguide Media"));
    assert!(
        waveguide
            .guide_wavelength()
            .expect("guide wavelength should be defined")
            .iter()
            .all(|value| value.re > 0.0)
    );
    assert!(
        waveguide
            .cutoff_wavelength()
            .expect("cutoff wavelength should be defined")
            .iter()
            .all(|value| *value > 0.0)
    );
}

#[test]
fn constructs_shared_media_components() {
    let media = defined_media(None);
    let points = media.frequency.points();
    let resistance = Array1::from_elem(points, 25.0);
    let capacitance = Array1::from_elem(points, 1.0e-12);
    let inductance = Array1::from_elem(points, 1.0e-9);

    assert_eq!(
        media
            .match_network(4, None)
            .expect("four-port match should be constructed")
            .ports(),
        4
    );
    assert_eq!(
        media
            .load_nports(
                &Array1::from_elem(points, Complex64::new(0.25, 0.0)),
                2,
                None,
            )
            .expect("two-port load should be constructed")
            .s[(0, 1, 1)],
        Complex64::new(0.25, 0.0)
    );
    for network in [
        media.resistor(&resistance),
        media.capacitor(&capacitance),
        media.inductor(&inductance),
        media.shunt_resistor(&resistance),
        media.shunt_capacitor(&capacitance),
        media.shunt_inductor(&inductance),
        media.capacitor_with_q(&capacitance, 2.0e9, 100.0),
        media.inductor_with_q(&inductance, 2.0e9, 100.0, 0.1),
    ] {
        assert_eq!(
            network
                .expect("lumped component should be constructed")
                .ports(),
            2
        );
    }
    assert_eq!(media.tee().expect("tee should be constructed").ports(), 3);
    assert_eq!(
        media
            .splitter(4)
            .expect("splitter should be constructed")
            .ports(),
        4
    );
    assert_eq!(
        media
            .floating_line(10.0, LengthUnit::Degree)
            .expect("floating line should be constructed")
            .ports(),
        4
    );
    assert_eq!(
        media
            .delay_short(30.0, LengthUnit::Degree)
            .expect("delayed short should be constructed")
            .ports(),
        1
    );
    assert_eq!(
        media
            .impedance_mismatch(
                &Array1::from_elem(points, 50.0),
                &Array1::from_elem(points, 75.0),
            )
            .expect("impedance mismatch should be constructed")
            .ports(),
        2
    );
}

#[test]
fn constructs_attenuators_mismatches_isolators_and_random_networks() {
    let media = defined_media(None);
    let attenuation = media
        .attenuator(&Array1::from_elem(3, -6.0), true, 0.0, LengthUnit::Meter)
        .expect("attenuator should be constructed");
    assert_relative_eq!(
        attenuation.s[(0, 1, 0)].re,
        10.0_f64.powf(-6.0 / 20.0),
        epsilon = 1.0e-12
    );
    assert_eq!(attenuation.s[(0, 0, 0)], Complex64::new(0.0, 0.0));

    let mismatch = media
        .lossless_mismatch(&Array1::from_elem(3, Complex64::new(0.6, 0.0)))
        .expect("lossless mismatch should be constructed");
    assert_complex_close(mismatch.s[(0, 0, 0)], Complex64::new(0.6, 0.0));
    assert_complex_close(mismatch.s[(0, 1, 0)], Complex64::new(0.0, 0.8));
    assert_relative_eq!(
        mismatch.s[(0, 0, 0)].norm_sqr() + mismatch.s[(0, 1, 0)].norm_sqr(),
        1.0,
        epsilon = 1.0e-12
    );

    let isolator = media.isolator(0).expect("isolator should be constructed");
    assert_eq!(isolator.s[(0, 0, 1)], Complex64::new(0.0, 0.0));
    assert_ne!(isolator.s[(0, 1, 0)], Complex64::new(0.0, 0.0));
    assert!(media.isolator(2).is_err());

    set_random_seed(42);
    let random = media
        .random_network(3, true, true, true)
        .expect("random network should be constructed");
    for point in 0..random.frequency_points() {
        for port in 0..3 {
            assert_eq!(random.s[(point, port, port)], Complex64::new(0.0, 0.0));
            for other in 0..3 {
                assert_eq!(
                    random.s[(point, port, other)],
                    random.s[(point, other, port)]
                );
            }
        }
    }
    set_random_seed(7);
    let noise = media
        .white_gaussian_polar(0.1, 0.2, 2)
        .expect("polar Gaussian noise should be constructed");
    assert_eq!(noise.s.dim(), (3, 2, 2));
    assert!(noise.s.iter().any(|value| value.norm() > 0.0));
}

#[test]
fn constructs_shunted_delayed_loads() {
    let media = defined_media(None);
    let open = media
        .shunt_delay_open(0.0, LengthUnit::Meter)
        .expect("shunted open should be constructed");
    for point in 0..open.frequency_points() {
        assert_complex_close(open.s[(point, 0, 0)], Complex64::new(0.0, 0.0));
        assert_complex_close(open.s[(point, 1, 0)], Complex64::new(1.0, 0.0));
    }
    let short = media
        .shunt_delay_short(0.0, LengthUnit::Meter)
        .expect("shunted short should be constructed");
    for point in 0..short.frequency_points() {
        assert_complex_close(short.s[(point, 0, 0)], Complex64::new(-1.0, 0.0));
        assert_complex_close(short.s[(point, 1, 0)], Complex64::new(0.0, 0.0));
    }
    let reflection = Complex64::new(0.25, -0.1);
    let shunted = media
        .shunt_delay_load(reflection, 0.0, LengthUnit::Meter)
        .expect("shunted load should be constructed");
    assert_complex_close(
        shunted.s[(0, 0, 0)],
        -(Complex64::new(1.0, 0.0) - reflection) / (3.0 + reflection),
    );
}

#[test]
fn calculates_media_velocities_and_extracts_reflection_distance() {
    let frequency = Frequency::new(1.0, 3.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let beta = frequency
        .values_hz()
        .mapv(|value| std::f64::consts::TAU * value / SPEED_OF_LIGHT);
    let media = DefinedGammaZ0::new(
        frequency.clone(),
        beta.mapv(|value| Complex64::new(0.0, value)),
        Array1::from_elem(3, Complex64::new(50.0, 0.0)),
        None,
    )
    .expect("media should be valid");
    for velocity in media
        .phase_velocity()
        .expect("phase velocity should be defined")
    {
        assert_relative_eq!(velocity.re, SPEED_OF_LIGHT, max_relative = 1.0e-12);
        assert_relative_eq!(velocity.im, 0.0, epsilon = 1.0e-12);
    }
    for velocity in media
        .group_velocity()
        .expect("group velocity should be defined")
    {
        assert_relative_eq!(velocity.re, 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(velocity.im, -SPEED_OF_LIGHT, max_relative = 1.0e-12);
    }

    let physical_distance = 0.02;
    let s = Array3::from_shape_fn((3, 1, 1), |(point, _, _)| {
        Complex64::new(0.0, -2.0 * beta[point] * physical_distance).exp()
    });
    let reflection = rust_rf::Network::new(
        frequency,
        s,
        Array2::from_elem((3, 1), Complex64::new(50.0, 0.0)),
    )
    .expect("reflection network should be valid");
    let extracted = media
        .extract_distance(&reflection)
        .expect("distance should be extracted");
    for distance in extracted {
        assert_relative_eq!(distance, 2.0 * physical_distance, epsilon = 1.0e-12);
    }
}

fn defined_media(port_z0: Option<Array1<Complex64>>) -> DefinedGammaZ0 {
    let frequency = Frequency::new(1.0, 3.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    DefinedGammaZ0::new(
        frequency,
        Array1::from_elem(3, Complex64::new(0.1, 2.0)),
        Array1::from_elem(3, Complex64::new(50.0, 0.0)),
        port_z0,
    )
    .expect("media should be valid")
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(actual.re, expected.re, epsilon = TOLERANCE);
    assert_relative_eq!(actual.im, expected.im, epsilon = TOLERANCE);
}
