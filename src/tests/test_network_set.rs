use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use approx::assert_relative_eq;
use ndarray::{Array2, Array3, array};
use num_complex::Complex64;
use rand::SeedableRng;
use rand::rngs::StdRng;
#[cfg(feature = "xlsx")]
use rust_rf::io::NetworkDataFormat;
use rust_rf::io::{StoredObject, read_object};
use rust_rf::{
    Frequency, Network, NetworkParameter, NetworkScalarAttribute, NetworkSet, NetworkSetAttribute,
    function_on_networks, get_set, tuner_constellation,
};

const TOLERANCE: f64 = 1.0e-12;

#[test]
fn validates_network_set_compatibility() {
    let first = one_port(&[1.0, 2.0], &[1.0, 2.0]);
    let different_frequency = one_port(&[1.0, 3.0], &[1.0, 2.0]);
    assert!(NetworkSet::new(vec![first, different_frequency], None).is_err());
}

#[test]
fn calculates_complex_mean_and_standard_deviation() {
    let first = one_port(&[1.0, 2.0], &[1.0, 3.0]);
    let second = one_port(&[1.0, 2.0], &[3.0, 7.0]);
    let set = NetworkSet::new(vec![first, second], Some("samples".to_owned()))
        .expect("network set should be valid");

    let mean = set.mean_s().expect("mean should be defined");
    assert_eq!(mean.name.as_deref(), Some("samples-mean"));
    assert_complex_close(mean.s[(0, 0, 0)], Complex64::new(2.0, 0.0));
    assert_complex_close(mean.s[(1, 0, 0)], Complex64::new(5.0, 0.0));

    let standard_deviation = set.std_s().expect("standard deviation should be defined");
    assert_relative_eq!(standard_deviation.s[(0, 0, 0)].re, 1.0, epsilon = TOLERANCE);
    assert_relative_eq!(standard_deviation.s[(1, 0, 0)].re, 2.0, epsilon = TOLERANCE);
}

#[test]
fn rejects_aggregates_of_empty_sets() {
    let set = NetworkSet::new(Vec::new(), None).expect("empty construction is supported");
    assert!(set.mean_s().is_err());
    assert!(set.std_s().is_err());
}

#[test]
fn interpolates_networks_along_parameter_axis() {
    let low = one_port(&[1.0, 2.0], &[1.0, 3.0]);
    let high = one_port(&[1.0, 2.0], &[5.0, 11.0]);
    let mut set = NetworkSet::new(vec![high, low], Some("sweep".to_owned()))
        .expect("network set should be valid");
    set.parameters
        .insert("temperature".to_owned(), vec![100.0, 0.0]);

    let interpolated = set
        .interpolate_from_network(25.0)
        .expect("parameter interpolation should succeed");
    assert_complex_close(interpolated.s[(0, 0, 0)], Complex64::new(2.0, 0.0));
    assert_complex_close(interpolated.s[(1, 0, 0)], Complex64::new(5.0, 0.0));
    assert_eq!(interpolated.name.as_deref(), Some("sweep-interpolated"));
    assert!(set.interpolate_from_network(150.0).is_err());
    assert!(set.interpolate_from_values(&[0.0], 0.0).is_err());
}

#[test]
fn selects_filters_and_sorts_parameterized_networks() {
    let mut high = one_port(&[1.0, 2.0], &[3.0, 3.0]);
    high.name = Some("gamma".to_owned());
    let mut low = one_port(&[1.0, 2.0], &[1.0, 1.0]);
    low.name = Some("alpha".to_owned());
    let mut middle = one_port(&[1.0, 2.0], &[2.0, 2.0]);
    middle.name = Some("beta".to_owned());
    let mut set = NetworkSet::new(vec![high, low, middle], Some("sweep".to_owned()))
        .expect("network set should be valid");
    set.set_parameter("temperature", vec![30.0, 10.0, 20.0])
        .expect("parameter should be valid");
    set.set_parameter("bias", vec![1.0, 1.0, 2.0])
        .expect("parameter should be valid");

    let selected = set
        .select(&BTreeMap::from([
            ("temperature".to_owned(), vec![10.0, 30.0]),
            ("bias".to_owned(), vec![1.0]),
        ]))
        .expect("selection should succeed");
    assert_eq!(selected.len(), 2);
    assert_eq!(selected.parameters["temperature"], vec![30.0, 10.0]);
    assert!(
        set.select(&BTreeMap::from([("missing".to_owned(), vec![1.0])]))
            .expect("unknown selection should be empty")
            .is_empty()
    );

    let filtered = set
        .filter_names("mm")
        .expect("name filtering should succeed");
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered.networks[0].name.as_deref(), Some("gamma"));

    set.sort_by_name();
    assert_eq!(set.networks[0].name.as_deref(), Some("alpha"));
    assert_eq!(set.networks[1].name.as_deref(), Some("beta"));
    assert_eq!(set.networks[2].name.as_deref(), Some("gamma"));
    assert_eq!(set.parameters["temperature"], vec![10.0, 20.0, 30.0]);
    assert_eq!(set.parameter_names(), vec!["bias", "temperature"]);
}

#[test]
fn interpolates_from_named_parameter_and_across_frequency() {
    let low = one_port(&[1.0, 3.0], &[1.0, 3.0]);
    let high = one_port(&[1.0, 3.0], &[5.0, 11.0]);
    let mut set = NetworkSet::new(vec![low, high], Some("sweep".to_owned()))
        .expect("network set should be valid");
    set.set_parameter("temperature", vec![0.0, 100.0])
        .expect("parameter should be valid");

    let interpolated = set
        .interpolate_from_parameter("temperature", 25.0, &BTreeMap::new())
        .expect("named interpolation should succeed");
    assert_complex_close(interpolated.s[(0, 0, 0)], Complex64::new(2.0, 0.0));
    assert_complex_close(interpolated.s[(1, 0, 0)], Complex64::new(5.0, 0.0));

    let frequency = Frequency::from_hz(array![1.0, 2.0, 3.0]).expect("frequency should be valid");
    let frequency_interpolated = set
        .interpolate_frequency(&frequency)
        .expect("frequency interpolation should succeed");
    assert_eq!(frequency_interpolated.networks[0].frequency_points(), 3);
    assert_complex_close(
        frequency_interpolated.networks[0].s[(1, 0, 0)],
        Complex64::new(2.0, 0.0),
    );
    assert_eq!(
        frequency_interpolated.parameters["temperature"],
        vec![0.0, 100.0]
    );
}

#[test]
fn rejects_invalid_parameter_lengths() {
    let mut set = NetworkSet::new(vec![one_port(&[1.0, 2.0], &[1.0, 2.0])], None)
        .expect("network set should be valid");
    assert!(set.set_parameter("temperature", vec![1.0, 2.0]).is_err());
}

#[test]
fn converts_named_scattering_maps_and_sorts_clones() {
    let frequency = Frequency::from_hz(array![1.0, 2.0]).expect("frequency should be valid");
    let alpha = Array3::from_shape_vec(
        (2, 1, 1),
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    )
    .expect("alpha shape should be valid");
    let beta = Array3::from_shape_vec(
        (2, 1, 1),
        vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)],
    )
    .expect("beta shape should be valid");
    let set = NetworkSet::from_s_map(
        BTreeMap::from([
            ("beta".to_owned(), beta.clone()),
            ("alpha".to_owned(), alpha.clone()),
        ]),
        frequency,
        Complex64::new(75.0, 0.0),
        Some("dictionary".to_owned()),
    )
    .expect("named scattering matrices should construct a set");

    assert_eq!(set.networks[0].name.as_deref(), Some("alpha"));
    assert_eq!(set.networks[1].name.as_deref(), Some("beta"));
    assert_eq!(set.networks[0].z0[(0, 0)], Complex64::new(75.0, 0.0));
    assert_eq!(
        set.to_s_map().expect("S map should convert")["alpha"],
        alpha
    );
    assert_eq!(
        set.to_network_map().expect("network map should convert")["beta"].s,
        beta
    );
    assert_eq!(set.sorted_by_name(), set);

    let unnamed = NetworkSet::new(vec![one_port(&[1.0, 2.0], &[1.0, 2.0])], None)
        .expect("unnamed set should construct");
    assert!(unnamed.to_network_map().is_err());
}

#[test]
fn calculates_component_statistics_and_uncertainty_bounds() {
    let first = one_port(&[1.0, 2.0], &[1.0, 3.0]);
    let second = one_port(&[1.0, 2.0], &[3.0, 7.0]);
    let set = NetworkSet::new(vec![first, second], Some("samples".to_owned()))
        .expect("network set should be valid");

    let mean_magnitude = set
        .mean_s_magnitude()
        .expect("magnitude mean should be defined");
    let std_magnitude = set
        .std_s_magnitude()
        .expect("magnitude deviation should be defined");
    assert_complex_close(mean_magnitude.s[(0, 0, 0)], Complex64::new(2.0, 0.0));
    assert_complex_close(std_magnitude.s[(1, 0, 0)], Complex64::new(2.0, 0.0));
    assert_relative_eq!(
        set.mean_s_db().expect("dB mean should be defined").s[(0, 0, 0)].re,
        20.0 * 2.0_f64.log10(),
        epsilon = TOLERANCE
    );

    let (mean, lower, upper) = set
        .uncertainty_network_triplet(NetworkSetAttribute::Magnitude, 2.0)
        .expect("uncertainty bounds should be defined");
    assert_complex_close(mean.s[(1, 0, 0)], Complex64::new(5.0, 0.0));
    assert_complex_close(lower.s[(1, 0, 0)], Complex64::new(1.0, 0.0));
    assert_complex_close(upper.s[(1, 0, 0)], Complex64::new(9.0, 0.0));
    assert!(
        set.uncertainty_network_triplet(NetworkSetAttribute::Scattering, f64::NAN)
            .is_err()
    );
}

#[test]
fn calculates_generated_statistics_for_every_parameter_and_component() {
    let frequency = Frequency::from_hz(array![1.0, 2.0]).expect("frequency");
    let s = Array3::from_shape_fn((2, 2, 2), |(_, output, input)| {
        Complex64::new(if output == input { 0.1 } else { 0.5 }, 0.05)
    });
    let network = Network::new(
        frequency,
        s,
        Array2::from_elem((2, 2), Complex64::new(50.0, 0.0)),
    )
    .expect("two-port");
    let set = NetworkSet::new(vec![network.clone(), network], Some("generated".to_owned()))
        .expect("network set");

    for parameter in [
        NetworkParameter::Scattering,
        NetworkParameter::Impedance,
        NetworkParameter::Admittance,
        NetworkParameter::Abcd,
        NetworkParameter::InverseHybrid,
        NetworkParameter::Hybrid,
        NetworkParameter::ScatteringTransfer,
    ] {
        assert_eq!(
            set.mean_parameter(parameter)
                .expect("parameter mean")
                .s
                .dim(),
            (2, 2, 2)
        );
        assert!(
            set.std_parameter(parameter)
                .expect("parameter deviation")
                .s
                .iter()
                .all(|value| value.norm() < TOLERANCE)
        );
        for component in [
            NetworkScalarAttribute::Magnitude,
            NetworkScalarAttribute::Decibel,
            NetworkScalarAttribute::Decibel10,
            NetworkScalarAttribute::PhaseDegrees,
            NetworkScalarAttribute::Real,
            NetworkScalarAttribute::Imaginary,
            NetworkScalarAttribute::Vswr,
        ] {
            assert_eq!(
                set.mean_parameter_component(parameter, component)
                    .expect("component mean")
                    .s
                    .dim(),
                (2, 2, 2)
            );
            assert!(
                set.std_parameter_component(parameter, component)
                    .expect("component deviation")
                    .s
                    .iter()
                    .all(|value| value.norm() < TOLERANCE)
            );
        }
    }
}

#[test]
fn exposes_parameter_and_datetime_metadata() {
    let mut first = one_port(&[1.0, 2.0], &[1.0, 2.0]);
    first.name = Some("2026.07.21.10.11.12.123456".to_owned());
    let mut second = one_port(&[1.0, 2.0], &[3.0, 4.0]);
    second.name = Some("2026.07.21.10.11.13.654321".to_owned());
    let mut set = NetworkSet::new(vec![first, second], None).expect("set should construct");
    assert!(!set.has_parameters());
    set.set_parameter("bias", vec![1.0, 2.0])
        .expect("parameter should be valid");
    set.set_text_parameter("mode", vec!["a".to_owned(), "b".to_owned()])
        .expect("text parameter should be valid");
    assert!(set.has_parameters());
    let dates = set.datetime_index().expect("network names should parse");
    assert_eq!(dates.len(), 2);
    assert_eq!(dates[0].and_utc().timestamp_subsec_micros(), 123_456);
}

#[test]
fn samples_maps_zips_and_adds_polar_noise() {
    let first = one_port(&[1.0, 2.0], &[1.0, 2.0]);
    let second = one_port(&[1.0, 2.0], &[3.0, 4.0]);
    let mut set = NetworkSet::new(
        vec![first.clone(), second.clone()],
        Some("operations".to_owned()),
    )
    .expect("set should construct");
    set.set_parameter("sample", vec![0.0, 1.0])
        .expect("parameter should be valid");

    let mut rng = StdRng::seed_from_u64(42);
    let samples = set
        .random_networks_with_rng(5, &mut rng)
        .expect("random samples should be drawn");
    assert_eq!(samples.len(), 5);
    assert!(
        samples
            .iter()
            .all(|network| network == &first || network == &second)
    );

    let doubled = set
        .map_networks(|network| {
            let mut mapped = network.clone();
            mapped.s.mapv_inplace(|value| value * 2.0);
            Ok(mapped)
        })
        .expect("typed element-wise operation should succeed");
    assert_eq!(doubled.parameters, set.parameters);
    assert_complex_close(doubled.networks[1].s[(1, 0, 0)], Complex64::new(8.0, 0.0));

    let summed = set
        .zip_networks(&doubled, |left, right| {
            let mut mapped = left.clone();
            mapped.s = &left.s + &right.s;
            Ok(mapped)
        })
        .expect("typed pairwise operation should succeed");
    assert_complex_close(summed.networks[0].s[(1, 0, 0)], Complex64::new(6.0, 0.0));

    let added = set.add_set(&doubled).expect("set addition should succeed");
    assert_eq!(added, summed);
    let multiplied = set
        .multiply_network(&first)
        .expect("single-network multiplication should succeed");
    assert_complex_close(
        multiplied.networks[1].s[(1, 0, 0)],
        Complex64::new(8.0, 0.0),
    );

    let zero_noise = NetworkSet::new(vec![first.clone(), first.clone()], None)
        .expect("zero-deviation set should construct");
    assert_eq!(
        zero_noise
            .add_polar_noise(&second)
            .expect("zero noise should be applicable")
            .s,
        second.s
    );
    let empty = NetworkSet::new(Vec::new(), None).expect("empty set should construct");
    assert!(empty.random_networks_with_rng(1, &mut rng).is_err());
}

#[test]
fn aggregates_filters_and_builds_tuner_constellations() {
    let mut first = one_port(&[1.0, 2.0], &[1.0, 3.0]);
    first.name = Some("cold-sample".to_owned());
    let mut second = one_port(&[1.0, 2.0], &[3.0, 7.0]);
    second.name = Some("hot-sample".to_owned());
    let aggregate = function_on_networks(
        &[first.clone(), second.clone()],
        Some("mean".to_owned()),
        |matrices| Ok((&matrices[0] + &matrices[1]) / 2.0),
    )
    .expect("aggregate should succeed");
    assert_eq!(aggregate.name.as_deref(), Some("mean"));
    assert_complex_close(aggregate.s[(1, 0, 0)], Complex64::new(5.0, 0.0));

    let dictionary = BTreeMap::from([("cold-1".to_owned(), first), ("hot-1".to_owned(), second)]);
    assert_eq!(
        get_set(&dictionary, "hot", Some("selected".to_owned()))
            .expect("dictionary selection should succeed")
            .expect("a matching set should exist")
            .len(),
        1
    );
    assert!(get_set(&dictionary, "missing", None).unwrap().is_none());

    let tuner = tuner_constellation("tuner", 76.0, 50.0, 3, 4).expect("constellation should build");
    assert_eq!(tuner.networks.len(), 12);
    assert_eq!(tuner.real.len(), 12);
    assert_eq!(tuner.imaginary.len(), 12);
    assert_eq!(tuner.reflection.len(), 12);
    assert_relative_eq!(tuner.reflection[0].norm(), 0.1, epsilon = TOLERANCE);
    assert_relative_eq!(tuner.reflection[2].norm(), 0.9, epsilon = TOLERANCE);
    assert_eq!(tuner.networks.networks[0].name.as_deref(), Some("tuner_0"));
}

#[test]
fn loads_directories_citi_and_mdif_and_serializes_sets() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should follow the Unix epoch")
        .as_nanos();
    let root = std::env::current_dir()
        .expect("current directory should exist")
        .join(".temp")
        .join(format!("network-set-{}-{unique}", std::process::id()));
    fs::create_dir_all(&root).expect("temporary directory should be created");

    let mut network = one_port(&[1.0, 2.0], &[0.1, 0.2]);
    network.name = Some("sample".to_owned());
    network
        .write_touchstone(root.join("sample.s1p"))
        .expect("Touchstone fixture should be written");
    let from_directory =
        NetworkSet::from_directory(&root).expect("directory should load supported networks");
    assert_eq!(from_directory.len(), 1);

    let set = NetworkSet::new(vec![network], Some("stored-set".to_owned()))
        .expect("set should construct");
    let object_path = set
        .write_to_path(root.join("stored"), true)
        .expect("set should serialize");
    assert!(matches!(
        read_object(object_path).expect("set should deserialize"),
        StoredObject::NetworkSet(restored) if restored == set
    ));

    let mdif_path = root.join("stored.mdf");
    set.write_mdif(&mdif_path, &["round trip".to_owned()])
        .expect("MDIF should write");
    let mdif_set = NetworkSet::from_mdif(&mdif_path).expect("MDIF should read");
    assert_eq!(mdif_set.networks[0].s, set.networks[0].s);

    let citi_path = root.join("stored.cti");
    fs::write(
        &citi_path,
        "CITIFILE A.01.00\nNAME stored\nVAR FREQ MAG 2\nDATA S RI\nVAR_LIST_BEGIN\n1\n2\nVAR_LIST_END\nBEGIN\n0.1,0\n0.2,0\nEND\n",
    )
    .expect("CITI fixture should be written");
    let citi_set = NetworkSet::from_citi(&citi_path).expect("CITI should read");
    assert_eq!(citi_set.len(), 1);
    assert_eq!(citi_set.networks[0].s[(1, 0, 0)], Complex64::new(0.2, 0.0));

    #[cfg(feature = "xlsx")]
    {
        let workbook = root.join("stored.xlsx");
        set.write_spreadsheet(&workbook, NetworkDataFormat::DecibelAngle)
            .expect("spreadsheet should write");
        assert!(workbook.is_file());
    }

    fs::remove_dir_all(&root).expect("temporary fixtures should be removed");
    let _ = fs::remove_dir(root.parent().expect("temporary root should have a parent"));
}

#[cfg(feature = "dataframe")]
#[test]
fn exports_parameterized_networks_to_dataframe() {
    let mut set = NetworkSet::new(
        vec![
            one_port(&[1.0, 2.0], &[1.0, 3.0]),
            one_port(&[1.0, 2.0], &[5.0, 11.0]),
        ],
        None,
    )
    .expect("network set should be valid");
    assert!(set.to_dataframe().is_err());
    set.parameters
        .insert("temperature".to_owned(), vec![20.0, 30.0]);

    let frame = set
        .to_dataframe()
        .expect("DataFrame conversion should succeed");
    assert_eq!(frame.height(), 4);
    assert_eq!(frame.width(), 7);
    let temperature = frame
        .column("temperature")
        .expect("parameter column should exist")
        .f64()
        .expect("parameter column should contain floats");
    assert_eq!(temperature.get(0), Some(20.0));
    assert_eq!(temperature.get(3), Some(30.0));

    let attributes = set
        .network_attribute_dataframe(NetworkScalarAttribute::Decibel, 0, 0)
        .expect("attribute DataFrame should succeed");
    assert_eq!(attributes.height(), 2);
    assert_eq!(attributes.width(), 3);
    assert_relative_eq!(
        attributes
            .column("Network0")
            .expect("first network column should exist")
            .f64()
            .expect("network column should contain floats")
            .get(0)
            .expect("first value should exist"),
        0.0,
        epsilon = TOLERANCE
    );
}

fn one_port(frequencies: &[f64], values: &[f64]) -> Network {
    let frequency = Frequency::from_hz(array![frequencies[0], frequencies[1]])
        .expect("frequency should be valid");
    let s = Array3::from_shape_fn((2, 1, 1), |(point, _, _)| {
        Complex64::new(values[point], 0.0)
    });
    let z0 = Array2::from_elem((2, 1), Complex64::new(50.0, 0.0));
    Network::new(frequency, s, z0).expect("network should be valid")
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(actual.re, expected.re, epsilon = TOLERANCE);
    assert_relative_eq!(actual.im, expected.im, epsilon = TOLERANCE);
}
use std::collections::BTreeMap;
