use std::collections::BTreeMap;
use std::io::Cursor;

use approx::assert_relative_eq;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rust_rf::io::{Mdif, MdifValue};
use rust_rf::{Frequency, Network, NetworkSet};

#[test]
fn parses_explicit_ri_data_parameters_and_comments() {
    let text = "! file comment\n\
VAR Cm(real) = 7e-16\n\
BEGIN Sweep.SP\n\
# Hz S RI R 50\n\
! network name: sample\n\
! block comment\n\
% freq(real) S[1,1](complex)\n\
710000000 0.999999951 -0.000312274302\n\
END\n";

    let mdif = Mdif::from_reader(Cursor::new(text)).unwrap();
    assert_eq!(mdif.comments, vec!["file comment"]);
    assert_eq!(mdif.parameters, vec!["Cm"]);
    assert_eq!(mdif.parameter_values[0]["Cm"], MdifValue::Number(7.0e-16));
    assert_eq!(mdif.networks[0].name.as_deref(), Some("sample"));
    assert_eq!(
        mdif.networks[0].comments,
        "network name: sample\nblock comment"
    );
    assert_eq!(
        mdif.networks[0].s[(0, 0, 0)],
        Complex64::new(0.999999951, -0.000312274302)
    );
    assert_eq!(mdif.networks[0].frequency.values_hz()[0], 710_000_000.0);
    assert_eq!(
        mdif.to_network_set().unwrap().parameters["Cm"],
        vec![7.0e-16]
    );
}

#[test]
fn parses_awr_component_order_and_magnitude_angle() {
    let text = "VAR bias=1\n\
BEGIN ACDATA\n\
# GHz S MA R 50\n\
% F N11X N11Y N21X N21Y N12X N12Y N22X N22Y\n\
1 0.1 0 0.2 10 0.3 20 0.4 30\n\
END\n";
    let network = &Mdif::parse(text).unwrap().networks[0];
    assert_eq!(network.frequency.values_hz()[0], 1.0e9);
    assert_relative_eq!(network.s[(0, 0, 0)].norm(), 0.1, epsilon = 1.0e-12);
    assert_relative_eq!(network.s[(0, 1, 0)].norm(), 0.2, epsilon = 1.0e-12);
    assert_relative_eq!(network.s[(0, 0, 1)].norm(), 0.3, epsilon = 1.0e-12);
    assert_relative_eq!(
        network.s[(0, 1, 1)].arg().to_degrees(),
        30.0,
        epsilon = 1.0e-12
    );
}

#[test]
fn parses_db_and_converts_z_and_y_parameters() {
    let db = "BEGIN ACDATA\n# Hz S DB R 50\n% F S[1,1](complex)\n1 -20 90\nEND\n";
    let db_network = &Mdif::parse(db).unwrap().networks[0];
    assert_relative_eq!(db_network.s[(0, 0, 0)].norm(), 0.1, epsilon = 1.0e-12);

    let z = "BEGIN data\n% freq(real) Z[1,1](complex)\n1 50 0\nEND\n";
    let y = "BEGIN data\n% freq(real) Y[1,1](complex)\n1 0.02 0\nEND\n";
    assert_relative_eq!(
        Mdif::parse(z).unwrap().networks[0].s[(0, 0, 0)].norm(),
        0.0,
        epsilon = 1.0e-12
    );
    assert_relative_eq!(
        Mdif::parse(y).unwrap().networks[0].s[(0, 0, 0)].norm(),
        0.0,
        epsilon = 1.0e-12
    );
}

#[test]
fn writer_round_trips_network_sets_and_parameters() {
    let frequency = Frequency::from_hz(ndarray::arr1(&[1.0, 2.0])).unwrap();
    let s = Array3::from_shape_vec(
        (2, 1, 1),
        vec![Complex64::new(0.1, 0.2), Complex64::new(0.3, 0.4)],
    )
    .unwrap();
    let z0 = Array2::from_elem((2, 1), Complex64::new(50.0, 0.0));
    let mut network = Network::new(frequency, s, z0).unwrap();
    network.name = Some("roundtrip".to_owned());
    let mut set = NetworkSet::new(vec![network], None).unwrap();
    set.parameters = BTreeMap::from([("bias".to_owned(), vec![2.5])]);
    let mut bytes = Vec::new();
    Mdif::write(&set, &mut bytes, &["generated".to_owned()]).unwrap();

    let parsed = Mdif::from_reader(Cursor::new(bytes)).unwrap();
    assert_eq!(parsed.comments, vec!["generated"]);
    assert_eq!(
        parsed.to_network_set().unwrap().parameters["bias"],
        vec![2.5]
    );
    assert_eq!(parsed.networks[0].s, set.networks[0].s);
    assert_eq!(parsed.networks[0].name, set.networks[0].name);
}

#[test]
fn reads_and_writes_noise_parameters_without_drift() {
    let text = "BEGIN ACDATA\n\
# GHz S RI R 50\n\
% F S[1,1](complex) S[1,2](complex) S[2,1](complex) S[2,2](complex)\n\
1 0 0 0 0 0 0 0 0\n\
END\n\
BEGIN NDATA\n\
%F nfmin n11x n11y rn\n\
# GHz S MA R 50\n\
1 0.5 0.25 45 0.2\n\
2 0.6 0.30 -30 0.4\n\
END\n";
    let mdif = Mdif::parse(text).unwrap();
    let noise = mdif.networks[0].noise.as_ref().unwrap();
    assert_eq!(noise.frequency.values_hz().to_vec(), vec![1.0e9, 2.0e9]);
    assert_eq!(noise.minimum_noise_figure_db.to_vec(), vec![0.5, 0.6]);
    assert_relative_eq!(noise.optimal_reflection[0].norm(), 0.25, epsilon = 1.0e-12);
    assert_eq!(noise.equivalent_noise_resistance.to_vec(), vec![10.0, 20.0]);
    let gamma = noise.optimal_reflection[0];
    let optimal_impedance = Complex64::new(50.0, 0.0) * (Complex64::new(1.0, 0.0) + gamma)
        / (Complex64::new(1.0, 0.0) - gamma);
    let factors = mdif.networks[0].noise_factor(optimal_impedance).unwrap();
    assert_relative_eq!(factors[0], 10.0_f64.powf(0.5 / 10.0), epsilon = 1.0e-12);

    let set = NetworkSet::new(mdif.networks, None).unwrap();
    let mut bytes = Vec::new();
    Mdif::write(&set, &mut bytes, &[]).unwrap();
    let round_trip = Mdif::from_reader(Cursor::new(bytes)).unwrap();
    assert_eq!(round_trip.networks[0].noise, set.networks[0].noise);
}

#[test]
fn option_strings_preserve_two_port_touchstone_order() {
    assert_eq!(
        Mdif::option_string(2),
        "%F n11x n11y n21x n21y n12x n12y n22x n22y"
    );
}

#[test]
fn string_parameters_round_trip_as_text_coordinates() {
    let text = "VAR mode = \"cold\"\nBEGIN data\n% F S[1,1](complex)\n1 0 0\nEND\n";
    let mdif = Mdif::parse(text).unwrap();
    assert_eq!(
        mdif.parameter_values[0]["mode"],
        MdifValue::Text("cold".to_owned())
    );
    let set = mdif.to_network_set().unwrap();
    assert_eq!(set.text_parameters["mode"], vec!["cold"]);
    let mut bytes = Vec::new();
    Mdif::write(&set, &mut bytes, &[]).unwrap();
    let parsed = Mdif::from_reader(Cursor::new(bytes)).unwrap();
    assert_eq!(
        parsed.parameter_values[0]["mode"],
        MdifValue::Text("cold".to_owned())
    );
}
