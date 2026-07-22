//! Circuit assembly, solution, validation, and component regressions.
//!
//! These consolidated tests cover the upstream constructor, graph, active
//! parameter, voltage/current, class-method, reduction, cascade, and Wilkinson
//! divider cases through representative complete circuits.

use std::collections::HashMap;
use std::sync::Arc;

use approx::assert_relative_eq;
use ndarray::{Array1, Array2, Array3, array};
use num_complex::Complex64;
use rust_rf::circuit::{Circuit, CircuitConnection};
use rust_rf::media::{DefinedGammaZ0, LengthUnit, Media};
use rust_rf::{Frequency, Network};

/// Connects a two-port between external ports and checks scattering, graph,
/// waves, voltages/currents, active quantities, reduction, and network updates.
#[test]
fn connects_a_two_port_between_external_ports() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency should be valid");
    let port_one = Circuit::port(frequency.clone(), "Port 1", Complex64::new(50.0, 0.0))
        .expect("port should be valid");
    let port_two = Circuit::port(frequency.clone(), "Port 2", Complex64::new(50.0, 0.0))
        .expect("port should be valid");
    let through = Arc::new(through_network(frequency).expect("through network should be valid"));
    let mut circuit = Circuit::new(vec![
        vec![port_one, CircuitConnection::new(Arc::clone(&through), 0)],
        vec![CircuitConnection::new(through, 1), port_two],
    ])
    .expect("circuit should be valid");
    circuit.name = Some("assembled".to_owned());

    let network = circuit.network().expect("circuit should be solved");
    assert_two_port_network(&network);
    assert_two_port_waves(&circuit).expect("two-port waves should be valid");
    assert_two_port_voltages_and_currents(&circuit)
        .expect("two-port voltages and currents should be valid");
    assert_two_port_active_parameters(&circuit)
        .expect("two-port active parameters should be valid");
    assert_two_port_structure(&circuit);
    assert_two_port_reduction_and_update(&circuit, &network)
        .expect("two-port reduction and update should be valid");
}

fn assert_two_port_network(network: &Network) {
    assert_eq!(network.name.as_deref(), Some("assembled"));
    assert_eq!(network.port_names, ["Port 1", "Port 2"]);
    for point in 0..network.frequency_points() {
        assert_complex_close(network.s[(point, 0, 0)], Complex64::new(0.0, 0.0));
        assert_complex_close(network.s[(point, 1, 1)], Complex64::new(0.0, 0.0));
        assert_complex_close(network.s[(point, 1, 0)], Complex64::new(1.0, 0.0));
        assert_complex_close(network.s[(point, 0, 1)], Complex64::new(1.0, 0.0));
    }
}

fn assert_two_port_waves(circuit: &Circuit) -> rust_rf::Result<()> {
    let graph = circuit.graph();
    assert_eq!(graph.node_count(), 5);
    assert_eq!(graph.edge_count(), 4);

    let incident = circuit.incident_waves(&array![1.0, 0.0], &array![0.0, 0.0])?;
    assert_eq!(incident.len(), 4);
    assert_complex_close(incident[0], Complex64::new(2.0_f64.sqrt(), 0.0));
    assert_complex_close(incident[3], Complex64::new(0.0, 0.0));
    let outgoing = circuit.outgoing_waves(&incident)?;
    assert_complex_close(outgoing[(0, 0)], Complex64::new(0.0, 0.0));
    assert_complex_close(outgoing[(0, 3)], Complex64::new(2.0_f64.sqrt(), 0.0));
    Ok(())
}

fn assert_two_port_voltages_and_currents(circuit: &Circuit) -> rust_rf::Result<()> {
    let (voltages, currents) =
        circuit.external_voltages_currents(&array![1.0, 0.0], &array![0.0, 0.0])?;
    assert_complex_close(voltages[(0, 0)], Complex64::new(10.0, 0.0));
    assert_complex_close(voltages[(0, 1)], Complex64::new(10.0, 0.0));
    assert_complex_close(currents[(0, 0)], Complex64::new(0.2, 0.0));
    assert_complex_close(currents[(0, 1)], Complex64::new(-0.2, 0.0));
    let (internal_voltages, internal_currents) =
        circuit.internal_voltages_currents(&array![1.0, 0.0], &array![0.0, 0.0])?;
    for connection in 0..4 {
        assert_complex_close(
            internal_voltages[(0, connection)],
            Complex64::new(10.0, 0.0),
        );
    }
    assert_complex_close(
        internal_currents[(0, 0)] + internal_currents[(0, 1)],
        Complex64::new(0.0, 0.0),
    );
    assert_complex_close(
        internal_currents[(0, 2)] + internal_currents[(0, 3)],
        Complex64::new(0.0, 0.0),
    );
    Ok(())
}

fn assert_two_port_active_parameters(circuit: &Circuit) -> rust_rf::Result<()> {
    let excitation = array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let active = circuit.active_s(&excitation)?;
    assert_complex_close(active[(0, 0)], Complex64::new(2.0, 0.0));
    assert_complex_close(active[(0, 1)], Complex64::new(0.5, 0.0));
    let active_z = circuit.active_z(&excitation)?;
    assert_complex_close(active_z[(0, 0)], Complex64::new(-150.0, 0.0));
    assert_complex_close(active_z[(0, 1)], Complex64::new(150.0, 0.0));
    let active_y = circuit.active_y(&excitation)?;
    assert_complex_close(active_y[(0, 0)], Complex64::new(-1.0 / 150.0, 0.0));
    assert_complex_close(active_y[(0, 1)], Complex64::new(1.0 / 150.0, 0.0));
    let active_vswr = circuit.active_vswr(&excitation)?;
    assert_relative_eq!(active_vswr[(0, 0)], -3.0, epsilon = 1.0e-12);
    assert_relative_eq!(active_vswr[(0, 1)], 3.0, epsilon = 1.0e-12);
    Ok(())
}

fn assert_two_port_structure(circuit: &Circuit) {
    assert_eq!(circuit.connection_count(), 4);
    assert_eq!(circuit.intersection_count(), 2);
    assert_eq!(circuit.dimension(), 4);
    assert_eq!(circuit.network_count(), 3);
    assert_eq!(circuit.networks_by_name().len(), 3);
    assert_eq!(circuit.port_indexes(), vec![0, 3]);
    assert_eq!(circuit.port_z0().dim(), (2, 2));
    assert!(circuit.is_connected());
    assert_eq!(circuit.intersections_by_name().len(), 2);
    assert_eq!(circuit.edges().len(), 4);
}

fn assert_two_port_reduction_and_update(
    circuit: &Circuit,
    network: &Network,
) -> rust_rf::Result<()> {
    let reduced = circuit.reduced()?;
    assert_eq!(reduced.network_count(), 3);
    let reduced_network = reduced.network()?;
    for (actual, expected) in reduced_network.s.iter().zip(network.s.iter()) {
        assert_complex_close(*actual, *expected);
    }

    let replacement = Arc::new(through_network(circuit.frequency().clone())?);
    let updated =
        circuit.updated_networks(&HashMap::from([("through".to_owned(), replacement)]))?;
    let updated_network = updated.network()?;
    for (actual, expected) in updated_network.s.iter().zip(network.s.iter()) {
        assert_complex_close(*actual, *expected);
    }
    Ok(())
}

/// Checks the ideal equal-impedance three-way junction matrix.
#[test]
fn produces_ideal_equal_impedance_three_way_intersection() {
    let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency should be valid");
    let circuit = Circuit::new(vec![vec![
        Circuit::port(frequency.clone(), "P1", Complex64::new(50.0, 0.0))
            .expect("port should be valid"),
        Circuit::port(frequency.clone(), "P2", Complex64::new(50.0, 0.0))
            .expect("port should be valid"),
        Circuit::port(frequency, "P3", Complex64::new(50.0, 0.0)).expect("port should be valid"),
    ]])
    .expect("circuit should be valid");

    let scattering = circuit.external_s().expect("circuit should be solved");
    for row in 0..3 {
        for column in 0..3 {
            let expected = if row == column { -1.0 / 3.0 } else { 2.0 / 3.0 };
            assert_relative_eq!(scattering[(0, row, column)].re, expected, epsilon = 1.0e-12);
            assert_relative_eq!(scattering[(0, row, column)].im, 0.0, epsilon = 1.0e-12);
        }
    }
}

/// Checks rejection of duplicate connection nodes and circuits without external ports.
#[test]
fn rejects_duplicate_ports_and_missing_external_ports() {
    let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency should be valid");
    let network = Arc::new(through_network(frequency).expect("through network should be valid"));
    assert!(
        Circuit::new(vec![vec![
            CircuitConnection::new(Arc::clone(&network), 0),
            CircuitConnection::new(network, 0),
        ]])
        .is_err()
    );

    let frequency = Frequency::from_hz(array![1.0e9]).expect("frequency should be valid");
    let network = Arc::new(through_network(frequency).expect("through network should be valid"));
    assert!(Circuit::new(vec![vec![CircuitConnection::new(network, 0)]]).is_err());
}

/// Checks ground, open, series-impedance, and shunt-admittance constructors.
#[test]
fn constructs_lumped_circuit_components() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9]).expect("frequency should be valid");
    let reference = Complex64::new(50.0, 0.0);
    let series = Circuit::series_impedance(
        frequency.clone(),
        &array![Complex64::new(50.0, 0.0), Complex64::new(50.0, 0.0)],
        "series",
        reference,
    )
    .expect("series impedance should be constructed");
    assert_eq!(series.name.as_deref(), Some("series"));
    assert_complex_close(series.s[(0, 0, 0)], Complex64::new(1.0 / 3.0, 0.0));
    assert_complex_close(series.s[(0, 1, 0)], Complex64::new(2.0 / 3.0, 0.0));

    let shunt = Circuit::shunt_admittance(
        frequency.clone(),
        &array![Complex64::new(0.02, 0.0), Complex64::new(0.02, 0.0)],
        "shunt",
        reference,
    )
    .expect("shunt admittance should be constructed");
    assert_eq!(shunt.name.as_deref(), Some("shunt"));
    assert_complex_close(shunt.s[(0, 0, 0)], Complex64::new(-1.0 / 3.0, 0.0));
    assert_complex_close(shunt.s[(0, 1, 0)], Complex64::new(2.0 / 3.0, 0.0));

    let ground = Circuit::ground(frequency.clone(), "ground", reference)
        .expect("ground should be constructed");
    let open = Circuit::open(frequency, "open", reference).expect("open should be constructed");
    assert_complex_close(ground.s[(0, 0, 0)], Complex64::new(-1.0, 0.0));
    assert_complex_close(open.s[(0, 0, 0)], Complex64::new(1.0, 0.0));
}

/// Compares a synthesized Wilkinson divider with theory and a Designer fixture.
#[test]
fn matches_theoretical_and_designer_wilkinson_divider() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/designer_wilkinson_splitter.s3p");
    let reference = Network::read_touchstone(fixture).expect("Designer fixture should load");
    let frequency = reference.frequency.clone();
    let points = frequency.points();
    let branch_impedance = 2.0_f64.sqrt() * 50.0;
    let branch_media = DefinedGammaZ0::new(
        frequency.clone(),
        Array1::from_elem(points, Complex64::new(0.0, 1.0)),
        Array1::from_elem(points, Complex64::new(branch_impedance, 0.0)),
        None,
    )
    .expect("branch media should be valid");
    let mut branch_one = branch_media
        .line(90.0, LengthUnit::Degree)
        .expect("branch should be constructed");
    branch_one.name = Some("branch1".to_owned());
    let mut branch_two = branch_one.clone();
    branch_two.name = Some("branch2".to_owned());
    let resistor_media = DefinedGammaZ0::new(
        frequency.clone(),
        Array1::from_elem(points, Complex64::new(0.0, 1.0)),
        Array1::from_elem(points, Complex64::new(50.0, 0.0)),
        None,
    )
    .expect("resistor media should be valid");
    let mut resistor = resistor_media
        .resistor(&Array1::from_elem(points, 100.0))
        .expect("resistor should be constructed");
    resistor.name = Some("resistor".to_owned());
    let branch_one = Arc::new(branch_one);
    let branch_two = Arc::new(branch_two);
    let resistor = Arc::new(resistor);
    let circuit = Circuit::new(vec![
        vec![
            Circuit::port(frequency.clone(), "port1", Complex64::new(50.0, 0.0))
                .expect("port should construct"),
            CircuitConnection::new(Arc::clone(&branch_one), 0),
            CircuitConnection::new(Arc::clone(&branch_two), 0),
        ],
        vec![
            Circuit::port(frequency.clone(), "port2", Complex64::new(50.0, 0.0))
                .expect("port should construct"),
            CircuitConnection::new(branch_one, 1),
            CircuitConnection::new(Arc::clone(&resistor), 0),
        ],
        vec![
            Circuit::port(frequency, "port3", Complex64::new(50.0, 0.0))
                .expect("port should construct"),
            CircuitConnection::new(branch_two, 1),
            CircuitConnection::new(resistor, 1),
        ],
    ])
    .expect("Wilkinson circuit should construct");
    let actual = circuit.network().expect("Wilkinson circuit should solve");
    let expected_transmission = Complex64::new(0.0, -1.0 / 2.0_f64.sqrt());
    assert_complex_close(actual.s[(0, 1, 0)], expected_transmission);
    assert_complex_close(actual.s[(0, 2, 0)], expected_transmission);
    assert!(actual.s[(0, 0, 0)].norm() < 1.0e-12);
    assert!(actual.s[(0, 2, 1)].norm() < 1.0e-12);
    for (actual, expected) in actual.s.iter().zip(reference.s.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-4);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-4);
    }
}

fn through_network(frequency: Frequency) -> rust_rf::Result<Network> {
    let points = frequency.points();
    let mut s = Array3::zeros((points, 2, 2));
    for point in 0..points {
        s[(point, 1, 0)] = Complex64::new(1.0, 0.0);
        s[(point, 0, 1)] = Complex64::new(1.0, 0.0);
    }
    let z0 = Array2::from_elem((points, 2), Complex64::new(50.0, 0.0));
    let mut network = Network::new(frequency, s, z0)?;
    network.name = Some("through".to_owned());
    Ok(network)
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
    assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
}
