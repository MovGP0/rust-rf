//! Arbitrary-topology RF circuits assembled from N-port networks.
//!
//! A [`Circuit`] connects any number of N-port [`Network`] objects at shared
//! intersections. One or more explicitly marked external ports define the
//! M-port network returned by [`Circuit::network`]. The module also exposes
//! graph, active-parameter, wave, voltage, current, and reduction operations.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;

use crate::math::left_solve;
use crate::network::{abcd_to_s, active_s, active_vswr, active_y, active_z};
use crate::{Error, Frequency, Network, Result};

/// A circuit assembled from N-port networks with arbitrary topology.
///
/// The resultant M-port network has one port for every
/// [`CircuitConnection::external`] entry, in order of appearance.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
///
/// use ndarray::array;
/// use rust_rf::{Complex64, Frequency};
/// use rust_rf::circuit::{Circuit, CircuitConnection};
///
/// # fn main() -> rust_rf::Result<()> {
/// let frequency = Frequency::from_hz(array![1.0e9])?;
/// let port1 = Circuit::port(frequency.clone(), "Port 1", Complex64::new(50.0, 0.0))?;
/// let port2 = Circuit::port(frequency.clone(), "Port 2", Complex64::new(50.0, 0.0))?;
/// let series = Arc::new(Circuit::series_impedance(
///     frequency,
///     &array![Complex64::new(10.0, 0.0)],
///     "R1",
///     Complex64::new(50.0, 0.0),
/// )?);
/// let circuit = Circuit::new(vec![
///     vec![port1, CircuitConnection::new(Arc::clone(&series), 0)],
///     vec![CircuitConnection::new(series, 1), port2],
/// ])?;
/// let network = circuit.network()?;
/// # assert_eq!(network.ports(), 2);
/// # Ok(())
/// # }
/// ```
///
/// # Reference
///
/// P. Hallbjörner, *Microwave and Optical Technology Letters*, vol. 38,
/// p. 99, 2003.
#[derive(Clone, Debug, Default)]
pub struct Circuit {
    /// Circuit intersections and their participating network ports.
    ///
    /// Each inner vector describes one electrical intersection.
    pub connections: Vec<Vec<CircuitConnection>>,
    /// Optional name copied to the network produced by [`Self::network`].
    pub name: Option<String>,
}

/// One network port participating in a circuit intersection.
#[derive(Clone, Debug)]
pub struct CircuitConnection {
    /// Shared network instance containing the connected port.
    pub network: Arc<Network>,
    /// Zero-based port index within [`Self::network`].
    pub port: usize,
    /// Whether this one-port network represents an external circuit port.
    pub external: bool,
}

impl CircuitConnection {
    /// Creates an internal circuit connection.
    #[must_use]
    pub const fn new(network: Arc<Network>, port: usize) -> Self {
        Self {
            network,
            port,
            external: false,
        }
    }

    /// Creates an explicitly marked external circuit connection.
    ///
    /// External connections must use port `0` of a one-port network.
    #[must_use]
    pub const fn external(network: Arc<Network>, port: usize) -> Self {
        Self {
            network,
            port,
            external: true,
        }
    }
}

impl Circuit {
    /// Creates and validates a circuit connection map.
    ///
    /// Every inner vector is one intersection. Each network must be named, all
    /// networks must share a frequency axis, and a `(network, port)` pair may
    /// appear only once. At least one connection must be explicitly external.
    ///
    /// # Errors
    ///
    /// Returns an error if the topology, network names, ports, or frequency axes are invalid.
    pub fn new(connections: Vec<Vec<CircuitConnection>>) -> Result<Self> {
        if connections.is_empty() || connections.iter().any(Vec::is_empty) {
            return Err(Error::IncompatibleShape(
                "a circuit requires at least one non-empty intersection".to_owned(),
            ));
        }
        let first = &connections[0][0].network;
        let mut names = HashMap::<String, usize>::new();
        let mut used_ports = HashSet::new();
        let mut external_ports = 0;
        for connection in connections.iter().flatten() {
            if connection.port >= connection.network.ports() {
                return Err(Error::InvalidPort {
                    port: connection.port,
                    ports: connection.network.ports(),
                });
            }
            if connection.network.frequency != first.frequency {
                return Err(Error::InvalidFrequency(
                    "all circuit networks must share the same frequency axis".to_owned(),
                ));
            }
            let name = connection
                .network
                .name
                .as_ref()
                .filter(|name| !name.is_empty())
                .ok_or_else(|| {
                    Error::Unsupported("all circuit networks must have a name".to_owned())
                })?;
            let identity = Arc::as_ptr(&connection.network) as usize;
            match names.get(name) {
                Some(previous) if *previous != identity => {
                    return Err(Error::Unsupported(format!(
                        "circuit network name {name:?} is not unique"
                    )));
                }
                _ => {
                    names.insert(name.clone(), identity);
                }
            }
            if !used_ports.insert((identity, connection.port)) {
                return Err(Error::Unsupported(format!(
                    "network {name:?} port {} appears more than once",
                    connection.port
                )));
            }
            if connection.external {
                if connection.network.ports() != 1 || connection.port != 0 {
                    return Err(Error::IncompatibleShape(
                        "an external circuit port must be represented by a one-port Network"
                            .to_owned(),
                    ));
                }
                external_ports += 1;
            }
        }
        if external_ports == 0 {
            return Err(Error::IncompatibleShape(
                "a circuit requires at least one explicitly marked external port".to_owned(),
            ));
        }
        Ok(Self {
            connections,
            name: None,
        })
    }

    /// Creates a one-port matched network marked as an external circuit port.
    ///
    /// The port's characteristic impedance is `z0`. External port numbering in
    /// the resulting network follows order of appearance in [`Self::connections`].
    ///
    /// # Errors
    ///
    /// Returns an error if `z0` is invalid or the one-port network cannot be constructed.
    pub fn port(
        frequency: Frequency,
        name: impl Into<String>,
        z0: Complex64,
    ) -> Result<CircuitConnection> {
        if !z0.re.is_finite() || !z0.im.is_finite() || z0.re <= 0.0 {
            return Err(Error::Unsupported(
                "circuit port impedance must be finite with positive real part".to_owned(),
            ));
        }
        let points = frequency.points();
        let mut network = Network::new(
            frequency,
            Array3::zeros((points, 1, 1)),
            Array2::from_elem((points, 1), z0),
        )?;
        network.name = Some(name.into());
        Ok(CircuitConnection::external(Arc::new(network), 0))
    }

    /// Creates a two-port series impedance network.
    ///
    /// ```text
    /// (Port 1)-----[Z]-----(Port 2)
    /// ```
    ///
    /// `impedance` supplies one complex value per frequency point.
    ///
    /// # Errors
    ///
    /// Returns an error if the component data or reference impedance is invalid.
    pub fn series_impedance(
        frequency: Frequency,
        impedance: &Array1<Complex64>,
        name: impl Into<String>,
        z0: Complex64,
    ) -> Result<Network> {
        validate_component_values(&frequency, impedance, z0)?;
        let points = frequency.points();
        let abcd =
            Array3::from_shape_fn((points, 2, 2), |(point, row, column)| match (row, column) {
                (0, 0) | (1, 1) => Complex64::new(1.0, 0.0),
                (0, 1) => impedance[point],
                _ => Complex64::new(0.0, 0.0),
            });
        component_from_abcd(frequency, &abcd, name, z0)
    }

    /// Creates a two-port shunt admittance network.
    ///
    /// ```text
    /// (Port 1)-----(Port 2)
    ///            |
    ///           [Y]
    ///            |
    ///           Gnd
    /// ```
    ///
    /// `admittance` supplies one complex value per frequency point.
    ///
    /// # Errors
    ///
    /// Returns an error if the component data or reference impedance is invalid.
    pub fn shunt_admittance(
        frequency: Frequency,
        admittance: &Array1<Complex64>,
        name: impl Into<String>,
        z0: Complex64,
    ) -> Result<Network> {
        validate_component_values(&frequency, admittance, z0)?;
        let points = frequency.points();
        let abcd =
            Array3::from_shape_fn((points, 2, 2), |(point, row, column)| match (row, column) {
                (0, 0) | (1, 1) => Complex64::new(1.0, 0.0),
                (1, 0) => admittance[point],
                _ => Complex64::new(0.0, 0.0),
            });
        component_from_abcd(frequency, &abcd, name, z0)
    }

    /// Creates a grounded one-port network with reflection coefficient $-1$.
    ///
    /// ```text
    /// (Port)-----Gnd
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the reference impedance or resulting network is invalid.
    pub fn ground(frequency: Frequency, name: impl Into<String>, z0: Complex64) -> Result<Network> {
        one_port_termination(frequency, name, z0, Complex64::new(-1.0, 0.0))
    }

    /// Creates an open one-port network with reflection coefficient $+1$.
    ///
    /// ```text
    /// (Port)-----Open
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the reference impedance or resulting network is invalid.
    pub fn open(frequency: Frequency, name: impl Into<String>, z0: Complex64) -> Result<Network> {
        one_port_termination(frequency, name, z0, Complex64::new(1.0, 0.0))
    }

    /// Generates an undirected graph of networks and intersections.
    ///
    /// Network names and synthetic intersection names (`X0`, `X1`, …) are
    /// nodes. Edge weights are zero-based network port indices.
    #[must_use]
    pub fn graph(&self) -> UnGraph<String, usize> {
        let mut graph = UnGraph::new_undirected();
        let mut network_nodes = HashMap::<usize, NodeIndex>::new();
        for connection in self.connections.iter().flatten() {
            let identity = Arc::as_ptr(&connection.network) as usize;
            network_nodes.entry(identity).or_insert_with(|| {
                graph.add_node(
                    connection
                        .network
                        .name
                        .clone()
                        .unwrap_or_else(|| "unnamed".to_owned()),
                )
            });
        }
        for (intersection, connections) in self.connections.iter().enumerate() {
            let intersection_node = graph.add_node(format!("X{intersection}"));
            for connection in connections {
                let network_node = network_nodes[&(Arc::as_ptr(&connection.network) as usize)];
                graph.add_edge(intersection_node, network_node, connection.port);
            }
        }
        graph
    }

    /// Returns the frequency axis shared by all circuit networks.
    #[must_use]
    pub fn frequency(&self) -> &Frequency {
        &self.connections[0][0].network.frequency
    }

    /// Returns the number of flattened `(network, port)` connections.
    #[must_use]
    pub fn connection_count(&self) -> usize {
        self.connections.iter().map(Vec::len).sum()
    }

    /// Returns the number of circuit intersections.
    #[must_use]
    pub fn intersection_count(&self) -> usize {
        self.connections.len()
    }

    /// Returns the dimension of the global circuit matrices.
    ///
    /// This is the sum of the number of ports participating in all
    /// intersections.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.connection_count()
    }

    /// Returns unique networks in connection order.
    #[must_use]
    pub fn networks(&self) -> Vec<Arc<Network>> {
        let mut seen = HashSet::new();
        self.connections
            .iter()
            .flatten()
            .filter_map(|connection| {
                let identity = Arc::as_ptr(&connection.network) as usize;
                seen.insert(identity)
                    .then(|| Arc::clone(&connection.network))
            })
            .collect()
    }

    /// Returns the circuit networks keyed by their unique names.
    #[must_use]
    pub fn networks_by_name(&self) -> HashMap<String, Arc<Network>> {
        self.networks()
            .into_iter()
            .filter_map(|network| network.name.clone().map(|name| (name, network)))
            .collect()
    }

    /// Returns the number of unique connected networks, including port networks.
    #[must_use]
    pub fn network_count(&self) -> usize {
        self.networks().len()
    }

    /// Returns global-matrix indexes of external circuit ports.
    #[must_use]
    pub fn port_indexes(&self) -> Vec<usize> {
        self.external_indexes()
    }

    /// Returns external-port characteristic impedances.
    ///
    /// The array shape is `(frequency points, external ports)`.
    #[must_use]
    pub fn port_z0(&self) -> Array2<Complex64> {
        let external = self.external_connections();
        Array2::from_shape_fn(
            (self.frequency().points(), external.len()),
            |(point, port)| external[port].network.z0[(point, external[port].port)],
        )
    }

    /// Returns `true` if every pair of vertices in the circuit graph is connected.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        petgraph::algo::connected_components(&self.graph()) == 1
    }

    /// Returns every intersection with its network names and port indices.
    #[must_use]
    pub fn intersections_by_name(&self) -> HashMap<usize, Vec<(String, usize)>> {
        self.connections
            .iter()
            .enumerate()
            .map(|(intersection, connections)| {
                (
                    intersection,
                    connections
                        .iter()
                        .filter_map(|connection| {
                            connection
                                .network
                                .name
                                .clone()
                                .map(|name| (name, connection.port))
                        })
                        .collect(),
                )
            })
            .collect()
    }

    /// Returns graph edges as `(node, node, network_port)` tuples.
    #[must_use]
    pub fn edges(&self) -> Vec<(String, String, usize)> {
        let graph = self.graph();
        graph
            .edge_references()
            .map(|edge| {
                (
                    graph[edge.source()].clone(),
                    graph[edge.target()].clone(),
                    *edge.weight(),
                )
            })
            .collect()
    }

    /// Returns a newly validated circuit with networks replaced by name.
    ///
    /// Names absent from `replacements` retain their existing network. The
    /// original circuit is not modified.
    ///
    /// # Errors
    ///
    /// Returns an error if a replacement makes the circuit topology invalid.
    pub fn updated_networks(&self, replacements: &HashMap<String, Arc<Network>>) -> Result<Self> {
        let connections = self
            .connections
            .iter()
            .map(|intersection| {
                intersection
                    .iter()
                    .map(|connection| {
                        let name = connection.network.name.as_ref().ok_or_else(|| {
                            Error::Unsupported(
                                "circuit network is missing its required name".to_owned(),
                            )
                        })?;
                        Ok(CircuitConnection {
                            network: replacements
                                .get(name)
                                .cloned()
                                .unwrap_or_else(|| Arc::clone(&connection.network)),
                            port: connection.port,
                            external: connection.external,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut updated = Self::new(connections)?;
        updated.name.clone_from(&self.name);
        Ok(updated)
    }

    /// Condenses the solved circuit to one internal N-port connected to the original ports.
    ///
    /// The equivalent circuit produces the same external S-parameters with
    /// fewer components, which makes repeated network calculations faster.
    /// Internal voltage and current distributions no longer represent the
    /// original topology.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit cannot be solved or the reduced topology is invalid.
    pub fn reduced(&self) -> Result<Self> {
        let external = self.external_connections();
        let mut assembled = self.network()?;
        let existing_names = self.networks_by_name().into_keys().collect::<HashSet<_>>();
        let mut reduced_name = "__reduced_circuit__".to_owned();
        while existing_names.contains(&reduced_name) {
            reduced_name.push('_');
        }
        assembled.name = Some(reduced_name);
        let assembled = Arc::new(assembled);
        let connections = external
            .into_iter()
            .enumerate()
            .map(|(port, external)| {
                vec![
                    external.clone(),
                    CircuitConnection::new(Arc::clone(&assembled), port),
                ]
            })
            .collect();
        let mut reduced = Self::new(connections)?;
        reduced.name.clone_from(&self.name);
        Ok(reduced)
    }

    /// Returns the M-port [`Network`] associated with the external ports.
    ///
    /// M is the number of explicitly external connections. Port names and
    /// reference impedances are inherited from their external port networks.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit cannot be solved or the network cannot be constructed.
    pub fn network(&self) -> Result<Network> {
        let scattering = self.external_s()?;
        let external = self.external_connections();
        let frequency = external[0].network.frequency.clone();
        let mut z0 = Array2::zeros((frequency.points(), external.len()));
        for (output, connection) in external.iter().enumerate() {
            for point in 0..frequency.points() {
                z0[(point, output)] = connection.network.z0[(point, connection.port)];
            }
        }
        let mut network = Network::new(frequency, scattering, z0)?;
        network.name.clone_from(&self.name);
        network.port_names = external
            .iter()
            .map(|connection| {
                connection.network.name.clone().ok_or_else(|| {
                    Error::Unsupported("circuit network is missing its required name".to_owned())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(network)
    }

    /// Returns scattering parameters for the external circuit ports.
    ///
    /// The array shape is `(frequency points, external ports, external ports)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the global circuit system cannot be solved.
    pub fn external_s(&self) -> Result<Array3<Complex64>> {
        let global_scattering = self.global_s()?;
        let flattened = self.flattened_connections();
        let points = flattened[0].network.frequency_points();
        let external_indexes = self.external_indexes();
        let mut external = Array3::zeros((points, external_indexes.len(), external_indexes.len()));
        for point in 0..points {
            for (row, global_row) in external_indexes.iter().enumerate() {
                for (column, global_column) in external_indexes.iter().enumerate() {
                    external[(point, row, column)] =
                        global_scattering[(point, *global_row, *global_column)];
                }
            }
        }
        Ok(external)
    }

    /// Returns the global scattering matrix for internal and external connections.
    ///
    /// If $X$ is the block-diagonal intersection matrix and $C$ is the global
    /// component scattering matrix, the circuit scattering matrix is
    ///
    /// $$S = \left(X^{-1} - C\right)^{-1}.$$
    ///
    /// The result is used by the wave, voltage, and current observables. Its
    /// shape is `(frequency points, dimension, dimension)`.
    ///
    /// # Errors
    ///
    /// Returns an error if an intersection is singular or the global system cannot be solved.
    pub fn global_s(&self) -> Result<Array3<Complex64>> {
        let flattened = self.flattened_connections();
        let dimension = flattened.len();
        let points = flattened[0].network.frequency_points();
        let mut intersection = Array3::zeros((points, dimension, dimension));
        let mut offset = 0;
        for connections in &self.connections {
            for point in 0..points {
                let admittances = connections
                    .iter()
                    .map(|connection| {
                        Complex64::new(1.0, 0.0) / connection.network.z0[(point, connection.port)]
                    })
                    .collect::<Vec<_>>();
                let total: Complex64 = admittances.iter().copied().sum();
                if total.norm_sqr() <= f64::EPSILON {
                    return Err(Error::Unsupported(
                        "a circuit intersection has zero total reference admittance".to_owned(),
                    ));
                }
                for row in 0..connections.len() {
                    for column in 0..connections.len() {
                        let mut value =
                            2.0 * (admittances[row] * admittances[column]).sqrt() / total;
                        if row == column {
                            value -= Complex64::new(1.0, 0.0);
                        }
                        intersection[(point, offset + row, offset + column)] = value;
                    }
                }
            }
            offset += connections.len();
        }

        let identity = identity_matrices(points, dimension);
        let inverse_intersection = left_solve(&intersection, &identity)?;
        let mut component = Array3::zeros((points, dimension, dimension));
        let mut network_ports = HashMap::<usize, Vec<(usize, usize, Arc<Network>)>>::new();
        for (global, connection) in flattened.iter().enumerate() {
            if !connection.external {
                network_ports
                    .entry(Arc::as_ptr(&connection.network) as usize)
                    .or_default()
                    .push((connection.port, global, Arc::clone(&connection.network)));
            }
        }
        for ports in network_ports.values() {
            let network = &ports[0].2;
            for point in 0..points {
                for (source_port, source_global, _) in ports {
                    for (destination_port, destination_global, _) in ports {
                        component[(point, *source_global, *destination_global)] =
                            network.s[(point, *source_port, *destination_port)];
                    }
                }
            }
        }
        let system = &inverse_intersection - &component;
        left_solve(&system, &identity)
    }

    /// Returns active S-parameters for the external-port excitation vector.
    ///
    /// The active reflection coefficient at port $m$ is
    ///
    /// $$\operatorname{active}(S)_m = \sum_{i} S_{m i}\frac{a_{i}}{a_{m}}.$$
    ///
    /// Active parameters are important for active phased-array antennas.
    ///
    /// # References
    ///
    /// - D. M. Pozar, *IEEE Transactions on Antennas and Propagation*, vol. 42,
    ///   p. 1176, 1994.
    /// - D. Williams, *IEEE Microwave Magazine*, vol. 14, p. 38, 2013.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit network or excitation vector is invalid.
    pub fn active_s(&self, excitation: &Array1<Complex64>) -> Result<Array2<Complex64>> {
        let network = self.network()?;
        active_s(&network.s, excitation)
    }

    /// Returns active impedances for the external-port excitation vector.
    ///
    /// $$\operatorname{active}(Z)_m = Z_{0,m}
    /// \frac{1 + \operatorname{active}(S)_m}
    /// {1 - \operatorname{active}(S)_m}.$$
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit network or excitation vector is invalid.
    pub fn active_z(&self, excitation: &Array1<Complex64>) -> Result<Array2<Complex64>> {
        let network = self.network()?;
        active_z(&network.s, &network.z0, excitation)
    }

    /// Returns active admittances for the external-port excitation vector.
    ///
    /// $$\operatorname{active}(Y)_m = Y_{0,m}
    /// \frac{1 - \operatorname{active}(S)_m}
    /// {1 + \operatorname{active}(S)_m}.$$
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit network or excitation vector is invalid.
    pub fn active_y(&self, excitation: &Array1<Complex64>) -> Result<Array2<Complex64>> {
        let network = self.network()?;
        active_y(&network.s, &network.z0, excitation)
    }

    /// Returns active voltage standing-wave ratios for the excitation vector.
    ///
    /// $$\operatorname{active}(\mathrm{VSWR})_m =
    /// \frac{1 + |\operatorname{active}(S)_m|}
    /// {1 - |\operatorname{active}(S)_m|}.$$
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit network or excitation vector is invalid.
    pub fn active_vswr(&self, excitation: &Array1<Complex64>) -> Result<Array2<f64>> {
        let network = self.network()?;
        active_vswr(&network.s, excitation)
    }

    /// Builds the global incident-wave vector from external powers and phases.
    ///
    /// Each external-port wave is
    ///
    /// $$a = \sqrt{2P_{\mathrm{in}}}\,e^{j\phi},$$
    ///
    /// where the factor of two produces peak values. Internal entries are zero.
    ///
    /// # Errors
    ///
    /// Returns an error if the power and phase vectors do not match the external ports or contain invalid values.
    pub fn incident_waves(
        &self,
        power_watts: &Array1<f64>,
        phase_radians: &Array1<f64>,
    ) -> Result<Array1<Complex64>> {
        let external_indexes = self.external_indexes();
        if power_watts.len() != external_indexes.len()
            || phase_radians.len() != external_indexes.len()
        {
            return Err(Error::IncompatibleShape(format!(
                "{} circuit ports received {} powers and {} phases",
                external_indexes.len(),
                power_watts.len(),
                phase_radians.len()
            )));
        }
        if power_watts
            .iter()
            .any(|power| !power.is_finite() || *power < 0.0)
            || phase_radians.iter().any(|phase| !phase.is_finite())
        {
            return Err(Error::Unsupported(
                "circuit excitation requires finite non-negative powers and finite phases"
                    .to_owned(),
            ));
        }
        let mut incident = Array1::zeros(self.flattened_connections().len());
        for (port, global) in external_indexes.iter().enumerate() {
            incident[*global] =
                Complex64::from_polar((2.0 * power_watts[port]).sqrt(), phase_radians[port]);
        }
        Ok(incident)
    }

    /// Calculates outgoing waves at every flattened circuit connection.
    ///
    /// `incident` must contain one entry per global circuit dimension. The
    /// result has shape `(frequency points, dimension)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit cannot be solved or `incident` has the wrong length.
    pub fn outgoing_waves(&self, incident: &Array1<Complex64>) -> Result<Array2<Complex64>> {
        let scattering = self.global_s()?;
        if incident.len() != scattering.dim().1 {
            return Err(Error::IncompatibleShape(format!(
                "global scattering dimension {} received {} incident waves",
                scattering.dim().1,
                incident.len()
            )));
        }
        Ok(Array2::from_shape_fn(
            (scattering.dim().0, scattering.dim().1),
            |(point, output)| {
                (0..scattering.dim().2)
                    .map(|input| scattering[(point, output, input)] * incident[input])
                    .sum()
            },
        ))
    }

    /// Returns peak voltages and currents at the external ports.
    ///
    /// Power is specified in watts and phase in radians. External current is
    /// positive when entering the circuit port. Both returned arrays have shape
    /// `(frequency points, external ports)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the excitation is invalid or the circuit cannot be solved.
    pub fn external_voltages_currents(
        &self,
        power_watts: &Array1<f64>,
        phase_radians: &Array1<f64>,
    ) -> Result<(Array2<Complex64>, Array2<Complex64>)> {
        let incident = self.incident_waves(power_watts, phase_radians)?;
        let outgoing = self.outgoing_waves(&incident)?;
        let flattened = self.flattened_connections();
        let external_indexes = self.external_indexes();
        let points = outgoing.dim().0;
        let mut voltages = Array2::zeros((points, external_indexes.len()));
        let mut currents = Array2::zeros((points, external_indexes.len()));
        for point in 0..points {
            for (port, global) in external_indexes.iter().enumerate() {
                let z0 = flattened[*global].network.z0[(point, flattened[*global].port)];
                let root_z0 = z0.sqrt();
                voltages[(point, port)] =
                    (incident[*global] + outgoing[(point, *global)]) * root_z0;
                currents[(point, port)] =
                    (incident[*global] - outgoing[(point, *global)]) / root_z0;
            }
        }
        Ok((voltages, currents))
    }

    /// Returns peak voltages and currents at every flattened circuit connection.
    ///
    /// Power is specified in watts and phase in radians. Voltage is common to
    /// every port at an intersection; current is positive when entering that
    /// intersection. Both arrays have shape `(frequency points, dimension)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the excitation is invalid or the circuit cannot be solved.
    pub fn internal_voltages_currents(
        &self,
        power_watts: &Array1<f64>,
        phase_radians: &Array1<f64>,
    ) -> Result<(Array2<Complex64>, Array2<Complex64>)> {
        let incident = self.incident_waves(power_watts, phase_radians)?;
        let outgoing = self.outgoing_waves(&incident)?;
        let flattened = self.flattened_connections();
        let points = outgoing.dim().0;
        let dimension = flattened.len();
        let mut voltages = Array2::zeros((points, dimension));
        let mut currents = Array2::zeros((points, dimension));
        let mut offset = 0;
        for connections in &self.connections {
            for point in 0..points {
                let references = connections
                    .iter()
                    .map(|connection| connection.network.z0[(point, connection.port)])
                    .collect::<Vec<_>>();
                let total_admittance = references
                    .iter()
                    .map(|reference| Complex64::new(1.0, 0.0) / reference)
                    .sum::<Complex64>();
                let mut output_currents = vec![Complex64::new(0.0, 0.0); connections.len()];
                let mut node_voltage = Complex64::new(0.0, 0.0);
                for (port, input_reference) in references.iter().copied().enumerate() {
                    let transmission = if connections.len() == 1 {
                        Complex64::new(2.0, 0.0)
                    } else {
                        let output_reference = Complex64::new(1.0, 0.0)
                            / (total_admittance - Complex64::new(1.0, 0.0) / input_reference);
                        2.0 * output_reference / (output_reference + input_reference)
                    };
                    output_currents[port] =
                        outgoing[(point, offset + port)] / input_reference.sqrt() * transmission;
                    node_voltage +=
                        outgoing[(point, offset + port)] * input_reference.sqrt() * transmission;
                }
                for port in 0..connections.len() {
                    voltages[(point, offset + port)] = node_voltage;
                    if connections.len() == 1 {
                        continue;
                    }
                    let input_reference = references[port];
                    let output_reference = Complex64::new(1.0, 0.0)
                        / (total_admittance - Complex64::new(1.0, 0.0) / input_reference);
                    for (other, other_reference) in references.iter().copied().enumerate() {
                        if port == other {
                            currents[(point, offset + port)] +=
                                output_currents[other] * other_reference / output_reference;
                        } else {
                            currents[(point, offset + port)] -=
                                output_currents[other] * other_reference / input_reference;
                        }
                    }
                }
            }
            offset += connections.len();
        }
        Ok((voltages, currents))
    }

    fn flattened_connections(&self) -> Vec<&CircuitConnection> {
        self.connections.iter().flatten().collect()
    }

    fn external_connections(&self) -> Vec<&CircuitConnection> {
        self.connections
            .iter()
            .flatten()
            .filter(|connection| connection.external)
            .collect()
    }

    fn external_indexes(&self) -> Vec<usize> {
        self.flattened_connections()
            .iter()
            .enumerate()
            .filter_map(|(index, connection)| connection.external.then_some(index))
            .collect()
    }
}

fn validate_component_values(
    frequency: &Frequency,
    values: &Array1<Complex64>,
    z0: Complex64,
) -> Result<()> {
    if values.len() != frequency.points() {
        return Err(Error::IncompatibleShape(format!(
            "component has {} values for {} frequency points",
            values.len(),
            frequency.points()
        )));
    }
    if !z0.re.is_finite() || !z0.im.is_finite() || z0.re <= 0.0 {
        return Err(Error::Unsupported(
            "component reference impedance must be finite with positive real part".to_owned(),
        ));
    }
    Ok(())
}

fn component_from_abcd(
    frequency: Frequency,
    abcd: &Array3<Complex64>,
    name: impl Into<String>,
    z0: Complex64,
) -> Result<Network> {
    let reference = Array2::from_elem((frequency.points(), 2), z0);
    let scattering = abcd_to_s(abcd, &reference)?;
    let mut network = Network::new(frequency, scattering, reference)?;
    network.name = Some(name.into());
    Ok(network)
}

fn one_port_termination(
    frequency: Frequency,
    name: impl Into<String>,
    z0: Complex64,
    reflection: Complex64,
) -> Result<Network> {
    if !z0.re.is_finite() || !z0.im.is_finite() || z0.re <= 0.0 {
        return Err(Error::Unsupported(
            "termination reference impedance must be finite with positive real part".to_owned(),
        ));
    }
    let points = frequency.points();
    let mut network = Network::new(
        frequency,
        Array3::from_elem((points, 1, 1), reflection),
        Array2::from_elem((points, 1), z0),
    )?;
    network.name = Some(name.into());
    Ok(network)
}

fn identity_matrices(points: usize, dimension: usize) -> Array3<Complex64> {
    let mut identity = Array3::zeros((points, dimension, dimension));
    for point in 0..points {
        for diagonal in 0..dimension {
            identity[(point, diagonal, diagonal)] = Complex64::new(1.0, 0.0);
        }
    }
    identity
}
