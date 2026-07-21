use approx::assert_relative_eq;
use ndarray::array;
use num_complex::Complex64;
use rust_rf::constants::FREE_SPACE_PERMEABILITY;
use rust_rf::transmission_line::{
    distance_from_electrical_length, distributed_circuit_from_propagation_and_impedance,
    electrical_length, impedance_from_reflection, input_impedance_at_electrical_length,
    load_reflection_at_electrical_length, propagate_voltage_current,
    propagation_and_impedance_from_distributed_circuit, propagation_constant_from_reflections,
    reflection_at_electrical_length, reflection_coefficient,
    reflection_to_impedance_at_electrical_length, skin_depth, standing_wave_ratio,
    standing_wave_ratio_from_impedance, surface_resistivity, total_loss,
};

const TOLERANCE: f64 = 1.0e-10;

#[test]
fn converts_between_impedance_and_reflection_coefficient() {
    let z0 = Complex64::new(100.0, 0.0);
    let input_impedance = Complex64::new(40.0, -280.0);

    let reflection = reflection_coefficient(z0, input_impedance);
    let expected = (input_impedance - z0) / (input_impedance + z0);
    assert_complex_close(reflection, expected, TOLERANCE);
    assert_complex_close(
        impedance_from_reflection(z0, reflection),
        input_impedance,
        TOLERANCE,
    );
}

#[test]
fn calculates_propagation_constant_from_reflections() {
    let z0 = Complex64::new(100.0, 0.0);
    let input_impedance = Complex64::new(40.0, -280.0);
    let input_reflection = reflection_coefficient(z0, input_impedance);

    let propagation =
        propagation_constant_from_reflections(input_reflection, Complex64::new(-1.0, 0.0), 1.5)
            .expect("propagation constant should be defined");

    assert_relative_eq!(propagation.re, 0.02971, epsilon = 1.5e-4);
    assert_relative_eq!(propagation.im, 1.272, epsilon = 1.5e-4);
}

#[test]
fn converts_between_physical_and_electrical_length() {
    let propagation = Complex64::new(0.2, 5.0);
    let distance = 1.5;

    let radians = electrical_length(propagation, distance, false)
        .expect("electrical length should be defined");
    assert_complex_close(radians, propagation * distance, TOLERANCE);
    assert_relative_eq!(
        distance_from_electrical_length(radians, propagation, false)
            .expect("physical distance should be defined"),
        distance,
        epsilon = TOLERANCE
    );

    let degrees = electrical_length(propagation, distance, true)
        .expect("degree electrical length should be defined");
    assert_relative_eq!(
        distance_from_electrical_length(degrees, propagation, true)
            .expect("distance from degrees should be defined"),
        distance,
        epsilon = TOLERANCE
    );
}

#[test]
fn propagates_voltage_and_current_over_special_lengths() {
    let voltage = Complex64::new(3.0, 0.0);
    let current = Complex64::new(2.0, 0.0);
    let impedance = Complex64::new(50.0, 0.0);

    let (same_voltage, same_current) =
        propagate_voltage_current(voltage, current, impedance, Complex64::new(0.0, 0.0))
            .expect("zero-length propagation should be valid");
    assert_complex_close(same_voltage, voltage, TOLERANCE);
    assert_complex_close(same_current, current, TOLERANCE);

    let (wavelength_voltage, wavelength_current) = propagate_voltage_current(
        voltage,
        current,
        impedance,
        Complex64::new(0.0, std::f64::consts::TAU),
    )
    .expect("full-wavelength propagation should be valid");
    assert_complex_close(wavelength_voltage, voltage, TOLERANCE);
    assert_complex_close(wavelength_current, current, TOLERANCE);

    let (half_voltage, half_current) = propagate_voltage_current(
        voltage,
        current,
        impedance,
        Complex64::new(0.0, std::f64::consts::PI),
    )
    .expect("half-wavelength propagation should be valid");
    assert_complex_close(half_voltage, -voltage, TOLERANCE);
    assert_complex_close(half_current, -current, TOLERANCE);
}

#[test]
fn calculates_skin_depth_and_surface_resistivity() {
    let frequencies = array![1.0e6, 10.0e6];
    let copper_resistivity = 1.68e-8;
    let depth = skin_depth(&frequencies, copper_resistivity, 1.0)
        .expect("positive material parameters should be valid");

    let expected_first = (copper_resistivity
        / (std::f64::consts::PI * frequencies[0] * FREE_SPACE_PERMEABILITY))
        .sqrt();
    assert_relative_eq!(depth[0], expected_first, epsilon = TOLERANCE);
    assert_relative_eq!(depth[0] / depth[1], 10.0_f64.sqrt(), epsilon = TOLERANCE);

    let surface = surface_resistivity(&frequencies, copper_resistivity, 1.0)
        .expect("surface resistivity should be defined");
    assert_relative_eq!(
        surface[0],
        copper_resistivity / depth[0],
        epsilon = TOLERANCE
    );
}

#[test]
fn rejects_non_physical_transmission_line_inputs() {
    assert!(skin_depth(&array![0.0], 1.68e-8, 1.0).is_err());
    assert!(skin_depth(&array![1.0e6], -1.0, 1.0).is_err());
    assert!(
        propagation_constant_from_reflections(
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
            1.0,
        )
        .is_err()
    );
}

#[test]
fn converts_distributed_circuit_and_wave_quantities() {
    let admittance = Complex64::new(0.0, 2.0e-3);
    let impedance = Complex64::new(0.5, 3.0);
    let (propagation, characteristic) =
        propagation_and_impedance_from_distributed_circuit(admittance, impedance)
            .expect("distributed circuit should be valid");
    let (recovered_admittance, recovered_impedance) =
        distributed_circuit_from_propagation_and_impedance(propagation, characteristic)
            .expect("wave quantities should be valid");
    assert_complex_close(recovered_admittance, admittance, TOLERANCE);
    assert_complex_close(recovered_impedance, impedance, TOLERANCE);
}

#[test]
fn converts_reflection_and_impedance_along_a_line() {
    let impedance = Complex64::new(50.0, 0.0);
    let load = Complex64::new(75.0, 10.0);
    let theta = Complex64::new(0.1, 0.7);
    let load_reflection = reflection_coefficient(impedance, load);
    let shifted_reflection = reflection_at_electrical_length(load_reflection, theta);

    assert_complex_close(
        load_reflection_at_electrical_length(impedance, load, theta),
        shifted_reflection,
        TOLERANCE,
    );
    assert_complex_close(
        input_impedance_at_electrical_length(impedance, load, theta),
        impedance_from_reflection(impedance, shifted_reflection),
        TOLERANCE,
    );
    assert_complex_close(
        reflection_to_impedance_at_electrical_length(impedance, load_reflection, theta),
        impedance_from_reflection(impedance, shifted_reflection),
        TOLERANCE,
    );
}

#[test]
fn calculates_standing_wave_ratio_and_total_loss() {
    let impedance = Complex64::new(50.0, 0.0);
    let matched_load = Complex64::new(50.0, 0.0);
    assert_relative_eq!(
        standing_wave_ratio(Complex64::new(0.0, 0.0)),
        1.0,
        epsilon = TOLERANCE
    );
    assert_relative_eq!(
        standing_wave_ratio_from_impedance(impedance, matched_load),
        1.0,
        epsilon = TOLERANCE
    );
    assert_relative_eq!(
        total_loss(
            impedance,
            matched_load,
            Complex64::new(0.0, std::f64::consts::PI)
        )
        .expect("matched lossless line should be valid"),
        1.0,
        epsilon = TOLERANCE
    );
}

fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
    assert_relative_eq!(actual.re, expected.re, epsilon = tolerance);
    assert_relative_eq!(actual.im, expected.im, epsilon = tolerance);
}
