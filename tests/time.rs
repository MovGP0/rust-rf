use approx::assert_relative_eq;
use ndarray::{Array2, Array3, array};
use num_complex::Complex64;
use rust_rf::time::{
    GateMethod, GateMode, TimeGateOptions, Window, detect_span, find_n_peaks, irfft, peak_indexes,
    time_gate, time_gate_with_options, window_samples,
};
use rust_rf::{Frequency, FrequencyUnit, Network};

#[test]
fn detects_peaks_and_enforces_minimum_distance() {
    let signal = [0.0, 1.0, 0.0, 0.8, 0.0, 0.9, 0.0];
    assert_eq!(
        peak_indexes(&signal, 0.3, 1).expect("peaks should be detected"),
        vec![1, 3, 5]
    );
    assert_eq!(
        peak_indexes(&signal, 0.3, 3).expect("peaks should be detected"),
        vec![1, 5]
    );
    assert_eq!(
        find_n_peaks(&signal, 2, 0.9, 1).expect("two peaks should be found"),
        vec![1, 5]
    );
    assert!(
        peak_indexes(&[1.0, 1.0, 1.0], 0.3, 1)
            .expect("flat signals are valid")
            .is_empty()
    );
}

#[test]
fn generates_periodic_windows() {
    let hann = window_samples(&Window::Hann, 4).expect("window should be generated");
    assert_relative_eq!(hann[0], 0.0, epsilon = 1.0e-12);
    assert_relative_eq!(hann[1], 0.5, epsilon = 1.0e-12);
    assert_relative_eq!(hann[2], 1.0, epsilon = 1.0e-12);
    assert_relative_eq!(hann[3], 0.5, epsilon = 1.0e-12);
    let kaiser =
        window_samples(&Window::Kaiser(6.0), 4).expect("Kaiser window should be generated");
    assert_relative_eq!(kaiser[2], 1.0, epsilon = 1.0e-12);
}

#[test]
fn performs_numpy_normalized_inverse_real_fft() {
    let spectrum = array![
        Complex64::new(6.0, 0.0),
        Complex64::new(-2.0, 2.0),
        Complex64::new(-2.0, 0.0),
    ];
    let time = irfft(&spectrum, 4).expect("inverse transform should succeed");
    for (actual, expected) in time.iter().zip([0.0, 1.0, 2.0, 3.0]) {
        assert_relative_eq!(*actual, expected, epsilon = 1.0e-12);
    }
    assert!(irfft(&spectrum, 6).is_err());
}

#[test]
fn detects_impulse_span_and_applies_time_gate() {
    const POINTS: usize = 9;
    let frequency = Frequency::from_values(
        ndarray::Array1::linspace(1.0, 9.0, POINTS),
        FrequencyUnit::GHz,
    )
    .expect("frequency should be valid");
    let two_impulses = Array3::from_shape_fn((POINTS, 1, 1), |(bin, _, _)| {
        let angle_one = -std::f64::consts::TAU
            * f64::from(u32::try_from(bin).expect("FFT bin should fit in u32"))
            / f64::from(u32::try_from(POINTS).expect("FFT point count should fit in u32"));
        let angle_three = -3.0
            * std::f64::consts::TAU
            * f64::from(u32::try_from(bin).expect("FFT bin should fit in u32"))
            / f64::from(u32::try_from(POINTS).expect("FFT point count should fit in u32"));
        Complex64::from_polar(1.0, angle_one) + Complex64::from_polar(0.5, angle_three)
    });
    let z0 = Array2::from_elem((POINTS, 1), Complex64::new(50.0, 0.0));
    let network =
        Network::new(frequency.clone(), two_impulses, z0.clone()).expect("network should be valid");
    assert_relative_eq!(
        detect_span(&network).expect("span should be detected"),
        2.0 / 9.0e9,
        epsilon = 1.0e-18
    );

    let constant = Network::new(
        frequency,
        Array3::from_elem((POINTS, 1, 1), Complex64::new(1.0, 0.0)),
        z0,
    )
    .expect("network should be valid");
    let passed = time_gate(
        &constant,
        None,
        None,
        Some(0.0),
        Some(0.0),
        Window::Rectangular,
    )
    .expect("centered impulse should pass");
    for value in &passed.s {
        assert_relative_eq!(value.re, 1.0, epsilon = 1.0e-12);
        assert_relative_eq!(value.im, 0.0, epsilon = 1.0e-12);
    }
    let rejected = time_gate(
        &constant,
        Some(3.0 / 9.0e9),
        Some(3.0 / 9.0e9),
        None,
        None,
        Window::Rectangular,
    )
    .expect("off-center gate should be applied");
    assert!(rejected.s.iter().all(|value| value.norm() < 1.0e-12));
}

#[test]
fn supports_all_gate_methods_and_bandstop_mode() {
    const POINTS: usize = 9;
    let frequency = Frequency::from_values(
        ndarray::Array1::linspace(0.0, 8.0, POINTS),
        FrequencyUnit::GHz,
    )
    .expect("frequency should be valid");
    let network = Network::new(
        frequency,
        Array3::from_elem((POINTS, 1, 1), Complex64::new(1.0, 0.0)),
        Array2::from_elem((POINTS, 1), Complex64::new(50.0, 0.0)),
    )
    .expect("network should be valid");

    for method in [
        GateMethod::Convolution,
        GateMethod::Fft,
        GateMethod::RealFft,
    ] {
        let passed = time_gate_with_options(
            &network,
            &TimeGateOptions {
                center: Some(0.0),
                span: Some(0.0),
                gate_window: Window::Rectangular,
                method,
                fft_window: None,
                ..TimeGateOptions::default()
            },
        )
        .expect("gate method should succeed");
        for value in &passed.s {
            assert_relative_eq!(value.re, 1.0, epsilon = 1.0e-12);
            assert_relative_eq!(value.im, 0.0, epsilon = 1.0e-12);
        }
    }

    let stopped = time_gate_with_options(
        &network,
        &TimeGateOptions {
            center: Some(0.0),
            span: Some(0.0),
            mode: GateMode::BandStop,
            gate_window: Window::Rectangular,
            fft_window: None,
            ..TimeGateOptions::default()
        },
    )
    .expect("band-stop gate should succeed");
    assert!(stopped.s.iter().all(|value| value.norm() < 1.0e-12));
}
