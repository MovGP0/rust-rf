//! Regression tests for de-embedding algorithms.
//!
//! The upstream tests use Touchstone fixtures representing a 1 nH device and
//! the following pseudo-netlists. The Rust tests build equivalent networks in
//! memory, avoiding a dependency on external fixture files.
//!
//! ## Open-short fixtures
//!
//! | File | Pseudo-netlist |
//! | --- | --- |
//! | `deemb_ind.s2p` | `P1 (1,0)`; `Cpad1 (1,0) 25 fF`; `Rline1 (1,2) 2 Ω`; `Dut_ind (2,3) 1 nH`; `Rline2 (3,4) 2 Ω`; `Cpad2 (4,0) 25 fF`; `Cp2p (1,4) 10 fF`; `P2 (4,0)` |
//! | `deemb_open.s2p` | `P1 (1,0)`; `Cpad1 (1,0) 25 fF`; `Rline1 (1,2) 2 Ω`; `Rline2 (3,4) 2 Ω`; `Cpad2 (4,0) 25 fF`; `Cp2p (1,4) 10 fF`; `P2 (4,0)` |
//! | `deemb_short.s2p` | `P1 (1,0)`; `Cpad1 (1,0) 25 fF`; `Rline1 (1,0) 2 Ω`; `Rline2 (0,4) 2 Ω`; `Cpad2 (4,0) 25 fF`; `Cp2p (1,4) 10 fF`; `P2 (4,0)` |
//!
//! ## Open fixtures
//!
//! | File | Pseudo-netlist |
//! | --- | --- |
//! | `deemb_ind7.s2p` | `P1 (1,0)`; `Cpad1 (1,0) 25 fF`; `Dut_ind (1,2) 1 nH`; `Cpad2 (2,0) 25 fF`; `Cp2p (1,2) 10 fF`; `P2 (2,0)` |
//! | `deemb_open7.s2p` | `P1 (1,0)`; `Cpad1 (1,0) 25 fF`; `Cpad2 (2,0) 25 fF`; `Cp2p (1,2) 10 fF`; `P2 (2,0)` |
//!
//! ## Short-open fixtures
//!
//! | File | Pseudo-netlist |
//! | --- | --- |
//! | `deemb_ind2.s2p` | `P1 (1,0)`; `Rline1 (1,2) 2 Ω`; `Cpad1 (2,0) 25 fF`; `Dut_ind (2,3) 1 nH`; `Cpad2 (3,0) 25 fF`; `Cp2p (2,3) 10 fF`; `Rline2 (3,4) 2 Ω`; `P2 (4,0)` |
//! | `deemb_open2.s2p` | `P1 (1,0)`; `Rline1 (1,2) 2 Ω`; `Cpad1 (2,0) 25 fF`; `Cpad2 (3,0) 25 fF`; `Cp2p (2,3) 10 fF`; `Rline2 (3,4) 2 Ω`; `P2 (4,0)` |
//! | `deemb_short2.s2p` | `P1 (1,0)`; `Rline1 (1,0) 2 Ω`; `Cpad1 (0,0) 25 fF`; `Cpad2 (0,0) 25 fF`; `Cp2p (0,0) 10 fF`; `Rline2 (0,4) 2 Ω`; `P2 (4,0)` |
//!
//! ## Short fixtures
//!
//! | File | Pseudo-netlist |
//! | --- | --- |
//! | `deemb_ind8.s2p` | `P1 (1,0)`; `Rline1 (1,2) 2 Ω`; `Dut_ind (2,3) 1 nH`; `Rline2 (3,4) 2 Ω`; `P2 (4,0)` |
//! | `deemb_short8.s2p` | `P1 (1,0)`; `Rline1 (1,0) 2 Ω`; `Rline2 (0,4) 2 Ω`; `P2 (4,0)` |
//!
//! ## Split-pi and split-tee fixtures
//!
//! | Method | Device fixture | Thru fixture |
//! | --- | --- | --- |
//! | Split pi | `deemb_ind3.s2p`: pad capacitors at `(1,0)` and `(4,0)`, 2 Ω series resistors, and a 1 nH DUT between nodes 2 and 3 | `deemb_thru3.s2p`: the same fixture without the DUT |
//! | Split tee | `deemb_ind4.s2p`: 2 Ω series resistors, pad capacitors at `(2,0)` and `(3,0)`, and a 1 nH DUT between nodes 2 and 3 | `deemb_thru4.s2p`: the same fixture without the DUT |
//!
//! ## Cancellation fixtures
//!
//! | Method | Device fixture | Thru fixture |
//! | --- | --- | --- |
//! | Admittance cancellation | `deemb_ind5.s2p`: a 1 nH DUT between nodes 1 and 2 with 25 fF shunt capacitors at both ports | `deemb_thru5.s2p`: two 25 fF shunt capacitors on the thru node |
//! | Impedance cancellation | `deemb_ind6.s2p`: a 1 nH DUT between two 2 Ω series resistors | `deemb_thru6.s2p`: the two 2 Ω resistors in series |

use approx::assert_relative_eq;
use ndarray::{Array2, Array3, array};
use num_complex::Complex64;
use rust_rf::calibration::{
    AdmittanceCancel, Deembedding, IeeeP370, IeeeP370FixtureElectricalRequirements,
    IeeeP370FrequencyDomainQuality, IeeeP370MmNzc2xThru, IeeeP370MmZc2xThru, IeeeP370PortOrder,
    IeeeP370SeNzc2xThru, IeeeP370SeZc2xThru, IeeeP370TimeDomainQuality, ImpedanceCancel, Open,
    OpenShort, QualityEvaluation, Short, ShortOpen, SplitPi, SplitTee,
};
use rust_rf::network::{concatenate_ports, s_to_y, s_to_z, y_to_s, z_to_s};
use rust_rf::{Frequency, Network, SParameterDefinition};

const TOLERANCE: f64 = 1.0e-10;
type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Verifies separate open and short correction recovers the expected device value.
///
/// This is the in-memory counterpart of the upstream 1 nH spot-frequency
/// checks for the open-only and short-only fixtures.
#[test]
fn removes_open_and_short_parasitics() -> TestResult<()> {
    let open = one_port_from_y(0.01)?;
    let measured = one_port_from_y(0.03)?;
    let corrected = Open::new(open, Some("open".to_owned()))
        .deembed(&measured)
        .expect("open de-embedding should succeed");
    assert_relative_eq!(admittance(&corrected)?, 0.02, epsilon = TOLERANCE);

    let short = one_port_from_z(5.0)?;
    let measured = one_port_from_z(20.0)?;
    let corrected = Short::new(short, Some("short".to_owned()))
        .deembed(&measured)
        .expect("short de-embedding should succeed");
    assert_relative_eq!(impedance(&corrected)?, 15.0, epsilon = TOLERANCE);
    Ok(())
}

/// Verifies open-short correction removes parallel parasitics before series parasitics.
///
/// The upstream Touchstone regression expects the remaining device to be a
/// pure 1 nH inductor at the selected spot frequency.
#[test]
fn removes_open_then_short_parasitics() -> TestResult<()> {
    let open = one_port_from_y(0.01)?;
    let measured_short = one_port_from_y(0.21)?;
    let measured_dut = one_port_from_y(0.05)?;
    let deembedding = OpenShort::new(open, measured_short, Some("open-short".to_owned()))
        .expect("dummies should be compatible");

    let corrected = deembedding
        .deembed(&measured_dut)
        .expect("open-short de-embedding should succeed");
    assert_relative_eq!(impedance(&corrected)?, 20.0, epsilon = TOLERANCE);
    Ok(())
}

/// Verifies short-open correction removes series parasitics before parallel parasitics.
///
/// The upstream Touchstone regression expects the remaining device to be a
/// pure 1 nH inductor at the selected spot frequency.
#[test]
fn removes_short_then_open_parasitics() -> TestResult<()> {
    let short = one_port_from_z(5.0)?;
    let measured_open = one_port_from_z(105.0)?;
    let measured_dut = one_port_from_z(65.0 / 3.0)?;
    let deembedding = ShortOpen::new(short, measured_open, Some("short-open".to_owned()))
        .expect("dummies should be compatible");

    let corrected = deembedding
        .deembed(&measured_dut)
        .expect("short-open de-embedding should succeed");
    assert_relative_eq!(admittance(&corrected)?, 0.05, epsilon = TOLERANCE);
    Ok(())
}

/// Verifies split-pi and split-tee fixture removal recovers the embedded device.
///
/// These are the in-memory counterparts of the upstream 1 nH spot-frequency
/// checks for the pi- and tee-shaped fixtures.
#[test]
fn splits_pi_and_tee_thru_fixtures() -> TestResult<()> {
    let intrinsic = matched_two_port(Complex64::new(0.65, 0.05))?;

    let pi_thru = two_port_from_y([[0.03, -0.02], [-0.02, 0.03]])?;
    let pi_left = two_port_from_y([[0.05, -0.04], [-0.04, 0.04]])?;
    let pi_right = pi_left.flipped().expect("fixture should flip");
    let pi_measured = pi_left
        .cascade(&intrinsic)
        .and_then(|network| network.cascade(&pi_right))
        .expect("fixture should embed");
    let pi_corrected = SplitPi::new(pi_thru, Some("pi".to_owned()))
        .expect("dummy should be valid")
        .deembed(&pi_measured)
        .expect("split-pi de-embedding should succeed");
    assert_network_close(&pi_corrected, &intrinsic);

    let tee_thru = two_port_from_z([[60.0, 20.0], [20.0, 60.0]])?;
    let tee_left = two_port_from_z([[80.0, 40.0], [40.0, 40.0]])?;
    let tee_right = tee_left.flipped().expect("fixture should flip");
    let tee_measured = tee_left
        .cascade(&intrinsic)
        .and_then(|network| network.cascade(&tee_right))
        .expect("fixture should embed");
    let tee_corrected = SplitTee::new(tee_thru, Some("tee".to_owned()))
        .expect("dummy should be valid")
        .deembed(&tee_measured)
        .expect("split-tee de-embedding should succeed");
    assert_network_close(&tee_corrected, &intrinsic);
    Ok(())
}

/// Verifies admittance and impedance cancellation recover a symmetric device.
///
/// This corresponds to the upstream 1 nH spot-frequency cancellation checks.
#[test]
fn cancels_symmetric_admittance_and_impedance_fixtures() -> TestResult<()> {
    let intrinsic = matched_two_port(Complex64::new(0.6, -0.1))?;
    let thru = matched_two_port(Complex64::new(0.8, 0.05))?;
    let measured = intrinsic.cascade(&thru).expect("fixture should cascade");

    let admittance_corrected = AdmittanceCancel::new(thru.clone(), None)
        .expect("dummy should be valid")
        .deembed(&measured)
        .expect("admittance cancellation should succeed");
    assert_network_close(&admittance_corrected, &intrinsic);

    let impedance_corrected = ImpedanceCancel::new(thru, None)
        .expect("dummy should be valid")
        .deembed(&measured)
        .expect("impedance cancellation should succeed");
    assert_network_close(&impedance_corrected, &intrinsic);
    Ok(())
}

/// Checks the IEEE 370 network, DC, Nyquist-rate-point, shift, and signal helpers.
#[test]
fn provides_ieee_370_network_and_signal_helpers() -> TestResult<()> {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9, 3.0e9]).expect("frequency");
    let mut scattering = Array3::zeros((3, 2, 2));
    for point in 0..3 {
        scattering[(point, 0, 1)] = Complex64::new(0.8, 0.0);
        scattering[(point, 1, 0)] = Complex64::new(0.8, 0.0);
    }
    let network = Network::new(
        frequency,
        scattering,
        Array2::from_elem((3, 2), Complex64::new(50.0, 0.0)),
    )
    .expect("network");
    let thru = IeeeP370::thru(&network).expect("thru");
    assert_eq!(thru.s[(0, 1, 0)], Complex64::new(1.0, 0.0));
    let requirements = IeeeP370FixtureElectricalRequirements::single_ended(&network)
        .expect("fixture electrical requirements");
    assert_relative_eq!(
        requirements.insertion_loss_forward_db[0],
        20.0 * 0.8_f64.log10(),
        epsilon = TOLERANCE
    );
    assert_relative_eq!(
        IeeeP370FixtureElectricalRequirements::FER6_MAXIMUM_DB,
        -15.0,
        epsilon = TOLERANCE
    );
    let with_dc = IeeeP370::add_dc(&network).expect("DC extrapolation");
    assert_relative_eq!(with_dc.frequency.values_hz()[0], 0.0, epsilon = TOLERANCE);
    let step = IeeeP370::make_step(&[1.0, 2.0, -1.0]);
    assert_eq!(step.as_slice().expect("contiguous"), &[1.0, 3.0, 2.0]);
    let zero_reflection = vec![Complex64::new(0.0, 0.0); 3];
    let frequencies = [1.0e9, 2.0e9, 3.0e9];
    assert_relative_eq!(
        IeeeP370::dc_interp(&zero_reflection, &frequencies).expect("DC interpolation"),
        0.0,
        epsilon = TOLERANCE
    );
    assert_relative_eq!(
        IeeeP370::dc(&zero_reflection, &frequencies, 1.0e-12).expect("reflective DC extraction"),
        0.0,
        epsilon = TOLERANCE
    );
    let impedance =
        IeeeP370::getz(&zero_reflection, &frequencies, 50.0).expect("impedance response");
    for value in impedance {
        assert_relative_eq!(value, 50.0, epsilon = TOLERANCE);
    }
    let conjugated = IeeeP370TimeDomainQuality::add_conjugates(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, 2.0),
    ]);
    assert_eq!(
        conjugated.as_slice().expect("contiguous"),
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(3.0, -2.0),
            Complex64::new(2.0, -1.0),
        ]
    );
    assert_eq!(
        IeeeP370TimeDomainQuality::align_signals(&[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0])
            .expect("alignment"),
        -1
    );

    let mut delayed = network.clone();
    for point in 0..delayed.frequency_points() {
        let point_number = f64::from(u32::try_from(point + 1)?);
        delayed.s[(point, 0, 0)] = Complex64::from_polar(0.2, 0.3 * point_number);
        delayed.s[(point, 1, 1)] = Complex64::from_polar(0.15, -0.2 * point_number);
    }
    let (nyquist_aligned, delays) = IeeeP370::nrp(&delayed, None, None).expect("NRP enforcement");
    let (restored, _) =
        IeeeP370::nrp(&nyquist_aligned, Some(&delays), None).expect("NRP restoration");
    assert_network_close(&restored, &delayed);
    let shifted = IeeeP370::shift_n_points(&delayed, 2).expect("sample shift");
    let unshifted = IeeeP370::shift_n_points(&shifted, -2).expect("inverse sample shift");
    assert_network_close(&unshifted, &delayed);

    let (remaining, left_box, right_box) =
        IeeeP370::peel_n_points_lossless(&network, 1, 50.0).expect("lossless peeling");
    let reconstructed = left_box
        .cascade(&remaining)
        .and_then(|network| network.cascade(&right_box))
        .expect("peeled network should reconstruct");
    assert_network_close(&reconstructed, &network);
    Ok(())
}

/// Checks IEEE 370 passivity, reciprocity, and causality metrics in the
/// frequency and time domains.
#[test]
fn calculates_ieee_370_frequency_domain_quality_metrics() {
    let frequency = Frequency::from_hz(array![1.0e9, 2.0e9, 3.0e9]).expect("frequency");
    let mut scattering = Array3::zeros((3, 2, 2));
    for point in 0..3 {
        scattering[(point, 0, 1)] = Complex64::new(0.8, 0.0);
        scattering[(point, 1, 0)] = Complex64::new(0.8, 0.0);
    }
    let network = Network::new(
        frequency,
        scattering,
        Array2::from_elem((3, 2), Complex64::new(50.0, 0.0)),
    )
    .expect("network");

    let quality =
        IeeeP370FrequencyDomainQuality::check_single_ended(&network).expect("quality metrics");

    assert_relative_eq!(quality.causality.value_percent, 100.0, epsilon = TOLERANCE);
    assert_relative_eq!(quality.passivity.value_percent, 100.0, epsilon = TOLERANCE);
    assert_relative_eq!(
        quality.reciprocity.value_percent,
        100.0,
        epsilon = TOLERANCE
    );
    assert_eq!(quality.causality.evaluation, QualityEvaluation::Good);
    assert_eq!(quality.passivity.evaluation, QualityEvaluation::Good);
    assert_eq!(quality.reciprocity.evaluation, QualityEvaluation::Good);

    let quality_options =
        IeeeP370TimeDomainQuality::new(1.0e9, 32, 0.4, 1, 2).expect("time-domain quality options");
    assert_eq!(quality_options.samples_per_unit_interval, 32);
    let mut non_passive = network;
    non_passive.s.mapv_inplace(|value| value * 2.0);
    let passive =
        IeeeP370TimeDomainQuality::create_passive(&non_passive).expect("passivity enforcement");
    assert_relative_eq!(
        IeeeP370FrequencyDomainQuality::check_passivity(&passive).expect("passivity metric"),
        100.0,
        epsilon = 1.0e-8
    );
    let time_quality = quality_options
        .check_single_ended(&non_passive)
        .expect("application-based quality metrics");
    assert!(time_quality.passivity.value_millivolts > 0.0);
    assert_eq!(time_quality.reciprocity.evaluation, QualityEvaluation::Good);
}

/// Verifies an IEEE 370 single-ended NZC 2x-thru splits into fixture models.
///
/// Self-de-embedding the 2x-thru must recover a perfect thru with negligible
/// transmission-magnitude and phase residuals.
#[test]
fn splits_and_self_deembeds_ieee_370_two_x_thru() -> TestResult<()> {
    let fixture = two_port_from_z([[70.0, 20.0], [20.0, 65.0]])?;
    let two_x_thru = fixture
        .cascade(&fixture.flipped().expect("flipped fixture"))
        .expect("2x-thru");
    let deembedding = IeeeP370SeNzc2xThru::new(two_x_thru.clone(), Some("2x-thru".to_owned()))
        .expect("fixture split");

    let residual = deembedding.deembed(&two_x_thru).expect("self de-embedding");
    let thru = IeeeP370::thru(&two_x_thru).expect("ideal thru");

    assert_network_close(&residual, &thru);
    Ok(())
}

/// Verifies IEEE 370 single-ended ZC and time-gated NZC extraction.
///
/// The regression covers the normal harmonic sweep, a DC point, and a
/// nonuniform sweep requiring interpolation. Interpolation should be avoided
/// when possible, but remains useful when a malformed measurement frequency
/// axis must still produce an approximate result.
#[test]
fn peels_and_self_deembeds_ieee_370_impedance_corrected_two_x_thru() {
    let frequency =
        Frequency::from_hz(array![1.0e9, 2.0e9, 3.0e9, 4.0e9, 5.0e9]).expect("frequency");
    let propagation = (1..=5)
        .map(|index| Complex64::new(0.01 * f64::from(index), 0.4 * f64::from(index)))
        .collect::<Vec<_>>();
    let scattering = IeeeP370::make_transmission_line(55.0, 50.0, &propagation, 0.5)
        .expect("fixture transmission line");
    let fixture = Network::new(
        frequency,
        scattering,
        Array2::from_elem((5, 2), Complex64::new(50.0, 0.0)),
    )
    .expect("fixture");
    let two_x_thru = fixture
        .cascade(&fixture.flipped().expect("flipped fixture"))
        .expect("2x-thru");
    let deembedding = IeeeP370SeZc2xThru::new(
        two_x_thru.clone(),
        two_x_thru.clone(),
        Some("ZC 2x-thru".to_owned()),
    )
    .expect("ZC fixture extraction");

    let residual = deembedding
        .deembed(&two_x_thru)
        .expect("ZC self de-embedding");
    let expected = IeeeP370::thru(&two_x_thru).expect("ideal thru");
    for (actual, expected) in residual.s.iter().zip(expected.s.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 2.0e-2);
        assert_relative_eq!(actual.im, expected.im, epsilon = 2.0e-2);
    }

    let time_gated = IeeeP370SeNzc2xThru::new_time_gated(
        two_x_thru.clone(),
        Some(55.0),
        Some("time-gated NZC 2x-thru".to_owned()),
    )
    .expect("time-gated NZC fixture extraction");
    let time_gated_residual = time_gated
        .deembed(&two_x_thru)
        .expect("time-gated NZC self de-embedding");
    for (actual, expected) in time_gated_residual.s.iter().zip(expected.s.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 5.0e-2);
        assert_relative_eq!(actual.im, expected.im, epsilon = 5.0e-2);
    }

    let with_dc = IeeeP370::add_dc(&two_x_thru).expect("DC 2x-thru");
    let dc_time_gated = IeeeP370SeNzc2xThru::new_time_gated(
        with_dc.clone(),
        Some(55.0),
        Some("DC time-gated NZC 2x-thru".to_owned()),
    )
    .expect("DC time-gated NZC extraction");
    let dc_residual = dc_time_gated
        .deembed(&with_dc)
        .expect("DC time-gated NZC self de-embedding");
    let dc_expected = IeeeP370::thru(&with_dc).expect("DC ideal thru");
    for (actual, expected) in dc_residual.s.iter().zip(dc_expected.s.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 7.0e-2);
        assert_relative_eq!(actual.im, expected.im, epsilon = 7.0e-2);
    }

    let nonuniform_frequency =
        Frequency::from_hz(array![1.0e9, 1.8e9, 3.1e9, 4.3e9, 5.0e9]).expect("nonuniform");
    let nonuniform = two_x_thru
        .interpolate(&nonuniform_frequency)
        .expect("nonuniform 2x-thru");
    IeeeP370SeNzc2xThru::new_time_gated(
        nonuniform.clone(),
        Some(55.0),
        Some("nonuniform time-gated NZC 2x-thru".to_owned()),
    )
    .expect("nonuniform time-gated NZC extraction");

    for (variant, label) in [
        (with_dc, "DC ZC 2x-thru"),
        (nonuniform, "nonuniform ZC 2x-thru"),
    ] {
        let deembedding =
            IeeeP370SeZc2xThru::new(variant.clone(), variant.clone(), Some(label.to_owned()))
                .expect("ZC fixture extraction should restore the original frequency axis");
        assert_eq!(deembedding.side1.frequency, variant.frequency);
        assert_eq!(deembedding.side2.frequency, variant.frequency);
        let residual = deembedding.deembed(&variant).expect("ZC self de-embedding");
        let expected = IeeeP370::thru(&variant).expect("ideal thru");
        for (actual, expected) in residual.s.iter().zip(expected.s.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 8.0e-2);
            assert_relative_eq!(actual.im, expected.im, epsilon = 8.0e-2);
        }
    }
}

/// Verifies mixed-mode IEEE 370 ZC and time-gated NZC extraction.
///
/// Self-de-embedding the combined differential/common 2x-thru must recover a
/// perfect mixed-mode thru within the extraction tolerance.
#[test]
fn peels_and_self_deembeds_ieee_370_mixed_mode_impedance_corrected_two_x_thru() {
    let frequency =
        Frequency::from_hz(array![1.0e9, 2.0e9, 3.0e9, 4.0e9, 5.0e9]).expect("frequency");
    let propagation = (1..=5)
        .map(|index| Complex64::new(0.008 * f64::from(index), 0.35 * f64::from(index)))
        .collect::<Vec<_>>();
    let differential = Network::new(
        frequency.clone(),
        IeeeP370::make_transmission_line(110.0, 100.0, &propagation, 0.5)
            .expect("differential fixture"),
        Array2::from_elem((5, 2), Complex64::new(100.0, 0.0)),
    )
    .expect("differential network");
    let common = Network::new(
        frequency,
        IeeeP370::make_transmission_line(27.0, 25.0, &propagation, 0.5).expect("common fixture"),
        Array2::from_elem((5, 2), Complex64::new(25.0, 0.0)),
    )
    .expect("common network");
    let differential_two_x_thru = differential
        .cascade(&differential.flipped().expect("differential flip"))
        .expect("differential 2x-thru");
    let common_two_x_thru = common
        .cascade(&common.flipped().expect("common flip"))
        .expect("common 2x-thru");
    let mixed_two_x_thru =
        concatenate_ports(&[differential_two_x_thru.clone(), common_two_x_thru.clone()])
            .expect("mixed 2x-thru");
    let single_ended_two_x_thru = mixed_two_x_thru
        .mixed_mode_to_single_ended(2)
        .expect("single-ended 2x-thru");
    let deembedding = IeeeP370MmZc2xThru::new(
        single_ended_two_x_thru.clone(),
        single_ended_two_x_thru.clone(),
        IeeeP370PortOrder::Second,
        Some("mixed ZC 2x-thru".to_owned()),
    )
    .expect("mixed ZC fixture extraction");

    let residual = deembedding
        .deembed(&single_ended_two_x_thru)
        .expect("mixed ZC self de-embedding");
    let expected = concatenate_ports(&[
        IeeeP370::thru(&differential_two_x_thru).expect("differential thru"),
        IeeeP370::thru(&common_two_x_thru).expect("common thru"),
    ])
    .expect("mixed ideal thru")
    .mixed_mode_to_single_ended(2)
    .expect("single-ended ideal thru");
    for (actual, expected) in residual.s.iter().zip(expected.s.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 2.0e-2);
        assert_relative_eq!(actual.im, expected.im, epsilon = 2.0e-2);
    }

    let time_gated = IeeeP370MmNzc2xThru::new_time_gated(
        single_ended_two_x_thru.clone(),
        IeeeP370PortOrder::Second,
        Some(110.0),
        Some(27.0),
        Some("mixed time-gated NZC 2x-thru".to_owned()),
    )
    .expect("mixed time-gated NZC fixture extraction");
    let time_gated_residual = time_gated
        .deembed(&single_ended_two_x_thru)
        .expect("mixed time-gated NZC self de-embedding");
    for (actual, expected) in time_gated_residual.s.iter().zip(expected.s.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = 5.0e-2);
        assert_relative_eq!(actual.im, expected.im, epsilon = 5.0e-2);
    }
}

/// Verifies an IEEE 370 mixed-mode NZC 2x-thru splits and self-de-embeds.
#[test]
fn splits_and_self_deembeds_ieee_370_mixed_mode_two_x_thru() -> TestResult<()> {
    let differential_fixture =
        two_port_from_z_with_reference([[130.0, 30.0], [30.0, 120.0]], 100.0)?;
    let common_fixture = two_port_from_z_with_reference([[35.0, 10.0], [10.0, 30.0]], 25.0)?;
    let differential_two_x_thru = differential_fixture
        .cascade(&differential_fixture.flipped().expect("differential flip"))
        .expect("differential 2x-thru");
    let common_two_x_thru = common_fixture
        .cascade(&common_fixture.flipped().expect("common flip"))
        .expect("common 2x-thru");
    let mixed_mode_two_x_thru =
        concatenate_ports(&[differential_two_x_thru.clone(), common_two_x_thru.clone()])
            .expect("mixed-mode block network");
    let single_ended_two_x_thru = mixed_mode_two_x_thru
        .mixed_mode_to_single_ended(2)
        .expect("single-ended conversion");
    let deembedding = IeeeP370MmNzc2xThru::new(
        single_ended_two_x_thru.clone(),
        IeeeP370PortOrder::Second,
        Some("mixed-mode 2x-thru".to_owned()),
    )
    .expect("mixed-mode fixture split");

    let residual = deembedding
        .deembed(&single_ended_two_x_thru)
        .expect("mixed-mode self de-embedding");
    let expected_mixed = concatenate_ports(&[
        IeeeP370::thru(&differential_two_x_thru).expect("differential thru"),
        IeeeP370::thru(&common_two_x_thru).expect("common thru"),
    ])
    .expect("ideal mixed-mode thru");
    let expected = expected_mixed
        .mixed_mode_to_single_ended(2)
        .expect("ideal single-ended thru");

    assert_network_close(&residual, &expected);

    let quality =
        IeeeP370FrequencyDomainQuality::check_mixed_mode(&expected).expect("mixed-mode quality");
    assert_relative_eq!(
        quality.differential.passivity.value_percent,
        100.0,
        epsilon = TOLERANCE
    );
    assert_relative_eq!(
        quality.common.reciprocity.value_percent,
        100.0,
        epsilon = TOLERANCE
    );
    Ok(())
}

fn one_port_from_y(value: f64) -> TestResult<Network> {
    let frequency = Frequency::from_hz(array![1.0e9])?;
    let reference = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
    let admittance = Array3::from_elem((1, 1, 1), Complex64::new(value, 0.0));
    let scattering = y_to_s(&admittance, &reference, SParameterDefinition::Power)?;
    Ok(Network::new(frequency, scattering, reference)?)
}

fn one_port_from_z(value: f64) -> TestResult<Network> {
    let frequency = Frequency::from_hz(array![1.0e9])?;
    let reference = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
    let impedance = Array3::from_elem((1, 1, 1), Complex64::new(value, 0.0));
    let scattering = z_to_s(&impedance, &reference, SParameterDefinition::Power)?;
    Ok(Network::new(frequency, scattering, reference)?)
}

fn two_port_from_y(values: [[f64; 2]; 2]) -> TestResult<Network> {
    let frequency = Frequency::from_hz(array![1.0e9])?;
    let reference = Array2::from_elem((1, 2), Complex64::new(50.0, 0.0));
    let admittance = Array3::from_shape_fn((1, 2, 2), |(_, row, column)| {
        Complex64::new(values[row][column], 0.0)
    });
    let scattering = y_to_s(&admittance, &reference, SParameterDefinition::Power)?;
    Ok(Network::new(frequency, scattering, reference)?)
}

fn two_port_from_z(values: [[f64; 2]; 2]) -> TestResult<Network> {
    two_port_from_z_with_reference(values, 50.0)
}

fn two_port_from_z_with_reference(
    values: [[f64; 2]; 2],
    reference_ohms: f64,
) -> TestResult<Network> {
    let frequency = Frequency::from_hz(array![1.0e9])?;
    let reference = Array2::from_elem((1, 2), Complex64::new(reference_ohms, 0.0));
    let impedance = Array3::from_shape_fn((1, 2, 2), |(_, row, column)| {
        Complex64::new(values[row][column], 0.0)
    });
    let scattering = z_to_s(&impedance, &reference, SParameterDefinition::Power)?;
    Ok(Network::new(frequency, scattering, reference)?)
}

fn matched_two_port(transmission: Complex64) -> TestResult<Network> {
    let frequency = Frequency::from_hz(array![1.0e9])?;
    let mut scattering = Array3::zeros((1, 2, 2));
    scattering[(0, 0, 1)] = transmission;
    scattering[(0, 1, 0)] = transmission;
    Ok(Network::new(
        frequency,
        scattering,
        Array2::from_elem((1, 2), Complex64::new(50.0, 0.0)),
    )?)
}

fn admittance(network: &Network) -> TestResult<f64> {
    let admittance = s_to_y(&network.s, &network.z0, network.s_definition)?;
    Ok(admittance[(0, 0, 0)].re)
}

fn impedance(network: &Network) -> TestResult<f64> {
    let impedance = s_to_z(&network.s, &network.z0, network.s_definition)?;
    Ok(impedance[(0, 0, 0)].re)
}

fn assert_network_close(actual: &Network, expected: &Network) {
    assert_eq!(actual.s.dim(), expected.s.dim());
    for (actual, expected) in actual.s.iter().zip(expected.s.iter()) {
        assert_relative_eq!(actual.re, expected.re, epsilon = TOLERANCE);
        assert_relative_eq!(actual.im, expected.im, epsilon = TOLERANCE);
    }
}
