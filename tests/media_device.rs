use approx::assert_relative_eq;
use num_complex::Complex64;
use rust_rf::instances::INSTANCES;
use rust_rf::media::device::{
    Device, DualCoupler, Hybrid, Hybrid180, MatchedSymmetricCoupler, QuadratureHybrid,
};

const TOLERANCE: f64 = 1.0e-12;

#[test]
fn builds_four_and_three_port_matched_symmetric_couplers() {
    let media = INSTANCES.air50().unwrap();
    let coupler = MatchedSymmetricCoupler::from_coupling(media.clone(), 0.5, 4).unwrap();
    let network = coupler.network().unwrap();
    assert_eq!(network.ports(), 4);
    assert_relative_eq!(network.s[(0, 0, 2)].re, 0.5, epsilon = TOLERANCE);
    assert_relative_eq!(
        network.s[(0, 0, 1)].re,
        (0.75_f64).sqrt(),
        epsilon = TOLERANCE
    );
    assert_eq!(network.s[(0, 0, 3)], Complex64::new(0.0, 0.0));
    assert_eq!(
        MatchedSymmetricCoupler::from_coupling(media, 0.5, 3)
            .unwrap()
            .network()
            .unwrap()
            .ports(),
        3
    );
}

#[test]
fn creates_db_hybrid_quadrature_and_180_degree_variants() {
    let media = INSTANCES.air50().unwrap();
    let db = MatchedSymmetricCoupler::from_db_degrees(media.clone(), -6.0, 30.0, 4)
        .unwrap()
        .network()
        .unwrap();
    assert_relative_eq!(
        db.s[(0, 0, 2)].norm(),
        10.0_f64.powf(-6.0 / 20.0),
        epsilon = TOLERANCE
    );

    let hybrid = Hybrid::new(media.clone(), 180.0, 0.0)
        .unwrap()
        .network()
        .unwrap();
    assert_relative_eq!(
        hybrid.s[(0, 0, 1)].re,
        -std::f64::consts::FRAC_1_SQRT_2,
        epsilon = TOLERANCE
    );
    let quadrature = QuadratureHybrid::new(media.clone(), 0.0)
        .unwrap()
        .network()
        .unwrap();
    assert_relative_eq!(
        quadrature.s[(0, 0, 2)].im,
        -std::f64::consts::FRAC_1_SQRT_2,
        epsilon = TOLERANCE
    );
    let hybrid180 = Hybrid180::new(media, 4).unwrap().network().unwrap();
    assert_relative_eq!(
        hybrid180.s[(0, 3, 1)].im,
        std::f64::consts::FRAC_1_SQRT_2,
        epsilon = TOLERANCE
    );
    assert_relative_eq!(
        hybrid180.s[(0, 3, 2)].im,
        -std::f64::consts::FRAC_1_SQRT_2,
        epsilon = TOLERANCE
    );
}

#[test]
fn connects_and_renumbers_a_dual_coupler() {
    let network = DualCoupler::new(INSTANCES.air50().unwrap(), 0.5, Some(0.25))
        .unwrap()
        .network()
        .unwrap();
    assert_eq!(network.ports(), 4);
    assert_eq!(network.frequency_points(), 101);
}

#[test]
fn validates_coupling_and_port_configuration() {
    let media = INSTANCES.air50().unwrap();
    assert!(MatchedSymmetricCoupler::from_coupling(media.clone(), 1.1, 4).is_err());
    assert!(MatchedSymmetricCoupler::from_coupling(media.clone(), 0.5, 2).is_err());
    assert!(Hybrid180::new(media, 5).is_err());
}
