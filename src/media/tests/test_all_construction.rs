//! Construction checks shared by every general-purpose media implementation.
//!
//! This is the Rust counterpart of the Python `MediaTestCase`: each medium
//! must be able to construct the common line, load, lumped-element, junction,
//! delayed, and shunt networks exposed by [`Media`].

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
use rust_rf::{Frequency, FrequencyUnit, Network, Result, SweepType};

/// Verifies that representative media support every shared component constructor.
#[test]
fn constructs_shared_components_for_every_general_media() -> Result<()> {
    let frequency = Frequency::new(75.0, 110.0, 3, FrequencyUnit::GHz, SweepType::Linear)
        .expect("frequency should be valid");
    let freespace = Freespace::vacuum(frequency.clone()).expect("vacuum should be valid");
    let cpw = Cpw::lossless(frequency.clone(), 10.0e-6, 5.0e-6, 1.0e-6, 11.7)
        .expect("CPW should be valid");
    let rectangular = RectangularWaveguide::dominant_mode(frequency.clone(), 100.0 * 0.000_025_4)
        .expect("rectangular waveguide should be valid");
    let distributed = DistributedCircuit::from_scalars(frequency, 0.0, 0.0, 1.0, 1.0)
        .expect("distributed circuit should be valid");
    assert_general_media_construction(&freespace)?;
    assert_general_media_construction(&cpw)?;
    assert_general_media_construction(&rectangular)?;
    assert_general_media_construction(&distributed)?;
    Ok(())
}

// Applies the common construction contract to one concrete medium.
fn assert_general_media_construction<M: Media>(media: &M) -> Result<()> {
    let points = media.points();
    assert_media_parameters(media, points)?;
    assert_one_port_components(media, points)?;
    assert_lumped_components(media, points)?;
    assert_multiport_components(media, points)?;
    assert_line_components(media)?;
    assert_delayed_components(media)?;
    Ok(())
}

fn assert_media_parameters<M: Media>(media: &M, points: usize) -> Result<()> {
    assert_eq!(media.propagation_constant()?.len(), points);
    assert_eq!(media.characteristic_impedance()?.len(), points);
    Ok(())
}

fn assert_one_port_components<M: Media>(media: &M, points: usize) -> Result<()> {
    assert_eq!(media.match_network(1, None)?.ports(), 1);
    assert_eq!(
        media
            .load_nports(
                &Array1::from_elem(points, Complex64::new(0.25, 0.0)),
                1,
                None
            )?
            .ports(),
        1
    );
    assert_eq!(media.short()?.ports(), 1);
    assert_eq!(media.open()?.ports(), 1);
    Ok(())
}

fn assert_lumped_components<M: Media>(media: &M, points: usize) -> Result<()> {
    let resistance = Array1::from_elem(points, 1.0);
    let capacitance = Array1::from_elem(points, 1.0e-12);
    let inductance = Array1::from_elem(points, 1.0e-9);
    for component in [
        media.resistor(&resistance),
        media.capacitor(&capacitance),
        media.inductor(&inductance),
        media.capacitor_with_q(&capacitance, 80.0e9, 100.0),
        media.inductor_with_q(&inductance, 80.0e9, 100.0, 0.0),
        media.shunt_capacitor(&capacitance),
        media.shunt_inductor(&inductance),
    ] {
        assert_eq!(component?.ports(), 2);
    }
    Ok(())
}

fn assert_multiport_components<M: Media>(media: &M, points: usize) -> Result<()> {
    assert_eq!(
        media
            .impedance_mismatch(
                &Array1::from_elem(points, 50.0),
                &Array1::from_elem(points, 75.0),
            )?
            .ports(),
        2
    );
    assert_eq!(media.tee()?.ports(), 3);
    assert_eq!(media.splitter(4)?.ports(), 4);
    assert_eq!(media.thru()?.ports(), 2);
    Ok(())
}

fn assert_line_components<M: Media>(media: &M) -> Result<()> {
    assert_eq!(media.line(1.0, LengthUnit::Millimeter)?.ports(), 2);
    assert_eq!(media.floating_line(1.0, LengthUnit::Millimeter)?.ports(), 4);
    Ok(())
}

fn assert_delayed_components<M: Media>(media: &M) -> Result<()> {
    assert_eq!(
        media
            .delay_load(Complex64::new(0.2, 0.0), 10.0, LengthUnit::Degree)?
            .ports(),
        1
    );
    assert_eq!(media.delay_short(10.0, LengthUnit::Degree)?.ports(), 1);
    assert_eq!(media.delay_open(10.0, LengthUnit::Degree)?.ports(), 1);
    assert_eq!(
        media
            .shunt_delay_load(Complex64::new(0.2, 0.0), 10.0, LengthUnit::Degree,)?
            .ports(),
        2
    );
    assert_eq!(media.shunt_delay_open(10.0, LengthUnit::Degree)?.ports(), 2);
    assert_eq!(
        media.shunt_delay_short(10.0, LengthUnit::Degree)?.ports(),
        2
    );
    Ok(())
}
