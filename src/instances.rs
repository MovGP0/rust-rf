use ndarray::Array1;
use num_complex::Complex64;

use crate::constants::MIL;
use crate::media::{Freespace, RectangularWaveguide, WaveguideMode};
use crate::{Frequency, FrequencyUnit, Result, SweepType};

/// Standard rectangular-waveguide designations exposed by scikit-rf.
///
/// Origin: `skrf/instances.py::StaticInstances`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaveguideBand {
    Wr51,
    Wr42,
    Wr34,
    Wr28,
    Wr22p4,
    Wr18p8,
    Wr14p8,
    Wr12p2,
    Wr10,
    Wr8,
    Wr6p5,
    Wr5p1,
    Wr4p3,
    Wr3p4,
    Wr2p8,
    Wr2p2,
    Wr1p9,
    Wr1p5,
    Wr1p2,
    Wr1,
    Wr0p8,
    Wr0p65,
    Wr0p51,
    Wm1295,
    Wm1092,
    Wm864,
    Wm710,
    Wm570,
    Wm470,
    Wm380,
    Wm310,
    Wm250,
    Wm200,
    Wm164,
    Wm130,
    Wm106,
    Wm86,
}

impl WaveguideBand {
    pub const ALL: [Self; 37] = [
        Self::Wr51,
        Self::Wr42,
        Self::Wr34,
        Self::Wr28,
        Self::Wr22p4,
        Self::Wr18p8,
        Self::Wr14p8,
        Self::Wr12p2,
        Self::Wr10,
        Self::Wr8,
        Self::Wr6p5,
        Self::Wr5p1,
        Self::Wr4p3,
        Self::Wr3p4,
        Self::Wr2p8,
        Self::Wr2p2,
        Self::Wr1p9,
        Self::Wr1p5,
        Self::Wr1p2,
        Self::Wr1,
        Self::Wr0p8,
        Self::Wr0p65,
        Self::Wr0p51,
        Self::Wm1295,
        Self::Wm1092,
        Self::Wm864,
        Self::Wm710,
        Self::Wm570,
        Self::Wm470,
        Self::Wm380,
        Self::Wm310,
        Self::Wm250,
        Self::Wm200,
        Self::Wm164,
        Self::Wm130,
        Self::Wm106,
        Self::Wm86,
    ];

    const fn frequency_range_ghz(self) -> (f64, f64) {
        match self {
            Self::Wr51 => (15.0, 22.0),
            Self::Wr42 => (17.5, 26.5),
            Self::Wr34 => (22.0, 33.0),
            Self::Wr28 => (26.5, 40.0),
            Self::Wr22p4 => (33.0, 50.5),
            Self::Wr18p8 => (40.0, 60.0),
            Self::Wr14p8 => (50.0, 75.0),
            Self::Wr12p2 => (60.0, 90.0),
            Self::Wr10 => (75.0, 110.0),
            Self::Wr8 => (90.0, 140.0),
            Self::Wr6p5 => (110.0, 170.0),
            Self::Wr5p1 | Self::Wm1295 => (140.0, 220.0),
            Self::Wr4p3 | Self::Wm1092 => (170.0, 260.0),
            Self::Wr3p4 | Self::Wm864 => (220.0, 330.0),
            Self::Wr2p8 | Self::Wm710 => (260.0, 400.0),
            Self::Wr2p2 | Self::Wm570 => (330.0, 500.0),
            Self::Wr1p9 | Self::Wm470 => (400.0, 600.0),
            Self::Wr1p5 | Self::Wm380 => (500.0, 750.0),
            Self::Wr1p2 | Self::Wm310 => (600.0, 900.0),
            Self::Wr1 | Self::Wm250 => (750.0, 1100.0),
            Self::Wr0p8 | Self::Wm200 => (900.0, 1400.0),
            Self::Wr0p65 | Self::Wm164 => (1100.0, 1700.0),
            Self::Wr0p51 | Self::Wm130 => (1400.0, 2200.0),
            Self::Wm106 => (1700.0, 2600.0),
            Self::Wm86 => (2200.0, 3300.0),
        }
    }

    const fn dimensions_meters(self) -> (f64, f64) {
        match self {
            Self::Wr51 => (510.0 * MIL, 255.0 * MIL),
            Self::Wr42 => (420.0 * MIL, 170.0 * MIL),
            Self::Wr34 => (340.0 * MIL, 170.0 * MIL),
            Self::Wr28 => (280.0 * MIL, 140.0 * MIL),
            Self::Wr22p4 => (224.0 * MIL, 112.0 * MIL),
            Self::Wr18p8 => (188.0 * MIL, 94.0 * MIL),
            Self::Wr14p8 => (148.0 * MIL, 74.0 * MIL),
            Self::Wr12p2 => (122.0 * MIL, 61.0 * MIL),
            Self::Wr10 => (100.0 * MIL, 50.0 * MIL),
            Self::Wr8 => (80.0 * MIL, 40.0 * MIL),
            Self::Wr6p5 => (65.0 * MIL, 32.5 * MIL),
            Self::Wr5p1 => (51.0 * MIL, 25.5 * MIL),
            Self::Wr4p3 => (43.0 * MIL, 21.5 * MIL),
            Self::Wr3p4 => (34.0 * MIL, 17.0 * MIL),
            Self::Wr2p8 => (28.0 * MIL, 14.0 * MIL),
            Self::Wr2p2 => (22.0 * MIL, 11.0 * MIL),
            Self::Wr1p9 => (19.0 * MIL, 9.5 * MIL),
            Self::Wr1p5 => (15.0 * MIL, 7.5 * MIL),
            Self::Wr1p2 => (12.0 * MIL, 6.0 * MIL),
            Self::Wr1 => (10.0 * MIL, 5.0 * MIL),
            Self::Wr0p8 => (8.0 * MIL, 4.0 * MIL),
            Self::Wr0p65 => (6.5 * MIL, 3.25 * MIL),
            Self::Wr0p51 => (5.1 * MIL, 2.55 * MIL),
            Self::Wm1295 => (1295.0e-6, 647.5e-6),
            Self::Wm1092 => (1092.0e-6, 546.0e-6),
            Self::Wm864 => (864.0e-6, 432.0e-6),
            Self::Wm710 => (710.0e-6, 355.0e-6),
            Self::Wm570 => (570.0e-6, 285.0e-6),
            Self::Wm470 => (470.0e-6, 235.0e-6),
            Self::Wm380 => (380.0e-6, 190.0e-6),
            Self::Wm310 => (310.0e-6, 155.0e-6),
            Self::Wm250 => (250.0e-6, 125.0e-6),
            Self::Wm200 => (200.0e-6, 100.0e-6),
            Self::Wm164 => (164.0e-6, 82.0e-6),
            Self::Wm130 => (130.0e-6, 65.0e-6),
            Self::Wm106 => (106.0e-6, 53.0e-6),
            Self::Wm86 => (86.0e-6, 43.0e-6),
        }
    }
}

/// Rust method-based counterpart to the Python module's lazy properties.
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticInstances;

macro_rules! named_instances {
    ($(($frequency:ident, $waveguide:ident, $band:ident)),+ $(,)?) => {
        $(
            pub fn $frequency(&self) -> Result<Frequency> {
                self.frequency(WaveguideBand::$band)
            }

            pub fn $waveguide(&self) -> Result<RectangularWaveguide> {
                self.waveguide(WaveguideBand::$band)
            }
        )+
    };
}

impl StaticInstances {
    pub fn air(&self) -> Result<Freespace> {
        Freespace::from_scalars(
            default_frequency()?,
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        )
    }

    pub fn air50(&self) -> Result<Freespace> {
        let frequency = default_frequency()?;
        let points = frequency.points();
        Freespace::new(
            frequency,
            Array1::from_elem(points, Complex64::new(1.0, 0.0)),
            Array1::from_elem(points, Complex64::new(1.0, 0.0)),
            None,
            None,
            None,
            None,
            Some(Array1::from_elem(points, Complex64::new(50.0, 0.0))),
        )
    }

    pub fn frequency(&self, band: WaveguideBand) -> Result<Frequency> {
        let (start, stop) = band.frequency_range_ghz();
        Frequency::new(start, stop, 1001, FrequencyUnit::GHz, SweepType::Linear)
    }

    pub fn waveguide(&self, band: WaveguideBand) -> Result<RectangularWaveguide> {
        let frequency = self.frequency(band)?;
        let points = frequency.points();
        let (width, height) = band.dimensions_meters();
        RectangularWaveguide::new(
            frequency,
            width,
            Some(height),
            WaveguideMode::TransverseElectric,
            1,
            0,
            Array1::ones(points),
            Array1::ones(points),
            None,
            None,
            None,
            Some(Array1::from_elem(points, Complex64::new(50.0, 0.0))),
        )
    }

    named_instances! {
        (f_wr51, wr51, Wr51), (f_wr42, wr42, Wr42), (f_wr34, wr34, Wr34),
        (f_wr28, wr28, Wr28), (f_wr22p4, wr22p4, Wr22p4),
        (f_wr18p8, wr18p8, Wr18p8), (f_wr14p8, wr14p8, Wr14p8),
        (f_wr12p2, wr12p2, Wr12p2), (f_wr10, wr10, Wr10), (f_wr8, wr8, Wr8),
        (f_wr6p5, wr6p5, Wr6p5), (f_wr5p1, wr5p1, Wr5p1),
        (f_wr4p3, wr4p3, Wr4p3), (f_wr3p4, wr3p4, Wr3p4),
        (f_wr2p8, wr2p8, Wr2p8), (f_wr2p2, wr2p2, Wr2p2),
        (f_wr1p9, wr1p9, Wr1p9), (f_wr1p5, wr1p5, Wr1p5),
        (f_wr1p2, wr1p2, Wr1p2), (f_wr1, wr1, Wr1),
        (f_wr0p8, wr0p8, Wr0p8), (f_wr0p65, wr0p65, Wr0p65),
        (f_wr0p51, wr0p51, Wr0p51), (f_wm1295, wm1295, Wm1295),
        (f_wm1092, wm1092, Wm1092), (f_wm864, wm864, Wm864),
        (f_wm710, wm710, Wm710), (f_wm570, wm570, Wm570),
        (f_wm470, wm470, Wm470), (f_wm380, wm380, Wm380),
        (f_wm310, wm310, Wm310), (f_wm250, wm250, Wm250),
        (f_wm200, wm200, Wm200), (f_wm164, wm164, Wm164),
        (f_wm130, wm130, Wm130), (f_wm106, wm106, Wm106),
        (f_wm86, wm86, Wm86)
    }
}

fn default_frequency() -> Result<Frequency> {
    Frequency::new(1.0, 10.0, 101, FrequencyUnit::GHz, SweepType::Linear)
}

pub const INSTANCES: StaticInstances = StaticInstances;
