//! VNA calibration algorithms ported from `skrf/calibration/calibration.py`.
//!
//! The module provides one-port, two-port, multi-port, and three-receiver
//! calibration implementations together with switch-term, standard-determination,
//! error-model conversion, standard-alignment, and coefficient-conversion helpers.

use std::collections::BTreeMap;

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use num_traits::ToPrimitive;

use super::ensure_nonzero;
use crate::math::sqrt_phase_unwrap;
use crate::{Error, Frequency, Network, NetworkSet, Result};
type ComplexMatrix2 = [[Complex64; 2]; 2];
type FourComplexMatrices = (
    ComplexMatrix2,
    ComplexMatrix2,
    ComplexMatrix2,
    ComplexMatrix2,
);
/// Four frequency-indexed $2 \times 2$ complex matrix arrays.
pub type FourComplexArrayMatrices = (
    Array3<Complex64>,
    Array3<Complex64>,
    Array3<Complex64>,
    Array3<Complex64>,
);

/// Common interface implemented by every calibration algorithm.
///
/// Implementations solve error coefficients from aligned measured and ideal
/// standards, apply those coefficients to measured networks, and can embed an
/// ideal response back into the estimated error network.
///
/// Measured and ideal standards must be aligned. [`align_measured_ideals`] can
/// align named standards for algorithms where order is not semantically significant;
/// do not use name-based alignment for order-dependent methods such as [`Trl`].
///
/// The coefficient accessors expose the conventional one-port, eight-term, and
/// twelve-term error models. Conversion between the two-port models is performed
/// by [`convert_8term_2_12term`] and [`convert_12term_2_8term`].
///
/// Origin: `skrf/calibration/calibration.py::Calibration`.
pub trait Calibration {
    /// Measured calibration standards.
    fn measured(&self) -> &[Network];

    /// Ideal responses corresponding to the measured standards.
    fn ideals(&self) -> &[Network];

    /// Solves the calibration coefficients from the measured and ideal standards.
    ///
    /// # Errors
    ///
    /// Returns an error when the standards cannot produce a valid coefficient solution.
    fn run(&mut self) -> Result<()>;

    /// Applies the solved calibration to a measured network.
    ///
    /// # Errors
    ///
    /// Returns an error when the network is incompatible with the calibration model.
    fn apply(&self, network: &Network) -> Result<Network>;

    /// Embeds an ideal network in the calibration error model.
    ///
    /// # Errors
    ///
    /// Returns an error when the network is incompatible with the calibration model.
    fn embed(&self, network: &Network) -> Result<Network>;

    /// Solved error coefficients keyed by their conventional names.
    fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>>;

    /// Applies the calibration to a measured network.
    ///
    /// # Errors
    ///
    /// Returns an error when the network is incompatible with the calibration model.
    fn apply_cal(&self, network: &Network) -> Result<Network> {
        self.apply(network)
    }

    /// Returns the frequency axis shared by the calibration standards.
    ///
    /// # Errors
    ///
    /// Returns an error when the calibration has no measured standards.
    fn frequency(&self) -> Result<&Frequency> {
        self.measured()
            .first()
            .map(|network| &network.frequency)
            .ok_or_else(|| {
                Error::IncompatibleShape("a calibration has no measured standards".to_owned())
            })
    }

    /// Returns the number of measured calibration standards.
    fn standards(&self) -> usize {
        self.measured().len()
    }

    /// Returns the number of measured calibration standards.
    fn nstandards(&self) -> usize {
        self.standards()
    }

    /// Returns the solved coefficient arrays using conventional error-term names.
    fn coefs(&self) -> &BTreeMap<String, Array1<Complex64>> {
        self.coefficients()
    }

    /// Applies the calibration to every network in a slice.
    ///
    /// # Errors
    ///
    /// Returns an error if any network is incompatible with the calibration model.
    fn apply_all(&self, networks: &[Network]) -> Result<Vec<Network>> {
        networks.iter().map(|network| self.apply(network)).collect()
    }

    /// Applies the calibration to every network in a slice.
    ///
    /// # Errors
    ///
    /// Returns an error if any network is incompatible with the calibration model.
    fn apply_cal_to_list(&self, networks: &[Network]) -> Result<Vec<Network>> {
        self.apply_all(networks)
    }

    /// Applies the calibration while preserving network-set metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if any network in the set is incompatible with the calibration model.
    fn apply_network_set(&self, set: &NetworkSet) -> Result<NetworkSet> {
        Ok(NetworkSet {
            networks: self.apply_all(&set.networks)?,
            name: set.name.clone(),
            parameters: set.parameters.clone(),
            text_parameters: set.text_parameters.clone(),
        })
    }

    /// Applies the calibration while preserving network-set metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if any network in the set is incompatible with the calibration model.
    fn apply_cal_to_network_set(&self, set: &NetworkSet) -> Result<NetworkSet> {
        self.apply_network_set(set)
    }

    /// Applies the calibration to the standards used to solve it.
    ///
    /// # Errors
    ///
    /// Returns an error if any measured standard is incompatible with the calibration model.
    fn calibrated_standards(&self) -> Result<Vec<Network>> {
        self.apply_all(self.measured())
    }

    /// Returns the calibrated standard measurements.
    ///
    /// # Errors
    ///
    /// Returns an error if any measured standard is incompatible with the calibration model.
    fn caled_ntwks(&self) -> Result<Vec<Network>> {
        self.calibrated_standards()
    }

    /// Returns the complex residual between each calibrated measurement and its
    /// corresponding ideal standard.
    ///
    /// # Errors
    ///
    /// Returns an error when standards are not aligned or have incompatible shapes.
    fn residual_networks(&self) -> Result<Vec<Network>> {
        if self.measured().len() != self.ideals().len() {
            return Err(Error::IncompatibleShape(
                "calibration standards are not aligned".to_owned(),
            ));
        }
        self.calibrated_standards()?
            .into_iter()
            .zip(self.ideals())
            .map(|(mut calibrated, ideal)| {
                if calibrated.s.dim() != ideal.s.dim() {
                    return Err(Error::IncompatibleShape(
                        "calibrated and ideal standards have different shapes".to_owned(),
                    ));
                }
                calibrated.s -= &ideal.s;
                calibrated.name = ideal.name.as_ref().map(|name| format!("{name}-residual"));
                Ok(calibrated)
            })
            .collect()
    }

    /// Returns the residual networks for all aligned standards.
    ///
    /// # Errors
    ///
    /// Returns an error when standards are not aligned or have incompatible shapes.
    fn residual_ntwks(&self) -> Result<Vec<Network>> {
        self.residual_networks()
    }

    /// Converts every coefficient array into a one-port [`Network`].
    ///
    /// # Errors
    ///
    /// Returns an error when no measured standard exists, a coefficient length is
    /// inconsistent, or a coefficient network cannot be constructed.
    fn coefficient_networks(&self) -> Result<BTreeMap<String, Network>> {
        let frequency = self.frequency()?.clone();
        let points = frequency.points();
        let reference = self
            .measured()
            .first()
            .map(|network| Array2::from_shape_fn((points, 1), |(point, _)| network.z0[(point, 0)]))
            .ok_or_else(|| {
                Error::IncompatibleShape("a calibration has no measured standards".to_owned())
            })?;
        self.coefficients()
            .iter()
            .map(|(name, values)| {
                if values.len() != points {
                    return Err(Error::IncompatibleShape(format!(
                        "coefficient {name} has {} values for {points} frequency points",
                        values.len()
                    )));
                }
                let s = Array3::from_shape_fn((points, 1, 1), |(point, _, _)| values[point]);
                let mut network = Network::new(frequency.clone(), s, reference.clone())?;
                network.name = Some(name.clone());
                Ok((name.clone(), network))
            })
            .collect()
    }

    /// Returns the error coefficients as one-port [`Network`] values.
    ///
    /// # Errors
    ///
    /// Returns an error when coefficient arrays cannot be represented as networks.
    fn coefs_ntwks(&self) -> Result<BTreeMap<String, Network>> {
        self.coefficient_networks()
    }

    /// Returns directivity, source match, and reflection tracking for the
    /// one-port three-term error model.
    ///
    /// # Errors
    ///
    /// Returns an error when any required three-term coefficient is unavailable.
    fn coefs_3term(&self) -> Result<BTreeMap<String, Array1<Complex64>>> {
        ["directivity", "source match", "reflection tracking"]
            .into_iter()
            .map(|name| {
                self.coefficients()
                    .get(name)
                    .cloned()
                    .map(|values| (name.to_owned(), values))
                    .ok_or_else(|| {
                        Error::Unsupported(format!("missing calibration coefficient '{name}'"))
                    })
            })
            .collect()
    }

    /// Returns the directional error-box coefficients, switch terms, isolation
    /// terms, and $k$ coefficient for the eight-term model.
    ///
    /// # Errors
    ///
    /// Returns an error when the coefficient set cannot be converted to the eight-term model.
    fn coefs_8term(&self) -> Result<BTreeMap<String, Array1<Complex64>>> {
        if self.coefficients().contains_key("directivity") {
            return Err(Error::Unsupported(
                "one-port coefficients cannot be converted to an eight-term model".to_owned(),
            ));
        }
        if TWELVE_TERM_COEFFICIENTS
            .iter()
            .all(|name| self.coefficients().contains_key(*name))
        {
            convert_12term_2_8term(self.coefficients(), false)
        } else {
            Ok(self.coefficients().clone())
        }
    }

    /// Returns the forward and reverse directivity, match, reflection tracking,
    /// transmission tracking, and isolation terms for the twelve-term model.
    ///
    /// # Errors
    ///
    /// Returns an error when the coefficient set cannot be converted to the twelve-term model.
    fn coefs_12term(&self) -> Result<BTreeMap<String, Array1<Complex64>>> {
        if self.coefficients().contains_key("directivity") {
            return Err(Error::Unsupported(
                "one-port coefficients cannot be converted to a twelve-term model".to_owned(),
            ));
        }
        if TWELVE_TERM_COEFFICIENTS
            .iter()
            .all(|name| self.coefficients().contains_key(*name))
        {
            Ok(self.coefficients().clone())
        } else {
            convert_8term_2_12term(self.coefficients())
        }
    }

    /// Evaluates the twelve-term consistency relation at each frequency point.
    ///
    /// # Errors
    ///
    /// Returns an error when required twelve-term coefficients are missing or inconsistent.
    fn verify_12term(&self) -> Result<Array1<Complex64>> {
        let coefficients = self.coefs_12term()?;
        let points = coefficient_points(&coefficients)?;
        let edf = required_coefficient(&coefficients, "forward directivity", points)?;
        let esf = required_coefficient(&coefficients, "forward source match", points)?;
        let erf = required_coefficient(&coefficients, "forward reflection tracking", points)?;
        let etf = required_coefficient(&coefficients, "forward transmission tracking", points)?;
        let elf = required_coefficient(&coefficients, "forward load match", points)?;
        let edr = required_coefficient(&coefficients, "reverse directivity", points)?;
        let elr = required_coefficient(&coefficients, "reverse load match", points)?;
        let err = required_coefficient(&coefficients, "reverse reflection tracking", points)?;
        let etr = required_coefficient(&coefficients, "reverse transmission tracking", points)?;
        let esr = required_coefficient(&coefficients, "reverse source match", points)?;
        Ok(Array1::from_shape_fn(points, |point| {
            etf[point] * etr[point]
                - (err[point] + edr[point] * (elf[point] - esr[point]))
                    * (erf[point] + edf[point] * (elr[point] - esf[point]))
        }))
    }

    /// Returns the twelve-term consistency relation as a one-port [`Network`].
    ///
    /// # Errors
    ///
    /// Returns an error when the consistency values cannot be computed or represented as a network.
    fn verify_12term_network(&self) -> Result<Network> {
        let values = self.verify_12term()?;
        coefficient_network(self.frequency()?.clone(), &values, "verify 12-term")
    }

    /// Groups residual networks by ideal-standard name.
    ///
    /// # Errors
    ///
    /// Returns an error when residuals cannot be computed or standards cannot be grouped.
    fn residual_ntwk_sets(&self) -> Result<BTreeMap<String, NetworkSet>> {
        group_networks_by_ideal_name(&self.residual_networks()?, self.ideals())
    }

    /// Groups calibrated standards by ideal-standard name.
    ///
    /// # Errors
    ///
    /// Returns an error when standards cannot be calibrated or grouped.
    fn caled_ntwk_sets(&self) -> Result<BTreeMap<String, NetworkSet>> {
        group_networks_by_ideal_name(&self.calibrated_standards()?, self.ideals())
    }

    /// Estimates biased error across repeated connections of each standard.
    ///
    /// $$
    /// \operatorname{mean}_s\left(\left|\operatorname{mean}_c(r)\right|\right)
    /// $$
    ///
    /// # Errors
    ///
    /// Returns an error when residual sets cannot be computed or averaged.
    fn biased_error(&self) -> Result<Network> {
        let standard_means = self
            .residual_ntwk_sets()?
            .into_values()
            .map(|set| set.mean_s())
            .collect::<Result<Vec<_>>>()?;
        let mut error = NetworkSet::new(standard_means, Some("biased-error-standards".to_owned()))?
            .mean_s_magnitude()?;
        error.name = Some("Biased Error".to_owned());
        Ok(error)
    }

    /// Estimates connection-to-connection variation for each standard.
    ///
    /// $$
    /// \operatorname{mean}_s\left(\operatorname{std}_c(r)\right)
    /// $$
    ///
    /// # Errors
    ///
    /// Returns an error when residual sets cannot be computed or their deviations averaged.
    fn unbiased_error(&self) -> Result<Network> {
        let standard_deviations = self
            .residual_ntwk_sets()?
            .into_values()
            .map(|set| set.std_s())
            .collect::<Result<Vec<_>>>()?;
        let mut error = NetworkSet::new(
            standard_deviations,
            Some("unbiased-error-standards".to_owned()),
        )?
        .mean_s_magnitude()?;
        error.name = Some("Unbiased Error".to_owned());
        Ok(error)
    }

    /// Estimates total residual error across connections and standards.
    ///
    /// $$
    /// \operatorname{std}_{cs}(r)
    /// $$
    ///
    /// # Errors
    ///
    /// Returns an error when residual networks cannot be computed or aggregated.
    fn total_error(&self) -> Result<Network> {
        let mut error = NetworkSet::new(
            self.residual_networks()?,
            Some("total-error-residuals".to_owned()),
        )?
        .mean_s_magnitude()?;
        error.name = Some("Total Error".to_owned());
        Ok(error)
    }

    /// Builds the estimated one-port or directional error network from the
    /// solved coefficients.
    ///
    /// # Errors
    ///
    /// Returns an error when coefficients or frequency data cannot form an error network.
    fn error_ntwk(&self, reciprocal: bool) -> Result<ErrorNetworkResult> {
        error_dict_2_network(self.coefficients(), self.frequency()?, reciprocal)
    }
}

const TWELVE_TERM_COEFFICIENTS: [&str; 12] = [
    "forward directivity",
    "forward source match",
    "forward reflection tracking",
    "forward transmission tracking",
    "forward load match",
    "forward isolation",
    "reverse directivity",
    "reverse load match",
    "reverse reflection tracking",
    "reverse transmission tracking",
    "reverse source match",
    "reverse isolation",
];

macro_rules! calibration_structure {
    ($name:ident, $origin:literal, $description:literal) => {
        #[doc = $description]
        ///
        #[doc = concat!("Origin: `", $origin, "`.")]
        #[derive(Clone, Debug, Default)]
        pub struct $name {
            /// Measured calibration standards.
            pub measured: Vec<Network>,
            /// Ideal standards aligned with [`Self::measured`].
            pub ideals: Vec<Network>,
            /// Solved calibration coefficients.
            pub coefficients: BTreeMap<String, Array1<Complex64>>,
        }
    };
}

calibration_structure!(
    OnePort,
    "skrf/calibration/calibration.py::OnePort",
    "Standard one-port calibration using three or more standards.\n\nThe solver determines directivity, source match, and reflection tracking from\n\n$$\ne_{11} i_n m_n - \\Delta e\\, i_{n} + e_{00} = m_{n}.\n$$\n\nWith more than three standards, it solves the overdetermined system by least squares.\n\n## References\n\n- [Agilent VNA Help: one-port calibration](http://na.tm.agilent.com/vnahelp/tip20.html)\n- R. F. Bauer Jr. and P. Penfield, \"De-Embedding and Unterminating,\" IEEE Transactions on Microwave Theory and Techniques, 1974, [doi:10.1109/TMTT.1974.1128212](https://doi.org/10.1109/TMTT.1974.1128212)."
);
calibration_structure!(
    SddlWeikle,
    "skrf/calibration/calibration.py::SDDLWeikle",
    "Liu-Weikle short-delay-delay-load one-port self-calibration.\n\nThe standards are a known short, two delay shorts with unknown phase, and a known\nreflective load. The short may have a known offset; the delay shorts have unit\nreflection magnitude and unknown electrical length. A perfectly matched load makes\nthe calibration singular.\n\n> **Note:** The method is bandwidth-limited by phase wrapping. Wideband use requires\nsplitting measurements into subbands with suitable delay lengths.\n\n## References\n\n- Z. Liu and R. M. Weikle, *A reflectometer calibration method resistant to waveguide flange misalignment*, IEEE Transactions on Microwave Theory and Techniques, 2006.\n- W. Sigg and J. Simon, *Reflectometer calibration using load, short and offset shorts with unknown phase*, Electronics Letters, 1991.\n- A. Lewandowski et al., *Accuracy and Bandwidth Optimization of the Over-Determined Offset-Short Reflectometer Calibration*, IEEE Transactions on Microwave Theory and Techniques, 2015, [doi:10.1109/TMTT.2015.2396496](https://doi.org/10.1109/TMTT.2015.2396496)."
);
calibration_structure!(
    Sddl,
    "skrf/calibration/calibration.py::SDDL",
    "Arsenovic short-delay-delay-load one-port self-calibration.\n\nThe standards are a known short, two unit-magnitude delay shorts with unknown phase,\nand a known load. The short may have a known offset. Unlike [`SddlWeikle`], the\nload may be matched or reflective.\n\n> **Note:** The method is bandwidth-limited by phase wrapping. Wideband use requires\nsplitting measurements into subbands with suitable delay lengths.\n\n## References\n\n- A. Arsenovic, R. M. Weikle, and J. L. Hesler, *Reflectometer calibration with a pair of partially known standards*, European Microwave Conference, 2015, [doi:10.1109/EUMC.2015.7345766](https://doi.org/10.1109/EUMC.2015.7345766).\n- Z. Liu and R. M. Weikle, *A reflectometer calibration method resistant to waveguide flange misalignment*, IEEE Transactions on Microwave Theory and Techniques, 2006.\n- W. Sigg and J. Simon, *Reflectometer calibration using load, short and offset shorts with unknown phase*, Electronics Letters, 1991.\n- A. Lewandowski et al., *Accuracy and Bandwidth Optimization of the Over-Determined Offset-Short Reflectometer Calibration*, IEEE Transactions on Microwave Theory and Techniques, 2015, [doi:10.1109/TMTT.2015.2396496](https://doi.org/10.1109/TMTT.2015.2396496)."
);
calibration_structure!(
    Phn,
    "skrf/calibration/calibration.py::PHN",
    "Pair-of-half-knowns one-port self-calibration.\n\nUses two fully known standards and two standards whose reflection magnitude is\nknown but whose phase is solved.\n\n> **Important:** A square-root sign ambiguity can make this method unstable for\narbitrary reflection coefficients, although it has proven reliable for rectangular\nwaveguide calibration.\n\n## Reference\n\nA. Arsenovic, R. M. Weikle, and J. L. Hesler, *Reflectometer calibration with a pair of partially known standards*, European Microwave Conference, 2015, [doi:10.1109/EUMC.2015.7345766](https://doi.org/10.1109/EUMC.2015.7345766)."
);
calibration_structure!(
    EightTerm,
    "skrf/calibration/calibration.py::EightTerm",
    "General eight-term, or error-box, two-port calibration.\n\nA least-squares estimator determines the error coefficients; no self-calibration\nis performed. Switched-source VNAs require switch terms to be unterminated before\napplying this model; use [`unterminate`] and [`terminate`] for that conversion.\n\n## References\n\n- R. A. Speciale, *A Generalization of the TSD Network-Analyzer Calibration Procedure, Covering n-Port Scattering-Parameter Measurements, Affected by Leakage Errors*, IEEE Transactions on Microwave Theory and Techniques, 1977.\n- D. Rytting, *Network Analyzer Error Models and Calibration Methods*, ARFTG/NIST Short Course Notes, 1996."
);
calibration_structure!(
    TwelveTerm,
    "skrf/calibration/calibration.py::TwelveTerm",
    "Traditional full twelve-term two-port calibration.\n\nIt accepts arbitrary, including non-flush, reflect and transmissive standards. More\nthan three reflect standards produce a least-squares one-port solution. Repeated\ntransmissive standards produce load-match and transmission-tracking estimates that\nare averaged.\n\n## Reference\n\nStig Rehnmark, *Calibration Process of Automatic Network Analyzer Systems*."
);
calibration_structure!(
    SixteenTerm,
    "skrf/calibration/calibration.py::SixteenTerm",
    "General sixteen-term two-port calibration for a leaky VNA.\n\nThe model includes crosstalk between all four receivers and requires at least five\ntwo-port measurements. Some thru, open, short, and load combinations are singular.\nOrdinary switch-term correction is not applicable when leakage is significant, so\nmeasurements are expected to have their switch termination handled beforehand.\n\n## Reference\n\nK. J. Silvonen, *Calibration of 16-term error model*, Electronics Letters, 1993."
);
calibration_structure!(
    Solt,
    "skrf/calibration/calibration.py::SOLT",
    "Short-open-load-thru full two-port calibration.\n\nDespite the name, the implementation accepts any number of reflect standards and\ncan use redundant thru measurements. More than three reflect standards produce a\nleast-squares one-port solution. It is the standard-specific form of [`TwelveTerm`].\n\n## Reference\n\nW. Kruppa and K. F. Sodomsky, *An Explicit Solution for the Scattering Parameters of a Linear Two-Port Measured with an Imperfect Test Set*, IEEE Transactions on Microwave Theory and Techniques, 1971."
);
/// Two-port one-path calibration for a switchless three-receiver system.
///
/// Full correction requires measuring the DUT in both orientations and passing
/// the forward and reverse measurements to [`Self::apply_pair`]. Applying the
/// calibration to only one network yields an enhanced-response partial correction.
///
/// Origin: `skrf/calibration/calibration.py::TwoPortOnePath`.
#[derive(Clone, Debug, Default)]
pub struct TwoPortOnePath {
    /// Measured calibration standards.
    pub measured: Vec<Network>,
    /// Ideal standards aligned with the measurements.
    pub ideals: Vec<Network>,
    /// Solved calibration coefficients.
    pub coefficients: BTreeMap<String, Array1<Complex64>>,
    /// Zero-based source-port selection.
    pub source_port: usize,
}

/// Enhanced-response partial two-port calibration.
///
/// Only the actively sourced path is fully corrected, so accuracy depends on a
/// good match at the passive DUT port. For full correction, measure both DUT
/// orientations and use [`TwoPortOnePath`].
///
/// Origin: `skrf/calibration/calibration.py::EnhancedResponse`.
#[derive(Clone, Debug, Default)]
pub struct EnhancedResponse {
    /// Measured calibration standards.
    pub measured: Vec<Network>,
    /// Ideal standards aligned with the measurements.
    pub ideals: Vec<Network>,
    /// Solved calibration coefficients.
    pub coefficients: BTreeMap<String, Array1<Complex64>>,
    /// Zero-based source-port selection.
    pub source_port: usize,
}
/// Thru-reflect-line self-calibration.
///
/// Standards are ordered as thru, one or more reflects, and one or more lines.
/// The calibration reference impedance is the characteristic impedance of the
/// line standards.
///
/// See [`determine_line`] and [`determine_reflect`] for the two standard-solving
/// stages.
///
/// ## References
///
/// - G. F. Engen and C. A. Hoer, *Thru-Reflect-Line: An Improved Technique for Calibrating the Dual Six-Port Automatic Network Analyzer*, IEEE Transactions on Microwave Theory and Techniques, 1979.
/// - H.-J. Eul and B. Schiek, *A generalized theory and new calibration procedures for network analyzer self-calibration*, IEEE Transactions on Microwave Theory and Techniques, 1991.
///
/// Origin: `skrf/calibration/calibration.py::TRL`.
#[derive(Clone, Debug, Default)]
pub struct Trl {
    /// Measured thru, reflect, and line standards.
    pub measured: Vec<Network>,
    /// Ideal thru, reflect, and line standards when known.
    pub ideals: Vec<Network>,
    /// Solved eight-term calibration coefficients.
    pub coefficients: BTreeMap<String, Array1<Complex64>>,
    /// Number of reflect standards.
    pub reflects: usize,
    /// Whether the line standard is estimated from its measurement.
    pub estimate_line: bool,
    /// Whether the reflect standard is solved rather than taken as exact.
    pub solve_reflect: bool,
}
macro_rules! multiline_trl_structure {
    ($name:ident, $origin:literal, $description:literal) => {
        #[doc = $description]
        ///
        #[doc = concat!("Origin: `", $origin, "`.")]
        #[derive(Clone, Debug, Default)]
        pub struct $name {
            /// Measured multiline TRL standards.
            pub measured: Vec<Network>,
            /// Ideal standards aligned with the measurements.
            pub ideals: Vec<Network>,
            /// Solved eight-term calibration coefficients.
            pub coefficients: BTreeMap<String, Array1<Complex64>>,
            /// Physical lengths of the measured line standards, in meters.
            pub line_lengths: Vec<f64>,
            /// Estimated reflection coefficients of the reflect standards.
            pub reflect_estimates: Vec<Complex64>,
            /// Initial effective-permittivity estimate; a negative imaginary part
            /// represents loss.
            pub effective_permittivity_estimate: Complex64,
            /// Solved complex propagation constant versus frequency,
            /// $\gamma = \alpha + j\beta$.
            pub propagation_constant: Option<Array1<Complex64>>,
        }
    };
}

multiline_trl_structure!(
    NistMultilineTrl,
    "skrf/calibration/calibration.py::NISTMultilineTRL",
    "NIST multiline TRL calibration.\n\nMultiple line standards extend calibration bandwidth and improve accuracy. At each\nfrequency, at least one line pair must have a phase difference that is neither\nzero nor a multiple of $180^\\circ$, otherwise the calibration system is singular.\nThe reference plane lies at the line edges and the default reference impedance is\nthe line characteristic impedance.\n\n## References\n\n- D. C. `DeGroot`, J. A. Jargon, and R. B. Marks, *Multiline TRL revealed*, 60th ARFTG Conference Digest, 2002.\n- K. Yau, *On the metrology of nanoscale Silicon transistors above 100 GHz*, Ph.D. dissertation, University of Toronto, 2011."
);
multiline_trl_structure!(
    TugMultilineTrl,
    "skrf/calibration/calibration.py::TUGMultilineTRL",
    "TUG multiline TRL calibration.\n\nSolves one weighted $4 \\times 4$ eigenvalue problem. The first measured line is\nthe thru and defines the reference plane. The default reference impedance is the\nline characteristic impedance. Without reflect data, only the transmission terms\ncan be calibrated accurately.\n\n## Example\n\n```rust,ignore\nlet mut calibration = TugMultilineTrl::new(\n    vec![thru, line_1, line_2],\n    vec![Complex64::new(-1.0, 0.0)],\n    vec![0.0, 1.0e-3, 5.0e-3],\n    Complex64::new(4.0, 0.0),\n)?;\ncalibration.run()?;\nlet calibrated = calibration.apply(&dut)?;\n```\n\n## References\n\n- Z. Hatab, M. Gadringer, and W. Bösch, *Improving The Reliability of The Multiline TRL Calibration Algorithm*, ARFTG, 2022, [doi:10.1109/ARFTG52954.2022.9844064](https://doi.org/10.1109/ARFTG52954.2022.9844064).\n- Z. Hatab, M. Gadringer, and W. Bösch, *Propagation of Linear Uncertainties Through Multiline Thru-Reflect-Line Calibration*, IEEE Transactions on Instrumentation and Measurement, 2023, [doi:10.1109/TIM.2023.3296123](https://doi.org/10.1109/TIM.2023.3296123).\n- [Multiline TRL calibration notes](https://ziadhatab.github.io/posts/multiline-trl-calibration/).\n- Z. Hatab, M. E. Gadringer, and W. Bösch, *A Thru-Free Multiline Calibration*, IEEE Transactions on Instrumentation and Measurement, 2023, [doi:10.1109/TIM.2023.3308226](https://doi.org/10.1109/TIM.2023.3308226).\n- Z. Hatab, M. E. Gadringer, and W. Bösch, *The Choice of Line Lengths in Multiline Thru-Reflect-Line Calibration*, [arXiv:2512.18641](https://arxiv.org/abs/2512.18641).\n\nSee also [`NistMultilineTrl`]."
);
calibration_structure!(
    UnknownThru,
    "skrf/calibration/calibration.py::UnknownThru",
    "Self-calibration using known reflect standards and a reciprocal but otherwise\nunknown thru. The approximate thru phase is used only to choose the square-root\nsign and therefore only needs to be known within $\\pi$. The solver uses the\n[`EightTerm`] error model.\n\n## Reference\n\nA. Ferrero and U. Pisani, *Two-port network analyzer calibration using an unknown thru*, IEEE Microwave and Guided Wave Letters, 1992."
);
calibration_structure!(
    Mrc,
    "skrf/calibration/calibration.py::MRC",
    "Misalignment-resistant waveguide calibration combining [`Sddl`] and\n[`UnknownThru`]. It uses a short, two delay shorts, a load, and a thru in that\norder. Solving the delay-short phases and reciprocal thru response makes the method\nresistant to waveguide-flange misalignment.\n\n## References\n\n- Z. Liu and R. M. Weikle, *A reflectometer calibration method resistant to waveguide flange misalignment*, IEEE Transactions on Microwave Theory and Techniques, 2006.\n- A. Ferrero and U. Pisani, *Two-port network analyzer calibration using an unknown thru*, IEEE Microwave and Guided Wave Letters, 1992."
);
/// Line-reflect-match self-calibration.
///
/// Uses fully known line and match standards with a reflect standard whose phase
/// is solved. The reflect phase must be known within $90^\circ$. Reflect and match
/// standards are assumed to be identical at both ports, and standards must be
/// supplied in line-reflect-match order.
///
/// ## Reference
///
/// W. Zhao et al., *A Unified Approach for Reformulations of LRM/LRMM/LRRM
/// Calibration Algorithms Based on the T-Matrix Representation*, Applied Sciences,
/// 2017.
///
/// Origin: `skrf/calibration/calibration.py::LRM`.
#[derive(Clone, Debug, Default)]
pub struct Lrm {
    /// Measured line, reflect, and match standards.
    pub measured: Vec<Network>,
    /// Ideal standards aligned with the measurements.
    pub ideals: Vec<Network>,
    /// Solved eight-term calibration coefficients.
    pub coefficients: BTreeMap<String, Array1<Complex64>>,
    /// Reflect standard solved by the LRM algorithm.
    pub solved_reflect: Option<Network>,
}
/// Model used to fit the LRRM match standard.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LrrmMatchFit {
    /// Solve the match independently at each frequency.
    #[default]
    PerFrequency,
    /// Fit a series-inductance model.
    Inductance,
    /// Fit a series-inductance and shunt-capacitance model.
    InductanceCapacitance,
}

/// Line-reflect-reflect-match self-calibration.
///
/// The two reflect standards need not be known, but must differ sufficiently in
/// phase. The first reflect magnitude may be unknown; the second magnitude must be
/// known, and both phases must be known within $90^\circ$. Reflects are assumed
/// identical at both ports. Only the first port of the match measurement is used.
/// The match may be solved independently at each frequency or fitted to an
/// inductance or inductance-capacitance model selected by [`LrrmMatchFit`].
///
/// ## References
///
/// - W. Zhao et al., *A Unified Approach for Reformulations of LRM/LRMM/LRRM Calibration Algorithms Based on the T-Matrix Representation*, Applied Sciences, 2017.
/// - F. Purroy and L. Pradell, *New theoretical analysis of the LRRM calibration technique for vector network analyzers*, IEEE Transactions on Instrumentation and Measurement, 2001.
/// - S. Liu et al., *An Improved Line-Reflect-Reflect-Match Calibration With an Enhanced Load Model*, IEEE Microwave and Wireless Components Letters, 2017.
///
/// Origin: `skrf/calibration/calibration.py::LRRM`.
#[derive(Clone, Debug, Default)]
pub struct Lrrm {
    /// Measured line, two reflect, and match standards.
    pub measured: Vec<Network>,
    /// Ideal standards aligned with the measurements.
    pub ideals: Vec<Network>,
    /// Solved eight-term calibration coefficients.
    pub coefficients: BTreeMap<String, Array1<Complex64>>,
    /// Reference impedance used by the match model.
    pub reference_impedance: f64,
    /// Selected match-standard fitting model.
    pub match_fit: LrrmMatchFit,
    /// Match standard solved by LRRM.
    pub solved_match: Option<Network>,
    /// First solved reflect standard.
    pub solved_reflect1: Option<Network>,
    /// Second solved reflect standard.
    pub solved_reflect2: Option<Network>,
    /// Solved series inductance versus frequency.
    pub solved_inductance: Option<Array1<f64>>,
    /// Solved shunt capacitance versus frequency.
    pub solved_capacitance: Option<Array1<f64>>,
}
/// Sixteen-term line-match-reflect self-calibration for a leaky VNA.
///
/// Measurements comprise thru, match-match, reflect-reflect, reflect-match, and
/// match-reflect standards. Either the thru or reflect response must be supplied;
/// the other is solved. The reflect must be highly reflective and consistent in
/// all measurements; the thru and match are assumed perfectly matched and the thru
/// lossless, although it may have nonzero length. The optional sign disambiguates
/// the quadratic solution. Switch termination must already have been handled.
///
/// ## Reference
///
/// K. Silvonen, *LMR 16-a self-calibration procedure for a leaky network analyzer*,
/// IEEE Transactions on Microwave Theory and Techniques, 1997.
///
/// Origin: `skrf/calibration/calibration.py::LMR16`.
#[derive(Clone, Debug, Default)]
pub struct Lmr16 {
    /// Measured line, match, and reflect standards.
    pub measured: Vec<Network>,
    /// Ideal standards aligned with the measurements.
    pub ideals: Vec<Network>,
    /// Solved sixteen-term calibration coefficients.
    pub coefficients: BTreeMap<String, Array1<Complex64>>,
    /// Whether the ideal standard supplied to the solver is the reflect standard.
    pub ideal_is_reflect: bool,
    /// Optional sign used to disambiguate the solution.
    pub sign: Option<f64>,
    /// Through standard solved by LMR16.
    pub solved_through: Option<Network>,
    /// Reflect standard solved by LMR16.
    pub solved_reflect: Option<Network>,
}

/// Two-port method used by [`MultiportCal`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MultiportPairMethod {
    /// Eight-term error model.
    #[default]
    EightTerm,
    /// Twelve-term error model.
    TwelveTerm,
}

/// A measured two-port calibration assigned to a pair of N-port indices.
///
/// Origin: `skrf/calibration/calibration.py::MultiportCal`.
#[derive(Clone, Debug)]
pub struct MultiportPairCalibration {
    /// Zero-based N-port indices calibrated by this pair.
    pub ports: [usize; 2],
    /// Measured two-port standards for the pair.
    pub measured: Vec<Network>,
    /// Ideal two-port standards for the pair.
    pub ideals: Vec<Network>,
    /// Error model used to solve the pair.
    pub method: MultiportPairMethod,
}

/// Multi-port calibration assembled from two-port calibrations sharing a common port.
///
/// Each [`MultiportPairCalibration`] solves one connected port pair. All pairs must
/// share at least one port so their error coefficients form a connected multi-port
/// calibration. Pair measurements may be two-port subnetworks extracted from a
/// larger measurement. An optional all-ports-matched network supplies isolation
/// terms; omit it to assume zero leakage.
///
/// Origin: `skrf/calibration/calibration.py::MultiportCal`.
#[derive(Clone, Debug, Default)]
pub struct MultiportCal {
    /// Two-port calibrations forming the multi-port calibration graph.
    pub pairs: Vec<MultiportPairCalibration>,
    /// Optional measured isolation network.
    pub isolation: Option<Network>,
    /// Solved one-port coefficients for each port.
    pub port_coefficients: Vec<BTreeMap<String, Array1<Complex64>>>,
    /// Solved transmission coefficients for each calibrated port pair.
    pub pair_coefficients: BTreeMap<(usize, usize), BTreeMap<String, Array1<Complex64>>>,
    /// Number of ports represented by the calibration.
    pub nports: usize,
}

/// SOLT convenience wrapper for [`MultiportCal`].
///
/// The measured standards begin with $N-1$ thru networks that connect every port
/// through a common port. Remaining standards are passed to the selected pairwise
/// calibration method. All standards are N-port networks. Use [`MultiportCal`]
/// directly for TRL-style methods whose line standards do not fit this interface.
///
/// Origin: `skrf/calibration/calibration.py::MultiportSOLT`.
#[derive(Clone, Debug)]
pub struct MultiportSolt {
    /// Underlying pairwise multi-port calibration.
    pub inner: MultiportCal,
    /// Measured SOLT standards.
    pub measured: Vec<Network>,
    /// Ideal SOLT standards.
    pub ideals: Vec<Network>,
    /// Port pairs associated with the measured thru standards.
    pub thru_ports: Vec<[usize; 2]>,
}
calibration_structure!(
    Normalization,
    "skrf/calibration/calibration.py::Normalization",
    "Simple thru normalization. Each measured scattering parameter is divided by\nthe average measured standard response; ideal standards are not used."
);

impl OnePort {
    /// Creates an unsolved one-port calibration from aligned measured and ideal
    /// standards. Three standards determine the model exactly; additional standards
    /// produce a least-squares solution.
    ///
    /// # Errors
    ///
    /// Returns an error when the measured and ideal standards are invalid or misaligned.
    pub fn new(measured: Vec<Network>, ideals: Vec<Network>) -> Result<Self> {
        validate_one_port_standards(&measured, &ideals)?;
        Ok(Self {
            measured,
            ideals,
            coefficients: BTreeMap::new(),
        })
    }
}

impl Calibration for OnePort {
    fn measured(&self) -> &[Network] {
        &self.measured
    }

    fn ideals(&self) -> &[Network] {
        &self.ideals
    }

    /// Solves directivity, source match, and reflection tracking by least squares.
    fn run(&mut self) -> Result<()> {
        validate_one_port_standards(&self.measured, &self.ideals)?;
        let points = self.measured[0].frequency_points();
        let mut directivity = Array1::zeros(points);
        let mut tracking = Array1::zeros(points);
        let mut source_match = Array1::zeros(points);
        for point in 0..points {
            let mut normal = [[Complex64::new(0.0, 0.0); 3]; 3];
            let mut right = [Complex64::new(0.0, 0.0); 3];
            for (measured, ideal) in self.measured.iter().zip(self.ideals.iter()) {
                let measured = measured.s[(point, 0, 0)];
                let ideal = ideal.s[(point, 0, 0)];
                let row = [ideal, Complex64::new(1.0, 0.0), ideal * measured];
                for column in 0..3 {
                    right[column] += row[column].conj() * measured;
                    for other in 0..3 {
                        normal[column][other] += row[column].conj() * row[other];
                    }
                }
            }
            let [a, b, c] = solve_three_by_three(normal, right).ok_or_else(|| {
                Error::Unsupported(
                    "one-port calibration standards form a singular least-squares system"
                        .to_owned(),
                )
            })?;
            directivity[point] = b;
            source_match[point] = c;
            tracking[point] = a + b * c;
        }
        self.coefficients
            .insert("directivity".to_owned(), directivity);
        self.coefficients
            .insert("reflection tracking".to_owned(), tracking);
        self.coefficients
            .insert("source match".to_owned(), source_match);
        Ok(())
    }

    /// Corrects a measured one-port reflection using the solved error coefficients.
    fn apply(&self, network: &Network) -> Result<Network> {
        validate_one_port_target(self, network)?;
        let directivity = coefficient(self, "directivity")?;
        let tracking = coefficient(self, "reflection tracking")?;
        let source_match = coefficient(self, "source match")?;
        let mut corrected = network.clone();
        for point in 0..network.frequency_points() {
            let measured = network.s[(point, 0, 0)];
            let numerator = measured - directivity[point];
            let denominator = tracking[point] + source_match[point] * numerator;
            if denominator.norm_sqr() <= f64::EPSILON {
                return Err(Error::Unsupported(
                    "one-port calibration correction has a zero denominator".to_owned(),
                ));
            }
            corrected.s[(point, 0, 0)] = numerator / denominator;
        }
        Ok(corrected)
    }

    /// Applies the solved one-port error model to an ideal reflection response.
    fn embed(&self, network: &Network) -> Result<Network> {
        validate_one_port_target(self, network)?;
        let directivity = coefficient(self, "directivity")?;
        let tracking = coefficient(self, "reflection tracking")?;
        let source_match = coefficient(self, "source match")?;
        let mut embedded = network.clone();
        for point in 0..network.frequency_points() {
            let ideal = network.s[(point, 0, 0)];
            let a = tracking[point] - directivity[point] * source_match[point];
            let denominator = Complex64::new(1.0, 0.0) - source_match[point] * ideal;
            if denominator.norm_sqr() <= f64::EPSILON {
                return Err(Error::Unsupported(
                    "one-port calibration embedding has a zero denominator".to_owned(),
                ));
            }
            embedded.s[(point, 0, 0)] = (directivity[point] + a * ideal) / denominator;
        }
        Ok(embedded)
    }

    fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
        &self.coefficients
    }
}

macro_rules! one_port_self_calibration {
    ($type:ty, $solver:ident, $constructor_doc:literal) => {
        impl $type {
            #[doc = $constructor_doc]
            #[doc = "\n\n# Errors\n\nReturns an error when the four measured and ideal standards are invalid or misaligned."]
            pub fn new(measured: Vec<Network>, ideals: Vec<Network>) -> Result<Self> {
                validate_four_one_port_standards(&measured, &ideals)?;
                Ok(Self {
                    measured,
                    ideals,
                    coefficients: BTreeMap::new(),
                })
            }
        }

        impl Calibration for $type {
            fn measured(&self) -> &[Network] {
                &self.measured
            }

            fn ideals(&self) -> &[Network] {
                &self.ideals
            }

            fn run(&mut self) -> Result<()> {
                $solver(self)?;
                Ok(())
            }

            fn apply(&self, network: &Network) -> Result<Network> {
                OnePort {
                    measured: self.measured.clone(),
                    ideals: self.ideals.clone(),
                    coefficients: self.coefficients.clone(),
                }
                .apply(network)
            }

            fn embed(&self, network: &Network) -> Result<Network> {
                OnePort {
                    measured: self.measured.clone(),
                    ideals: self.ideals.clone(),
                    coefficients: self.coefficients.clone(),
                }
                .embed(network)
            }

            fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
                &self.coefficients
            }
        }
    };
}

one_port_self_calibration!(
    Sddl,
    solve_sddl,
    "Creates an SDDL calibration. Standards must be ordered as short, first delay\nshort, second delay short, and load. The two ideal delay-short phases are solved."
);
one_port_self_calibration!(
    SddlWeikle,
    solve_sddl_weikle,
    "Creates a Liu-Weikle SDDL calibration. Standards must be ordered as short,\nfirst delay short, second delay short, and reflective load."
);
one_port_self_calibration!(
    Phn,
    solve_phn,
    "Creates a pair-of-half-knowns calibration. Standards must be ordered as the\ntwo half-known standards followed by the two fully known standards."
);

macro_rules! twelve_term_calibration {
    ($type:ty) => {
        impl $type {
            /// Creates an unsolved full two-port calibration from aligned measured
            /// and ideal standards.
            ///
            /// # Errors
            ///
            /// Returns an error when the measured and ideal standards are invalid or misaligned.
            pub fn new(measured: Vec<Network>, ideals: Vec<Network>) -> Result<Self> {
                validate_two_port_calibration_standards(&measured, &ideals)?;
                Ok(Self {
                    measured,
                    ideals,
                    coefficients: BTreeMap::new(),
                })
            }
        }

        impl Calibration for $type {
            fn measured(&self) -> &[Network] {
                &self.measured
            }

            fn ideals(&self) -> &[Network] {
                &self.ideals
            }

            fn run(&mut self) -> Result<()> {
                self.coefficients = solve_twelve_term(&self.measured, &self.ideals)?;
                Ok(())
            }

            fn apply(&self, network: &Network) -> Result<Network> {
                apply_twelve_term(&self.measured, &self.coefficients, network)
            }

            fn embed(&self, network: &Network) -> Result<Network> {
                embed_twelve_term(&self.measured, &self.coefficients, network)
            }

            fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
                &self.coefficients
            }
        }
    };
}

twelve_term_calibration!(TwelveTerm);
twelve_term_calibration!(Solt);

macro_rules! one_path_calibration {
    ($type:ty) => {
        impl $type {
            /// Creates a one-path calibration for `source_port`, using zero-based
            /// port indices. Full correction requires [`Self::apply_pair`].
            ///
            /// # Errors
            ///
            /// Returns an error when standards are invalid or `source_port` is not zero or one.
            pub fn new(
                measured: Vec<Network>,
                ideals: Vec<Network>,
                source_port: usize,
            ) -> Result<Self> {
                validate_two_port_calibration_standards(&measured, &ideals)?;
                if source_port > 1 {
                    return Err(Error::Unsupported(
                        "one-path calibration source port must be zero or one".to_owned(),
                    ));
                }
                let receive_port = 1 - source_port;
                let mut symmetric = measured;
                for network in &mut symmetric {
                    for point in 0..network.frequency_points() {
                        network.s[(point, source_port, receive_port)] =
                            network.s[(point, receive_port, source_port)];
                        network.s[(point, receive_port, receive_port)] =
                            network.s[(point, source_port, source_port)];
                    }
                }
                Ok(Self {
                    measured: symmetric,
                    ideals,
                    coefficients: BTreeMap::new(),
                    source_port,
                })
            }

            /// Ports the two-orientation correction performed by `TwoPortOnePath.apply_cal`.
            ///
            /// # Errors
            ///
            /// Returns an error when either orientation is incompatible with the calibration.
            pub fn apply_pair(&self, forward: &Network, reverse: &Network) -> Result<Network> {
                validate_two_port_target(&self.measured, forward)?;
                validate_two_port_target(&self.measured, reverse)?;
                let source = self.source_port;
                let receive = 1 - source;
                let mut composite = forward.clone();
                for point in 0..composite.frequency_points() {
                    composite.s[(point, source, source)] = forward.s[(point, source, source)];
                    composite.s[(point, receive, source)] = forward.s[(point, receive, source)];
                    composite.s[(point, receive, receive)] = reverse.s[(point, source, source)];
                    composite.s[(point, source, receive)] = reverse.s[(point, receive, source)];
                }
                apply_twelve_term(&self.measured, &self.coefficients, &composite)
            }
        }

        impl Calibration for $type {
            fn measured(&self) -> &[Network] {
                &self.measured
            }

            fn ideals(&self) -> &[Network] {
                &self.ideals
            }

            fn run(&mut self) -> Result<()> {
                let solved = solve_twelve_term(&self.measured, &self.ideals)?;
                self.coefficients = duplicate_one_path_coefficients(solved, self.source_port)?;
                Ok(())
            }

            fn apply(&self, network: &Network) -> Result<Network> {
                validate_two_port_target(&self.measured, network)?;
                let source = self.source_port;
                let receive = 1 - source;
                let mut partial = network.clone();
                for point in 0..partial.frequency_points() {
                    partial.s[(point, receive, receive)] = Complex64::new(0.0, 0.0);
                    partial.s[(point, source, receive)] = Complex64::new(0.0, 0.0);
                }
                let mut corrected =
                    apply_twelve_term(&self.measured, &self.coefficients, &partial)?;
                for point in 0..corrected.frequency_points() {
                    corrected.s[(point, receive, receive)] = Complex64::new(0.0, 0.0);
                    corrected.s[(point, source, receive)] = Complex64::new(0.0, 0.0);
                }
                Ok(corrected)
            }

            fn embed(&self, network: &Network) -> Result<Network> {
                embed_twelve_term(&self.measured, &self.coefficients, network)
            }

            fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
                &self.coefficients
            }
        }
    };
}

one_path_calibration!(TwoPortOnePath);
one_path_calibration!(EnhancedResponse);

impl EightTerm {
    /// Creates an unsolved eight-term calibration from aligned measured and ideal standards.
    ///
    /// # Errors
    ///
    /// Returns an error when the measured and ideal standards are invalid or misaligned.
    pub fn new(measured: Vec<Network>, ideals: Vec<Network>) -> Result<Self> {
        validate_eight_term_standards(&measured, &ideals)?;
        Ok(Self {
            measured,
            ideals,
            coefficients: BTreeMap::new(),
        })
    }
}

impl Calibration for EightTerm {
    fn measured(&self) -> &[Network] {
        &self.measured
    }

    fn ideals(&self) -> &[Network] {
        &self.ideals
    }

    fn run(&mut self) -> Result<()> {
        self.coefficients = solve_eight_term(&self.measured, &self.ideals)?;
        Ok(())
    }

    fn apply(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, false)
    }

    fn embed(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, true)
    }

    fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
        &self.coefficients
    }
}

impl Trl {
    /// Creates a TRL calibration with both `measured` and `ideals` ordered as thru,
    /// `reflects` reflect standards, and the remaining line standards. Multiple
    /// reflects and lines are supported. The implementation uses the [`EightTerm`]
    /// error model; switch-term effects must already be handled by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error when the standard ordering, count, or network shapes are invalid.
    pub fn new(measured: Vec<Network>, ideals: Vec<Network>, reflects: usize) -> Result<Self> {
        if reflects == 0 || measured.len() < reflects + 2 {
            return Err(Error::IncompatibleShape(
                "TRL requires a thru, at least one reflect, and at least one line".to_owned(),
            ));
        }
        validate_eight_term_standards(&measured, &ideals)?;
        Ok(Self {
            measured,
            ideals,
            coefficients: BTreeMap::new(),
            reflects,
            estimate_line: false,
            solve_reflect: true,
        })
    }

    /// Creates the conventional single-reflect TRL calibration.
    ///
    /// # Errors
    ///
    /// Returns an error when the standard ordering, count, or network shapes are invalid.
    pub fn single_reflect(measured: Vec<Network>, ideals: Vec<Network>) -> Result<Self> {
        Self::new(measured, ideals, 1)
    }
}

impl Calibration for Trl {
    fn measured(&self) -> &[Network] {
        &self.measured
    }

    fn ideals(&self) -> &[Network] {
        &self.ideals
    }

    fn run(&mut self) -> Result<()> {
        validate_eight_term_standards(&self.measured, &self.ideals)?;
        for index in self.reflects + 1..self.measured.len() {
            let approximation = (!self.estimate_line).then_some(&self.ideals[index]);
            self.ideals[index] =
                determine_line(&self.measured[0], &self.measured[index], approximation)?;
        }
        if self.solve_reflect {
            let line_index = self.measured.len() - 1;
            for index in 1..=self.reflects {
                let reflect = determine_reflect(
                    &self.measured[0],
                    &self.measured[index],
                    &self.measured[line_index],
                    Some(&self.ideals[index]),
                    Some(&self.ideals[line_index]),
                )?;
                self.ideals[index] = two_port_reflect_network(&reflect)?;
            }
        }
        self.coefficients = solve_eight_term(&self.measured, &self.ideals)?;
        Ok(())
    }

    fn apply(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, false)
    }

    fn embed(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, true)
    }

    fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
        &self.coefficients
    }
}

macro_rules! multiline_trl_calibration {
    ($type:ty, $name:literal) => {
        impl $type {
            /// Creates a multiline TRL calibration with measurements strictly ordered
            /// as thru, reflect standards, and line standards. `reflect_estimates`
            /// normally contains $-1$ for shorts or $+1$ for opens; `line_lengths`
            /// are in meters and include the thru length.
            ///
            /// # Errors
            ///
            /// Returns an error when the measurements or standard estimates are invalid.
            pub fn new(
                measured: Vec<Network>,
                reflect_estimates: Vec<Complex64>,
                line_lengths: Vec<f64>,
                effective_permittivity_estimate: Complex64,
            ) -> Result<Self> {
                validate_multiline_trl_inputs(&measured, &reflect_estimates, &line_lengths, $name)?;
                let ideals = multiline_trl_ideals(
                    &measured,
                    &reflect_estimates,
                    &line_lengths,
                    effective_permittivity_estimate,
                )?;
                Ok(Self {
                    measured,
                    ideals,
                    coefficients: BTreeMap::new(),
                    line_lengths,
                    reflect_estimates,
                    effective_permittivity_estimate,
                    propagation_constant: None,
                })
            }
        }

        impl Calibration for $type {
            fn measured(&self) -> &[Network] {
                &self.measured
            }

            fn ideals(&self) -> &[Network] {
                &self.ideals
            }

            fn run(&mut self) -> Result<()> {
                let mut calibration = Trl::new(
                    self.measured.clone(),
                    self.ideals.clone(),
                    self.reflect_estimates.len(),
                )?;
                calibration.run()?;
                self.propagation_constant = Some(multiline_propagation_constant(
                    &calibration.ideals,
                    self.reflect_estimates.len(),
                    &self.line_lengths,
                    self.effective_permittivity_estimate,
                )?);
                self.ideals = calibration.ideals;
                self.coefficients = calibration.coefficients;
                Ok(())
            }

            fn apply(&self, network: &Network) -> Result<Network> {
                transform_eight_term(&self.measured, &self.coefficients, network, false)
            }

            fn embed(&self, network: &Network) -> Result<Network> {
                transform_eight_term(&self.measured, &self.coefficients, network, true)
            }

            fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
                &self.coefficients
            }
        }
    };
}

multiline_trl_calibration!(NistMultilineTrl, "NIST multiline TRL");
multiline_trl_calibration!(TugMultilineTrl, "TUG multiline TRL");

fn validate_multiline_trl_inputs(
    measured: &[Network],
    reflect_estimates: &[Complex64],
    line_lengths: &[f64],
    name: &str,
) -> Result<()> {
    if reflect_estimates.is_empty() || line_lengths.len() < 2 {
        return Err(Error::IncompatibleShape(format!(
            "{name} requires at least one reflect and two line measurements"
        )));
    }
    if measured.len() != reflect_estimates.len() + line_lengths.len() {
        return Err(Error::IncompatibleShape(format!(
            "{name} measurement count must equal reflect count plus line count"
        )));
    }
    if line_lengths.iter().any(|length| !length.is_finite()) {
        return Err(Error::Unsupported(format!(
            "{name} line lengths must be finite"
        )));
    }
    let frequency = &measured[0].frequency;
    if measured
        .iter()
        .any(|network| network.ports() != 2 || network.frequency != *frequency)
    {
        return Err(Error::IncompatibleShape(format!(
            "{name} requires frequency-compatible two-port measurements"
        )));
    }
    Ok(())
}

fn multiline_trl_ideals(
    measured: &[Network],
    reflect_estimates: &[Complex64],
    line_lengths: &[f64],
    effective_permittivity_estimate: Complex64,
) -> Result<Vec<Network>> {
    let frequency = &measured[0].frequency;
    let points = frequency.points();
    let line = |length: f64| {
        Network::new(
            frequency.clone(),
            Array3::from_shape_fn((points, 2, 2), |(point, row, column)| {
                if row == column {
                    Complex64::new(0.0, 0.0)
                } else {
                    let angular = 2.0 * std::f64::consts::PI * frequency.values_hz()[point];
                    let gamma = multiline_gamma_estimate(angular, effective_permittivity_estimate);
                    (-gamma * length).exp()
                }
            }),
            measured[0].z0.clone(),
        )
    };
    let mut ideals = Vec::with_capacity(measured.len());
    ideals.push(line(line_lengths[0])?);
    for estimate in reflect_estimates {
        ideals.push(Network::new(
            frequency.clone(),
            Array3::from_shape_fn((points, 2, 2), |(_, row, column)| {
                if row == column {
                    *estimate
                } else {
                    Complex64::new(0.0, 0.0)
                }
            }),
            measured[0].z0.clone(),
        )?);
    }
    for length in &line_lengths[1..] {
        ideals.push(line(*length)?);
    }
    Ok(ideals)
}

fn multiline_propagation_constant(
    ideals: &[Network],
    reflect_count: usize,
    line_lengths: &[f64],
    effective_permittivity_estimate: Complex64,
) -> Result<Array1<Complex64>> {
    let points = ideals[0].frequency_points();
    let mut propagation = Array1::zeros(points);
    for point in 0..points {
        let angular = 2.0 * std::f64::consts::PI * ideals[0].frequency.values_hz()[point];
        let estimate = multiline_gamma_estimate(angular, effective_permittivity_estimate);
        let thru = ideals[0].s[(point, 1, 0)];
        ensure_nonzero(thru, "multiline TRL thru transmission is zero")?;
        let mut candidates = Vec::with_capacity(line_lengths.len() - 1);
        for (line, length) in line_lengths.iter().enumerate().skip(1) {
            let delta = length - line_lengths[0];
            if delta.abs() <= f64::EPSILON {
                continue;
            }
            let transmission = ideals[reflect_count + line].s[(point, 1, 0)];
            ensure_nonzero(transmission, "multiline TRL line transmission is zero")?;
            let mut gamma = -(transmission / thru).ln() / delta;
            let periods = ((estimate.im - gamma.im) * delta / (2.0 * std::f64::consts::PI)).round();
            gamma += Complex64::new(0.0, 2.0 * std::f64::consts::PI * periods / delta);
            candidates.push(gamma);
        }
        if candidates.is_empty() {
            return Err(Error::Unsupported(
                "multiline TRL needs at least one distinct line length".to_owned(),
            ));
        }
        let candidate_count = u32::try_from(candidates.len()).map_err(|_| {
            Error::Unsupported("multiline TRL has too many line candidates".to_owned())
        })?;
        propagation[point] =
            candidates.iter().copied().sum::<Complex64>() / f64::from(candidate_count);
    }
    Ok(propagation)
}

fn multiline_gamma_estimate(
    angular_frequency: f64,
    effective_permittivity: Complex64,
) -> Complex64 {
    let mut root = (-effective_permittivity).sqrt();
    if root.im < 0.0 || (root.im == 0.0 && root.re < 0.0) {
        root = -root;
    }
    angular_frequency / 299_792_458.0 * root
}

impl SixteenTerm {
    /// Creates an unsolved sixteen-term calibration from aligned measured and ideal
    /// standards. Switch termination must already be corrected because the ordinary
    /// switch-term equations are invalid when crosstalk is significant.
    ///
    /// # Errors
    ///
    /// Returns an error when the measured and ideal standards are invalid or misaligned.
    pub fn new(measured: Vec<Network>, ideals: Vec<Network>) -> Result<Self> {
        validate_sixteen_term_standards(&measured, &ideals)?;
        Ok(Self {
            measured,
            ideals,
            coefficients: BTreeMap::new(),
        })
    }
}

impl Calibration for SixteenTerm {
    fn measured(&self) -> &[Network] {
        &self.measured
    }

    fn ideals(&self) -> &[Network] {
        &self.ideals
    }

    fn run(&mut self) -> Result<()> {
        self.coefficients = solve_sixteen_term(&self.measured, &self.ideals)?;
        Ok(())
    }

    fn apply(&self, network: &Network) -> Result<Network> {
        transform_sixteen_term(&self.measured, &self.coefficients, network, false)
    }

    fn embed(&self, network: &Network) -> Result<Network> {
        transform_sixteen_term(&self.measured, &self.coefficients, network, true)
    }

    fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
        &self.coefficients
    }
}

impl UnknownThru {
    /// Creates an unknown-thru calibration from aligned measured and ideal standards.
    /// The reciprocal thru must be last; its approximate transmission phase is used
    /// only to choose a square-root sign and need only be known within $\pi$.
    ///
    /// # Errors
    ///
    /// Returns an error when the standards are invalid, misaligned, or incorrectly ordered.
    pub fn new(measured: Vec<Network>, ideals: Vec<Network>) -> Result<Self> {
        validate_unknown_thru_standards(&measured, &ideals)?;
        Ok(Self {
            measured,
            ideals,
            coefficients: BTreeMap::new(),
        })
    }
}

impl Calibration for UnknownThru {
    fn measured(&self) -> &[Network] {
        &self.measured
    }

    fn ideals(&self) -> &[Network] {
        &self.ideals
    }

    fn run(&mut self) -> Result<()> {
        self.coefficients = solve_unknown_thru(&self.measured, &self.ideals)?;
        Ok(())
    }

    fn apply(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, false)
    }

    fn embed(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, true)
    }

    fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
        &self.coefficients
    }
}

impl Mrc {
    /// Creates a misalignment-resistant calibration from exactly five standards,
    /// ordered as short, first delay short, second delay short, load, and thru.
    /// The ideal delay-short phases are solved, and the approximate thru phase need
    /// only be known within $\pi$.
    ///
    /// # Errors
    ///
    /// Returns an error unless exactly five compatible measured and ideal standards are provided.
    pub fn new(measured: Vec<Network>, ideals: Vec<Network>) -> Result<Self> {
        if measured.len() != 5 || ideals.len() != 5 {
            return Err(Error::IncompatibleShape(
                "MRC requires exactly four reflective standards and a final thru".to_owned(),
            ));
        }
        validate_unknown_thru_standards(&measured, &ideals)?;
        Ok(Self {
            measured,
            ideals,
            coefficients: BTreeMap::new(),
        })
    }
}

impl Calibration for Mrc {
    fn measured(&self) -> &[Network] {
        &self.measured
    }

    fn ideals(&self) -> &[Network] {
        &self.ideals
    }

    fn run(&mut self) -> Result<()> {
        self.coefficients = solve_mrc(&self.measured, &mut self.ideals)?;
        Ok(())
    }

    fn apply(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, false)
    }

    fn embed(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, true)
    }

    fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
        &self.coefficients
    }
}

impl Lrm {
    /// Creates a line-reflect-match calibration with standards ordered as line,
    /// reflect, and match.
    ///
    /// # Errors
    ///
    /// Returns an error unless exactly three compatible measured and ideal standards are provided.
    pub fn new(measured: Vec<Network>, ideals: Vec<Network>) -> Result<Self> {
        validate_named_standard_count(&measured, &ideals, 3, "LRM")?;
        Ok(Self {
            measured,
            ideals,
            coefficients: BTreeMap::new(),
            solved_reflect: None,
        })
    }
}

impl Calibration for Lrm {
    fn measured(&self) -> &[Network] {
        &self.measured
    }

    fn ideals(&self) -> &[Network] {
        &self.ideals
    }

    fn run(&mut self) -> Result<()> {
        let (coefficients, solved_reflect) = solve_lrm(&self.measured, &self.ideals)?;
        self.coefficients = coefficients;
        self.solved_reflect = Some(solved_reflect);
        Ok(())
    }

    fn apply(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, false)
    }

    fn embed(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, true)
    }

    fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
        &self.coefficients
    }
}

impl Lrrm {
    /// Creates a line-reflect-reflect-match calibration and selects the match fitting
    /// model. Standards must be ordered as line, first reflect, second reflect, and match.
    ///
    /// # Errors
    ///
    /// Returns an error when the standards or reference impedance are invalid.
    pub fn new(
        measured: Vec<Network>,
        ideals: Vec<Network>,
        reference_impedance: f64,
        match_fit: LrrmMatchFit,
    ) -> Result<Self> {
        validate_named_standard_count(&measured, &ideals, 4, "LRRM")?;
        if !reference_impedance.is_finite() || reference_impedance <= 0.0 {
            return Err(Error::Unsupported(
                "LRRM reference impedance must be finite and positive".to_owned(),
            ));
        }
        Ok(Self {
            measured,
            ideals,
            coefficients: BTreeMap::new(),
            reference_impedance,
            match_fit,
            solved_match: None,
            solved_reflect1: None,
            solved_reflect2: None,
            solved_inductance: None,
            solved_capacitance: None,
        })
    }
}

impl Calibration for Lrrm {
    fn measured(&self) -> &[Network] {
        &self.measured
    }

    fn ideals(&self) -> &[Network] {
        &self.ideals
    }

    fn run(&mut self) -> Result<()> {
        let solved = solve_lrrm(
            &self.measured,
            &self.ideals,
            self.reference_impedance,
            self.match_fit,
        )?;
        self.coefficients = solved.coefficients;
        self.solved_match = Some(solved.matched);
        self.solved_reflect1 = Some(solved.reflect1);
        self.solved_reflect2 = Some(solved.reflect2);
        self.solved_inductance = Some(solved.inductance);
        self.solved_capacitance = Some(solved.capacitance);
        Ok(())
    }

    fn apply(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, false)
    }

    fn embed(&self, network: &Network) -> Result<Network> {
        transform_eight_term(&self.measured, &self.coefficients, network, true)
    }

    fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
        &self.coefficients
    }
}

impl Lmr16 {
    /// Creates a sixteen-term line-match-reflect calibration. `sign` may be $+1$
    /// or $-1$ to choose the quadratic root. When omitted, the solver chooses the
    /// sign that makes $k = \frac{t_{15}}{t_{12}}$ closest to $+1$, as expected
    /// for a symmetric fixture.
    ///
    /// # Errors
    ///
    /// Returns an error when the measurement count, frequency axes, or ideal are invalid.
    pub fn new(
        measured: Vec<Network>,
        ideal: Network,
        ideal_is_reflect: bool,
        sign: Option<f64>,
    ) -> Result<Self> {
        if measured.len() != 5 {
            return Err(Error::IncompatibleShape(
                "LMR16 requires through, match-match, reflect-reflect, reflect-match, and match-reflect measurements"
                    .to_owned(),
            ));
        }
        let frequency = &measured[0].frequency;
        if measured
            .iter()
            .any(|network| network.ports() != 2 || network.frequency != *frequency)
            || ideal.frequency != *frequency
            || (ideal_is_reflect && ideal.ports() != 1)
            || (!ideal_is_reflect && ideal.ports() != 2)
        {
            return Err(Error::IncompatibleShape(
                "LMR16 measurements and ideal must have compatible frequencies and ports"
                    .to_owned(),
            ));
        }
        if sign.is_some_and(|value| {
            value.to_bits() != 1.0f64.to_bits() && value.to_bits() != (-1.0f64).to_bits()
        }) {
            return Err(Error::Unsupported(
                "LMR16 root sign must be +1, -1, or automatic".to_owned(),
            ));
        }
        Ok(Self {
            measured,
            ideals: vec![ideal],
            coefficients: BTreeMap::new(),
            ideal_is_reflect,
            sign,
            solved_through: None,
            solved_reflect: None,
        })
    }
}

impl Calibration for Lmr16 {
    fn measured(&self) -> &[Network] {
        &self.measured
    }

    fn ideals(&self) -> &[Network] {
        &self.ideals
    }

    fn run(&mut self) -> Result<()> {
        let (coefficients, through, reflect) = solve_lmr16(
            &self.measured,
            &self.ideals[0],
            self.ideal_is_reflect,
            self.sign,
        )?;
        self.coefficients = coefficients;
        self.solved_through = Some(through);
        self.solved_reflect = Some(reflect);
        Ok(())
    }

    fn apply(&self, network: &Network) -> Result<Network> {
        transform_sixteen_term(&self.measured, &self.coefficients, network, false)
    }

    fn embed(&self, network: &Network) -> Result<Network> {
        transform_sixteen_term(&self.measured, &self.coefficients, network, true)
    }

    fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
        &self.coefficients
    }
}

fn solve_sddl(calibration: &mut Sddl) -> Result<()> {
    validate_four_one_port_standards(&calibration.measured, &calibration.ideals)?;
    let points = calibration.measured[0].frequency_points();
    for point in 0..points {
        let d = normalized_impedance(calibration.measured[0].s[(point, 0, 0)])?;
        let a = normalized_impedance(calibration.measured[1].s[(point, 0, 0)])?;
        let b = normalized_impedance(calibration.measured[2].s[(point, 0, 0)])?;
        let c = normalized_impedance(calibration.measured[3].s[(point, 0, 0)])?;
        let load = normalized_impedance(calibration.ideals[3].s[(point, 0, 0)])?;
        let alpha_ratio = cross_ratio(b, a, c, d)?;
        let beta_ratio = cross_ratio(a, b, c, d)?;
        let alpha_denominator = (alpha_ratio / load).re;
        let beta_denominator = (beta_ratio / load).re;
        if alpha_denominator.abs() <= f64::EPSILON || beta_denominator.abs() <= f64::EPSILON {
            return Err(Error::Unsupported(
                "SDDL standards produce a singular cross-ratio solution".to_owned(),
            ));
        }
        let alpha = Complex64::new(0.0, alpha_ratio.im / alpha_denominator);
        let beta = Complex64::new(0.0, beta_ratio.im / beta_denominator);
        calibration.ideals[1].s[(point, 0, 0)] = normalized_reflection(alpha)?;
        calibration.ideals[2].s[(point, 0, 0)] = normalized_reflection(beta)?;
    }
    solve_updated_one_port(
        &calibration.measured,
        &calibration.ideals,
        &mut calibration.coefficients,
    )
}

fn solve_sddl_weikle(calibration: &mut SddlWeikle) -> Result<()> {
    validate_four_one_port_standards(&calibration.measured, &calibration.ideals)?;
    let points = calibration.measured[0].frequency_points();
    let mut directivity = Array1::zeros(points);
    let mut tracking = Array1::zeros(points);
    let mut source_match = Array1::zeros(points);
    for point in 0..points {
        let short = calibration.measured[0].s[(point, 0, 0)];
        let delay1 = calibration.measured[1].s[(point, 0, 0)];
        let delay2 = calibration.measured[2].s[(point, 0, 0)];
        let load_measured = calibration.measured[3].s[(point, 0, 0)];
        let mut load = calibration.ideals[3].s[(point, 0, 0)];
        if load.norm_sqr() <= f64::EPSILON {
            load = Complex64::new(1.0e-12, 0.0);
        }
        let delay1_prime = delay1 - short;
        let delay2_prime = delay2 - short;
        let load_prime = load_measured - short;
        ensure_nonzero(delay1_prime, "SDDLWeikle first delay equals the short")?;
        ensure_nonzero(delay2_prime, "SDDLWeikle second delay equals the short")?;
        ensure_nonzero(load_prime, "SDDLWeikle load equals the short")?;
        let phase_argument =
            Complex64::new(1.0, 0.0) / delay2_prime - Complex64::new(1.0, 0.0) / delay1_prime;
        let alpha = Complex64::from_polar(1.0, 2.0 * phase_argument.arg());
        let p_denominator = Complex64::new(1.0, 0.0) / delay1_prime
            - alpha / delay1_prime.conj()
            - (Complex64::new(1.0, 0.0) + load) / (load * load_prime);
        ensure_nonzero(p_denominator, "SDDLWeikle p denominator is zero")?;
        let equation_p = alpha / p_denominator;
        ensure_nonzero(alpha * load, "SDDLWeikle reflective load is required")?;
        let equation_q = equation_p / (alpha * load);
        let real_q_minus_p = (equation_q - equation_p).re;
        let real_p_plus_q = (equation_p + equation_q).re;
        if real_q_minus_p.abs() <= f64::EPSILON || real_p_plus_q.abs() <= f64::EPSILON {
            return Err(Error::Unsupported(
                "SDDLWeikle standards produce a singular real-valued solution".to_owned(),
            ));
        }
        let first_ratio = (equation_p + equation_q).im / real_q_minus_p;
        let second_ratio = (equation_q - equation_p).im / real_p_plus_q;
        let b_prime_real = -((1.0 + first_ratio * second_ratio)
            / first_ratio.mul_add(first_ratio, 1.0))
            * real_p_plus_q;
        let b_prime = Complex64::new(
            b_prime_real,
            (equation_q + equation_p).im / real_q_minus_p * b_prime_real,
        );
        ensure_nonzero(b_prime.conj(), "SDDLWeikle B coefficient is zero")?;
        let coefficient_b = b_prime + short;
        let coefficient_c = b_prime
            * (Complex64::new(1.0, 0.0) / delay1_prime - alpha / delay1_prime.conj())
            + alpha * b_prime / b_prime.conj();
        let coefficient_a = coefficient_b - short + short * coefficient_c;
        directivity[point] = coefficient_b;
        source_match[point] = -coefficient_c;
        tracking[point] = coefficient_a + coefficient_b * source_match[point];
    }
    calibration.coefficients = BTreeMap::from([
        ("directivity".to_owned(), directivity),
        ("reflection tracking".to_owned(), tracking),
        ("source match".to_owned(), source_match),
    ]);
    Ok(())
}

fn solve_phn(calibration: &mut Phn) -> Result<()> {
    validate_four_one_port_standards(&calibration.measured, &calibration.ideals)?;
    let points = calibration.measured[0].frequency_points();
    for point in 0..points {
        let ideal_a = normalized_impedance(calibration.ideals[0].s[(point, 0, 0)])?;
        let ideal_b = normalized_impedance(calibration.ideals[1].s[(point, 0, 0)])?;
        let ideal_c = normalized_impedance(calibration.ideals[2].s[(point, 0, 0)])?;
        let ideal_d = normalized_impedance(calibration.ideals[3].s[(point, 0, 0)])?;
        let measured = (0..4)
            .map(|index| normalized_impedance(calibration.measured[index].s[(point, 0, 0)]))
            .collect::<Result<Vec<_>>>()?;
        let ratio = cross_ratio(measured[0], measured[1], measured[2], measured[3])?;
        let e = ideal_c - ideal_d - ideal_c * ratio;
        let f = ideal_d - ideal_c - ideal_d * ratio;
        let g = ideal_c * ideal_d * ratio;
        let quadratic_a = Complex64::new(-(f * ratio.conj()).re, 0.0);
        let quadratic_b = Complex64::new(0.0, (f * e.conj() + g.conj() * ratio).im);
        let quadratic_c = Complex64::new((g * e.conj()).re, 0.0);
        let (root1, root2) = quadratic_roots(quadratic_a, quadratic_b, quadratic_c)?;
        let found = <[Complex64; 2]>::from((root1, root2))
            .map(|found_b| {
                let denominator = ratio * found_b + e;
                ensure_nonzero(denominator, "PHN half-known solution is singular")?;
                let found_a = -(f * found_b + g) / denominator;
                let distance = (normalized_reflection(found_a)? - normalized_reflection(ideal_a)?)
                    .norm()
                    + (normalized_reflection(found_b)? - normalized_reflection(ideal_b)?).norm();
                Ok((distance, found_a, found_b))
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let (_, found_a, found_b) = if found[0].0 < found[1].0 {
            found[0]
        } else {
            found[1]
        };
        calibration.ideals[0].s[(point, 0, 0)] = normalized_reflection(found_a)?;
        calibration.ideals[1].s[(point, 0, 0)] = normalized_reflection(found_b)?;
    }
    solve_updated_one_port(
        &calibration.measured,
        &calibration.ideals,
        &mut calibration.coefficients,
    )
}

fn solve_updated_one_port(
    measured: &[Network],
    ideals: &[Network],
    coefficients: &mut BTreeMap<String, Array1<Complex64>>,
) -> Result<()> {
    let mut calibration = OnePort::new(measured.to_vec(), ideals.to_vec())?;
    calibration.run()?;
    *coefficients = calibration.coefficients;
    Ok(())
}

fn normalized_impedance(reflection: Complex64) -> Result<Complex64> {
    let denominator = Complex64::new(1.0, 0.0) - reflection;
    ensure_nonzero(
        denominator,
        "reflection maps to infinite normalized impedance",
    )?;
    Ok((Complex64::new(1.0, 0.0) + reflection) / denominator)
}

fn normalized_reflection(impedance: Complex64) -> Result<Complex64> {
    let denominator = impedance + Complex64::new(1.0, 0.0);
    ensure_nonzero(
        denominator,
        "normalized impedance maps to infinite reflection",
    )?;
    Ok((impedance - Complex64::new(1.0, 0.0)) / denominator)
}

fn cross_ratio(a: Complex64, b: Complex64, c: Complex64, d: Complex64) -> Result<Complex64> {
    let denominator = (a - d) * (c - b);
    ensure_nonzero(denominator, "cross-ratio denominator is zero")?;
    Ok((a - b) * (c - d) / denominator)
}

fn quadratic_roots(a: Complex64, b: Complex64, c: Complex64) -> Result<(Complex64, Complex64)> {
    if a.norm_sqr() <= f64::EPSILON {
        ensure_nonzero(b, "quadratic has no finite roots")?;
        let root = -c / b;
        return Ok((root, root));
    }
    let discriminant = (b * b - 4.0 * a * c).sqrt();
    let denominator = 2.0 * a;
    Ok((
        (-b + discriminant) / denominator,
        (-b - discriminant) / denominator,
    ))
}

/// Determines the matched-line response from measured thru and line standards.
///
/// Combining the two measurements eliminates one error box and produces similar
/// matrices:
///
/// $$
/// \begin{aligned}
/// M_{t} &= X A_{t} Y, \\
/// M_{l} &= X A_{l} Y, \\
/// M_{t} M_l^{-1} &= X A_{t} A_l^{-1} X^{-1}.
/// \end{aligned}
/// $$
///
/// Their eigenvalues determine the line's $S_{21}$. `line_approximation` selects
/// the physically correct root; if it is omitted, the measured line-to-thru
/// transmission ratio is used. The measurements must already have switch terms
/// unterminated for the eight-term relationship to hold.
///
/// # Errors
///
/// Returns an error when the networks are incompatible or the line solution is singular.
///
/// Origin: `skrf.calibration/calibration.py::determine_line`.
pub fn determine_line(
    thru_measured: &Network,
    line_measured: &Network,
    line_approximation: Option<&Network>,
) -> Result<Network> {
    validate_two_networks(thru_measured, line_measured, "line determination")?;
    if let Some(approximation) = line_approximation {
        validate_two_networks(thru_measured, approximation, "line approximation")?;
    }
    let relative = thru_measured.inverse()?.cascade(line_measured)?;
    let transfer = relative.scattering_transfer()?;
    let mut found = line_measured.clone();
    for point in 0..found.frequency_points() {
        let trace = transfer[(point, 0, 0)] + transfer[(point, 1, 1)];
        let determinant = transfer[(point, 0, 0)] * transfer[(point, 1, 1)]
            - transfer[(point, 0, 1)] * transfer[(point, 1, 0)];
        let discriminant = (trace * trace - 4.0 * determinant).sqrt();
        let candidates = [(trace + discriminant) / 2.0, (trace - discriminant) / 2.0];
        let approximation = if let Some(network) = line_approximation {
            network.s[(point, 1, 0)]
        } else {
            ensure_nonzero(
                thru_measured.s[(point, 1, 0)],
                "thru transmission is zero during line estimation",
            )?;
            line_measured.s[(point, 1, 0)] / thru_measured.s[(point, 1, 0)]
        };
        let transmission =
            if (candidates[0] - approximation).norm() <= (candidates[1] - approximation).norm() {
                candidates[0]
            } else {
                candidates[1]
            };
        found.s[(point, 0, 0)] = Complex64::new(0.0, 0.0);
        found.s[(point, 1, 1)] = Complex64::new(0.0, 0.0);
        found.s[(point, 0, 1)] = transmission;
        found.s[(point, 1, 0)] = transmission;
    }
    Ok(found)
}

/// Determines a reflect standard from measured thru, reflect, and line standards.
///
/// This is the reflect-solving stage of TRL. `line_approximation` selects the
/// line root and `reflect_approximation` selects between the two possible reflect
/// solutions. A flush short is used as the reflect approximation when none is
/// supplied.
///
/// # Errors
///
/// Returns an error when the networks are incompatible or the reflect solution is singular.
///
/// Origin: `skrf/calibration/calibration.py::determine_reflect`.
pub fn determine_reflect(
    thru_measured: &Network,
    reflect_measured: &Network,
    line_measured: &Network,
    reflect_approximation: Option<&Network>,
    line_approximation: Option<&Network>,
) -> Result<Network> {
    validate_two_networks(thru_measured, reflect_measured, "reflect determination")?;
    validate_two_networks(thru_measured, line_measured, "reflect determination")?;
    let mut thru = thru_measured.clone();
    for point in 0..thru.frequency_points() {
        thru.s[(point, 0, 0)] = regularize(thru.s[(point, 0, 0)], 1.0e-7);
        thru.s[(point, 1, 1)] = regularize(thru.s[(point, 1, 1)], 1.0e-7);
    }
    let line = determine_line(&thru, line_measured, line_approximation)?;
    let thru_transfer = thru.scattering_transfer()?;
    let line_transfer = line_measured.scattering_transfer()?;
    let mut reflection = Array1::zeros(thru.frequency_points());
    for point in 0..thru.frequency_points() {
        let thru_matrix = matrix_from_array(&thru_transfer, point);
        let line_matrix = matrix_from_array(&line_transfer, point);
        let relation = matrix_multiply(line_matrix, matrix_inverse(thru_matrix)?);
        let quadratic_a = relation[1][0];
        let quadratic_b = relation[1][1] - relation[0][0];
        let quadratic_c = -relation[0][1];
        let (solution1, solution2) = quadratic_roots(quadratic_a, quadratic_b, quadratic_c)?;
        ensure_nonzero(solution1, "TRL reflect first root is zero")?;
        ensure_nonzero(solution2, "TRL reflect second root is zero")?;
        let denominator1 = relation[0][1] / solution2 + relation[0][0];
        let denominator2 = relation[0][1] / solution1 + relation[0][0];
        ensure_nonzero(denominator1, "TRL reflect first comparison is singular")?;
        ensure_nonzero(denominator2, "TRL reflect second comparison is singular")?;
        let x1 = (relation[1][0] * solution1 + relation[1][1]) / denominator1;
        let x2 = (relation[1][0] * solution2 + relation[1][1]) / denominator2;
        let expected = line.s[(point, 0, 1)].powu(2);
        let first_is_closer = (x1 - expected).norm() < (x2 - expected).norm();
        let selected_root = if first_is_closer {
            solution1
        } else {
            solution2
        };
        let alternate_root = if first_is_closer {
            solution2
        } else {
            solution1
        };
        ensure_nonzero(selected_root, "TRL reflect selected root is zero")?;
        let thru_s11 = thru.s[(point, 0, 0)];
        let determinant = thru.s[(point, 0, 0)] * thru.s[(point, 1, 1)]
            - thru.s[(point, 0, 1)] * thru.s[(point, 1, 0)];
        let negated_determinant = -determinant;
        let negated_thru_s22 = -thru.s[(point, 1, 1)];
        let gamma_denominator = Complex64::new(1.0, 0.0) - thru_s11 / selected_root;
        ensure_nonzero(gamma_denominator, "TRL reflect gamma is singular")?;
        let gamma = (negated_thru_s22 - negated_determinant / selected_root) / gamma_denominator;
        let beta_denominator = negated_determinant - alternate_root * negated_thru_s22;
        ensure_nonzero(beta_denominator, "TRL reflect beta/alpha is singular")?;
        let beta_over_alpha = (thru_s11 - alternate_root) / beta_denominator;
        let w1 = reflect_measured.s[(point, 0, 0)];
        let w2 = reflect_measured.s[(point, 1, 1)];
        let alpha_denominator =
            (w2 + gamma) * (Complex64::new(1.0, 0.0) - w1 / selected_root) * gamma_denominator;
        ensure_nonzero(alpha_denominator, "TRL reflect alpha is singular")?;
        let alpha = (((w1 - alternate_root)
            * (Complex64::new(1.0, 0.0) + w2 * beta_over_alpha)
            * beta_denominator)
            / alpha_denominator)
            .sqrt();
        let output_denominator = alpha * (Complex64::new(1.0, 0.0) - w1 / selected_root);
        ensure_nonzero(output_denominator, "TRL reflect solution is singular")?;
        let candidates = [
            (w1 - alternate_root) / output_denominator,
            -(w1 - alternate_root) / output_denominator,
        ];
        let approximation = reflect_approximation.map_or(Complex64::new(-1.0, 0.0), |network| {
            network.s[(point, 0, 0)]
        });
        reflection[point] =
            if (candidates[0] - approximation).norm() <= (candidates[1] - approximation).norm() {
                candidates[0]
            } else {
                candidates[1]
            };
    }
    let z0 = Array2::from_shape_fn((thru.frequency_points(), 1), |(point, _)| {
        thru.z0[(point, 0)]
    });
    Network::new(
        thru.frequency.clone(),
        Array3::from_shape_fn((thru.frequency_points(), 1, 1), |(point, _, _)| {
            reflection[point]
        }),
        z0,
    )
}

fn validate_two_networks(first: &Network, second: &Network, operation: &str) -> Result<()> {
    if first.ports() != 2 || second.ports() != 2 || first.frequency != second.frequency {
        return Err(Error::IncompatibleShape(format!(
            "{operation} requires frequency-compatible two-port networks"
        )));
    }
    Ok(())
}

fn regularize(value: Complex64, epsilon: f64) -> Complex64 {
    if value.norm() < epsilon {
        (Complex64::from_polar(epsilon, value.arg()) + value) / 2.0
    } else {
        value
    }
}

fn matrix_from_array(array: &Array3<Complex64>, point: usize) -> [[Complex64; 2]; 2] {
    [
        [array[(point, 0, 0)], array[(point, 0, 1)]],
        [array[(point, 1, 0)], array[(point, 1, 1)]],
    ]
}

fn two_port_reflect_network(reflection: &Network) -> Result<Network> {
    if reflection.ports() != 1 {
        return Err(Error::IncompatibleShape(
            "reflect reconstruction requires a one-port network".to_owned(),
        ));
    }
    let points = reflection.frequency_points();
    Network::new(
        reflection.frequency.clone(),
        Array3::from_shape_fn((points, 2, 2), |(point, row, column)| {
            if row == column {
                reflection.s[(point, 0, 0)]
            } else {
                Complex64::new(0.0, 0.0)
            }
        }),
        Array2::from_shape_fn((points, 2), |(point, _)| reflection.z0[(point, 0)]),
    )
}

fn solve_eight_term(
    measured: &[Network],
    ideals: &[Network],
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    validate_eight_term_standards(measured, ideals)?;
    let points = measured[0].frequency_points();
    let mut error = Array2::<Complex64>::zeros((points, 7));
    for point in 0..points {
        let mut design = Vec::with_capacity(4 * measured.len());
        let mut right = Vec::with_capacity(4 * measured.len());
        for (measured, ideal) in measured.iter().zip(ideals) {
            let m = |row, column| measured.s[(point, row, column)];
            let i = |row, column| ideal.s[(point, row, column)];
            design.extend([
                vec![
                    Complex64::new(1.0, 0.0),
                    i(0, 0) * m(0, 0),
                    -i(0, 0),
                    Complex64::new(0.0, 0.0),
                    i(1, 0) * m(0, 1),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
                vec![
                    Complex64::new(0.0, 0.0),
                    i(0, 1) * m(0, 0),
                    -i(0, 1),
                    Complex64::new(0.0, 0.0),
                    i(1, 1) * m(0, 1),
                    Complex64::new(0.0, 0.0),
                    -m(0, 1),
                ],
                vec![
                    Complex64::new(0.0, 0.0),
                    i(0, 0) * m(1, 0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    i(1, 0) * m(1, 1),
                    -i(1, 0),
                    Complex64::new(0.0, 0.0),
                ],
                vec![
                    Complex64::new(0.0, 0.0),
                    i(0, 1) * m(1, 0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    i(1, 1) * m(1, 1),
                    -i(1, 1),
                    -m(1, 1),
                ],
            ]);
            right.extend([
                m(0, 0),
                Complex64::new(0.0, 0.0),
                m(1, 0),
                Complex64::new(0.0, 0.0),
            ]);
        }
        let solution = solve_complex_least_squares(&design, &right)?;
        for coefficient in 0..7 {
            error[(point, coefficient)] = solution[coefficient];
        }
    }
    let column = |index| Array1::from_iter((0..points).map(|point| error[(point, index)]));
    let e0 = column(0);
    let e1 = column(1);
    let e2 = column(2);
    let e3 = column(3);
    let e4 = column(4);
    let e5 = column(5);
    let k = column(6);
    let reverse_directivity = &e3 / &k;
    let reverse_source_match = &e4 / &k;
    let mut coefficients = BTreeMap::from([
        ("forward directivity".to_owned(), e0.clone()),
        ("forward source match".to_owned(), e1.clone()),
        ("forward reflection tracking".to_owned(), &e0 * &e1 - &e2),
        (
            "reverse directivity".to_owned(),
            reverse_directivity.clone(),
        ),
        (
            "reverse source match".to_owned(),
            reverse_source_match.clone(),
        ),
        (
            "reverse reflection tracking".to_owned(),
            &reverse_source_match * &reverse_directivity - &e5 / &k,
        ),
        ("k".to_owned(), k),
    ]);
    coefficients.insert("forward isolation".to_owned(), Array1::zeros(points));
    coefficients.insert("reverse isolation".to_owned(), Array1::zeros(points));
    coefficients.insert("forward switch term".to_owned(), Array1::zeros(points));
    coefficients.insert("reverse switch term".to_owned(), Array1::zeros(points));
    Ok(coefficients)
}

fn transform_eight_term(
    measured: &[Network],
    coefficients: &BTreeMap<String, Array1<Complex64>>,
    network: &Network,
    embed: bool,
) -> Result<Network> {
    validate_two_port_target(measured, network)?;
    let required = |name: &str| {
        coefficients.get(name).ok_or_else(|| {
            Error::Unsupported("the eight-term calibration has not been run".to_owned())
        })
    };
    let edf = required("forward directivity")?;
    let esf = required("forward source match")?;
    let erf = required("forward reflection tracking")?;
    let edr = required("reverse directivity")?;
    let esr = required("reverse source match")?;
    let err = required("reverse reflection tracking")?;
    let k = required("k")?;
    let forward_isolation = required("forward isolation")?;
    let reverse_isolation = required("reverse isolation")?;
    let mut result = network.clone();
    for point in 0..network.frequency_points() {
        let determinant_x = edf[point] * esf[point] - erf[point];
        let determinant_y = edr[point] * esr[point] - err[point];
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let t1 = [[-determinant_x, zero], [zero, -k[point] * determinant_y]];
        let t2 = [[edf[point], zero], [zero, k[point] * edr[point]]];
        let t3 = [[-esf[point], zero], [zero, -k[point] * esr[point]]];
        let t4 = [[one, zero], [zero, k[point]]];
        let mut value = [
            [network.s[(point, 0, 0)], network.s[(point, 0, 1)]],
            [network.s[(point, 1, 0)], network.s[(point, 1, 1)]],
        ];
        if embed {
            value = matrix_multiply(
                matrix_add(matrix_multiply(t1, value), t2),
                matrix_inverse(matrix_add(matrix_multiply(t3, value), t4))?,
            );
            value[1][0] += forward_isolation[point];
            value[0][1] += reverse_isolation[point];
        } else {
            value[1][0] -= forward_isolation[point];
            value[0][1] -= reverse_isolation[point];
            value = matrix_multiply(
                matrix_inverse(matrix_add(
                    matrix_scale(matrix_multiply(value, t3), -1.0),
                    t1,
                ))?,
                matrix_add(matrix_multiply(value, t4), matrix_scale(t2, -1.0)),
            );
        }
        for (row, values) in value.iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                result.s[(point, row, column)] = *value;
            }
        }
    }
    Ok(result)
}

fn matrix_multiply(left: [[Complex64; 2]; 2], right: [[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..2)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

fn matrix_add(left: [[Complex64; 2]; 2], right: [[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    std::array::from_fn(|row| std::array::from_fn(|column| left[row][column] + right[row][column]))
}

fn matrix_scale(matrix: [[Complex64; 2]; 2], scale: f64) -> [[Complex64; 2]; 2] {
    matrix.map(|row| row.map(|value| value * scale))
}

fn matrix_complex_scale(matrix: [[Complex64; 2]; 2], scale: Complex64) -> [[Complex64; 2]; 2] {
    matrix.map(|row| row.map(|value| value * scale))
}

fn matrix_determinant(matrix: [[Complex64; 2]; 2]) -> Complex64 {
    matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
}

fn transfer_to_scattering_matrix(transfer: [[Complex64; 2]; 2]) -> Result<[[Complex64; 2]; 2]> {
    ensure_nonzero(
        transfer[1][1],
        "scattering transfer matrix has zero lower-right element",
    )?;
    Ok([
        [
            transfer[0][1] / transfer[1][1],
            transfer[0][0] - transfer[0][1] * transfer[1][0] / transfer[1][1],
        ],
        [
            Complex64::new(1.0, 0.0) / transfer[1][1],
            -transfer[1][0] / transfer[1][1],
        ],
    ])
}

fn matrix_inverse(matrix: [[Complex64; 2]; 2]) -> Result<[[Complex64; 2]; 2]> {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if determinant.norm_sqr() <= f64::EPSILON {
        return Err(Error::Unsupported(
            "eight-term transform matrix is singular".to_owned(),
        ));
    }
    Ok([
        [matrix[1][1] / determinant, -matrix[0][1] / determinant],
        [-matrix[1][0] / determinant, matrix[0][0] / determinant],
    ])
}

fn solve_complex_least_squares(
    design: &[Vec<Complex64>],
    right: &[Complex64],
) -> Result<Vec<Complex64>> {
    let columns = design.first().map_or(0, Vec::len);
    if columns == 0 || design.len() != right.len() || design.iter().any(|row| row.len() != columns)
    {
        return Err(Error::IncompatibleShape(
            "complex least-squares input has incompatible shape".to_owned(),
        ));
    }
    let mut normal = vec![vec![Complex64::new(0.0, 0.0); columns]; columns];
    let mut projected = vec![Complex64::new(0.0, 0.0); columns];
    for (row, value) in design.iter().zip(right) {
        for column in 0..columns {
            projected[column] += row[column].conj() * value;
            for other in 0..columns {
                normal[column][other] += row[column].conj() * row[other];
            }
        }
    }
    solve_complex_system(normal, projected).ok_or_else(|| {
        Error::Unsupported("calibration least-squares system is singular".to_owned())
    })
}

fn solve_complex_system(
    mut matrix: Vec<Vec<Complex64>>,
    mut right: Vec<Complex64>,
) -> Option<Vec<Complex64>> {
    let dimension = right.len();
    for pivot in 0..dimension {
        let best = (pivot..dimension).max_by(|left, right_index| {
            matrix[*left][pivot]
                .norm_sqr()
                .total_cmp(&matrix[*right_index][pivot].norm_sqr())
        })?;
        if matrix[best][pivot].norm_sqr() <= f64::EPSILON {
            return None;
        }
        matrix.swap(pivot, best);
        right.swap(pivot, best);
        let pivot_row = matrix[pivot].clone();
        let pivot_right = right[pivot];
        for row in pivot + 1..dimension {
            let multiplier = matrix[row][pivot] / pivot_row[pivot];
            for column in pivot..dimension {
                matrix[row][column] -= multiplier * pivot_row[column];
            }
            right[row] -= multiplier * pivot_right;
        }
    }
    let mut solution = vec![Complex64::new(0.0, 0.0); dimension];
    for row in (0..dimension).rev() {
        let tail = (row + 1..dimension)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<Complex64>();
        solution[row] = (right[row] - tail) / matrix[row][row];
    }
    Some(solution)
}

struct LrrmSolution {
    coefficients: BTreeMap<String, Array1<Complex64>>,
    matched: Network,
    reflect1: Network,
    reflect2: Network,
    inductance: Array1<f64>,
    capacitance: Array1<f64>,
}

fn solve_lrrm(
    measured: &[Network],
    ideals: &[Network],
    reference_impedance: f64,
    match_fit: LrrmMatchFit,
) -> Result<LrrmSolution> {
    validate_named_standard_count(measured, ideals, 4, "LRRM")?;
    let points = measured[0].frequency_points();
    let (relations, initial) = lrrm_initial_relations(measured, ideals, points)?;
    let (angular, resistance) = lrrm_frequency_data(measured, ideals, reference_impedance, points);
    let inductance =
        lrrm_initial_inductance(&initial, ideals, &angular, &resistance, reference_impedance)?;
    let (inductance, capacitance) = fit_lrrm_match(
        match_fit,
        &initial,
        ideals,
        &angular,
        &resistance,
        reference_impedance,
        inductance,
    )?;
    let matched_values = Array1::from_iter((0..points).map(|point| {
        lrrm_match_reflection(
            reference_impedance,
            resistance[point],
            angular[point],
            inductance[point],
            capacitance[point],
        )
    }));
    let final_solutions = (0..points)
        .map(|point| {
            solve_lrrm_point(
                &relations[point],
                matched_values[point],
                ideals[1].s[(point, 0, 0)],
                ideals[2].s[(point, 0, 0)],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let coefficients = lrrm_coefficients(&relations, &final_solutions, points)?;
    let one_port = |values: &Array1<Complex64>| {
        Network::new(
            measured[0].frequency.clone(),
            Array3::from_shape_fn((points, 1, 1), |(point, _, _)| values[point]),
            Array2::from_shape_fn((points, 1), |(point, _)| measured[0].z0[(point, 0)]),
        )
    };
    Ok(LrrmSolution {
        coefficients,
        matched: one_port(&matched_values)?,
        reflect1: one_port(&Array1::from_iter(
            final_solutions.iter().map(|solution| solution.reflect1),
        ))?,
        reflect2: one_port(&Array1::from_iter(
            final_solutions.iter().map(|solution| solution.reflect2),
        ))?,
        inductance,
        capacitance,
    })
}

fn lrrm_initial_relations(
    measured: &[Network],
    ideals: &[Network],
    points: usize,
) -> Result<(Vec<LrrmPointRelation>, Vec<LrrmPointSolution>)> {
    let line_transfer = ideals[0].scattering_transfer()?;
    let mut relations = Vec::with_capacity(points);
    let mut initial = Vec::with_capacity(points);
    for point in 0..points {
        let line_measured = &measured[0];
        let reflect1_measured = &measured[1];
        let reflect2_measured = &measured[2];
        let match_measured = &measured[3];
        let r11 = reflect1_measured.s[(point, 0, 0)];
        let r12 = reflect1_measured.s[(point, 1, 1)];
        let r21 = reflect2_measured.s[(point, 0, 0)];
        let r22 = reflect2_measured.s[(point, 1, 1)];
        // Upstream scikit-rf variable: `wlr1`.
        let reflect_input_matrix = [[Complex64::new(1.0, 0.0); 2], [r11, r21]];
        // Upstream scikit-rf variable: `wll1`.
        let line_input_matrix = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [
                line_measured.s[(point, 0, 0)],
                line_measured.s[(point, 0, 1)],
            ],
        ];
        // Upstream scikit-rf variable: `wll2`.
        let line_output_matrix = [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [
                line_measured.s[(point, 1, 0)],
                line_measured.s[(point, 1, 1)],
            ],
        ];
        // Upstream scikit-rf variable: `wlr2`.
        let reflect_output_matrix = [[Complex64::new(1.0, 0.0); 2], [r12, r22]];
        let wl = matrix_multiply(
            matrix_multiply(
                matrix_multiply(matrix_inverse(reflect_input_matrix)?, line_input_matrix),
                matrix_inverse(line_output_matrix)?,
            ),
            reflect_output_matrix,
        );
        let tl = matrix_from_array(&line_transfer, point);
        let xyz2 = -matrix_determinant(tl) / matrix_determinant(wl);
        let roots = quadratic_roots(wl[0][0], -tl[1][0] - tl[0][1], wl[1][1] * xyz2)?;
        let match_vector = matrix_vector_multiply(
            matrix_inverse(reflect_input_matrix)?,
            [Complex64::new(1.0, 0.0), match_measured.s[(point, 0, 0)]],
        );
        let relation = LrrmPointRelation {
            wl,
            tl,
            xyz2,
            roots: roots.into(),
            match_vector,
            measured_reflections: [r11, r12, r21, r22],
        };
        let result = solve_lrrm_point(
            &relation,
            ideals[3].s[(point, 0, 0)],
            ideals[1].s[(point, 0, 0)],
            ideals[2].s[(point, 0, 0)],
        )?;
        relations.push(relation);
        initial.push(result);
    }
    Ok((relations, initial))
}

fn lrrm_frequency_data(
    measured: &[Network],
    ideals: &[Network],
    reference_impedance: f64,
    points: usize,
) -> (Array1<f64>, Array1<f64>) {
    let angular = Array1::from_iter(
        measured[0]
            .frequency
            .values_hz()
            .iter()
            .map(|frequency| 2.0 * std::f64::consts::PI * frequency),
    );
    let resistance = Array1::from_iter((0..points).map(|point| {
        let gamma = ideals[3].s[(point, 0, 0)];
        (reference_impedance * (Complex64::new(1.0, 0.0) + gamma)
            / (Complex64::new(1.0, 0.0) - gamma))
            .re
    }));
    (angular, resistance)
}

fn lrrm_initial_inductance(
    initial: &[LrrmPointSolution],
    ideals: &[Network],
    angular: &Array1<f64>,
    resistance: &Array1<f64>,
    reference_impedance: f64,
) -> Result<Array1<f64>> {
    let mut inductance = Array1::zeros(initial.len());
    for (point, solution) in initial.iter().enumerate() {
        let reflect2 = solution.reflect2;
        let line_transmission = ideals[0].s[(point, 1, 0)];
        ensure_nonzero(line_transmission, "LRRM ideal line transmission is zero")?;
        let adjusted = reflect2 / line_transmission.powu(2);
        let a = 2.0f64.mul_add(
            -adjusted.re,
            2.0f64.mul_add(reflect2.re, reflect2.norm_sqr()),
        ) - adjusted.norm_sqr();
        let b = 4.0 * resistance[point] * (reflect2.im + adjusted.im);
        let c = 4.0 * resistance[point].powi(2) * (reflect2.norm_sqr() - 1.0);
        if a.abs() <= f64::EPSILON || angular[point].abs() <= f64::EPSILON {
            return Err(Error::Unsupported(
                "LRRM match inductance equation is singular".to_owned(),
            ));
        }
        let discriminant = (4.0 * a).mul_add(-c, b * b).max(0.0).sqrt();
        let reactances = [
            (-b + discriminant) / (2.0 * a),
            (-b - discriminant) / (2.0 * a),
        ];
        let ideal_match = ideals[3].s[(point, 0, 0)];
        let match_candidates = reactances.map(|reactance| {
            let impedance = Complex64::new(resistance[point], reactance);
            (impedance - reference_impedance) / (impedance + reference_impedance)
        });
        let selected = if (match_candidates[0] - ideal_match).norm()
            <= (match_candidates[1] - ideal_match).norm()
        {
            reactances[0]
        } else {
            reactances[1]
        };
        inductance[point] = selected / angular[point];
    }
    Ok(inductance)
}

fn fit_lrrm_match(
    match_fit: LrrmMatchFit,
    initial: &[LrrmPointSolution],
    ideals: &[Network],
    angular: &Array1<f64>,
    resistance: &Array1<f64>,
    reference_impedance: f64,
    mut inductance: Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let points = angular.len();
    let mut capacitance = Array1::zeros(points);
    let objective_l = |candidate: f64| {
        lrrm_inductance_objective(
            candidate,
            initial,
            ideals,
            angular,
            resistance,
            reference_impedance,
        )
    };
    if match_fit == LrrmMatchFit::Inductance {
        let weight = angular.sum();
        let weighted = angular
            .iter()
            .zip(inductance.iter())
            .map(|(angular, value)| angular * value)
            .sum::<f64>()
            / weight;
        let maximum = 10.0 / angular[points - 1].abs();
        let (grid_best, grid_error) = (0..10)
            .map(|index| -maximum + 2.0 * maximum * f64::from(index) / 9.0)
            .map(|candidate| (candidate, objective_l(candidate)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or_else(|| Error::Unsupported("LRRM fit grid is empty".to_owned()))?;
        let initial = if grid_error < objective_l(weighted) {
            grid_best
        } else {
            weighted
        };
        let fitted = if objective_l(initial) <= f64::EPSILON {
            initial
        } else {
            let span = 2.0 * maximum / 9.0;
            minimize_scalar(objective_l, initial - span, initial + span)
        };
        inductance.fill(fitted);
    } else if match_fit == LrrmMatchFit::InductanceCapacitance {
        let maximum_l = 20.0 / angular[points - 1].abs();
        let worst_match = 0.4_f64;
        let maximum_c = 2.0 * worst_match
            / (worst_match.mul_add(-worst_match, 1.0).sqrt()
                * angular[points - 1].abs()
                * reference_impedance);
        let objective = |candidate: [f64; 2]| {
            lrrm_inductance_capacitance_objective(
                candidate,
                initial,
                angular,
                resistance,
                reference_impedance,
            )
        };
        let (initial_guess, _) = (0..10)
            .flat_map(|l_index| {
                (0..10).map(move |c_index| {
                    [
                        maximum_l * f64::from(l_index) / 9.0,
                        maximum_c * f64::from(c_index) / 9.0,
                    ]
                })
            })
            .map(|candidate| (candidate, objective(candidate)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or_else(|| Error::Unsupported("LRRM fit grid is empty".to_owned()))?;
        let fitted = minimize_two_variables(
            objective,
            initial_guess,
            [maximum_l / 20.0, maximum_c / 20.0],
        );
        inductance.fill(fitted[0]);
        capacitance.fill(fitted[1]);
    }
    Ok((inductance, capacitance))
}

fn lrrm_coefficients(
    relations: &[LrrmPointRelation],
    final_solutions: &[LrrmPointSolution],
    points: usize,
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    let names = [
        "forward directivity",
        "forward source match",
        "forward reflection tracking",
        "reverse directivity",
        "reverse source match",
        "reverse reflection tracking",
        "k",
    ];
    let mut coefficients = names
        .iter()
        .map(|name| ((*name).to_owned(), Array1::zeros(points)))
        .collect::<BTreeMap<_, _>>();
    for (point, solved) in final_solutions.iter().enumerate() {
        let relation = &relations[point];
        let [r11, r12, r21, r22] = relation.measured_reflections;
        let t10 = matrix_multiply(
            [
                [Complex64::new(1.0, 0.0), solved.x],
                [solved.reflect1, solved.reflect2 * solved.x],
            ],
            matrix_inverse([[Complex64::new(1.0, 0.0); 2], [r11, r21]])?,
        );
        let t23 = matrix_multiply(
            matrix_complex_scale(
                [
                    [Complex64::new(1.0, 0.0), solved.y],
                    [solved.reflect1, solved.reflect2 * solved.y],
                ],
                Complex64::new(1.0, 0.0) / solved.z,
            ),
            matrix_inverse([[Complex64::new(1.0, 0.0); 2], [r12, r22]])?,
        );
        let error1 = transfer_to_scattering_matrix(t10)?;
        let error2 = transfer_to_scattering_matrix(t23)?;
        let values = [
            error1[1][1],
            error1[0][0],
            error1[0][0] * error1[1][1] - matrix_determinant(error1),
            error2[1][1],
            error2[0][0],
            error2[1][1] * error2[0][0] - matrix_determinant(error2),
            error1[0][1] / error2[0][1],
        ];
        for (name, value) in names.iter().zip(values) {
            let coefficient = coefficients.get_mut(*name).ok_or_else(|| {
                Error::Unsupported(format!("missing calibration coefficient: {name}"))
            })?;
            coefficient[point] = value;
        }
    }
    coefficients.insert("forward isolation".to_owned(), Array1::zeros(points));
    coefficients.insert("reverse isolation".to_owned(), Array1::zeros(points));
    coefficients.insert("forward switch term".to_owned(), Array1::zeros(points));
    coefficients.insert("reverse switch term".to_owned(), Array1::zeros(points));
    Ok(coefficients)
}

struct LrrmPointRelation {
    wl: ComplexMatrix2,
    tl: ComplexMatrix2,
    xyz2: Complex64,
    roots: [Complex64; 2],
    match_vector: [Complex64; 2],
    measured_reflections: [Complex64; 4],
}

#[derive(Clone, Copy)]
struct LrrmPointSolution {
    reflect1: Complex64,
    reflect2: Complex64,
    x: Complex64,
    y: Complex64,
    z: Complex64,
    error_relations: [Complex64; 4],
}

fn solve_lrrm_point(
    relation: &LrrmPointRelation,
    matched: Complex64,
    reflect1_approximation: Complex64,
    reflect2_approximation: Complex64,
) -> Result<LrrmPointSolution> {
    let candidates = relation.roots.map(|z| -> Result<(f64, LrrmPointSolution)> {
        ensure_nonzero(z, "LRRM root is zero")?;
        let xyz = relation.xyz2 / z;
        let w11 = relation.wl[0][0] * z;
        let w21 = relation.wl[1][0] * z;
        let w12 = relation.wl[0][1] * xyz;
        let w22 = relation.wl[1][1] * xyz;
        let e1 = relation.tl[1][1].powu(2) * relation.match_vector[0];
        let e0 = relation.tl[1][1] * (relation.tl[1][0] - w22) * relation.match_vector[0]
            + relation.tl[1][1] * relation.match_vector[1] * w12;
        let f1 = relation.tl[1][1] * (w11 - relation.tl[1][0]) * relation.match_vector[0]
            + relation.tl[1][1] * relation.match_vector[1] * w12;
        let f0 = (w11 - relation.tl[1][0]) * (relation.tl[1][0] - w22) * relation.match_vector[0]
            + relation.match_vector[0] * w21 * w12;
        let denominator = f1 - e1 * matched;
        ensure_nonzero(denominator, "LRRM second reflect solution is singular")?;
        let reflect2 = -(f0 - e0 * matched) / denominator;
        let x_denominator = relation.tl[1][0] + relation.tl[1][1] * reflect2 - w22;
        ensure_nonzero(x_denominator, "LRRM x solution is singular")?;
        let x = w12 / x_denominator;
        let reflect1 = ((w11 - relation.tl[1][0]) * x_denominator + w21 * w12)
            / (relation.tl[1][1] * x_denominator);
        let y = x * z.powu(2) / relation.xyz2;
        let error =
            (reflect1 - reflect1_approximation).norm() + (reflect2 - reflect2_approximation).norm();
        Ok((
            error,
            LrrmPointSolution {
                reflect1,
                reflect2,
                x,
                y,
                z,
                error_relations: [e1, e0, f1, f0],
            },
        ))
    });
    let [first, second] = candidates;
    let first = first?;
    let second = second?;
    Ok(if first.0 < second.0 {
        first.1
    } else {
        second.1
    })
}

fn lrrm_match_reflection(
    reference_impedance: f64,
    resistance: f64,
    angular_frequency: f64,
    inductance: f64,
    capacitance: f64,
) -> Complex64 {
    let imaginary = Complex64::new(0.0, 1.0);
    let numerator = reference_impedance
        + resistance
            * (-Complex64::new(1.0, 0.0)
                + imaginary * capacitance * angular_frequency * reference_impedance)
        - inductance
            * angular_frequency
            * (imaginary + capacitance * angular_frequency * reference_impedance);
    let denominator = -reference_impedance
        + resistance
            * (-Complex64::new(1.0, 0.0)
                - imaginary * capacitance * angular_frequency * reference_impedance)
        + inductance
            * angular_frequency
            * (-imaginary + capacitance * angular_frequency * reference_impedance);
    numerator / denominator
}

fn lrrm_reflect2(solution: &LrrmPointSolution, matched: Complex64) -> Complex64 {
    let [e1, e0, f1, f0] = solution.error_relations;
    -(f0 - e0 * matched) / (f1 - e1 * matched)
}

fn lrrm_inductance_objective(
    inductance: f64,
    solutions: &[LrrmPointSolution],
    ideals: &[Network],
    angular: &Array1<f64>,
    resistance: &Array1<f64>,
    reference_impedance: f64,
) -> f64 {
    solutions
        .iter()
        .enumerate()
        .map(|(point, solution)| {
            let matched = lrrm_match_reflection(
                reference_impedance,
                resistance[point],
                angular[point],
                inductance,
                0.0,
            );
            let error = ideals[2].s[(point, 0, 0)].norm() - lrrm_reflect2(solution, matched).norm();
            error * error
        })
        .sum::<f64>()
        / solutions.len().to_f64().unwrap_or(f64::INFINITY)
}

fn lrrm_inductance_capacitance_objective(
    candidate: [f64; 2],
    solutions: &[LrrmPointSolution],
    angular: &Array1<f64>,
    resistance: &Array1<f64>,
    reference_impedance: f64,
) -> f64 {
    let reflects = solutions
        .iter()
        .enumerate()
        .map(|(point, solution)| {
            let matched = lrrm_match_reflection(
                reference_impedance,
                resistance[point],
                angular[point],
                candidate[0],
                candidate[1],
            );
            lrrm_reflect2(solution, matched)
        })
        .collect::<Vec<_>>();
    let imaginary = Complex64::new(0.0, 1.0);
    let reflect_capacitance = reflects
        .iter()
        .enumerate()
        .map(|(point, reflection)| {
            (imaginary * (-Complex64::new(1.0, 0.0) + reflection)
                / ((Complex64::new(1.0, 0.0) + reflection) * angular[point] * reference_impedance))
                .re
        })
        .sum::<f64>()
        / reflects.len().to_f64().unwrap_or(f64::INFINITY);
    reflects
        .iter()
        .enumerate()
        .map(|(point, reflection)| {
            let ideal = (imaginary + reflect_capacitance * angular[point] * reference_impedance)
                / (imaginary - reflect_capacitance * angular[point] * reference_impedance);
            (ideal - reflection).norm_sqr()
        })
        .sum::<f64>()
        / reflects.len().to_f64().unwrap_or(f64::INFINITY)
}

fn minimize_scalar(objective: impl Fn(f64) -> f64, mut left: f64, mut right: f64) -> f64 {
    let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut first = right - ratio * (right - left);
    let mut second = left + ratio * (right - left);
    let mut first_value = objective(first);
    let mut second_value = objective(second);
    for _ in 0..120 {
        if first_value < second_value {
            right = second;
            second = first;
            second_value = first_value;
            first = right - ratio * (right - left);
            first_value = objective(first);
        } else {
            left = first;
            first = second;
            first_value = second_value;
            second = left + ratio * (right - left);
            second_value = objective(second);
        }
    }
    f64::midpoint(left, right)
}

fn minimize_two_variables(
    objective: impl Fn([f64; 2]) -> f64,
    initial: [f64; 2],
    step: [f64; 2],
) -> [f64; 2] {
    let mut simplex = [
        initial,
        [initial[0] + step[0], initial[1]],
        [initial[0], initial[1] + step[1]],
    ];
    for _ in 0..160 {
        simplex.sort_by(|left, right| objective(*left).total_cmp(&objective(*right)));
        let centroid = [
            f64::midpoint(simplex[0][0], simplex[1][0]),
            f64::midpoint(simplex[0][1], simplex[1][1]),
        ];
        let reflected = [
            2.0f64.mul_add(centroid[0], -simplex[2][0]).max(0.0),
            2.0f64.mul_add(centroid[1], -simplex[2][1]).max(0.0),
        ];
        if objective(reflected) < objective(simplex[0]) {
            let expanded = [
                2.0f64.mul_add(-simplex[2][0], 3.0 * centroid[0]).max(0.0),
                2.0f64.mul_add(-simplex[2][1], 3.0 * centroid[1]).max(0.0),
            ];
            simplex[2] = if objective(expanded) < objective(reflected) {
                expanded
            } else {
                reflected
            };
        } else if objective(reflected) < objective(simplex[1]) {
            simplex[2] = reflected;
        } else {
            let contracted = [
                f64::midpoint(centroid[0], simplex[2][0]).max(0.0),
                f64::midpoint(centroid[1], simplex[2][1]).max(0.0),
            ];
            if objective(contracted) < objective(simplex[2]) {
                simplex[2] = contracted;
            } else {
                simplex[1] = [
                    f64::midpoint(simplex[0][0], simplex[1][0]),
                    f64::midpoint(simplex[0][1], simplex[1][1]),
                ];
                simplex[2] = [
                    f64::midpoint(simplex[0][0], simplex[2][0]),
                    f64::midpoint(simplex[0][1], simplex[2][1]),
                ];
            }
        }
    }
    simplex.sort_by(|left, right| objective(*left).total_cmp(&objective(*right)));
    simplex[0]
}

fn matrix_vector_multiply(matrix: ComplexMatrix2, vector: [Complex64; 2]) -> [Complex64; 2] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

fn solve_lmr16(
    measured: &[Network],
    ideal: &Network,
    ideal_is_reflect: bool,
    requested_sign: Option<f64>,
) -> Result<(BTreeMap<String, Array1<Complex64>>, Network, Network)> {
    let points = measured[0].frequency_points();
    let mut through_s = Array3::zeros((points, 2, 2));
    let mut reflect_s = Array3::zeros((points, 1, 1));
    let names = sixteen_term_coefficient_names();
    let mut coefficients = names
        .iter()
        .map(|name| ((*name).to_owned(), Array1::zeros(points)))
        .collect::<BTreeMap<_, _>>();
    for point in 0..points {
        let matrix = |index: usize| matrix_from_array(&measured[index].s, point);
        let ma = matrix(0);
        let mb = matrix(1);
        let mc = matrix(2);
        let md = matrix(3);
        let me = matrix(4);
        let nn = matrix_multiply(
            matrix_inverse(matrix_add(me, matrix_scale(ma, -1.0)))?,
            matrix_add(mb, matrix_scale(me, -1.0)),
        );
        let mm = matrix_multiply(matrix_add(ma, matrix_scale(mc, -1.0)), nn);
        let oo = matrix_add(mb, matrix_scale(mc, -1.0));
        let rr = matrix_multiply(
            matrix_inverse(matrix_add(md, matrix_scale(ma, -1.0)))?,
            matrix_add(mb, matrix_scale(md, -1.0)),
        );
        let pp = matrix_multiply(matrix_add(ma, matrix_scale(mc, -1.0)), rr);
        let m = (pp[1][0] + oo[1][0]) * mm[1][1] - (pp[1][1] + oo[1][1]) * mm[1][0];
        let n = oo[1][0] * pp[0][1] - oo[1][1] * pp[0][0];
        let o = (mm[0][1] + oo[0][1]) * pp[0][0] - (mm[0][0] + oo[0][0]) * pp[0][1];
        let p = oo[0][1] * mm[1][0] - oo[0][0] * mm[1][1];
        ensure_nonzero(n * p, "LMR16 root equation is singular")?;
        ensure_nonzero(o * mm[1][1], "LMR16 normalization is singular")?;
        let root = (m * o / (n * p)).sqrt();
        let choose = |sign: f64| -> Result<(f64, Complex64, Complex64, Complex64)> {
            let gamma_times_thru = sign * root;
            let (gamma, thru) = if ideal_is_reflect {
                let gamma = ideal.s[(point, 0, 0)];
                ensure_nonzero(gamma_times_thru, "LMR16 reflect/thru ratio is zero")?;
                (gamma, gamma / gamma_times_thru)
            } else {
                let thru = ideal.s[(point, 1, 0)];
                (gamma_times_thru * thru, thru)
            };
            ensure_nonzero(gamma, "LMR16 reflect is zero")?;
            ensure_nonzero(thru, "LMR16 thru is zero")?;
            let t15 = -(p / o) * (pp[0][0] / mm[1][1]) * gamma_times_thru;
            Ok(((Complex64::new(1.0, 0.0) - t15).norm(), gamma, thru, t15))
        };
        let selected = if let Some(sign) = requested_sign {
            choose(sign)?
        } else {
            let positive = choose(1.0)?;
            let negative = choose(-1.0)?;
            if positive.0 < negative.0 {
                positive
            } else {
                negative
            }
        };
        let (_, gamma, thru, t15) = selected;
        through_s[(point, 0, 1)] = thru;
        through_s[(point, 1, 0)] = thru;
        reflect_s[(point, 0, 0)] = gamma;
        let t12 = Complex64::new(1.0, 0.0);
        ensure_nonzero(pp[0][0], "LMR16 pp coefficient is zero")?;
        let t13 = -pp[0][1] / pp[0][0] * t15;
        let t14 = -mm[1][0] / mm[1][1] * t12;
        let normalization_denominator = t15 - t13 * t14;
        ensure_nonzero(normalization_denominator, "LMR16 T4 matrix is singular")?;
        let scale = Complex64::new(1.0, 0.0) / normalization_denominator;
        let t8 = (rr[0][0] * t12 + rr[0][1] * t14) / gamma - t13 / thru;
        let t9 = (nn[0][0] * t13 + nn[0][1] * t15) / gamma - t12 / thru;
        let t10 = (rr[1][0] * t12 + rr[1][1] * t14) / gamma - t15 / thru;
        let t11 = (nn[1][0] * t13 + nn[1][1] * t15) / gamma - t14 / thru;
        let t0 = mc[0][0] * t8 + mc[0][1] * t10 - (oo[0][0] * t12 + oo[0][1] * t14) / gamma;
        let t1 = mc[0][0] * t9 + mc[0][1] * t11 - (oo[0][0] * t13 + oo[0][1] * t15) / gamma;
        let t2 = mc[1][0] * t8 + mc[1][1] * t10 - (oo[1][0] * t12 + oo[1][1] * t14) / gamma;
        let t3 = mc[1][0] * t9 + mc[1][1] * t11 - (oo[1][0] * t13 + oo[1][1] * t15) / gamma;
        let t4 = mb[0][0] * t12 + mb[0][1] * t14;
        let t5 = mb[0][0] * t13 + mb[0][1] * t15;
        let t6 = mb[1][0] * t12 + mb[1][1] * t14;
        let t7 = mb[1][0] * t13 + mb[1][1] * t15;
        let values = sixteen_values_from_t(
            matrix_complex_scale([[t0, t1], [t2, t3]], scale),
            matrix_complex_scale([[t4, t5], [t6, t7]], scale),
            matrix_complex_scale([[t8, t9], [t10, t11]], scale),
            matrix_complex_scale([[t12, t13], [t14, t15]], scale),
        )?;
        for (name, value) in names.iter().zip(values) {
            let coefficient = coefficients.get_mut(*name).ok_or_else(|| {
                Error::Unsupported(format!("missing calibration coefficient: {name}"))
            })?;
            coefficient[point] = value;
        }
    }
    coefficients.insert("forward switch term".to_owned(), Array1::zeros(points));
    coefficients.insert("reverse switch term".to_owned(), Array1::zeros(points));
    let (through, reflect) = lmr16_networks(measured, through_s, reflect_s)?;
    Ok((coefficients, through, reflect))
}

fn lmr16_networks(
    measured: &[Network],
    through_s: Array3<Complex64>,
    reflect_s: Array3<Complex64>,
) -> Result<(Network, Network)> {
    let points = measured[0].frequency_points();
    let through = Network::new(
        measured[0].frequency.clone(),
        through_s,
        measured[0].z0.clone(),
    )?;
    let reflect = Network::new(
        measured[0].frequency.clone(),
        reflect_s,
        Array2::from_shape_fn((points, 1), |(point, _)| measured[0].z0[(point, 0)]),
    )?;
    Ok((through, reflect))
}

const fn sixteen_term_coefficient_names() -> [&'static str; 15] {
    [
        "forward directivity",
        "reverse directivity",
        "forward source match",
        "reverse source match",
        "forward reflection tracking",
        "reverse reflection tracking",
        "k",
        "forward isolation",
        "reverse isolation",
        "forward port 1 isolation",
        "reverse port 1 isolation",
        "forward port 2 isolation",
        "reverse port 2 isolation",
        "forward port isolation",
        "reverse port isolation",
    ]
}

fn sixteen_values_from_t(
    t1: [[Complex64; 2]; 2],
    t2: [[Complex64; 2]; 2],
    t3: [[Complex64; 2]; 2],
    t4: [[Complex64; 2]; 2],
) -> Result<[Complex64; 15]> {
    let inverse_t4 = matrix_inverse(t4)?;
    let e1 = matrix_multiply(t2, inverse_t4);
    let e2 = matrix_add(
        t1,
        matrix_scale(matrix_multiply(matrix_multiply(t2, inverse_t4), t3), -1.0),
    );
    let e3 = inverse_t4;
    let e4 = matrix_scale(matrix_multiply(inverse_t4, t3), -1.0);
    Ok([
        e1[0][0],
        e1[1][1],
        e4[0][0],
        e4[1][1],
        e2[0][0] * e3[0][0],
        e2[1][1],
        e3[0][0],
        e1[1][0],
        e1[0][1],
        e3[1][0],
        e2[0][1],
        e2[1][0],
        e3[0][1],
        e4[1][0],
        e4[0][1],
    ])
}

fn solve_lrm(
    measured: &[Network],
    ideals: &[Network],
) -> Result<(BTreeMap<String, Array1<Complex64>>, Network)> {
    validate_named_standard_count(measured, ideals, 3, "LRM")?;
    let points = measured[0].frequency_points();
    let line_measured = &measured[0];
    let reflect_measured = &measured[1];
    let match_measured = &measured[2];
    let line_transfer = ideals[0].scattering_transfer()?;
    let mut solved_reflection = Array1::zeros(points);
    let names = [
        "forward directivity",
        "forward source match",
        "forward reflection tracking",
        "reverse directivity",
        "reverse source match",
        "reverse reflection tracking",
        "k",
    ];
    let mut coefficients = names
        .iter()
        .map(|name| ((*name).to_owned(), Array1::zeros(points)))
        .collect::<BTreeMap<_, _>>();
    for point in 0..points {
        let gm = ideals[2].s[(point, 0, 0)];
        let r1 = reflect_measured.s[(point, 0, 0)];
        let r2 = reflect_measured.s[(point, 1, 1)];
        let m1 = match_measured.s[(point, 0, 0)];
        let m2 = match_measured.s[(point, 1, 1)];
        let wl = lrm_measured_line_relation(line_measured, point, r1, m1, r2, m2)?;
        let tl = matrix_from_array(&line_transfer, point);
        let wl_determinant = matrix_determinant(wl);
        ensure_nonzero(wl_determinant, "LRM measured line relation is singular")?;
        let xyz2 = -matrix_determinant(tl) / wl_determinant;
        let (z0, z1) = quadratic_roots(wl[0][0], -tl[1][0] - tl[0][1], wl[1][1] * xyz2)?;
        let candidates = <[Complex64; 2]>::from((z0, z1)).map(
            |z| -> Result<(f64, Complex64, Complex64, Complex64)> {
                ensure_nonzero(z, "LRM root is zero")?;
                let xyz = xyz2 / z;
                let w11 = wl[0][0] * z;
                let w21 = wl[1][0] * z;
                let w12 = wl[0][1] * xyz;
                let w22 = wl[1][1] * xyz;
                let denominator = tl[1][0] + tl[1][1] * gm - w22;
                ensure_nonzero(denominator, "LRM reflect solution is singular")?;
                ensure_nonzero(tl[1][1], "LRM ideal line transfer is singular")?;
                let x = w12 / denominator;
                let reflect = (w11 - tl[1][0] + w21 * x) / tl[1][1];
                Ok(((reflect - ideals[1].s[(point, 0, 0)]).norm(), reflect, x, z))
            },
        );
        let [first, second] = candidates;
        let first = first?;
        let second = second?;
        let selected = if first.0 < second.0 { first } else { second };
        let (_, reflect, x, z) = selected;
        ensure_nonzero(xyz2, "LRM xyz relation is zero")?;
        let y = x * z.powu(2) / xyz2;
        solved_reflection[point] = reflect;
        let t10 = matrix_multiply(
            [[Complex64::new(1.0, 0.0), x], [reflect, gm * x]],
            matrix_inverse([[Complex64::new(1.0, 0.0); 2], [r1, m1]])?,
        );
        let t23 = matrix_multiply(
            matrix_complex_scale(
                [[Complex64::new(1.0, 0.0), y], [reflect, gm * y]],
                Complex64::new(1.0, 0.0) / z,
            ),
            matrix_inverse([[Complex64::new(1.0, 0.0); 2], [r2, m2]])?,
        );
        let error1 = transfer_to_scattering_matrix(t10)?;
        let error2 = transfer_to_scattering_matrix(t23)?;
        let values = [
            error1[1][1],
            error1[0][0],
            error1[0][0] * error1[1][1] - matrix_determinant(error1),
            error2[1][1],
            error2[0][0],
            error2[1][1] * error2[0][0] - matrix_determinant(error2),
            error1[0][1] / error2[0][1],
        ];
        for (name, value) in names.iter().zip(values) {
            let coefficient = coefficients.get_mut(*name).ok_or_else(|| {
                Error::Unsupported(format!("missing calibration coefficient: {name}"))
            })?;
            coefficient[point] = value;
        }
    }
    coefficients.insert("forward isolation".to_owned(), Array1::zeros(points));
    coefficients.insert("reverse isolation".to_owned(), Array1::zeros(points));
    coefficients.insert("forward switch term".to_owned(), Array1::zeros(points));
    coefficients.insert("reverse switch term".to_owned(), Array1::zeros(points));
    let solved = Network::new(
        measured[0].frequency.clone(),
        Array3::from_shape_fn((points, 1, 1), |(point, _, _)| solved_reflection[point]),
        Array2::from_shape_fn((points, 1), |(point, _)| measured[0].z0[(point, 0)]),
    )?;
    Ok((coefficients, solved))
}

fn lrm_measured_line_relation(
    line_measured: &Network,
    point: usize,
    r1: Complex64,
    m1: Complex64,
    r2: Complex64,
    m2: Complex64,
) -> Result<[[Complex64; 2]; 2]> {
    // Upstream scikit-rf variable: `wlr1`.
    let reflect_input_matrix = [[Complex64::new(1.0, 0.0); 2], [r1, m1]];
    // Upstream scikit-rf variable: `wll1`.
    let line_input_matrix = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [
            line_measured.s[(point, 0, 0)],
            line_measured.s[(point, 0, 1)],
        ],
    ];
    // Upstream scikit-rf variable: `wll2`.
    let line_output_matrix = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [
            line_measured.s[(point, 1, 0)],
            line_measured.s[(point, 1, 1)],
        ],
    ];
    // Upstream scikit-rf variable: `wlr2`.
    let reflect_output_matrix = [[Complex64::new(1.0, 0.0); 2], [r2, m2]];
    Ok(matrix_multiply(
        matrix_multiply(
            matrix_multiply(matrix_inverse(reflect_input_matrix)?, line_input_matrix),
            matrix_inverse(line_output_matrix)?,
        ),
        reflect_output_matrix,
    ))
}

fn solve_unknown_thru(
    measured: &[Network],
    ideals: &[Network],
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    validate_unknown_thru_standards(measured, ideals)?;
    let reflect_count = measured.len() - 1;
    let port1_measured = measured[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[0]))
        .collect::<Result<Vec<_>>>()?;
    let port2_measured = measured[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[1]))
        .collect::<Result<Vec<_>>>()?;
    let port1_ideals = ideals[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[0]))
        .collect::<Result<Vec<_>>>()?;
    let port2_ideals = ideals[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[1]))
        .collect::<Result<Vec<_>>>()?;
    let mut port1 = OnePort::new(port1_measured, port1_ideals)?;
    let mut port2 = OnePort::new(port2_measured, port2_ideals)?;
    port1.run()?;
    port2.run()?;
    finish_unknown_thru(measured, ideals, &port1.coefficients, &port2.coefficients)
}

fn solve_mrc(
    measured: &[Network],
    ideals: &mut [Network],
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    validate_unknown_thru_standards(measured, ideals)?;
    let reflect_count = measured.len() - 1;
    let port1_measured = measured[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[0]))
        .collect::<Result<Vec<_>>>()?;
    let port2_measured = measured[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[1]))
        .collect::<Result<Vec<_>>>()?;
    let port1_ideals = ideals[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[0]))
        .collect::<Result<Vec<_>>>()?;
    let port2_ideals = ideals[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[1]))
        .collect::<Result<Vec<_>>>()?;
    let mut port1 = Sddl::new(port1_measured, port1_ideals)?;
    let mut port2 = Sddl::new(port2_measured, port2_ideals)?;
    port1.run()?;
    port2.run()?;
    for (standard, ideal) in ideals.iter_mut().enumerate().take(reflect_count) {
        for point in 0..ideal.frequency_points() {
            ideal.s[(point, 0, 0)] = port1.ideals[standard].s[(point, 0, 0)];
            ideal.s[(point, 1, 1)] = port2.ideals[standard].s[(point, 0, 0)];
        }
    }
    finish_unknown_thru(measured, ideals, &port1.coefficients, &port2.coefficients)
}

fn finish_unknown_thru(
    measured: &[Network],
    ideals: &[Network],
    port1_coefficients: &BTreeMap<String, Array1<Complex64>>,
    port2_coefficients: &BTreeMap<String, Array1<Complex64>>,
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    let reflect_count = measured.len() - 1;
    let points = measured[0].frequency_points();
    let mut coefficients = BTreeMap::new();
    for (name, values) in port1_coefficients {
        coefficients.insert(format!("forward {name}"), values.clone());
    }
    for (name, values) in port2_coefficients {
        coefficients.insert(format!("reverse {name}"), values.clone());
    }
    coefficients.insert("forward isolation".to_owned(), Array1::zeros(points));
    coefficients.insert("reverse isolation".to_owned(), Array1::zeros(points));
    coefficients.insert("forward switch term".to_owned(), Array1::zeros(points));
    coefficients.insert("reverse switch term".to_owned(), Array1::zeros(points));
    let forward_tracking = &port1_coefficients["reflection tracking"];
    let reverse_tracking = &port2_coefficients["reflection tracking"];
    let thru_measured = &measured[reflect_count];
    let thru_ideal = &ideals[reflect_count];
    let mut k = Array1::zeros(points);
    for point in 0..points {
        ensure_nonzero(
            thru_measured.s[(point, 0, 1)],
            "unknown-thru reverse transmission is zero",
        )?;
        ensure_nonzero(
            reverse_tracking[point],
            "unknown-thru reverse tracking is zero",
        )?;
        let ambiguous =
            (forward_tracking[point] * reverse_tracking[point] * thru_measured.s[(point, 1, 0)]
                / thru_measured.s[(point, 0, 1)])
                .sqrt()
                / reverse_tracking[point];
        let mut first_coefficients = coefficients.clone();
        let mut second_coefficients = coefficients.clone();
        let mut first_k = Array1::zeros(points);
        let mut second_k = Array1::zeros(points);
        first_k.fill(ambiguous);
        second_k.fill(-ambiguous);
        first_coefficients.insert("k".to_owned(), first_k);
        second_coefficients.insert("k".to_owned(), second_k);
        let first = transform_eight_term(measured, &first_coefficients, thru_measured, false)?;
        let second = transform_eight_term(measured, &second_coefficients, thru_measured, false)?;
        ensure_nonzero(
            thru_ideal.s[(point, 1, 0)],
            "unknown-thru approximation has zero transmission",
        )?;
        let first_phase = (first.s[(point, 1, 0)] / thru_ideal.s[(point, 1, 0)])
            .arg()
            .abs();
        let second_phase = (second.s[(point, 1, 0)] / thru_ideal.s[(point, 1, 0)])
            .arg()
            .abs();
        k[point] = if first_phase < second_phase {
            ambiguous
        } else {
            -ambiguous
        };
    }
    coefficients.insert("k".to_owned(), k);
    Ok(coefficients)
}

fn solve_sixteen_term(
    measured: &[Network],
    ideals: &[Network],
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    validate_sixteen_term_standards(measured, ideals)?;
    let points = measured[0].frequency_points();
    let mut solved = vec![[Complex64::new(0.0, 0.0); 15]; points];
    for (point, solved_point) in solved.iter_mut().enumerate() {
        let (design, right) = sixteen_term_design(measured, ideals, point);
        let mut error = solve_complex_least_squares(&design, &right)?;
        let normalization_denominator = error[12] - error[13] * error[14];
        ensure_nonzero(
            normalization_denominator,
            "sixteen-term normalization is singular",
        )?;
        let normalization = error[12] / normalization_denominator;
        for value in &mut error {
            *value *= normalization;
        }
        solved_point.copy_from_slice(&error);
    }
    let mut coefficients = BTreeMap::<String, Array1<Complex64>>::new();
    let names = [
        "forward directivity",
        "reverse directivity",
        "forward source match",
        "reverse source match",
        "forward reflection tracking",
        "reverse reflection tracking",
        "k",
        "forward isolation",
        "reverse isolation",
        "forward port 1 isolation",
        "reverse port 1 isolation",
        "forward port 2 isolation",
        "reverse port 2 isolation",
        "forward port isolation",
        "reverse port isolation",
    ];
    for name in names {
        coefficients.insert(name.to_owned(), Array1::zeros(points));
    }
    for (point, error) in solved.iter().copied().enumerate() {
        let c = error[12] / (error[12] - error[13] * error[14]);
        let t1 = [[error[0], error[1]], [error[2], error[3]]];
        let t2 = [[error[4], error[5]], [error[6], error[7]]];
        let t3 = [[error[8], error[9]], [error[10], error[11]]];
        let t4 = [[error[12], error[13]], [error[14], c]];
        let inverse_t4 = matrix_inverse(t4)?;
        let e1 = matrix_multiply(t2, inverse_t4);
        let e2 = matrix_add(
            t1,
            matrix_scale(matrix_multiply(matrix_multiply(t2, inverse_t4), t3), -1.0),
        );
        let e3 = inverse_t4;
        let e4 = matrix_scale(matrix_multiply(inverse_t4, t3), -1.0);
        let values = [
            e1[0][0],
            e1[1][1],
            e4[0][0],
            e4[1][1],
            e2[0][0] * e3[0][0],
            e2[1][1],
            e3[0][0],
            e1[1][0],
            e1[0][1],
            e3[1][0],
            e2[0][1],
            e2[1][0],
            e3[0][1],
            e4[1][0],
            e4[0][1],
        ];
        for (name, value) in names.iter().zip(values) {
            let coefficient = coefficients.get_mut(*name).ok_or_else(|| {
                Error::Unsupported(format!("missing calibration coefficient: {name}"))
            })?;
            coefficient[point] = value;
        }
    }
    coefficients.insert("forward switch term".to_owned(), Array1::zeros(points));
    coefficients.insert("reverse switch term".to_owned(), Array1::zeros(points));
    Ok(coefficients)
}

fn sixteen_term_design(
    measured: &[Network],
    ideals: &[Network],
    point: usize,
) -> (Vec<Vec<Complex64>>, Vec<Complex64>) {
    let mut design = Vec::with_capacity(4 * measured.len());
    let mut right = Vec::with_capacity(4 * measured.len());
    for (measured, ideal) in measured.iter().zip(ideals) {
        let m = |row, column| measured.s[(point, row, column)];
        let i = |row, column| ideal.s[(point, row, column)];
        design.extend([
            vec![
                i(0, 0),
                i(1, 0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                -m(0, 0) * i(0, 0),
                -m(0, 0) * i(1, 0),
                -m(0, 1) * i(0, 0),
                -m(0, 1) * i(1, 0),
                -m(0, 0),
                Complex64::new(0.0, 0.0),
                -m(0, 1),
            ],
            vec![
                i(0, 1),
                i(1, 1),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                -m(0, 0) * i(0, 1),
                -m(0, 0) * i(1, 1),
                -m(0, 1) * i(0, 1),
                -m(0, 1) * i(1, 1),
                Complex64::new(0.0, 0.0),
                -m(0, 0),
                Complex64::new(0.0, 0.0),
            ],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                i(0, 0),
                i(1, 0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                -m(1, 0) * i(0, 0),
                -m(1, 0) * i(1, 0),
                -m(1, 1) * i(0, 0),
                -m(1, 1) * i(1, 0),
                -m(1, 0),
                Complex64::new(0.0, 0.0),
                -m(1, 1),
            ],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                i(0, 1),
                i(1, 1),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                -m(1, 0) * i(0, 1),
                -m(1, 0) * i(1, 1),
                -m(1, 1) * i(0, 1),
                -m(1, 1) * i(1, 1),
                Complex64::new(0.0, 0.0),
                -m(1, 0),
                Complex64::new(0.0, 0.0),
            ],
        ]);
        right.extend([
            Complex64::new(0.0, 0.0),
            m(0, 1),
            Complex64::new(0.0, 0.0),
            m(1, 1),
        ]);
    }
    (design, right)
}

fn transform_sixteen_term(
    measured: &[Network],
    coefficients: &BTreeMap<String, Array1<Complex64>>,
    network: &Network,
    embed: bool,
) -> Result<Network> {
    validate_two_port_target(measured, network)?;
    let mut result = network.clone();
    for point in 0..network.frequency_points() {
        let (t1, t2, t3, t4) = sixteen_term_matrices(coefficients, point)?;
        let value = matrix_from_array(&network.s, point);
        let transformed = if embed {
            matrix_multiply(
                matrix_add(matrix_multiply(t1, value), t2),
                matrix_inverse(matrix_add(matrix_multiply(t3, value), t4))?,
            )
        } else {
            matrix_multiply(
                matrix_inverse(matrix_add(
                    matrix_scale(matrix_multiply(value, t3), -1.0),
                    t1,
                ))?,
                matrix_add(matrix_multiply(value, t4), matrix_scale(t2, -1.0)),
            )
        };
        for (row, values) in transformed.iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                result.s[(point, row, column)] = *value;
            }
        }
    }
    Ok(result)
}

fn sixteen_term_matrices(
    coefficients: &BTreeMap<String, Array1<Complex64>>,
    point: usize,
) -> Result<FourComplexMatrices> {
    let value = |name: &str| {
        coefficients
            .get(name)
            .and_then(|values| values.get(point))
            .copied()
            .ok_or_else(|| Error::Unsupported(format!("missing sixteen-term coefficient {name}")))
    };
    let k = value("k")?;
    ensure_nonzero(k, "sixteen-term k coefficient is zero")?;
    let e1 = [
        [value("forward directivity")?, value("reverse isolation")?],
        [value("forward isolation")?, value("reverse directivity")?],
    ];
    let e2 = [
        [
            value("forward reflection tracking")? / k,
            value("reverse port 1 isolation")?,
        ],
        [
            value("forward port 2 isolation")?,
            value("reverse reflection tracking")?,
        ],
    ];
    let e3 = [
        [k, value("reverse port 2 isolation")?],
        [value("forward port 1 isolation")?, Complex64::new(1.0, 0.0)],
    ];
    let e4 = [
        [
            value("forward source match")?,
            value("reverse port isolation")?,
        ],
        [
            value("forward port isolation")?,
            value("reverse source match")?,
        ],
    ];
    let inverse_e3 = matrix_inverse(e3)?;
    let t1 = matrix_add(
        e2,
        matrix_scale(matrix_multiply(matrix_multiply(e1, inverse_e3), e4), -1.0),
    );
    let t2 = matrix_multiply(e1, inverse_e3);
    let t3 = matrix_scale(matrix_multiply(inverse_e3, e4), -1.0);
    Ok((t1, t2, t3, inverse_e3))
}

fn solve_twelve_term(
    measured: &[Network],
    ideals: &[Network],
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    validate_two_port_calibration_standards(measured, ideals)?;
    let reflect_count = measured.len() - 1;
    let measured_port1 = measured[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[0]))
        .collect::<Result<Vec<_>>>()?;
    let measured_port2 = measured[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[1]))
        .collect::<Result<Vec<_>>>()?;
    let ideal_port1 = ideals[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[0]))
        .collect::<Result<Vec<_>>>()?;
    let ideal_port2 = ideals[..reflect_count]
        .iter()
        .map(|network| network.subnetwork(&[1]))
        .collect::<Result<Vec<_>>>()?;
    let mut port1 = OnePort::new(measured_port1, ideal_port1)?;
    let mut port2 = OnePort::new(measured_port2, ideal_port2)?;
    port1.run()?;
    port2.run()?;
    let thru = &measured[reflect_count];
    let ideal_thru = &ideals[reflect_count];
    let corrected_s11 = port1.apply(&thru.subnetwork(&[0])?)?;
    let corrected_s22 = port2.apply(&thru.subnetwork(&[1])?)?;
    let load_match_forward = ideal_thru.inverse()?.connect(1, &corrected_s11, 0)?.s;
    let load_match_reverse = ideal_thru
        .flipped()?
        .inverse()?
        .connect(1, &corrected_s22, 0)?
        .s;
    let points = thru.frequency_points();
    let mut forward_tracking = Array1::zeros(points);
    let mut reverse_tracking = Array1::zeros(points);
    for point in 0..points {
        forward_tracking[point] = transmission_tracking(
            point,
            thru,
            ideal_thru,
            &load_match_forward,
            &port1.coefficients,
            false,
        )?;
        reverse_tracking[point] = transmission_tracking(
            point,
            thru,
            ideal_thru,
            &load_match_reverse,
            &port2.coefficients,
            true,
        )?;
    }
    let mut coefficients = BTreeMap::new();
    for (name, values) in &port1.coefficients {
        coefficients.insert(format!("forward {name}"), values.clone());
    }
    for (name, values) in &port2.coefficients {
        coefficients.insert(format!("reverse {name}"), values.clone());
    }
    coefficients.insert("forward isolation".to_owned(), Array1::zeros(points));
    coefficients.insert("reverse isolation".to_owned(), Array1::zeros(points));
    coefficients.insert(
        "forward load match".to_owned(),
        Array1::from_iter((0..points).map(|point| load_match_forward[(point, 0, 0)])),
    );
    coefficients.insert(
        "reverse load match".to_owned(),
        Array1::from_iter((0..points).map(|point| load_match_reverse[(point, 0, 0)])),
    );
    coefficients.insert("forward transmission tracking".to_owned(), forward_tracking);
    coefficients.insert("reverse transmission tracking".to_owned(), reverse_tracking);
    Ok(coefficients)
}

fn duplicate_one_path_coefficients(
    coefficients: BTreeMap<String, Array1<Complex64>>,
    source_port: usize,
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    let mut output = coefficients;
    let (source_prefix, reverse_prefix) = if source_port == 0 {
        ("forward", "reverse")
    } else {
        ("reverse", "forward")
    };
    let replacements = output
        .iter()
        .filter(|(name, _)| name.starts_with(source_prefix))
        .map(|(name, values)| {
            (
                name.replacen(source_prefix, reverse_prefix, 1),
                values.clone(),
            )
        })
        .collect::<Vec<_>>();
    output.extend(replacements);
    let points =
        output.values().next().map(Array1::len).ok_or_else(|| {
            Error::Unsupported("one-path calibration has no coefficients".to_owned())
        })?;
    let value = |name: &str, point: usize| {
        output
            .get(name)
            .and_then(|values| values.get(point))
            .copied()
            .ok_or_else(|| Error::Unsupported(format!("missing calibration coefficient {name}")))
    };
    let mut forward_switch = Array1::zeros(points);
    let mut reverse_switch = Array1::zeros(points);
    let mut k = Array1::zeros(points);
    for point in 0..points {
        let edf = value("forward directivity", point)?;
        let esf = value("forward source match", point)?;
        let erf = value("forward reflection tracking", point)?;
        let etf = value("forward transmission tracking", point)?;
        let elf = value("forward load match", point)?;
        let edr = value("reverse directivity", point)?;
        let esr = value("reverse source match", point)?;
        let err = value("reverse reflection tracking", point)?;
        let etr = value("reverse transmission tracking", point)?;
        let elr = value("reverse load match", point)?;
        let forward_denominator = err + edr * (elf - esr);
        let reverse_denominator = erf + edf * (elr - esf);
        ensure_nonzero(
            forward_denominator,
            "forward one-path conversion is singular",
        )?;
        ensure_nonzero(
            reverse_denominator,
            "reverse one-path conversion is singular",
        )?;
        ensure_nonzero(etr, "reverse one-path transmission tracking is zero")?;
        forward_switch[point] = (elf - esr) / forward_denominator;
        reverse_switch[point] = (elr - esf) / reverse_denominator;
        let first = etf / forward_denominator;
        let second = reverse_denominator / etr;
        k[point] = (first + second) / 2.0;
    }
    output.insert("forward switch term".to_owned(), forward_switch);
    output.insert("reverse switch term".to_owned(), reverse_switch);
    output.insert("k".to_owned(), k);
    Ok(output)
}

fn transmission_tracking(
    point: usize,
    measured: &Network,
    ideal: &Network,
    load_match: &Array3<Complex64>,
    one_port: &BTreeMap<String, Array1<Complex64>>,
    reverse: bool,
) -> Result<Complex64> {
    let source_match = one_port["source match"][point];
    let load_match = load_match[(point, 0, 0)];
    let (e, f, b, h, measured_transmission) = if reverse {
        (
            ideal.s[(point, 1, 1)],
            ideal.s[(point, 0, 0)],
            ideal.s[(point, 0, 1)],
            ideal.s[(point, 1, 0)],
            measured.s[(point, 0, 1)],
        )
    } else {
        (
            ideal.s[(point, 0, 0)],
            ideal.s[(point, 1, 1)],
            ideal.s[(point, 1, 0)],
            ideal.s[(point, 0, 1)],
            measured.s[(point, 1, 0)],
        )
    };
    if b.norm_sqr() <= f64::EPSILON {
        return Err(Error::Unsupported(
            "twelve-term calibration requires a transmissive final standard".to_owned(),
        ));
    }
    Ok(measured_transmission / b
        * (Complex64::new(1.0, 0.0)
            - (source_match * e + f * load_match + b * load_match * h * source_match)
            + source_match * e * f * load_match))
}

fn apply_twelve_term(
    measured: &[Network],
    coefficients: &BTreeMap<String, Array1<Complex64>>,
    network: &Network,
) -> Result<Network> {
    validate_two_port_target(measured, network)?;
    let coefficient = |name: &str| {
        coefficients.get(name).ok_or_else(|| {
            Error::Unsupported("the twelve-term calibration has not been run".to_owned())
        })
    };
    let edf = coefficient("forward directivity")?;
    let esf = coefficient("forward source match")?;
    let erf = coefficient("forward reflection tracking")?;
    let etf = coefficient("forward transmission tracking")?;
    let elf = coefficient("forward load match")?;
    let eif = coefficient("forward isolation")?;
    let edr = coefficient("reverse directivity")?;
    let elr = coefficient("reverse load match")?;
    let err = coefficient("reverse reflection tracking")?;
    let etr = coefficient("reverse transmission tracking")?;
    let esr = coefficient("reverse source match")?;
    let eir = coefficient("reverse isolation")?;
    let mut corrected = network.clone();
    for point in 0..network.frequency_points() {
        let s11 = network.s[(point, 0, 0)];
        let s12 = network.s[(point, 0, 1)];
        let s21 = network.s[(point, 1, 0)];
        let s22 = network.s[(point, 1, 1)];
        let forward_reflection = (s11 - edf[point]) / erf[point];
        let reverse_reflection = (s22 - edr[point]) / err[point];
        let forward_transmission = (s21 - eif[point]) / etf[point];
        let reverse_transmission = (s12 - eir[point]) / etr[point];
        let denominator = (Complex64::new(1.0, 0.0) + forward_reflection * esf[point])
            * (Complex64::new(1.0, 0.0) + reverse_reflection * esr[point])
            - forward_transmission * reverse_transmission * elf[point] * elr[point];
        if denominator.norm_sqr() <= f64::EPSILON {
            return Err(Error::Unsupported(
                "twelve-term correction has a zero denominator".to_owned(),
            ));
        }
        corrected.s[(point, 0, 0)] = (forward_reflection
            * (Complex64::new(1.0, 0.0) + reverse_reflection * esr[point])
            - elf[point] * forward_transmission * reverse_transmission)
            / denominator;
        corrected.s[(point, 1, 1)] = (reverse_reflection
            * (Complex64::new(1.0, 0.0) + forward_reflection * esf[point])
            - elr[point] * forward_transmission * reverse_transmission)
            / denominator;
        corrected.s[(point, 1, 0)] = forward_transmission
            * (Complex64::new(1.0, 0.0) + reverse_reflection * (esr[point] - elf[point]))
            / denominator;
        corrected.s[(point, 0, 1)] = reverse_transmission
            * (Complex64::new(1.0, 0.0) + forward_reflection * (esf[point] - elr[point]))
            / denominator;
    }
    Ok(corrected)
}

impl MultiportCal {
    /// Creates a multi-port calibration from pairwise two-port calibrations.
    ///
    /// # Errors
    ///
    /// Returns an error when the pair topology is invalid, measurements and ideals are missing or
    /// incompatible, or the optional isolation network does not match the calibrated network.
    pub fn new(pairs: Vec<MultiportPairCalibration>, isolation: Option<Network>) -> Result<Self> {
        if pairs.len() < 2 {
            return Err(Error::IncompatibleShape(
                "multiport calibration requires at least two port pairs".to_owned(),
            ));
        }
        let nports = pairs
            .iter()
            .flat_map(|pair| pair.ports)
            .max()
            .map_or(0, |port| port + 1);
        if nports < 3 {
            return Err(Error::IncompatibleShape(
                "multiport calibration requires at least three ports".to_owned(),
            ));
        }
        let common_port_count = pairs[0]
            .ports
            .into_iter()
            .filter(|port| pairs.iter().all(|pair| pair.ports.contains(port)))
            .count();
        if common_port_count != 1 {
            return Err(Error::Unsupported(
                "multiport calibration pairs must share exactly one common port".to_owned(),
            ));
        }
        let first = pairs
            .iter()
            .flat_map(|pair| pair.measured.iter())
            .next()
            .ok_or_else(|| {
                Error::IncompatibleShape("multiport calibration has no measurements".to_owned())
            })?;
        for pair in &pairs {
            if pair.ports[0] == pair.ports[1]
                || pair.measured.is_empty()
                || pair.measured.len() != pair.ideals.len()
            {
                return Err(Error::IncompatibleShape(
                    "invalid multiport pair calibration inputs".to_owned(),
                ));
            }
            if pair.measured.iter().any(|network| {
                !matches!(network.ports(), 2) && network.ports() != nports
                    || network.frequency != first.frequency
            }) || pair.ideals.iter().any(|network| {
                !matches!(network.ports(), 2) && network.ports() != nports
                    || network.frequency != first.frequency
            }) {
                return Err(Error::IncompatibleShape(
                    "multiport pair standards must have compatible frequencies and port counts"
                        .to_owned(),
                ));
            }
        }
        let isolation = if let Some(mut network) = isolation {
            if network.ports() != nports || network.frequency != first.frequency {
                return Err(Error::IncompatibleShape(
                    "multiport isolation must match the calibrated network".to_owned(),
                ));
            }
            for point in 0..network.frequency_points() {
                for port in 0..nports {
                    network.s[(point, port, port)] = Complex64::new(0.0, 0.0);
                }
            }
            Some(network)
        } else {
            None
        };
        Ok(Self {
            pairs,
            isolation,
            port_coefficients: vec![BTreeMap::new(); nports],
            pair_coefficients: BTreeMap::new(),
            nports,
        })
    }

    /// Solves each pairwise calibration and assembles the multi-port coefficients.
    ///
    /// # Errors
    ///
    /// Returns an error when a pair subnetwork cannot be extracted, a pairwise calibration cannot
    /// be solved, or the resulting coefficients do not cover every calibrated port.
    pub fn run(&mut self) -> Result<()> {
        let mut counts = vec![0_usize; self.nports];
        for pair in &self.pairs {
            counts[pair.ports[0]] += 1;
            counts[pair.ports[1]] += 1;
        }
        let points = self.pairs[0].measured[0].frequency_points();
        self.port_coefficients = (0..self.nports)
            .map(|_| BTreeMap::from([("k".to_owned(), Array1::ones(points))]))
            .collect();
        self.pair_coefficients.clear();
        for pair in &self.pairs {
            let ports = pair.ports;
            let mut measured = pair
                .measured
                .iter()
                .map(|network| {
                    if network.ports() == 2 {
                        Ok(network.clone())
                    } else {
                        network.subnetwork(&ports)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let ideals = pair
                .ideals
                .iter()
                .map(|network| {
                    if network.ports() == 2 {
                        Ok(network.clone())
                    } else {
                        network.subnetwork(&ports)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            if let Some(isolation) = &self.isolation {
                let isolation = isolation.subnetwork(&ports)?;
                for network in &mut measured {
                    network.s -= &isolation.s;
                }
            }
            let coefficients = match pair.method {
                MultiportPairMethod::EightTerm => {
                    let mut calibration = EightTerm::new(measured, ideals)?;
                    calibration.run()?;
                    calibration.coefficients
                }
                MultiportPairMethod::TwelveTerm => {
                    let mut calibration = TwelveTerm::new(measured, ideals)?;
                    calibration.run()?;
                    eight_term_from_twelve(&calibration.coefficients)?
                }
            };
            for (name, values) in &coefficients {
                if let Some(name) = name.strip_prefix("forward ") {
                    if name == "switch term" {
                        self.port_coefficients[ports[1]].insert(name.to_owned(), values.clone());
                    } else {
                        self.port_coefficients[ports[0]].insert(name.to_owned(), values.clone());
                    }
                } else if let Some(name) = name.strip_prefix("reverse ") {
                    if name == "switch term" {
                        self.port_coefficients[ports[0]].insert(name.to_owned(), values.clone());
                    } else {
                        self.port_coefficients[ports[1]].insert(name.to_owned(), values.clone());
                    }
                }
            }
            let k_side = usize::from(counts[ports[0]] > counts[ports[1]]);
            self.port_coefficients[ports[k_side]].insert("k".to_owned(), coefficients["k"].clone());
            self.pair_coefficients.insert(ports.into(), coefficients);
        }
        for coefficients in &self.port_coefficients {
            for name in ["directivity", "source match", "reflection tracking", "k"] {
                if !coefficients.contains_key(name) {
                    return Err(Error::Unsupported(format!(
                        "multiport calibration did not solve {name} for every port"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Applies the assembled calibration to an uncalibrated multi-port network.
    ///
    /// # Errors
    ///
    /// Returns an error when the target is incompatible, the calibration has not been run, or a
    /// pairwise error transformation is singular.
    pub fn apply(&self, network: &Network) -> Result<Network> {
        self.transform(network, false)
    }

    /// Embeds a calibrated multi-port response in the assembled error model.
    ///
    /// # Errors
    ///
    /// Returns an error when the target is incompatible, the calibration has not been run, or a
    /// pairwise error transformation is singular.
    pub fn embed(&self, network: &Network) -> Result<Network> {
        self.transform(network, true)
    }

    fn transform(&self, network: &Network, embed: bool) -> Result<Network> {
        if network.ports() != self.nports
            || self.port_coefficients.len() != self.nports
            || network.frequency != self.pairs[0].measured[0].frequency
        {
            return Err(Error::IncompatibleShape(
                "multiport calibration target is incompatible or calibration has not been run"
                    .to_owned(),
            ));
        }
        let mut source = network.clone();
        if !embed {
            if let Some(isolation) = &self.isolation {
                source.s -= &isolation.s;
            }
        }
        let mut result = source.clone();
        for first in 0..self.nports {
            for second in first + 1..self.nports {
                let pair = source.subnetwork(&[first, second])?;
                let transformed = transform_multiport_pair(
                    &pair,
                    &self.port_coefficients[first],
                    &self.port_coefficients[second],
                    embed,
                )?;
                for point in 0..network.frequency_points() {
                    result.s[(point, first, first)] = transformed.s[(point, 0, 0)];
                    result.s[(point, first, second)] = transformed.s[(point, 0, 1)];
                    result.s[(point, second, first)] = transformed.s[(point, 1, 0)];
                    result.s[(point, second, second)] = transformed.s[(point, 1, 1)];
                }
            }
        }
        if embed {
            if let Some(isolation) = &self.isolation {
                result.s += &isolation.s;
            }
        }
        Ok(result)
    }
}

impl MultiportSolt {
    /// Creates a multi-port SOLT calibration from measured and ideal standards.
    ///
    /// # Errors
    ///
    /// Returns an error when the standards or thru-port definitions are incompatible, a required
    /// two-port subnetwork cannot be extracted, or the resulting pair topology is invalid.
    pub fn new(
        measured: Vec<Network>,
        ideals: Vec<Network>,
        thru_ports: Vec<[usize; 2]>,
        isolation: Option<Network>,
    ) -> Result<Self> {
        if measured.len() != ideals.len() || ideals.is_empty() {
            return Err(Error::IncompatibleShape(
                "multiport SOLT measured and ideal standards must align".to_owned(),
            ));
        }
        let nports = ideals[0].ports();
        if nports < 3 || thru_ports.len() != nports - 1 || ideals.len() <= thru_ports.len() {
            return Err(Error::IncompatibleShape(
                "multiport SOLT requires N-1 thrus followed by reflect standards".to_owned(),
            ));
        }
        if measured
            .iter()
            .chain(ideals.iter())
            .any(|network| network.ports() != nports || network.frequency != ideals[0].frequency)
        {
            return Err(Error::IncompatibleShape(
                "multiport SOLT standards must be compatible N-port networks".to_owned(),
            ));
        }
        let thru_count = thru_ports.len();
        let pairs = thru_ports
            .iter()
            .enumerate()
            .map(|(index, ports)| {
                let mut pair_measured = measured[thru_count..]
                    .iter()
                    .map(|network| network.subnetwork(ports))
                    .collect::<Result<Vec<_>>>()?;
                let mut pair_ideals = ideals[thru_count..]
                    .iter()
                    .map(|network| network.subnetwork(ports))
                    .collect::<Result<Vec<_>>>()?;
                pair_measured.push(measured[index].subnetwork(ports)?);
                pair_ideals.push(ideals[index].subnetwork(ports)?);
                Ok(MultiportPairCalibration {
                    ports: *ports,
                    measured: pair_measured,
                    ideals: pair_ideals,
                    method: MultiportPairMethod::TwelveTerm,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inner: MultiportCal::new(pairs, isolation)?,
            measured,
            ideals,
            thru_ports,
        })
    }

    /// Solves the underlying pairwise multi-port calibration.
    ///
    /// # Errors
    ///
    /// Returns an error when a pair subnetwork or calibration cannot be solved, or the assembled
    /// coefficients do not cover every calibrated port.
    pub fn run(&mut self) -> Result<()> {
        self.inner.run()
    }

    /// Applies the solved SOLT calibration to a multi-port network.
    ///
    /// # Errors
    ///
    /// Returns an error when the target is incompatible, the calibration has not been run, or a
    /// pairwise error transformation is singular.
    pub fn apply(&self, network: &Network) -> Result<Network> {
        self.inner.apply(network)
    }

    /// Embeds a calibrated multi-port response in the SOLT error model.
    ///
    /// # Errors
    ///
    /// Returns an error when the target is incompatible, the calibration has not been run, or a
    /// pairwise error transformation is singular.
    pub fn embed(&self, network: &Network) -> Result<Network> {
        self.inner.embed(network)
    }
}

fn eight_term_from_twelve(
    coefficients: &BTreeMap<String, Array1<Complex64>>,
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    let points = coefficients
        .values()
        .next()
        .map(Array1::len)
        .ok_or_else(|| Error::Unsupported("twelve-term coefficient set is empty".to_owned()))?;
    let required = |name: &str| {
        coefficients
            .get(name)
            .ok_or_else(|| Error::Unsupported(format!("missing twelve-term coefficient {name}")))
    };
    let mut output = BTreeMap::new();
    for name in [
        "forward directivity",
        "forward source match",
        "forward reflection tracking",
        "reverse directivity",
        "reverse source match",
        "reverse reflection tracking",
        "forward isolation",
        "reverse isolation",
    ] {
        output.insert(name.to_owned(), required(name)?.clone());
    }
    let mut forward_switch = Array1::zeros(points);
    let mut reverse_switch = Array1::zeros(points);
    let mut k = Array1::zeros(points);
    for point in 0..points {
        let value = |name: &str| required(name).map(|values| values[point]);
        let edf = value("forward directivity")?;
        let esf = value("forward source match")?;
        let erf = value("forward reflection tracking")?;
        let etf = value("forward transmission tracking")?;
        let elf = value("forward load match")?;
        let edr = value("reverse directivity")?;
        let esr = value("reverse source match")?;
        let err = value("reverse reflection tracking")?;
        let etr = value("reverse transmission tracking")?;
        let elr = value("reverse load match")?;
        let forward_denominator = err + edr * (elf - esr);
        let reverse_denominator = erf + edf * (elr - esf);
        ensure_nonzero(
            forward_denominator,
            "multiport forward conversion is singular",
        )?;
        ensure_nonzero(
            reverse_denominator,
            "multiport reverse conversion is singular",
        )?;
        ensure_nonzero(etr, "multiport reverse tracking is zero")?;
        forward_switch[point] = (elf - esr) / forward_denominator;
        reverse_switch[point] = (elr - esf) / reverse_denominator;
        k[point] = (etf / forward_denominator + reverse_denominator / etr) / 2.0;
    }
    output.insert("forward switch term".to_owned(), forward_switch);
    output.insert("reverse switch term".to_owned(), reverse_switch);
    output.insert("k".to_owned(), k);
    Ok(output)
}

fn transform_multiport_pair(
    network: &Network,
    first: &BTreeMap<String, Array1<Complex64>>,
    second: &BTreeMap<String, Array1<Complex64>>,
    embed: bool,
) -> Result<Network> {
    let mut result = network.clone();
    for point in 0..network.frequency_points() {
        let edf = multiport_coefficient(first, "directivity")?[point];
        let esf = multiport_coefficient(first, "source match")?[point];
        let erf = multiport_coefficient(first, "reflection tracking")?[point];
        let kf = multiport_coefficient(first, "k")?[point];
        let edr = multiport_coefficient(second, "directivity")?[point];
        let esr = multiport_coefficient(second, "source match")?[point];
        let err = multiport_coefficient(second, "reflection tracking")?[point];
        let kr = multiport_coefficient(second, "k")?[point];
        let forward_switch = second
            .get("switch term")
            .map_or(Complex64::new(0.0, 0.0), |values| values[point]);
        let reverse_switch = first
            .get("switch term")
            .map_or(Complex64::new(0.0, 0.0), |values| values[point]);
        let zero = Complex64::new(0.0, 0.0);
        let t1 = [
            [-kf * (edf * esf - erf), zero],
            [zero, -kr * (edr * esr - err)],
        ];
        let t2 = [[kf * edf, zero], [zero, kr * edr]];
        let t3 = [[-kf * esf, zero], [zero, -kr * esr]];
        let t4 = [[kf, zero], [zero, kr]];
        let mut value = matrix_from_array(&network.s, point);
        if !embed {
            value = unterminate_switch_matrix(value, forward_switch, reverse_switch)?;
        }
        value = if embed {
            matrix_multiply(
                matrix_add(matrix_multiply(t1, value), t2),
                matrix_inverse(matrix_add(matrix_multiply(t3, value), t4))?,
            )
        } else {
            matrix_multiply(
                matrix_inverse(matrix_add(
                    matrix_scale(matrix_multiply(value, t3), -1.0),
                    t1,
                ))?,
                matrix_add(matrix_multiply(value, t4), matrix_scale(t2, -1.0)),
            )
        };
        if embed {
            value = terminate_switch_matrix(value, forward_switch, reverse_switch)?;
        }
        for (row, values) in value.iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                result.s[(point, row, column)] = *value;
            }
        }
    }
    Ok(result)
}

fn multiport_coefficient<'a>(
    coefficients: &'a BTreeMap<String, Array1<Complex64>>,
    name: &str,
) -> Result<&'a Array1<Complex64>> {
    coefficients
        .get(name)
        .ok_or_else(|| Error::Unsupported(format!("missing multiport coefficient {name}")))
}

fn embed_twelve_term(
    measured: &[Network],
    coefficients: &BTreeMap<String, Array1<Complex64>>,
    network: &Network,
) -> Result<Network> {
    validate_two_port_target(measured, network)?;
    let coefficient = |name: &str| {
        coefficients.get(name).ok_or_else(|| {
            Error::Unsupported("the twelve-term calibration has not been run".to_owned())
        })
    };
    let edf = coefficient("forward directivity")?;
    let esf = coefficient("forward source match")?;
    let erf = coefficient("forward reflection tracking")?;
    let etf = coefficient("forward transmission tracking")?;
    let elf = coefficient("forward load match")?;
    let eif = coefficient("forward isolation")?;
    let edr = coefficient("reverse directivity")?;
    let elr = coefficient("reverse load match")?;
    let err = coefficient("reverse reflection tracking")?;
    let etr = coefficient("reverse transmission tracking")?;
    let esr = coefficient("reverse source match")?;
    let eir = coefficient("reverse isolation")?;
    let mut embedded = network.clone();
    for point in 0..network.frequency_points() {
        let s11 = network.s[(point, 0, 0)];
        let s12 = network.s[(point, 0, 1)];
        let s21 = network.s[(point, 1, 0)];
        let s22 = network.s[(point, 1, 1)];
        let determinant = s11 * s22 - s12 * s21;
        let forward_denominator = Complex64::new(1.0, 0.0) - esf[point] * s11 - elf[point] * s22
            + esf[point] * elf[point] * determinant;
        let reverse_denominator = Complex64::new(1.0, 0.0) - elr[point] * s11 - esr[point] * s22
            + esr[point] * elr[point] * determinant;
        embedded.s[(point, 0, 0)] =
            edf[point] + erf[point] * (s11 - elf[point] * determinant) / forward_denominator;
        embedded.s[(point, 1, 0)] = eif[point] + etf[point] * s21 / forward_denominator;
        embedded.s[(point, 1, 1)] =
            edr[point] + err[point] * (s22 - elr[point] * determinant) / reverse_denominator;
        embedded.s[(point, 0, 1)] = eir[point] + etr[point] * s12 / reverse_denominator;
    }
    Ok(embedded)
}

impl Normalization {
    /// Creates a normalization calibration from measured standards.
    ///
    /// # Errors
    ///
    /// Returns an error when no standards are supplied or their frequencies, port counts, or
    /// scattering shapes are incompatible.
    pub fn new(measured: Vec<Network>) -> Result<Self> {
        validate_normalization_networks(&measured)?;
        Ok(Self {
            measured,
            ideals: Vec::new(),
            coefficients: BTreeMap::new(),
        })
    }

    fn average(&self) -> Result<Array3<Complex64>> {
        validate_normalization_networks(&self.measured)?;
        let mut average = Array3::zeros(self.measured[0].s.dim());
        for network in &self.measured {
            average += &network.s;
        }
        let measured_count = self.measured.len().to_f64().ok_or_else(|| {
            Error::Unsupported("normalization standard count cannot be represented".to_owned())
        })?;
        average.mapv_inplace(|value| value / measured_count);
        Ok(average)
    }
}

impl Calibration for Normalization {
    fn measured(&self) -> &[Network] {
        &self.measured
    }

    fn ideals(&self) -> &[Network] {
        &self.ideals
    }

    /// Computes the average measured response used as the normalization divisor.
    fn run(&mut self) -> Result<()> {
        validate_normalization_networks(&self.measured)
    }

    /// Divides a measured response by the average normalization standard.
    fn apply(&self, network: &Network) -> Result<Network> {
        validate_normalization_target(&self.measured, network)?;
        let average = self.average()?;
        let mut corrected = network.clone();
        for (value, normalizer) in corrected.s.iter_mut().zip(average.iter()) {
            if normalizer.norm_sqr() <= f64::EPSILON {
                return Err(Error::Unsupported(
                    "normalization average contains a zero scattering value".to_owned(),
                ));
            }
            *value /= *normalizer;
        }
        Ok(corrected)
    }

    fn embed(&self, network: &Network) -> Result<Network> {
        validate_normalization_target(&self.measured, network)?;
        let average = self.average()?;
        let mut embedded = network.clone();
        embedded.s *= &average;
        Ok(embedded)
    }

    fn coefficients(&self) -> &BTreeMap<String, Array1<Complex64>> {
        &self.coefficients
    }
}

fn validate_one_port_standards(measured: &[Network], ideals: &[Network]) -> Result<()> {
    if measured.len() != ideals.len() || measured.len() < 3 {
        return Err(Error::IncompatibleShape(format!(
            "one-port calibration requires at least three aligned standards, got {} measured and {} ideal",
            measured.len(),
            ideals.len()
        )));
    }
    let frequency = &measured[0].frequency;
    for network in measured.iter().chain(ideals.iter()) {
        if network.ports() != 1 {
            return Err(Error::IncompatibleShape(
                "one-port calibration standards must be one-port Networks".to_owned(),
            ));
        }
        if network.frequency != *frequency {
            return Err(Error::InvalidFrequency(
                "one-port calibration standards must share a frequency axis".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_four_one_port_standards(measured: &[Network], ideals: &[Network]) -> Result<()> {
    validate_one_port_standards(measured, ideals)?;
    if measured.len() != 4 {
        return Err(Error::IncompatibleShape(format!(
            "self-calibration requires exactly four aligned standards, got {}",
            measured.len()
        )));
    }
    Ok(())
}

fn validate_two_port_calibration_standards(measured: &[Network], ideals: &[Network]) -> Result<()> {
    if measured.len() != ideals.len() || measured.len() < 4 {
        return Err(Error::IncompatibleShape(
            "twelve-term calibration requires at least three reflects and one thru".to_owned(),
        ));
    }
    let frequency = &measured[0].frequency;
    if measured
        .iter()
        .chain(ideals)
        .any(|network| network.ports() != 2 || network.frequency != *frequency)
    {
        return Err(Error::IncompatibleShape(
            "twelve-term standards must be frequency-compatible two-port networks".to_owned(),
        ));
    }
    Ok(())
}

fn validate_eight_term_standards(measured: &[Network], ideals: &[Network]) -> Result<()> {
    if measured.len() != ideals.len() || measured.len() < 2 {
        return Err(Error::IncompatibleShape(
            "eight-term calibration requires at least two aligned standards".to_owned(),
        ));
    }
    let frequency = &measured[0].frequency;
    if measured
        .iter()
        .chain(ideals)
        .any(|network| network.ports() != 2 || network.frequency != *frequency)
    {
        return Err(Error::IncompatibleShape(
            "eight-term standards must be frequency-compatible two-port networks".to_owned(),
        ));
    }
    Ok(())
}

fn validate_sixteen_term_standards(measured: &[Network], ideals: &[Network]) -> Result<()> {
    if measured.len() != ideals.len() || measured.len() < 5 {
        return Err(Error::IncompatibleShape(
            "sixteen-term calibration requires at least five aligned standards".to_owned(),
        ));
    }
    let frequency = &measured[0].frequency;
    if measured
        .iter()
        .chain(ideals)
        .any(|network| network.ports() != 2 || network.frequency != *frequency)
    {
        return Err(Error::IncompatibleShape(
            "sixteen-term standards must be frequency-compatible two-port networks".to_owned(),
        ));
    }
    Ok(())
}

fn validate_unknown_thru_standards(measured: &[Network], ideals: &[Network]) -> Result<()> {
    if measured.len() != ideals.len() || measured.len() < 4 {
        return Err(Error::IncompatibleShape(
            "unknown-thru calibration requires at least three reflects and a final thru".to_owned(),
        ));
    }
    let frequency = &measured[0].frequency;
    if measured
        .iter()
        .chain(ideals)
        .any(|network| network.ports() != 2 || network.frequency != *frequency)
    {
        return Err(Error::IncompatibleShape(
            "unknown-thru standards must be frequency-compatible two-port networks".to_owned(),
        ));
    }
    Ok(())
}

fn validate_named_standard_count(
    measured: &[Network],
    ideals: &[Network],
    expected: usize,
    family: &str,
) -> Result<()> {
    if measured.len() != expected || ideals.len() != expected {
        return Err(Error::IncompatibleShape(format!(
            "{family} requires exactly {expected} aligned standards"
        )));
    }
    let frequency = &measured[0].frequency;
    if measured
        .iter()
        .chain(ideals)
        .any(|network| network.ports() != 2 || network.frequency != *frequency)
    {
        return Err(Error::IncompatibleShape(format!(
            "{family} standards must be frequency-compatible two-port networks"
        )));
    }
    Ok(())
}

fn validate_two_port_target(measured: &[Network], network: &Network) -> Result<()> {
    let first = measured.first().ok_or_else(|| {
        Error::IncompatibleShape("calibration has no measured standards".to_owned())
    })?;
    if network.ports() != 2 || network.frequency != first.frequency {
        return Err(Error::IncompatibleShape(
            "two-port calibration target must share frequency and port count".to_owned(),
        ));
    }
    Ok(())
}

fn validate_normalization_networks(measured: &[Network]) -> Result<()> {
    let first = measured.first().ok_or_else(|| {
        Error::IncompatibleShape("normalization requires measured networks".to_owned())
    })?;
    for network in measured.iter().skip(1) {
        if network.frequency != first.frequency {
            return Err(Error::InvalidFrequency(
                "normalization measurements must share a frequency axis".to_owned(),
            ));
        }
        if network.s.dim() != first.s.dim() {
            return Err(Error::IncompatibleShape(
                "normalization measurements must have equal scattering shapes".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_normalization_target(measured: &[Network], network: &Network) -> Result<()> {
    validate_normalization_networks(measured)?;
    let first = &measured[0];
    if network.frequency != first.frequency {
        return Err(Error::InvalidFrequency(
            "normalization and target must share a frequency axis".to_owned(),
        ));
    }
    if network.s.dim() != first.s.dim() {
        return Err(Error::IncompatibleShape(
            "normalization and target must have equal scattering shapes".to_owned(),
        ));
    }
    Ok(())
}

fn validate_one_port_target(calibration: &OnePort, network: &Network) -> Result<()> {
    if network.ports() != 1 {
        return Err(Error::IncompatibleShape(
            "one-port calibration can only process a one-port Network".to_owned(),
        ));
    }
    if network.frequency != calibration.measured[0].frequency {
        return Err(Error::InvalidFrequency(
            "calibration and target must share a frequency axis".to_owned(),
        ));
    }
    Ok(())
}

fn coefficient<'a>(calibration: &'a OnePort, name: &str) -> Result<&'a Array1<Complex64>> {
    calibration
        .coefficients
        .get(name)
        .ok_or_else(|| Error::Unsupported("the one-port calibration has not been run".to_owned()))
}

fn solve_three_by_three(
    mut matrix: [[Complex64; 3]; 3],
    mut right: [Complex64; 3],
) -> Option<[Complex64; 3]> {
    for pivot in 0..3 {
        let best = (pivot..3).max_by(|left, right_index| {
            matrix[*left][pivot]
                .norm_sqr()
                .total_cmp(&matrix[*right_index][pivot].norm_sqr())
        })?;
        if matrix[best][pivot].norm_sqr() <= f64::EPSILON {
            return None;
        }
        matrix.swap(pivot, best);
        right.swap(pivot, best);
        let pivot_row = matrix[pivot];
        for row in pivot + 1..3 {
            let multiplier = matrix[row][pivot] / pivot_row[pivot];
            for (value, pivot_value) in matrix[row][pivot..]
                .iter_mut()
                .zip(pivot_row[pivot..].iter())
            {
                *value -= multiplier * pivot_value;
            }
            right[row] -= multiplier * right[pivot];
        }
    }
    let mut solution = [Complex64::new(0.0, 0.0); 3];
    for row in (0..3).rev() {
        let tail = matrix[row][row + 1..]
            .iter()
            .zip(solution[row + 1..].iter())
            .map(|(coefficient, value)| coefficient * value)
            .sum::<Complex64>();
        solution[row] = (right[row] - tail) / matrix[row][row];
    }
    Some(solution)
}

/// Removes forward and reverse switch-term loading from a raw two-port measurement.
///
/// The switch terms are
///
/// $$
/// \Gamma_{f} = \frac{a_{2}}{b_{2}}, \qquad \Gamma_{r} = \frac{a_{1}}{b_{1}}.
/// $$
///
/// Four-sampler VNAs can measure these ratios directly. Otherwise they can be
/// obtained from a two-tier calibration or estimated with [`compute_switch_terms`].
///
/// # Errors
///
/// Returns an error when the network is not a two-port, the switch-term arrays do not match its
/// frequency points, or a switch-term correction matrix is singular.
///
/// ## Reference
///
/// Roger B. Marks, *Formulations of the Basic Vector Network Analyzer Error Model
/// including Switch Terms*.
///
/// Origin: `skrf/calibration/calibration.py::unterminate`.
pub fn unterminate(
    network: &Network,
    forward_switch: &Array1<Complex64>,
    reverse_switch: &Array1<Complex64>,
) -> Result<Network> {
    validate_switch_terms(network, forward_switch, reverse_switch)?;
    let mut result = network.clone();
    for point in 0..network.frequency_points() {
        let value = unterminate_switch_matrix(
            matrix_from_array(&network.s, point),
            forward_switch[point],
            reverse_switch[point],
        )?;
        for (row, values) in value.iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                result.s[(point, row, column)] = *value;
            }
        }
    }
    Ok(result)
}

/// Applies forward and reverse VNA switch-term loading to a two-port network.
///
/// This is the inverse of [`unterminate`]. The supplied arrays contain one value
/// of $\Gamma_{f}$ and $\Gamma_{r}$ per frequency point.
///
/// # Errors
///
/// Returns an error when the network is not a two-port, the switch-term arrays do not match its
/// frequency points, or a switch-term loading matrix is singular.
///
/// ## Reference
///
/// Roger B. Marks, *Formulations of the Basic Vector Network Analyzer Error Model
/// including Switch Terms*.
///
/// Origin: `skrf/calibration/calibration.py::terminate`.
pub fn terminate(
    network: &Network,
    forward_switch: &Array1<Complex64>,
    reverse_switch: &Array1<Complex64>,
) -> Result<Network> {
    validate_switch_terms(network, forward_switch, reverse_switch)?;
    let mut result = network.clone();
    for point in 0..network.frequency_points() {
        let value = terminate_switch_matrix(
            matrix_from_array(&network.s, point),
            forward_switch[point],
            reverse_switch[point],
        )?;
        for (row, values) in value.iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                result.s[(point, row, column)] = *value;
            }
        }
    }
    Ok(result)
}

fn validate_switch_terms(
    network: &Network,
    forward_switch: &Array1<Complex64>,
    reverse_switch: &Array1<Complex64>,
) -> Result<()> {
    if network.ports() != 2
        || forward_switch.len() != network.frequency_points()
        || reverse_switch.len() != network.frequency_points()
    {
        return Err(Error::IncompatibleShape(
            "switch-term correction requires a two-port network and one value per frequency"
                .to_owned(),
        ));
    }
    Ok(())
}

fn unterminate_switch_matrix(
    measured: ComplexMatrix2,
    forward_switch: Complex64,
    reverse_switch: Complex64,
) -> Result<ComplexMatrix2> {
    let denominator = Complex64::new(1.0, 0.0)
        - measured[0][1] * measured[1][0] * reverse_switch * forward_switch;
    ensure_nonzero(denominator, "switch-term untermination is singular")?;
    Ok([
        [
            (measured[0][0] - measured[0][1] * measured[1][0] * forward_switch) / denominator,
            (measured[0][1] - measured[0][0] * measured[0][1] * reverse_switch) / denominator,
        ],
        [
            (measured[1][0] - measured[1][1] * measured[1][0] * forward_switch) / denominator,
            (measured[1][1] - measured[0][1] * measured[1][0] * reverse_switch) / denominator,
        ],
    ])
}

fn terminate_switch_matrix(
    network: ComplexMatrix2,
    forward_switch: Complex64,
    reverse_switch: Complex64,
) -> Result<ComplexMatrix2> {
    let forward_denominator = Complex64::new(1.0, 0.0) - network[1][1] * forward_switch;
    let reverse_denominator = Complex64::new(1.0, 0.0) - network[0][0] * reverse_switch;
    ensure_nonzero(
        forward_denominator,
        "forward switch-term termination is singular",
    )?;
    ensure_nonzero(
        reverse_denominator,
        "reverse switch-term termination is singular",
    )?;
    Ok([
        [
            network[0][0] + network[1][0] * network[0][1] * forward_switch / forward_denominator,
            network[0][1] / reverse_denominator,
        ],
        [
            network[1][0] / forward_denominator,
            network[1][1] + network[1][0] * network[0][1] * reverse_switch / reverse_denominator,
        ],
    ])
}

/// Returns the ideal twelve-term calibration coefficients for a frequency axis.
///
/// Origin: `skrf/calibration/calibration.py::ideal_coefs_12term`.
#[must_use]
pub fn ideal_coefs_12term(frequency: &Frequency) -> BTreeMap<String, Array1<Complex64>> {
    let zeros = Array1::zeros(frequency.points());
    let ones = Array1::from_elem(frequency.points(), Complex64::new(1.0, 0.0));
    let mut coefficients = BTreeMap::new();
    for name in [
        "forward directivity",
        "forward source match",
        "forward load match",
        "reverse directivity",
        "reverse load match",
        "reverse source match",
        "forward isolation",
        "reverse isolation",
    ] {
        coefficients.insert(name.to_owned(), zeros.clone());
    }
    for name in [
        "forward reflection tracking",
        "forward transmission tracking",
        "reverse reflection tracking",
        "reverse transmission tracking",
    ] {
        coefficients.insert(name.to_owned(), ones.clone());
    }
    coefficients
}

/// Applies switch-term loading to every sourced column of an N-port network.
///
/// `gammas[port]` is $a_{p} / b_{p}$ while any other port is sourced.
/// For two ports the order is therefore reverse then forward, matching
/// `skrf.calibration.terminate_nport`.
///
/// # Errors
///
/// Returns an error when the number or shape of the switch-term networks is incompatible with the
/// target, or an N-port loading system is singular.
///
/// ## Reference
///
/// Roger B. Marks, *Formulations of the Basic Vector Network Analyzer Error Model
/// including Switch Terms*.
///
/// Origin: `skrf/calibration/calibration.py::terminate_nport`.
pub fn terminate_nport(network: &Network, gammas: &[Network]) -> Result<Network> {
    let ports = network.ports();
    if gammas.len() != ports {
        return Err(Error::IncompatibleShape(format!(
            "{} switch terms were supplied for a {ports}-port network",
            gammas.len()
        )));
    }
    for gamma in gammas {
        if gamma.ports() != 1
            || gamma.frequency_points() != network.frequency_points()
            || gamma.frequency != network.frequency
        {
            return Err(Error::IncompatibleShape(
                "every N-port switch term must be a one-port Network on the target frequency axis"
                    .to_owned(),
            ));
        }
    }

    let mut terminated = network.clone();
    for point in 0..network.frequency_points() {
        for source in 0..ports {
            let matrix = (0..ports)
                .map(|row| {
                    (0..ports)
                        .map(|column| {
                            let identity = if row == column {
                                Complex64::new(1.0, 0.0)
                            } else {
                                Complex64::new(0.0, 0.0)
                            };
                            if column == source {
                                identity
                            } else {
                                identity
                                    - network.s[(point, row, column)]
                                        * gammas[column].s[(point, 0, 0)]
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let incident = (0..ports)
                .map(|row| network.s[(point, row, source)])
                .collect::<Vec<_>>();
            let outgoing = solve_complex_system(matrix, incident).ok_or_else(|| {
                Error::Unsupported(format!(
                    "N-port switch-term termination is singular at frequency point {point}"
                ))
            })?;
            for (row, value) in outgoing.into_iter().enumerate() {
                terminated.s[(point, row, source)] = value;
            }
        }
    }
    Ok(terminated)
}

/// Indirectly estimates forward and reverse switch terms from at least three
/// measured reciprocal two-port devices.
///
/// The return order is $(\Gamma_{21}, \Gamma_{12})$, or forward then reverse.
/// Accuracy depends on the reciprocal devices being sufficiently distinct; using
/// more than three devices produces an overdetermined estimate.
///
/// # Errors
///
/// Returns an error when fewer than three aligned two-port measurements are supplied, a device has
/// zero forward transmission, or the switch-term null-space solution is singular.
///
/// ## References
///
/// - Z. Hatab, M. E. Gadringer, and W. Bösch, *Indirect Measurement of Switch Terms of a Vector Network Analyzer with Reciprocal Devices*, [arXiv:2306.07066](https://arxiv.org/abs/2306.07066).
/// - [VNA switch-term notes](https://ziadhatab.github.io/posts/vna-switch-terms/).
///
/// Origin: `skrf/calibration/calibration.py::compute_switch_terms`.
pub fn compute_switch_terms(networks: &[Network]) -> Result<(Network, Network)> {
    if networks.len() < 3 {
        return Err(Error::IncompatibleShape(
            "at least three reciprocal-device measurements are required".to_owned(),
        ));
    }
    let reference = &networks[0];
    if reference.ports() != 2 {
        return Err(Error::IncompatibleShape(
            "switch-term estimation requires two-port measurements".to_owned(),
        ));
    }
    for network in networks {
        if network.ports() != 2 || network.frequency != reference.frequency {
            return Err(Error::IncompatibleShape(
                "switch-term measurements must be aligned two-port Networks".to_owned(),
            ));
        }
    }

    let points = reference.frequency_points();
    let mut forward = Array1::zeros(points);
    let mut reverse = Array1::zeros(points);
    for point in 0..points {
        if networks
            .iter()
            .any(|network| network.s[(point, 1, 0)].norm_sqr() <= f64::EPSILON)
        {
            return Err(Error::Unsupported(format!(
                "switch-term measurement has zero forward transmission at frequency point {point}"
            )));
        }
        let system = faer::Mat::<Complex64>::from_fn(networks.len(), 4, |row, column| {
            let network = &networks[row];
            let s10 = network.s[(point, 1, 0)];
            match column {
                0 => -network.s[(point, 0, 0)] * network.s[(point, 0, 1)] / s10,
                1 => -network.s[(point, 1, 1)],
                2 => Complex64::new(1.0, 0.0),
                3 => network.s[(point, 0, 1)] / s10,
                _ => unreachable!(),
            }
        });
        let decomposition = system
            .svd()
            .map_err(|error| Error::Unsupported(format!("switch-term SVD failed: {error:?}")))?;
        let vectors = decomposition.V();
        let null_index = vectors.ncols() - 1;
        ensure_nonzero(
            vectors[(2, null_index)],
            "switch-term forward normalization is singular",
        )?;
        ensure_nonzero(
            vectors[(3, null_index)],
            "switch-term reverse normalization is singular",
        )?;
        forward[point] = vectors[(1, null_index)] / vectors[(2, null_index)];
        reverse[point] = vectors[(0, null_index)] / vectors[(3, null_index)];
    }

    Ok((
        switch_term_network(reference, &forward, "Gamma21")?,
        switch_term_network(reference, &reverse, "Gamma12")?,
    ))
}

/// Converts twelve-term coefficients to the equivalent eight-term model.
///
/// When `redundant_k` is true, the independently derived $k$ estimates are also
/// returned as `k first` and `k second`.
///
/// # Errors
///
/// Returns an error when required coefficients are missing or have inconsistent lengths, or when
/// the twelve-to-eight-term conversion is singular.
///
/// ## Reference
///
/// Roger B. Marks, *Formulations of the Basic Vector Network Analyzer Error Model
/// including Switch-Terms*, ARFTG Conference Digest, 1997.
///
/// Origin: `skrf/calibration/calibration.py::convert_12term_2_8term`.
pub fn convert_12term_2_8term(
    coefficients: &BTreeMap<String, Array1<Complex64>>,
    redundant_k: bool,
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    let points = coefficient_points(coefficients)?;
    let edf = required_coefficient(coefficients, "forward directivity", points)?;
    let esf = required_coefficient(coefficients, "forward source match", points)?;
    let erf = required_coefficient(coefficients, "forward reflection tracking", points)?;
    let etf = required_coefficient(coefficients, "forward transmission tracking", points)?;
    let elf = required_coefficient(coefficients, "forward load match", points)?;
    let edr = required_coefficient(coefficients, "reverse directivity", points)?;
    let esr = required_coefficient(coefficients, "reverse source match", points)?;
    let err = required_coefficient(coefficients, "reverse reflection tracking", points)?;
    let elr = required_coefficient(coefficients, "reverse load match", points)?;
    let etr = required_coefficient(coefficients, "reverse transmission tracking", points)?;

    let mut gamma_f = Array1::zeros(points);
    let mut gamma_r = Array1::zeros(points);
    let mut k_first = Array1::zeros(points);
    let mut k_second = Array1::zeros(points);
    for point in 0..points {
        let forward_denominator = err[point] + edr[point] * (elf[point] - esr[point]);
        let reverse_denominator = erf[point] + edf[point] * (elr[point] - esf[point]);
        ensure_nonzero(
            forward_denominator,
            "twelve-to-eight forward conversion is singular",
        )?;
        ensure_nonzero(
            reverse_denominator,
            "twelve-to-eight reverse conversion is singular",
        )?;
        ensure_nonzero(etr[point], "twelve-to-eight reverse tracking is zero")?;
        gamma_f[point] = (elf[point] - esr[point]) / forward_denominator;
        gamma_r[point] = (elr[point] - esf[point]) / reverse_denominator;
        k_first[point] = etf[point] / forward_denominator;
        k_second[point] = reverse_denominator / etr[point];
    }
    let k = Array1::from_shape_fn(points, |point| (k_first[point] + k_second[point]) / 2.0);

    let mut converted = BTreeMap::new();
    for name in [
        "forward directivity",
        "forward source match",
        "forward reflection tracking",
        "reverse directivity",
        "reverse reflection tracking",
        "reverse source match",
        "forward isolation",
        "reverse isolation",
    ] {
        converted.insert(
            name.to_owned(),
            required_coefficient(coefficients, name, points)?.clone(),
        );
    }
    converted.insert("forward switch term".to_owned(), gamma_f);
    converted.insert("reverse switch term".to_owned(), gamma_r);
    converted.insert("k".to_owned(), k);
    if redundant_k {
        converted.insert("k first".to_owned(), k_first);
        converted.insert("k second".to_owned(), k_second);
    }
    Ok(converted)
}

/// Converts eight-term coefficients to the equivalent twelve-term model.
///
/// The conversion uses the switch terms and $k$ coefficient and preserves the
/// directional isolation terms.
///
/// # Errors
///
/// Returns an error when required coefficients are missing or have inconsistent lengths, or when
/// the eight-to-twelve-term conversion is singular.
///
/// Origin: `skrf/calibration/calibration.py::convert_8term_2_12term`.
pub fn convert_8term_2_12term(
    coefficients: &BTreeMap<String, Array1<Complex64>>,
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    let points = coefficient_points(coefficients)?;
    let edf = required_coefficient(coefficients, "forward directivity", points)?;
    let esf = required_coefficient(coefficients, "forward source match", points)?;
    let erf = required_coefficient(coefficients, "forward reflection tracking", points)?;
    let edr = required_coefficient(coefficients, "reverse directivity", points)?;
    let esr = required_coefficient(coefficients, "reverse source match", points)?;
    let err = required_coefficient(coefficients, "reverse reflection tracking", points)?;
    let gamma_f = required_coefficient(coefficients, "forward switch term", points)?;
    let gamma_r = required_coefficient(coefficients, "reverse switch term", points)?;
    let k = required_coefficient(coefficients, "k", points)?;
    let k_first = coefficients.get("k first").unwrap_or(k);
    let k_second = coefficients.get("k second").unwrap_or(k);
    if k_first.len() != points || k_second.len() != points {
        return Err(Error::IncompatibleShape(
            "redundant k coefficients must have the common coefficient length".to_owned(),
        ));
    }

    let forward_zero = gamma_f.iter().all(|value| value.norm() <= 1.0e-12);
    let reverse_zero = gamma_r.iter().all(|value| value.norm() <= 1.0e-12);
    let mut elf = Array1::zeros(points);
    let mut elr = Array1::zeros(points);
    let mut etf = Array1::zeros(points);
    let mut etr = Array1::zeros(points);
    for point in 0..points {
        if forward_zero {
            elf[point] = esr[point];
            etf[point] = err[point] * k_first[point];
        } else {
            let denominator = Complex64::new(1.0, 0.0) - edr[point] * gamma_f[point];
            ensure_nonzero(
                denominator,
                "eight-to-twelve forward conversion is singular",
            )?;
            ensure_nonzero(
                gamma_f[point],
                "forward switch term contains an isolated zero",
            )?;
            elf[point] = esr[point] + err[point] * gamma_f[point] / denominator;
            etf[point] = (elf[point] - esr[point]) / gamma_f[point] * k_first[point];
        }
        if reverse_zero {
            elr[point] = esf[point];
            ensure_nonzero(k_second[point], "reverse k coefficient is zero")?;
            etr[point] = erf[point] / k_second[point];
        } else {
            let denominator = Complex64::new(1.0, 0.0) - edf[point] * gamma_r[point];
            ensure_nonzero(
                denominator,
                "eight-to-twelve reverse conversion is singular",
            )?;
            ensure_nonzero(
                gamma_r[point],
                "reverse switch term contains an isolated zero",
            )?;
            ensure_nonzero(k_second[point], "reverse k coefficient is zero")?;
            elr[point] = esf[point] + erf[point] * gamma_r[point] / denominator;
            etr[point] = (elr[point] - esf[point]) / gamma_r[point] / k_second[point];
        }
    }

    let mut converted = BTreeMap::new();
    converted.insert("forward load match".to_owned(), elf);
    converted.insert("reverse load match".to_owned(), elr);
    converted.insert("forward transmission tracking".to_owned(), etf);
    converted.insert("reverse transmission tracking".to_owned(), etr);
    for name in [
        "forward directivity",
        "forward source match",
        "forward reflection tracking",
        "reverse directivity",
        "reverse reflection tracking",
        "reverse source match",
        "forward isolation",
        "reverse isolation",
    ] {
        converted.insert(
            name.to_owned(),
            required_coefficient(coefficients, name, points)?.clone(),
        );
    }
    Ok(converted)
}

fn coefficient_points(coefficients: &BTreeMap<String, Array1<Complex64>>) -> Result<usize> {
    coefficients
        .values()
        .next()
        .map(Array1::len)
        .filter(|points| *points > 0)
        .ok_or_else(|| Error::IncompatibleShape("coefficient dictionary is empty".to_owned()))
}

fn required_coefficient<'a>(
    coefficients: &'a BTreeMap<String, Array1<Complex64>>,
    name: &str,
    points: usize,
) -> Result<&'a Array1<Complex64>> {
    let values = coefficients
        .get(name)
        .ok_or_else(|| Error::Unsupported(format!("missing calibration coefficient '{name}'")))?;
    if values.len() != points {
        return Err(Error::IncompatibleShape(format!(
            "calibration coefficient '{name}' has {} points instead of {points}",
            values.len()
        )));
    }
    Ok(values)
}

fn coefficient_network(
    frequency: Frequency,
    values: &Array1<Complex64>,
    name: &str,
) -> Result<Network> {
    let points = frequency.points();
    if values.len() != points {
        return Err(Error::IncompatibleShape(format!(
            "coefficient network has {} values for {points} frequency points",
            values.len()
        )));
    }
    let scattering = Array3::from_shape_fn((points, 1, 1), |(point, _, _)| values[point]);
    let z0 = Array2::from_elem((points, 1), Complex64::new(50.0, 0.0));
    let mut network = Network::new(frequency, scattering, z0)?;
    network.name = Some(name.to_owned());
    Ok(network)
}

fn group_networks_by_ideal_name(
    networks: &[Network],
    ideals: &[Network],
) -> Result<BTreeMap<String, NetworkSet>> {
    let mut names = ideals
        .iter()
        .map(|ideal| {
            ideal.name.clone().ok_or_else(|| {
                Error::Unsupported(
                    "calibration error grouping requires named ideal standards".to_owned(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    names.sort();
    names.dedup();
    names
        .into_iter()
        .map(|name| {
            let members = networks
                .iter()
                .filter(|network| {
                    network
                        .name
                        .as_deref()
                        .is_some_and(|network_name| network_name.starts_with(&name))
                })
                .cloned()
                .collect::<Vec<_>>();
            let set = NetworkSet::new(members, Some(name.clone()))?;
            Ok((name, set))
        })
        .collect()
}

fn switch_term_network(
    reference: &Network,
    values: &Array1<Complex64>,
    name: &str,
) -> Result<Network> {
    let points = reference.frequency_points();
    let scattering = Array3::from_shape_fn((points, 1, 1), |(point, _, _)| values[point]);
    let z0 = Array2::from_shape_fn((points, 1), |(point, _)| reference.z0[(point, 0)]);
    let mut network = Network::new(reference.frequency.clone(), scattering, z0)?;
    network.name = Some(name.to_owned());
    Ok(network)
}

/// Converts Keysight PNA coefficient names to scikit-rf coefficient names.
///
/// # Errors
///
/// Returns an error when a PNA coefficient name or term is invalid, or the coefficients describe
/// an unsupported number of ports.
///
/// Origin: `skrf/calibration/calibration.py::convert_pnacoefs_2_skrf`.
pub fn convert_pnacoefs_2_skrf(
    coefficients: &BTreeMap<String, Array1<Complex64>>,
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    let mut parsed = Vec::with_capacity(coefficients.len());
    for (name, values) in coefficients {
        let (term, source, receiver) = parse_pna_coefficient_name(name)?;
        parsed.push((term, source, receiver, values));
    }
    let mut converted = BTreeMap::new();
    if coefficients.len() == 3 {
        for (term, _, _, values) in parsed {
            converted.insert(pna_term_to_skrf(term)?.to_owned(), values.clone());
        }
        return Ok(converted);
    }

    let mut ports = parsed
        .iter()
        .flat_map(|(_, source, receiver, _)| [*source, *receiver])
        .collect::<Vec<_>>();
    ports.sort_unstable();
    ports.dedup();
    if ports.len() != 2 {
        return Err(Error::Unsupported(format!(
            "PNA coefficient conversion requires one or two ports, found {ports:?}"
        )));
    }
    for (term, source, receiver, values) in parsed {
        let direction_port = if matches!(term, "Directivity" | "SourceMatch" | "ReflectionTracking")
        {
            source
        } else {
            receiver
        };
        let direction = if direction_port == ports[0] {
            "forward"
        } else {
            "reverse"
        };
        converted.insert(
            format!("{direction} {}", pna_term_to_skrf(term)?),
            values.clone(),
        );
    }
    Ok(converted)
}

/// Converts scikit-rf coefficient names to Keysight PNA coefficient names.
///
/// `ports` is `(forward, reverse)` and defaults to `(1, 2)` in upstream.
///
/// # Errors
///
/// Returns an error when the port indices are invalid or a coefficient name, direction, or term
/// cannot be converted to the PNA convention.
///
/// Origin: `skrf/calibration/calibration.py::convert_skrfcoefs_2_pna`.
pub fn convert_skrfcoefs_2_pna(
    coefficients: &BTreeMap<String, Array1<Complex64>>,
    ports: (usize, usize),
) -> Result<BTreeMap<String, Array1<Complex64>>> {
    if ports.0 == ports.1 || ports.0 == 0 || ports.1 == 0 {
        return Err(Error::Unsupported(
            "PNA coefficient ports must be distinct positive indices".to_owned(),
        ));
    }
    let mut converted = BTreeMap::new();
    if coefficients.len() == 3 {
        for (name, values) in coefficients {
            let term = skrf_term_to_pna(name)?;
            converted.insert(format!("{term}({},{})", ports.0, ports.0), values.clone());
        }
        return Ok(converted);
    }

    for (name, values) in coefficients {
        let (direction, term_name) = name.split_once(' ').ok_or_else(|| {
            Error::Unsupported(format!("invalid directional coefficient name '{name}'"))
        })?;
        let term = skrf_term_to_pna(term_name)?;
        let (source, other) = match direction {
            "forward" => (ports.0, ports.1),
            "reverse" => (ports.1, ports.0),
            _ => {
                return Err(Error::Unsupported(format!(
                    "invalid coefficient direction '{direction}'"
                )));
            }
        };
        let (receiver, source) =
            if matches!(term, "Directivity" | "SourceMatch" | "ReflectionTracking") {
                (source, source)
            } else {
                (other, source)
            };
        converted.insert(format!("{term}({receiver},{source})"), values.clone());
    }
    Ok(converted)
}

/// Aligns measured and ideal standards by ideal-name containment in the
/// measured network name.
///
/// Origin: `skrf/calibration/calibration.py::align_measured_ideals`.
#[must_use]
pub fn align_measured_ideals(
    measured: &[Network],
    ideals: &[Network],
) -> (Vec<Network>, Vec<Network>) {
    let aligned_measured = measured
        .iter()
        .flat_map(|measurement| {
            ideals.iter().filter_map(move |ideal| {
                let ideal_name = ideal.name.as_deref()?;
                let measured_name = measurement.name.as_deref()?;
                measured_name
                    .contains(ideal_name)
                    .then(|| measurement.clone())
            })
        })
        .collect::<Vec<_>>();
    let aligned_ideals = aligned_measured
        .iter()
        .flat_map(|measurement| {
            ideals.iter().filter_map(move |ideal| {
                let ideal_name = ideal.name.as_deref()?;
                let measured_name = measurement.name.as_deref()?;
                measured_name.contains(ideal_name).then(|| ideal.clone())
            })
        })
        .collect();
    (aligned_measured, aligned_ideals)
}

/// Converts the compact two-port error vector to four T-matrix arrays.
///
/// # Errors
///
/// Returns an error when the coefficient map is empty or a required coefficient is missing or has
/// a different length from the other coefficients.
///
/// Origin: `skrf/calibration/calibration.py::two_port_error_vector_2_Ts`.
pub fn two_port_error_vector_2_ts(
    coefficients: &BTreeMap<String, Array1<Complex64>>,
) -> Result<FourComplexArrayMatrices> {
    let points = coefficient_points(coefficients)?;
    let det_x = required_coefficient(coefficients, "det_X", points)?;
    let det_y = required_coefficient(coefficients, "det_Y", points)?;
    let e00 = required_coefficient(coefficients, "e00", points)?;
    let e11 = required_coefficient(coefficients, "e11", points)?;
    let e22 = required_coefficient(coefficients, "e22", points)?;
    let e33 = required_coefficient(coefficients, "e33", points)?;
    let k = required_coefficient(coefficients, "k", points)?;
    let diagonal = |first: &Array1<Complex64>, second: &Array1<Complex64>| {
        Array3::from_shape_fn((points, 2, 2), |(point, row, column)| {
            if row != column {
                Complex64::new(0.0, 0.0)
            } else if row == 0 {
                first[point]
            } else {
                second[point]
            }
        })
    };
    let ones = Array1::from_elem(points, Complex64::new(1.0, 0.0));
    Ok((
        diagonal(
            &det_x.mapv(|value| -value),
            &Array1::from_shape_fn(points, |point| -k[point] * det_y[point]),
        ),
        diagonal(
            e00,
            &Array1::from_shape_fn(points, |point| k[point] * e33[point]),
        ),
        diagonal(
            &e11.mapv(|value| -value),
            &Array1::from_shape_fn(points, |point| -k[point] * e22[point]),
        ),
        diagonal(&ones, k),
    ))
}

/// Typed result of `error_dict_2_network`, whose Python return shape depends
/// on whether three-term or directional coefficients were supplied.
#[derive(Clone, Debug, PartialEq)]
pub enum ErrorNetworkResult {
    /// A single reciprocal two-port error network for a three-term calibration.
    One(Box<Network>),
    /// Directional forward and reverse error networks.
    Pair {
        /// Error network for forward excitation.
        forward: Box<Network>,
        /// Error network for reverse excitation.
        reverse: Box<Network>,
    },
}

/// Creates one or two error networks from standard calibration coefficients.
///
/// # Errors
///
/// Returns an error when required coefficients are missing or do not match the frequency length,
/// or an error network cannot be constructed from the supplied data.
///
/// Origin: `skrf/calibration/calibration.py::error_dict_2_network`.
pub fn error_dict_2_network(
    coefficients: &BTreeMap<String, Array1<Complex64>>,
    frequency: &Frequency,
    reciprocal: bool,
) -> Result<ErrorNetworkResult> {
    if coefficients.len() == 3 {
        let points = frequency.points();
        let directivity = required_coefficient(coefficients, "directivity", points)?;
        let source_match = required_coefficient(coefficients, "source match", points)?;
        let tracking = required_coefficient(coefficients, "reflection tracking", points)?;
        let (forward, reverse) = if reciprocal {
            let root = sqrt_phase_unwrap(tracking);
            (root.clone(), root)
        } else {
            (
                tracking.clone(),
                Array1::from_elem(points, Complex64::new(1.0, 0.0)),
            )
        };
        let scattering =
            Array3::from_shape_fn((points, 2, 2), |(point, row, column)| match (row, column) {
                (0, 0) => directivity[point],
                (0, 1) => reverse[point],
                (1, 0) => forward[point],
                (1, 1) => source_match[point],
                _ => unreachable!(),
            });
        let z0 = Array2::from_elem((points, 2), Complex64::new(50.0, 0.0));
        return Ok(ErrorNetworkResult::One(Box::new(Network::new(
            frequency.clone(),
            scattering,
            z0,
        )?)));
    }

    let directional = |direction: &str| {
        ["source match", "directivity", "reflection tracking"]
            .into_iter()
            .map(|term| {
                let name = format!("{direction} {term}");
                coefficients
                    .get(&name)
                    .cloned()
                    .map(|value| (term.to_owned(), value))
                    .ok_or_else(|| {
                        Error::Unsupported(format!("missing calibration coefficient '{name}'"))
                    })
            })
            .collect::<Result<BTreeMap<_, _>>>()
    };
    let forward = match error_dict_2_network(&directional("forward")?, frequency, reciprocal)? {
        ErrorNetworkResult::One(mut network) => {
            network.name = Some("forward".to_owned());
            *network
        }
        ErrorNetworkResult::Pair { .. } => unreachable!(),
    };
    let reverse = match error_dict_2_network(&directional("reverse")?, frequency, reciprocal)? {
        ErrorNetworkResult::One(mut network) => {
            network.name = Some("reverse".to_owned());
            *network
        }
        ErrorNetworkResult::Pair { .. } => unreachable!(),
    };
    Ok(ErrorNetworkResult::Pair {
        forward: Box::new(forward),
        reverse: Box::new(reverse),
    })
}

fn parse_pna_coefficient_name(name: &str) -> Result<(&str, usize, usize)> {
    let open = name
        .rfind('(')
        .ok_or_else(|| Error::Unsupported(format!("invalid PNA coefficient name '{name}'")))?;
    let ports = name[open + 1..]
        .strip_suffix(')')
        .ok_or_else(|| Error::Unsupported(format!("invalid PNA coefficient name '{name}'")))?;
    let (source, receiver) = ports
        .split_once(',')
        .ok_or_else(|| Error::Unsupported(format!("invalid PNA coefficient ports in '{name}'")))?;
    Ok((
        &name[..open],
        source.trim().parse().map_err(|error| {
            Error::Unsupported(format!("invalid PNA source port in '{name}': {error}"))
        })?,
        receiver.trim().parse().map_err(|error| {
            Error::Unsupported(format!("invalid PNA receiver port in '{name}': {error}"))
        })?,
    ))
}

fn pna_term_to_skrf(term: &str) -> Result<&'static str> {
    match term {
        "Directivity" => Ok("directivity"),
        "SourceMatch" => Ok("source match"),
        "ReflectionTracking" => Ok("reflection tracking"),
        "LoadMatch" => Ok("load match"),
        "TransmissionTracking" => Ok("transmission tracking"),
        "CrossTalk" => Ok("isolation"),
        _ => Err(Error::Unsupported(format!(
            "unknown PNA coefficient term '{term}'"
        ))),
    }
}

fn skrf_term_to_pna(term: &str) -> Result<&'static str> {
    match term {
        "directivity" => Ok("Directivity"),
        "source match" => Ok("SourceMatch"),
        "reflection tracking" => Ok("ReflectionTracking"),
        "load match" => Ok("LoadMatch"),
        "transmission tracking" => Ok("TransmissionTracking"),
        "isolation" => Ok("CrossTalk"),
        _ => Err(Error::Unsupported(format!(
            "unknown scikit-rf coefficient term '{term}'"
        ))),
    }
}
