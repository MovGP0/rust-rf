//! Transmission-line theory functions.
//!
//! The module relates characteristic impedance, load and input impedance,
//! reflection coefficient, electrical length, and distributed circuit values.
//!
//! | Symbol | Rust term |
//! | --- | --- |
//! | $Z_{l}$ | load impedance |
//! | $Z_{in}$ | input impedance |
//! | $\Gamma_{0}$ | load reflection coefficient |
//! | $\Gamma_{in}$ | input reflection coefficient |
//! | $\theta$ | electrical length |
//!
//! For a uniform line of characteristic impedance $Z_{0}$ and electrical length
//! $\theta$, $Z_{l}$ and $\Gamma_{0}$ are evaluated at the load. $Z_{in}$ and
//! $\Gamma_{in}$ look toward that load from the other end of the line. Therefore,
//! at $\theta = 0$:
//!
//! $$
//! Z_{in} = Z_{l}, \qquad \Gamma_{in} = \Gamma_{0}.
//! $$

use ndarray::Array1;
use num_complex::Complex64;

use crate::constants::FREE_SPACE_PERMEABILITY;
use crate::{Error, Result};

/// Calculates the skin depth of a material.
///
/// $$
/// \delta = \sqrt{\frac{\rho}{\pi f \mu_{r} \mu_{0}}}
/// $$
///
/// `frequency_hz` is the frequency in hertz, `resistivity` is the bulk
/// resistivity in $\Omega\,\mathrm{m}$, and `relative_permeability` is $\mu_{r}$.
/// The returned skin depth is in metres.
///
/// # Errors
///
/// Returns an error if a frequency, resistivity, or relative permeability is
/// non-finite or non-positive.
///
/// # References
///
/// - [Microwaves101: Skin Depth](https://www.microwaves101.com/encyclopedias/skin-depth)
/// - [Wikipedia: Skin effect](https://en.wikipedia.org/wiki/Skin_effect)
pub fn skin_depth(
    frequency_hz: &Array1<f64>,
    resistivity: f64,
    relative_permeability: f64,
) -> Result<Array1<f64>> {
    if !resistivity.is_finite() || resistivity <= 0.0 {
        return Err(Error::Unsupported(
            "resistivity must be finite and positive".to_owned(),
        ));
    }
    if !relative_permeability.is_finite() || relative_permeability <= 0.0 {
        return Err(Error::Unsupported(
            "relative permeability must be finite and positive".to_owned(),
        ));
    }
    if frequency_hz
        .iter()
        .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return Err(Error::InvalidFrequency(
            "skin depth requires finite, positive frequencies".to_owned(),
        ));
    }

    Ok(frequency_hz.mapv(|frequency| {
        (resistivity
            / (std::f64::consts::PI * frequency * relative_permeability * FREE_SPACE_PERMEABILITY))
            .sqrt()
    }))
}

/// Calculates surface resistivity in ohms per square.
///
/// $$
/// `R_s` = \frac{\rho}{\delta}
/// $$
///
/// Here $\delta$ is calculated by [`skin_depth`].
///
/// # Errors
///
/// Returns the validation errors reported by [`skin_depth`].
///
/// # References
///
/// - [Microwaves101: Sheet Resistance](https://www.microwaves101.com/encyclopedias/sheet-resistance)
/// - [Wikipedia: Sheet resistance](https://en.wikipedia.org/wiki/Sheet_resistance)
pub fn surface_resistivity(
    frequency_hz: &Array1<f64>,
    resistivity: f64,
    relative_permeability: f64,
) -> Result<Array1<f64>> {
    Ok(
        skin_depth(frequency_hz, resistivity, relative_permeability)?
            .mapv(|depth| resistivity / depth),
    )
}

/// Converts distributed circuit quantities to wave quantities.
///
/// For distributed impedance $Z'$ and admittance $Y'$:
///
/// $$
/// \gamma = \sqrt{Z'Y'}, \qquad `Z_0` = \sqrt{\frac{Z'}{Y'}}.
/// $$
///
/// Returns `(propagation_constant, characteristic_impedance)`.
///
/// # Errors
///
/// Returns an error when the distributed admittance is zero.
pub fn propagation_and_impedance_from_distributed_circuit(
    distributed_admittance: Complex64,
    distributed_impedance: Complex64,
) -> Result<(Complex64, Complex64)> {
    if distributed_admittance.norm_sqr() == 0.0 {
        return Err(Error::Unsupported(
            "distributed admittance must be non-zero".to_owned(),
        ));
    }
    Ok((
        (distributed_impedance * distributed_admittance).sqrt(),
        (distributed_impedance / distributed_admittance).sqrt(),
    ))
}

/// Converts wave quantities to distributed circuit quantities.
///
/// $$
/// Y' = \frac{\gamma}{`Z_0`}, \qquad Z' = \gamma `Z_0`.
/// $$
///
/// Returns `(distributed_admittance, distributed_impedance)`.
///
/// # Errors
///
/// Returns an error when the characteristic impedance is zero.
pub fn distributed_circuit_from_propagation_and_impedance(
    propagation_constant: Complex64,
    characteristic_impedance: Complex64,
) -> Result<(Complex64, Complex64)> {
    if characteristic_impedance.norm_sqr() == 0.0 {
        return Err(Error::Unsupported(
            "characteristic impedance must be non-zero".to_owned(),
        ));
    }
    Ok((
        propagation_constant / characteristic_impedance,
        propagation_constant * characteristic_impedance,
    ))
}

/// Calculates the reflection coefficient of a load impedance.
///
/// $$
/// \Gamma_{0} = \frac{Z_{l} - Z_{0}}{Z_{l} + Z_{0}}.
/// $$
///
/// An infinite load impedance is treated as an open circuit and returns
/// $\Gamma_{0} = 1$.
#[must_use]
pub fn reflection_coefficient(z0: Complex64, load: Complex64) -> Complex64 {
    if load.re.is_infinite() || load.im.is_infinite() {
        Complex64::new(1.0, 0.0)
    } else {
        (load - z0) / (load + z0)
    }
}

/// Calculates input impedance from a reflection coefficient.
///
/// $$
/// Z_{in} = `Z_0` \frac{1 + \Gamma}{1 - \Gamma}.
/// $$
#[must_use]
pub fn impedance_from_reflection(z0: Complex64, reflection: Complex64) -> Complex64 {
    z0 * (Complex64::new(1.0, 0.0) + reflection) / (Complex64::new(1.0, 0.0) - reflection)
}

/// Calculates the reflection coefficient at an electrical length.
///
/// $$
/// \Gamma_{in} = \Gamma_{0} e^{-2\theta}.
/// $$
///
/// `electrical_length` may be complex.
#[must_use]
pub fn reflection_at_electrical_length(
    load_reflection: Complex64,
    electrical_length: Complex64,
) -> Complex64 {
    load_reflection * (-2.0 * electrical_length).exp()
}

/// Calculates the input impedance of a load at an electrical length.
///
/// The load impedance is first converted to its reflection coefficient,
/// propagated through `electrical_length`, and converted back to impedance.
#[must_use]
pub fn input_impedance_at_electrical_length(
    characteristic_impedance: Complex64,
    load_impedance: Complex64,
    electrical_length: Complex64,
) -> Complex64 {
    let load_reflection = reflection_coefficient(characteristic_impedance, load_impedance);
    impedance_from_reflection(
        characteristic_impedance,
        reflection_at_electrical_length(load_reflection, electrical_length),
    )
}

/// Calculates a load's reflection coefficient at an electrical length.
#[must_use]
pub fn load_reflection_at_electrical_length(
    characteristic_impedance: Complex64,
    load_impedance: Complex64,
    electrical_length: Complex64,
) -> Complex64 {
    reflection_at_electrical_length(
        reflection_coefficient(characteristic_impedance, load_impedance),
        electrical_length,
    )
}

/// Calculates input impedance from a load reflection coefficient at an
/// electrical length.
#[must_use]
pub fn reflection_to_impedance_at_electrical_length(
    characteristic_impedance: Complex64,
    load_reflection: Complex64,
    electrical_length: Complex64,
) -> Complex64 {
    impedance_from_reflection(
        characteristic_impedance,
        reflection_at_electrical_length(load_reflection, electrical_length),
    )
}

/// Calculates the electrical length of a transmission-line section.
///
/// $$
/// \theta = \gamma d.
/// $$
///
/// The result is in radians unless `degrees` is `true`. Forward propagation is
/// represented by the positive imaginary part of the propagation constant.
///
/// # Errors
///
/// Returns an error when `distance_m` is not finite.
pub fn electrical_length(
    propagation_constant: Complex64,
    distance_m: f64,
    degrees: bool,
) -> Result<Complex64> {
    validate_distance(distance_m)?;
    let radians = propagation_constant * distance_m;
    Ok(if degrees {
        radians * (180.0 / std::f64::consts::PI)
    } else {
        radians
    })
}

/// Converts electrical length to physical distance in metres.
///
/// $$
/// d = \Re\left\{\frac{\theta}{\gamma}\right\}.
/// $$
///
/// `electrical_length` is interpreted as degrees when `degrees` is `true` and
/// as radians otherwise. Forward propagation is represented by the positive
/// imaginary part of the propagation constant.
///
/// # Errors
///
/// Returns an error when the propagation constant is zero.
pub fn distance_from_electrical_length(
    electrical_length: Complex64,
    propagation_constant: Complex64,
    degrees: bool,
) -> Result<f64> {
    if propagation_constant.norm_sqr() == 0.0 {
        return Err(Error::Unsupported(
            "propagation constant must be non-zero".to_owned(),
        ));
    }
    let radians = if degrees {
        electrical_length * (std::f64::consts::PI / 180.0)
    } else {
        electrical_length
    };
    Ok((radians / propagation_constant).re)
}

/// Calculates propagation constant from input and load reflection coefficients.
///
/// $$
/// \Gamma_{in} = \Gamma_{l} e^{-2\gamma d}, \qquad
/// \gamma = -\frac{1}{2d}\ln\left(\frac{\Gamma_{in}}{\Gamma_{l}}\right).
/// $$
///
/// The phase is selected so forward propagation has a positive imaginary part.
///
/// # Errors
///
/// Returns an error when `distance_m` is not finite and positive or the load
/// reflection coefficient is zero.
pub fn propagation_constant_from_reflections(
    input_reflection: Complex64,
    load_reflection: Complex64,
    distance_m: f64,
) -> Result<Complex64> {
    validate_positive_distance(distance_m)?;
    if load_reflection.norm_sqr() == 0.0 {
        return Err(Error::Unsupported(
            "load reflection coefficient must be non-zero".to_owned(),
        ));
    }

    let mut propagation_constant = -(input_reflection / load_reflection).ln() / (2.0 * distance_m);
    propagation_constant.im = propagation_constant
        .im
        .rem_euclid(std::f64::consts::PI / distance_m);
    Ok(propagation_constant)
}

/// Calculates voltage standing-wave ratio from a reflection coefficient.
///
/// $$
/// \mathrm{VSWR} = \frac{1 + |\Gamma_{0}|}{1 - |\Gamma_{0}|}.
/// $$
#[must_use]
pub fn standing_wave_ratio(reflection: Complex64) -> f64 {
    (1.0 + reflection.norm()) / (1.0 - reflection.norm())
}

/// Calculates voltage standing-wave ratio from characteristic and load
/// impedances.
///
/// The load impedance is converted with [`reflection_coefficient`] before the
/// standing-wave ratio is evaluated.
#[must_use]
pub fn standing_wave_ratio_from_impedance(
    characteristic_impedance: Complex64,
    load_impedance: Complex64,
) -> f64 {
    standing_wave_ratio(reflection_coefficient(
        characteristic_impedance,
        load_impedance,
    ))
}

/// Propagates voltage and current through a transmission-line section.
///
/// `voltage` and `current` are the total values at $\theta = 0$. The returned
/// pair contains the voltage and outward-directed current at
/// `electrical_length`. The calculation uses the inverse ABCD parameters of a
/// uniform transmission line.
///
/// # Errors
///
/// Returns an error when the characteristic impedance is zero.
pub fn propagate_voltage_current(
    voltage: Complex64,
    current: Complex64,
    characteristic_impedance: Complex64,
    electrical_length: Complex64,
) -> Result<(Complex64, Complex64)> {
    if characteristic_impedance.norm_sqr() == 0.0 {
        return Err(Error::Unsupported(
            "characteristic impedance must be non-zero".to_owned(),
        ));
    }

    let a = electrical_length.cosh();
    let b = characteristic_impedance * electrical_length.sinh();
    let c = electrical_length.sinh() / characteristic_impedance;
    let d = a;

    Ok((d * voltage - b * current, -c * voltage + a * current))
}

/// Calculates the total loss of a terminated transmission line in natural units.
///
/// $$
/// \mathrm{TL} = \frac{R_{`in`}}{`R_L`}
/// \left|\cosh\theta + \frac{`Z_L`}{`Z_0`}\sinh\theta\right|^2.
/// $$
///
/// # Errors
///
/// Returns an error when the characteristic impedance or load resistance is
/// zero.
///
/// # Reference
///
/// Steve Stearns (K6OIK), *Transmission Line Power Paradox and Its Resolution*,
/// [ARRL Pacificon Antenna Seminar, 2014](https://www.fars.k6ya.org/docs/K6OIK-A_Transmission_Line_Power_Paradox_and_Its_Resolution.pdf).
pub fn total_loss(
    characteristic_impedance: Complex64,
    load_impedance: Complex64,
    electrical_length: Complex64,
) -> Result<f64> {
    if characteristic_impedance.norm_sqr() == 0.0 {
        return Err(Error::Unsupported(
            "characteristic impedance must be non-zero".to_owned(),
        ));
    }
    if load_impedance.re == 0.0 {
        return Err(Error::Unsupported(
            "load resistance must be non-zero".to_owned(),
        ));
    }
    let input_impedance = input_impedance_at_electrical_length(
        characteristic_impedance,
        load_impedance,
        electrical_length,
    );
    let transfer = electrical_length.cosh()
        + load_impedance / characteristic_impedance * electrical_length.sinh();
    Ok(input_impedance.re / load_impedance.re * transfer.norm_sqr())
}

fn validate_distance(distance_m: f64) -> Result<()> {
    if distance_m.is_finite() {
        Ok(())
    } else {
        Err(Error::Unsupported("distance must be finite".to_owned()))
    }
}

fn validate_positive_distance(distance_m: f64) -> Result<()> {
    validate_distance(distance_m)?;
    if distance_m > 0.0 {
        Ok(())
    } else {
        Err(Error::Unsupported("distance must be positive".to_owned()))
    }
}
