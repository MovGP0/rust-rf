use ndarray::Array1;
use num_complex::Complex64;

use crate::constants::FREE_SPACE_PERMEABILITY;
use crate::{Error, Result};

/// Port of `skrf.tlineFunctions.skin_depth`.
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

/// Port of `skrf.tlineFunctions.surface_resistivity`.
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

/// Port of `skrf.tlineFunctions.distributed_circuit_2_propagation_impedance`.
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

/// Port of `skrf.tlineFunctions.propagation_impedance_2_distributed_circuit`.
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

/// Port of `skrf.tlineFunctions.zl_2_Gamma0`.
pub fn reflection_coefficient(z0: Complex64, load: Complex64) -> Complex64 {
    if load.re.is_infinite() || load.im.is_infinite() {
        Complex64::new(1.0, 0.0)
    } else {
        (load - z0) / (load + z0)
    }
}

/// Port of `skrf.tlineFunctions.Gamma0_2_zl`.
pub fn impedance_from_reflection(z0: Complex64, reflection: Complex64) -> Complex64 {
    z0 * (Complex64::new(1.0, 0.0) + reflection) / (Complex64::new(1.0, 0.0) - reflection)
}

/// Port of `skrf.tlineFunctions.reflection_coefficient_at_theta`.
pub fn reflection_at_electrical_length(
    load_reflection: Complex64,
    electrical_length: Complex64,
) -> Complex64 {
    load_reflection * (-2.0 * electrical_length).exp()
}

/// Port of `skrf.tlineFunctions.input_impedance_at_theta`.
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

/// Port of `skrf.tlineFunctions.load_impedance_2_reflection_coefficient_at_theta`.
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

/// Port of `skrf.tlineFunctions.reflection_coefficient_2_input_impedance_at_theta`.
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

/// Port of `skrf.tlineFunctions.electrical_length` for a scalar propagation
/// constant and distance.
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

/// Port of `skrf.tlineFunctions.electrical_length_2_distance` for scalar
/// inputs. As in scikit-rf, the physically meaningful real component is
/// returned.
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

/// Port of `skrf.tlineFunctions.reflection_coefficient_2_propagation_constant`.
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

/// Port of `skrf.tlineFunctions.Gamma0_2_swr`.
pub fn standing_wave_ratio(reflection: Complex64) -> f64 {
    (1.0 + reflection.norm()) / (1.0 - reflection.norm())
}

/// Port of `skrf.tlineFunctions.zl_2_swr`.
pub fn standing_wave_ratio_from_impedance(
    characteristic_impedance: Complex64,
    load_impedance: Complex64,
) -> f64 {
    standing_wave_ratio(reflection_coefficient(
        characteristic_impedance,
        load_impedance,
    ))
}

/// Port of `skrf.tlineFunctions.voltage_current_propagation` for scalar
/// voltage, current, impedance, and electrical length.
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

/// Port of `skrf.tlineFunctions.zl_2_total_loss`.
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
