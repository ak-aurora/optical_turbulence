"""
Scintillation index, Rytov variance, and Fried parameter following the flat-earth earth model. Downlink and Uplink.
"""

# ----------------- IMPORTS ----------------- #

import numpy as np
from scipy import integrate
from ._definitions import *
from typing import Callable

# ----------------- CONSTANTS ----------------- #

EARTH_RADIUS = 6371e3 # [m]

# ----------------- FUNCTIONS DEFINITIONS ----------------- #

#region Utilities

def get_link_distance(sat_altitude: real_t, zenith_angle: real_t, lct_altitude: real_t = 0) -> np.float64:
    """Get the link distance ($L$) between the satellite and the laser communication terminal.

    Args:
        sat_altitude (real_t): Satellite altitude from sea level [m]
        zenith_angle (real_t): Zenith angle between the LCT and SAT [rad]
        lct_altitude (real_t, optional): Laser Communication Terminal altitude from sea level [m]. Default value: 0 m.

    Raises:
        ValueError: the altitudes are not physically valid.

    Returns:
        np.float64: link distance between LCT and SAT [m]
    """
    
    if (lct_altitude < 0) or (sat_altitude < 0):
        raise ValueError(f"Altitudes have to be positive. LCT alt: {lct_altitude:.2e} | SAT alt: {sat_altitude:.2e}")

    if (lct_altitude >= sat_altitude):
        raise ValueError("LCT Altitude has to be less than the satellite altitude.")
    
    left_term = -np.cos(zenith_angle) * (EARTH_RADIUS + lct_altitude)
    right_term_1 = np.pow(np.cos(zenith_angle) * (EARTH_RADIUS + lct_altitude), 2)
    right_term_2 = (sat_altitude - lct_altitude) * (2 * EARTH_RADIUS + sat_altitude + lct_altitude)
    
    return left_term + np.sqrt( right_term_1 + right_term_2 )


def get_altitude_from_distance[T: real_t | real_array_t](distance: T, zenith_angle: real_t, lct_distance: real_t = -1) -> T:
    """From a distance in the round-earth model, get an altitude.

    Args:
        distance (real_t or real_array_t): distance in the round-earth model [m].
        zenith_angle (real_t): zenith angle between the LCT and SAT [rad].
        lct_distance (real_t, optional): distance of the lct w.r.t sea level. Defaults to 0.

    Returns:
        real_t or real_array_t: altitude to the corresponding round-earth-model distance [m].
    """
    
    if (np.min(distance) < 0):
        raise ValueError(f"Distances have to be positive. Minimum distance: {np.min(distance):.2e}")

    if lct_distance == -1:
        if isinstance(distance, np.ndarray):
            lct_distance = np.min(distance)
        else: 
            lct_distance = 0

    if (lct_distance > np.min(distance)):
        raise ValueError("LCT Distance has to be less than the distances to calculate altitude.")

    lct_altitude = lct_distance * np.cos(zenith_angle)

    sq_term1 = np.pow(EARTH_RADIUS + lct_altitude, 2) + np.pow(distance - lct_distance, 2)
    sq_term2 = 2 * (distance - lct_distance) * (lct_altitude + EARTH_RADIUS) * np.cos(zenith_angle)
    sq_term = np.sqrt( sq_term1 + sq_term2 )

    return sq_term - EARTH_RADIUS


def get_distance_from_altitude[T: real_t | real_array_t](altitude: T, zenith_angle: real_t, lct_altitude: real_t = -1) -> T:
    """From an altitude, get the corresponding distance in the round-earth model.

    Args:
        sat_altitude (real_t or real_array_t): Satellite altitude from sea level [m]
        zenith_angle (real_t): Zenith angle between the LCT and SAT [rad]
        lct_altitude (real_t, optional): altitude of the lct w.r.t sea level. Defaults to -1. If the value is -1, then \
            LCT distance is calculated as the minimum of the altitude array or set as 0.
        
    Raises:
        ValueError: the altitudes are not physically valid.

    Returns:
        real_t or real_array_t: distance to a point in the LCT-SAT link [m]
    """

    if (np.min(altitude) < 0):
        raise ValueError(f"Altitudes have to be positive. Minimum altitude: {np.min(altitude):.2e}")

    if lct_altitude == -1:
        if isinstance(altitude, np.ndarray):
            lct_altitude = np.min(altitude)
        else:
            lct_altitude = 0

    if (lct_altitude > np.min(altitude)):
        raise ValueError("LCT Altitude has to be less than the altitudes to calculate altitude.")
    
    left_term = -np.cos(zenith_angle) * (EARTH_RADIUS + lct_altitude) + lct_altitude / np.cos(zenith_angle)
    right_term_1 = np.pow(np.cos(zenith_angle) * (EARTH_RADIUS + lct_altitude), 2)
    right_term_2 = (altitude - lct_altitude) * (2 * EARTH_RADIUS + altitude + lct_altitude)
    
    return left_term + np.sqrt( right_term_1 + right_term_2 )


#endregion Utilities

# ---------

#region DOWNLINK


def scint_index_DL_PR_weak(wavelength: real_t, \
                           zenith_angle: real_t, \
                           distance: real_array_t, \
                           ris_model: Callable[[real_array_t], real_array_t],
                           **_) \
                           -> np.float64:
    """Calculate the scintillation index (in this case equivalent to rytov variance) for the case of \
    a **downlink** wave being captured by a **point receiver under weak turbulence**.

    Note:
        This function is for a round-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m].
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, \
            no. v. PM152. Bellingham: SPIE, 2005. p. 495
        
    """

    lct_distance = np.min(distance)
    altitude = get_altitude_from_distance(distance=distance, zenith_angle=zenith_angle, lct_distance=lct_distance)

    left_side = 2.25 * np.pow( (2 * np.pi / wavelength), 7/6 )
    rs_inner = ris_model(altitude) * np.pow((distance - lct_distance), 5/6)
    right_side = integrate.simpson(rs_inner, distance)
    
    return left_side * right_side


def scint_index_DL_PR_general(wavelength: real_t, \
                           zenith_angle: real_t, \
                           distance: real_array_t, \
                           ris_model: Callable[[real_array_t], real_array_t],
                           **_) \
                           -> np.float64:
    """Calculate the scintillation index for the case of \
    a **downlink** wave being captured by a **point receiver** under **all turbulence conditions** \
    weak, moderate, or strong.

    Note:
        This function is for a round-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, \
            no. v. PM152. Bellingham: SPIE, 2005. p. 497
        
    """

    rytov_var = scint_index_DL_PR_weak(wavelength=wavelength, zenith_angle=zenith_angle, distance=distance,\
                                            ris_model=ris_model)
    rvar_exp = np.pow(rytov_var, 6/5) # (sigma^2)^(6/5) = sigma^(12/5)
    in_exp_l = (0.49 * rytov_var) / np.pow(1 + 1.11 * rvar_exp, 7/6)
    in_exp_r = (0.51 * rytov_var) / np.pow(1 + 0.69 * rvar_exp, 5/6)

    return np.exp(in_exp_l + in_exp_r) - 1


def scint_index_DL_AA_general(wavelength: real_t, \
                            zenith_angle: real_t, \
                            distance: real_array_t, \
                            ris_model: Callable[[real_array_t], real_array_t], \
                            aperture_diameter: real_t,
                            **_) \
                            -> np.float64:
    """Calculate the scintillation index for the case of \
    a **downlink** wave being captured by a an aperture wit diameter such that **aperture averaging** \
    occrus under **all turbulence conditions** weak, moderate, or strong.

    Note:
        This function is for a round-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        aperture_diameter (real_t): diameter of the receiver aperture [m].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, \
            no. v. PM152. Bellingham: SPIE, 2005. p. 497
    """
    # To not re-calculate
    lct_distance = np.min(distance)
    altitude = get_altitude_from_distance(distance=distance, zenith_angle=zenith_angle, lct_distance=lct_distance)

    wavenum = 2 * np.pi / wavelength

    # We divide into left (out of Re function) and right (inside Re function)
    left = 8.70 * np.pow(wavenum, 7/6)
    # We calculate terms inside the integral
    in_integ_r1 = np.float_power( wavenum * aperture_diameter ** 2 / 16 + 1j*(distance - lct_distance), 5/6, dtype=complex)
    in_integ_r2 = np.pow(wavenum * aperture_diameter ** 2 / 16, 5/6)
    in_integral = ris_model(altitude) * (in_integ_r1 - in_integ_r2)
    # We calculate the integral en keep its real part
    right = np.real(integrate.simpson(in_integral, distance))

    return left * right


@warn_not_tested
def fried_parameter_DL(wavelength: real_t, zenith_angle: real_t, distance: real_array_t, ris_model: Callable[[real_array_t], real_array_t], **_) -> np.float64:
    """Calculate the Fried paramater ($r_0$) for a downlink wave, assuming plane wave.

    Args:
        wavelength (real_t): wavelenght of the wave [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Note:
        This function is for a round-earth model

    Returns:
        np.float64: the fried parameter for the specified SATCOM system [m].

    Source:
        L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, \
            no. v. PM152. Bellingham: SPIE, 2005. p. 492

    """
    altitude = get_altitude_from_distance(distance=distance, zenith_angle=zenith_angle, lct_distance=np.min(distance))

    wavenum_sq = np.pow( 2 * np.pi / wavelength, 2)
    base_val = 0.42 * wavenum_sq * integrate.simpson(ris_model(altitude), distance)
    return np.pow(base_val, -3/5)


#endregion DOWNLINK

#region UPLINK


@warn_not_tested
def isonoplanatic_angle_UL(wavelength: real_t, \
                            zenith_angle: real_t, \
                            distance: real_array_t, \
                            ris_model: Callable[[real_array_t], real_array_t],
                            Lambda: real_t,
                            Theta: real_t,
                            neg_Theta: real_t,
                            **_) \
                            -> np.float64:
    """Calculate the isoplanatic angle of an **uplink gaussian-beam wave**.

    Note:
        This function is for a flat-earth model
        Also known as $\\sigma_{Bu}$

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        Lambda (real_t): diffractive parameter of the beam at the receiver plane [unitless]
        Theta (real_t): refractive parameter of the beam at the receiver plane [unitless]
        neg_Theta (real_t): overbar refractive parameter of the beam at the receiver plane [unitless] (1 - Theta)

    Returns:
        out (np.float64): the isoplanatic angle for the uplink gaussian beam wave [rad].

    Source
        Andrews, Larry C. Laser Beam Propagation Through Random Media, \
            2nd ed. Press Monograph Series, v. PM152. SPIE, 2005. p. 493

    """
    
    # pre calculate
    lct_distance = np.min(distance)
    link_distance = np.max(distance) - lct_distance
    altitude = get_altitude_from_distance(distance=distance, zenith_angle=zenith_angle, lct_distance=lct_distance)
    
    wavenumber_sq = np.pow( 2 * np.pi / wavelength, 2)

    mu_frac = (distance - lct_distance) / link_distance

    mu1u = integrate.simpson(ris_model(altitude) * 
        np.pow( Theta - neg_Theta * mu_frac , 5/3 ),
        distance
    )

    mu2u = integrate.simpson(ris_model(altitude)* 
        np.pow( 1 - mu_frac, 5/3 ),
        distance
    )

    outside_pow = np.pow( link_distance, -1 )
    inside_pow = 2.91 * wavenumber_sq * (mu1u + 0.62 * mu2u * np.pow( Lambda, 11/6 ))
    pow_term = np.pow( inside_pow, -3/5 )

    return outside_pow * pow_term


def fried_parameter_UL_TX(wavelength: real_t, zenith_angle: real_t, distance: real_array_t, ris_model: Callable[[real_array_t], real_array_t], **_) -> np.float64:
    """Calculate the Fried paramater ($r_{0T}$) for an uplink beam, assuming an spherical wave \
        as seen from the transmitter.

    Args:
        wavelength (real_t): wavelenght of the wave [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link  [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Note:
        This function is for a round-earth model

    Returns:
        np.float64: the fried parameter for the specified SATCOM system [m].

    Source:
        [1] (p. 492) L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
        [2] (p. 189) F. G. Smith, J. S. Accetta, and D. L. Shumaker, Eds., “Propagation Throught Atmospheric Optical Turbulence,” in The infrared & electro-optical systems handbook, vol. 2: atmospheric propagation of radiation, in PM / SPIE, no. 1002. , Ann Arbor, Michigan: Infrared Information Analysis Center, 2023. doi: 10.1117/3.2543821.
        [3] L. B. Stotts and L. C. Andrews, “Optical communications in turbulence: a tutorial,” Opt. Eng., vol. 63, no. 04, Dec. 2023, doi: 10.1117/1.OE.63.4.041207.
        [4] L. B. Stotts, M. Toyoshima, and L. C. Andrews, “Effect of satellite slew rate on bit error rate model under atmospheric turbulence,” Opt. Eng., vol. 64, no. 05, May 2025, doi: 10.1117/1.OE.64.5.058104.
    """
    lct_distance = np.min(distance)
    link_distance = np.max(distance) - lct_distance
    altitude = get_altitude_from_distance(distance=distance, zenith_angle=zenith_angle, lct_distance=lct_distance)

    # divide
    left = 0.42 * np.pow(2 * np.pi / wavelength, 2)
    right = integrate.simpson( ris_model(altitude) * np.pow(1 - (distance - lct_distance) / link_distance, 5/3), distance )

    return np.pow(left * right, -3/5)


@warn_not_tested
def fried_parameter_UL_RX(wavelength: real_t, zenith_angle: real_t, distance: real_array_t, ris_model: Callable[[real_array_t], real_array_t], **_) -> np.float64:
    """Calculate the Fried paramater ($r_{0R}$) for an uplink beam, assuming an spherical wave \
        as seen from the receiver.

    Args:
        wavelength (real_t): wavelenght of the wave [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Note:
        This function is for a round-earth model

    Returns:
        np.float64: the fried parameter for the specified SATCOM system [m].

    Source:
        [1] (p. 492) L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
        [2] (p. 189) F. G. Smith, J. S. Accetta, and D. L. Shumaker, Eds., “Propagation Throught Atmospheric Optical Turbulence,” in The infrared & electro-optical systems handbook, vol. 2: atmospheric propagation of radiation, in PM / SPIE, no. 1002. , Ann Arbor, Michigan: Infrared Information Analysis Center, 2023. doi: 10.1117/3.2543821.
        [3] L. B. Stotts and L. C. Andrews, “Optical communications in turbulence: a tutorial,” Opt. Eng., vol. 63, no. 04, Dec. 2023, doi: 10.1117/1.OE.63.4.041207.
        [4] L. B. Stotts, M. Toyoshima, and L. C. Andrews, “Effect of satellite slew rate on bit error rate model under atmospheric turbulence,” Opt. Eng., vol. 64, no. 05, May 2025, doi: 10.1117/1.OE.64.5.058104.
    """
    # pre-calculate
    lct_distance = np.min(distance)
    link_distance = np.max(distance) - lct_distance
    altitude = get_altitude_from_distance(distance=distance, zenith_angle=zenith_angle, lct_distance=lct_distance)

    # divide
    left = 0.42 * np.pow(2 * np.pi / wavelength, 2)
    right = integrate.simpson( ris_model(altitude) * np.pow( (distance - lct_distance) / link_distance, 5/3), distance )

    return np.pow(left * right, -3/5)


@warn_not_tested
def rytov_variance_UL_spherical(wavelength: real_t, \
                            zenith_angle: real_t, \
                            distance: real_array_t, \
                            ris_model: Callable[[real_array_t], real_array_t],
                            **_) \
                            -> np.float64:
    """Calculate the Rytov variance for the case of an **uplink spherical wave**.

    Note:
        This function is for a round-earth model
        Also known as $\\sigma_{Bu}$

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Returns:
        out (np.float64): the Rytov variance for an spherical wave [unitless].

    Source
        L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 52

    """

    # to not re-calculate
    lct_distance = np.min(distance)
    link_distance = np.max(distance) - lct_distance
    altitude = get_altitude_from_distance(distance=distance, zenith_angle=zenith_angle, lct_distance=lct_distance)
    wavenum = 2 * np.pi / wavelength

    # we divide into left and right to calculate
    left = 2.25 * np.pow(wavenum, 7/6)
    inner_right = ris_model(altitude) * np.pow(distance - lct_distance, 5/6) * np.pow(1 - (distance - lct_distance) / link_distance, 5/6)
    right = integrate.simpson(inner_right, distance)

    return left * right


@warn_not_tested
def scint_index_UL_spherical(wavelength: real_t, \
                            zenith_angle: real_t, \
                            distance: real_array_t, \
                            ris_model: Callable[[real_array_t], real_array_t],
                            **_) \
                            -> np.float64:
    """Calculate the scintillation index for the case of \
    a tracked **uplink** spherical wave being captured by a **point receiver** \
    under **all turbulence conditions** weak, moderate, or strong.

    Note:
        This function is for a round-earth model
        Also known as $\\sigma_{Bu}$

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 52

    """

    rytov_var = rytov_variance_UL_spherical(wavelength=wavelength, zenith_angle=zenith_angle, distance=distance, ris_model=ris_model)

    in_exp_1 = 0.49 * rytov_var / np.pow(1 + 0.56 * np.pow(rytov_var, 6/5), 7/6)
    in_exp_2 = 0.51 * rytov_var / np.pow(1 + 0.69 * np.pow(rytov_var, 6/5), 5/6)
    return np.exp(in_exp_1 + in_exp_2) - 1


def rytov_variance_UL_gaussian(wavelength: real_t, \
                            zenith_angle: real_t, \
                            distance: real_array_t, \
                            ris_model: Callable[[real_array_t], real_array_t],
                            Lambda: real_t,
                            Theta: real_t,
                            **_) \
                            -> np.float64:
    """Calculate the Rytov variance for the case of an **uplink collimated gaussian-beam wave**.

    Note:
        This function is for a round-earth model
        Also known as $\\sigma_{Bu}$

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        Lambda (real_t): diffractive parameter of the beam at the receiver plane [unitless]
        Theta (real_t): refractive parameter of the beam at the receiver plane [unitless]

    Returns:
        out (np.float64): the Rytov variance for a collimated gaussian-beam wave [unitless].

    Source
        L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 110

    """
    # to not re-calculate
    lct_distance = np.min(distance)
    link_distance = np.max(distance) - lct_distance
    altitude = get_altitude_from_distance(distance=distance, zenith_angle=zenith_angle, lct_distance=lct_distance)

    wavenum = 2 * np.pi / wavelength
    xi = 1 - (distance - lct_distance) / (link_distance)
    neg_Theta = 1 - Theta
    
    # divide by sectors
    left = 8.70 * np.pow(wavenum, 7/6) * np.pow(link_distance, 5/6)

    right_in = np.pow(xi, 5/6) * np.pow(Lambda * xi + 1j * (1 - neg_Theta * xi), 5/6) - np.pow(Lambda, 5/6) * np.pow(xi, 5/3)
    right = integrate.simpson(ris_model(altitude) * right_in, distance)
    return left * np.real(right)


def _total_beam_wander_variance_UL_gaussian(zenith_angle: real_t, \
                                            distance: real_array_t, \
                                            ris_model: Callable[[real_array_t], real_array_t],
                                            beam_radius: real_t,
                                            pfront_radius: real_t) \
                                            -> np.float64:
    """Calculate the uplink total-beam-wander variance ($\\langle r_c^2\rangle$) for a collimated or convergent beam

    Note:
        flat-earth model
        Adapted by me for round-earth

    Args:
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (Callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        beam_radius (real_t): $W_0$ effective beam radius (spot size) [m]
        pfront_radius (real_t): phase front radius of curvature at the transmitter [m].

    Returns:
        np.float64: total beam wander variance for the given beam type [m^2]

    Source:
        L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. \
        Bellingham, Washington, USA: SPIE Press, 2023. p. 180

    """
    # to not re-calculate
    lct_distance = np.min(distance)
    link_distance = np.max(distance) - lct_distance
    altitude = get_altitude_from_distance(distance=distance, zenith_angle=zenith_angle, lct_distance=lct_distance)

    # divide in sides
    left = 7.25 * np.pow(link_distance, 2) / ( np.pow(beam_radius, 1/3) )

    right_in1 = np.pow(1 - (distance - lct_distance) / link_distance , 2)
    right_in2 = np.pow(np.abs( 1 - (distance - lct_distance) / (pfront_radius) ), 1/3)

    right = integrate.simpson(ris_model(altitude) * right_in1 / right_in2, distance)

    return left * right


def _bw_pointing_error_var_UL_gaussian(wavelength: real_t, \
                                        zenith_angle: real_t, \
                                        distance: real_array_t, \
                                        ris_model: Callable[[real_array_t], real_array_t],
                                        beam_radius: real_t,
                                        pfront_radius: real_t) \
                                        -> np.float64:
    """Calculate the beam-wander induced pointing-error variance for a collimated or convergent beam for \
        the uplink case. This pointing-error variance is NOT tracked/corrected and applies uniquely to LEO.

    Note:
        round-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (Callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        beam_radius (real_t): $W_0$ effective beam radius (spot size) [m]
        pfront_radius (real_t): phase front radius of curvature at the transmitter [m].

    Returns:
        np.float64: total beam wander variance for the given beam type [m^2]

    Source:
        L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. \
        Bellingham, Washington, USA: SPIE Press, 2023. p. 181

    """
    beam_wander_variance = _total_beam_wander_variance_UL_gaussian(zenith_angle=zenith_angle,\
                                                                   distance=distance,
                                                                   ris_model=ris_model,
                                                                   beam_radius=beam_radius,
                                                                   pfront_radius=pfront_radius)
    
    tx_fried_param = fried_parameter_UL_TX(wavelength=wavelength, zenith_angle=zenith_angle,
                                           distance=distance, ris_model=ris_model)

    # pre-calculate
    radius_to_fried_sq_LEO = np.pow( np.pi * beam_radius / tx_fried_param, 2)

    second_term = 1 - np.pow( radius_to_fried_sq_LEO / (1 + radius_to_fried_sq_LEO), 1/6 )

    return beam_wander_variance * second_term


def _bw_pointing_error_var_UL_gaussian_TT(wavelength: real_t, \
                                            zenith_angle: real_t, \
                                            distance: real_array_t, \
                                            ris_model: Callable[[real_array_t], real_array_t],
                                            beam_radius: real_t,
                                            pfront_radius: real_t) \
                                            -> np.float64:
    """Calculate the beam-wander induced pointing-error variance for a collimated or convergent beam for \
        the tilt-corrected uplink case. This pointing-error variance IS tilt-corrected and applies uniquely to LEO.

    Note:
        round-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (Callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        beam_radius (real_t): $W_0$ effective beam radius (spot size) [m]
        pfront_radius (real_t): phase front radius of curvature at the transmitter [m].

    Returns:
        np.float64: total beam wander variance for the given beam type [m^2]

    Source:
        L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. \
        Bellingham, Washington, USA: SPIE Press, 2023. p. 181

    """
    beam_wander_variance = _total_beam_wander_variance_UL_gaussian(zenith_angle=zenith_angle,\
                                                                   distance=distance,
                                                                   ris_model=ris_model,
                                                                   beam_radius=beam_radius,
                                                                   pfront_radius=pfront_radius)
    
    tx_fried_param = fried_parameter_UL_TX(wavelength=wavelength, zenith_angle=zenith_angle,
                                           distance=distance, ris_model=ris_model)

    # pre-calculate
    radius_to_fried_sq_LEO = np.pow( np.pi * beam_radius / tx_fried_param, 2)
    link_distance = np.max(distance) - np.min(distance)

    # Calculate zernike tilt variance and the two terms separately
    zernike_tilt_variance = 0.57 * ( wavelength / (2 * beam_radius) ) * np.pow(2 * beam_radius / tx_fried_param, 5/6)
    first_term = np.pow(np.sqrt(beam_wander_variance) - zernike_tilt_variance * ( link_distance ), 2)
    second_term = 1 - np.pow( radius_to_fried_sq_LEO / (1 + radius_to_fried_sq_LEO), 1/6 )

    return first_term * second_term


def scint_index_UL_untracked_gaussian(wavelength: real_t, \
                                        zenith_angle: real_t,
                                        distance: real_array_t,
                                        ris_model: Callable[[real_array_t], real_array_t],
                                        beam_radius: real_t,
                                        pfront_radius: real_t,
                                        Lambda: real_t,
                                        Theta: real_t,
                                        rx_spot_size: real_t,
                                        **_) \
                                        -> np.float64:
    """Calculate the scintillation index for the case of \
    an **untracked uplink** collimated or convergent wave being captured by a **point receiver** \
    under **all turbulence conditions** weak, moderate, or strong.

    Note:
        This function is for a round-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (Callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        beam_radius (real_t): $W_0$ effective beam radius (spot size) [m]
        pfront_radius (real_t): $F_0$ phase front radius of curvature at the transmitter [m].
        Lambda (real_t): diffractive parameter of the beam at the receiver plane [unitless]
        Theta (real_t): refractive parameter of the beam at the receiver plane [unitless]
        rx_spot_size (real_t): $W$ spot size (effective beam radius) of the beam at the receiver plane [m]

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        [1] L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 110
        [2] L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. \
        Bellingham, Washington, USA: SPIE Press, 2023.

    """
    # pre-calculate some values
    tx_fried_radius = fried_parameter_UL_TX(wavelength=wavelength, zenith_angle=zenith_angle, distance=distance, ris_model=ris_model)
    perror_variance = _bw_pointing_error_var_UL_gaussian(wavelength=wavelength, zenith_angle=zenith_angle, distance=distance, ris_model=ris_model,\
                                                         beam_radius=beam_radius, pfront_radius=pfront_radius)
    rytov_var = rytov_variance_UL_gaussian(wavelength=wavelength, zenith_angle=zenith_angle, distance=distance, ris_model=ris_model,
                                           Lambda=Lambda, Theta=Theta)

    # to not re-calculate
    link_distance = np.max(distance) - np.min(distance)
    wavenum = 2 * np.pi / wavelength

    # separate in the sum
    left = 34.29 * np.pow( Lambda * link_distance / ( wavenum * np.pow(tx_fried_radius, 2) ), 5/6 ) * ( perror_variance / np.pow(rx_spot_size, 2) )
    
    right_1 = 0.49 * rytov_var / np.pow( 1 + (1 + Theta) * 0.56 * np.pow(rytov_var, 6/5), 7/6 )
    right_2 = 0.51 * rytov_var / np.pow( 1 + 0.69 * np.pow(rytov_var, 6/5), 5/6 )
    right = np.exp(right_1 + right_2) - 1

    return left + right


def scint_index_UL_tracked_gaussian(wavelength: real_t, \
                                        zenith_angle: real_t,
                                        distance: real_array_t,
                                        ris_model: Callable[[real_array_t], real_array_t],
                                        beam_radius: real_t,
                                        pfront_radius: real_t,
                                        Lambda: real_t,
                                        Theta: real_t,
                                        rx_spot_size: real_t,
                                        **_) \
                                        -> np.float64:
    """Calculate the scintillation index for the case of \
    a **tracked uplink** collimated or convergent wave being captured by a **point receiver** \
    under **all turbulence conditions** weak, moderate, or strong.

    Note:
        This function is for a round-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        distance (real_array_t): array with distances that cover the SATCOM link [m]
        ris_model (Callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        beam_radius (real_t): $W_0$ effective beam radius (spot size) [m]
        pfront_radius (real_t): $F_0$ phase front radius of curvature at the transmitter [m].
        Lambda (real_t): diffractive parameter of the beam at the receiver plane [unitless]
        Theta (real_t): refractive parameter of the beam at the receiver plane [unitless]
        rx_spot_size (real_t): $W$ spot size (effective beam radius) of the beam at the receiver plane [m]

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        [1] L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 110
        [2] L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. \
        Bellingham, Washington, USA: SPIE Press, 2023.

    """
    # pre-calculate some values
    tx_fried_radius = fried_parameter_UL_TX(wavelength=wavelength, zenith_angle=zenith_angle, distance=distance, ris_model=ris_model)
    perror_variance = _bw_pointing_error_var_UL_gaussian_TT(wavelength=wavelength, zenith_angle=zenith_angle, distance=distance, ris_model=ris_model,\
                                                         beam_radius=beam_radius, pfront_radius=pfront_radius)
    rytov_var = rytov_variance_UL_gaussian(wavelength=wavelength, zenith_angle=zenith_angle, distance=distance, ris_model=ris_model,
                                           Lambda=Lambda, Theta=Theta)
    wavenum = 2 * np.pi / wavelength
    link_distance = np.max(distance) - np.min(distance)

    # Long-term spot/beam radius for H >> 20 km ([2] p. 180)
    lt_spot_radius = rx_spot_size * np.pow( 1 + np.pow( 2 * np.sqrt(2) * beam_radius / tx_fried_radius, 5/3) , 3/5 )

    # Long-term diffractive parameter ([2] p. 182)
    lt_Lambda = 2 * link_distance / ( wavenum * np.pow(lt_spot_radius, 2) )

    # separate in the sum
    left = 34.29 * np.pow( lt_Lambda * link_distance / ( wavenum * np.pow(tx_fried_radius, 2) ), 5/6 ) * ( perror_variance / np.pow(lt_spot_radius, 2) )
    right_1 = 0.49 * rytov_var / np.pow( 1 + (1 + Theta) * 0.56 * np.pow(rytov_var, 6/5), 7/6 )
    right_2 = 0.51 * rytov_var / np.pow( 1 + 0.69 * np.pow(rytov_var, 6/5), 5/6 )
    right = np.exp(right_1 + right_2) - 1

    return left + right


#endregion UPLINK
