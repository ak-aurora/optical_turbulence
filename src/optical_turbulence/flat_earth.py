"""
Scintillation index, Rytov variance, and Fried parameter following the flat-earth earth model. Downlink and Uplink.
"""

# ----------------- IMPORTS ----------------- #

import numpy as np
from scipy import integrate
from ._definitions import *
from typing import Callable

# ----------------- FUNCTIONS DEFINITIONS ----------------- #

#region DOWNLINK

def scint_index_DL_PR_weak(wavelength: real_t, \
                           zenith_angle: real_t, \
                           altitude: real_array_t, \
                           ris_model: Callable[[real_array_t], real_array_t],
                           **_) \
                           -> np.float64:
    """Calculate the scintillation index (in this case equivalent to rytov variance) for the case of \
    a **downlink** wave being captured by a **point receiver under weak turbulence**.

    Note:
        This function is for a flat-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, \
            no. v. PM152. Bellingham: SPIE, 2005. p. 495
        
    """

    lct_altitude = np.min(altitude)

    left_side = 2.25 * np.pow( (2 * np.pi / wavelength), 7/6 ) / np.pow( np.cos(zenith_angle), 11/6 )
    rs_inner = ris_model(altitude) * np.pow((altitude - lct_altitude), 5/6)
    right_side = integrate.simpson(rs_inner, altitude)
    
    return left_side * right_side


def scint_index_DL_PR_general(wavelength: real_t, \
                           zenith_angle: real_t, \
                           altitude: real_array_t, \
                           ris_model: Callable[[real_array_t], real_array_t],
                           **_) \
                           -> np.float64:
    """Calculate the scintillation index for the case of \
    a **downlink** wave being captured by a **point receiver** under **all turbulence conditions** \
    weak, moderate, or strong.

    Note:
        This function is for a flat-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, \
            no. v. PM152. Bellingham: SPIE, 2005. p. 497
        
    """

    rytov_var = scint_index_DL_PR_weak(wavelength=wavelength, zenith_angle=zenith_angle, altitude=altitude,\
                                            ris_model=ris_model)
    rvar_exp = np.pow(rytov_var, 6/5) # (sigma^2)^(6/5) = sigma^(12/5)
    in_exp_l = (0.49 * rytov_var) / np.pow(1 + 1.11 * rvar_exp, 7/6)
    in_exp_r = (0.51 * rytov_var) / np.pow(1 + 0.69 * rvar_exp, 5/6)

    return np.exp(in_exp_l + in_exp_r) - 1


def scint_index_DL_AA_general(wavelength: real_t, \
                            zenith_angle: real_t, \
                            altitude: real_array_t, \
                            ris_model: Callable[[real_array_t], real_array_t], \
                            aperture_diameter: real_t,
                            **_) \
                            -> np.float64:
    """Calculate the scintillation index for the case of \
    a **downlink** wave being captured by a an aperture wit diameter such that **aperture averaging** \
    occrus under **all turbulence conditions** weak, moderate, or strong.

    Note:
        This function is for a flat-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
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
    lct_altitude = np.min(altitude)
    sat_altitude = np.max(altitude)

    wavenum = 2 * np.pi / wavelength
    link_length = (sat_altitude - lct_altitude) / np.cos(zenith_angle)

    # We divide into left (out of Re function) and right (inside Re function)
    left = 8.70 * np.pow(wavenum, 7/6) * np.pow(sat_altitude - lct_altitude, 5/6) / np.pow(np.cos(zenith_angle), 11/6)
    # We calculate terms inside the integral
    in_integ_r1 = np.float_power( wavenum * aperture_diameter ** 2 / (16 * link_length) +  \
                                 1j*(altitude - lct_altitude) / (sat_altitude - lct_altitude), 5/6, dtype=complex)
    in_integ_r2 = np.pow(wavenum * aperture_diameter ** 2 / (16 * link_length), 5/6)
    in_integral = ris_model(altitude) * (in_integ_r1 - in_integ_r2)
    # We calculate the integral en keep its real part
    right = np.real(integrate.simpson(in_integral, altitude))

    return left * right


@warn_not_tested
def fried_parameter_DL(wavelength: real_t, zenith_angle: real_t, altitude: real_array_t, ris_model: Callable[[real_array_t], real_array_t], **_) -> np.float64:
    """Calculate the Fried paramater ($r_0$) for a downlink wave, assuming plane wave.

    Args:
        wavelength (real_t): wavelenght of the wave [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Note:
        This function is for a flat-earth model

    Returns:
        np.float64: the fried parameter for the specified SATCOM system [m].

    Source:
        L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, \
            no. v. PM152. Bellingham: SPIE, 2005. p. 492

    """
    wavenum_sq = np.pow( 2 * np.pi / wavelength, 2)
    base_val = 0.42 * wavenum_sq / np.cos(zenith_angle) * integrate.simpson(ris_model(altitude), altitude)
    return np.pow(base_val, -3/5)

#endregion DOWNLINK

#region UPLINK

def fried_parameter_UL_TX(wavelength: real_t, zenith_angle: real_t, altitude: real_array_t, ris_model: Callable[[real_array_t], real_array_t], **_) -> np.float64:
    """Calculate the Fried paramater ($r_{0T}$) for an uplink beam, assuming an spherical wave \
        as seen from the transmitter.

    Args:
        wavelength (real_t): wavelenght of the wave [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Note:
        This function is for a flat-earth model

    Returns:
        np.float64: the fried parameter for the specified SATCOM system [m].

    Source:
        [1] (p. 492) L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
        [2] (p. 189) F. G. Smith, J. S. Accetta, and D. L. Shumaker, Eds., “Propagation Throught Atmospheric Optical Turbulence,” in The infrared & electro-optical systems handbook, vol. 2: atmospheric propagation of radiation, in PM / SPIE, no. 1002. , Ann Arbor, Michigan: Infrared Information Analysis Center, 2023. doi: 10.1117/3.2543821.
        [3] L. B. Stotts and L. C. Andrews, “Optical communications in turbulence: a tutorial,” Opt. Eng., vol. 63, no. 04, Dec. 2023, doi: 10.1117/1.OE.63.4.041207.
        [4] L. B. Stotts, M. Toyoshima, and L. C. Andrews, “Effect of satellite slew rate on bit error rate model under atmospheric turbulence,” Opt. Eng., vol. 64, no. 05, May 2025, doi: 10.1117/1.OE.64.5.058104.
    """
    # pre-calculate
    lct_altitude = np.min(altitude)
    sat_altitude = np.max(altitude)
    link_len = sat_altitude - lct_altitude

    # divide
    left = 0.42 * np.pow(2 * np.pi / wavelength, 2) / np.cos(zenith_angle)
    right = integrate.simpson( ris_model(altitude) * np.pow(1 - (altitude - lct_altitude) / link_len, 5/3), altitude )

    return np.pow(left * right, -3/5)


@warn_not_tested
def fried_parameter_UL_RX(wavelength: real_t, zenith_angle: real_t, altitude: real_array_t, ris_model: Callable[[real_array_t], real_array_t], **_) -> np.float64:
    """Calculate the Fried paramater ($r_{0R}$) for an uplink beam, assuming an spherical wave \
        as seen from the receiver.

    Args:
        wavelength (real_t): wavelenght of the wave [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Note:
        This function is for a flat-earth model

    Returns:
        np.float64: the fried parameter for the specified SATCOM system [m].

    Source:
        [1] (p. 492) L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
        [2] (p. 189) F. G. Smith, J. S. Accetta, and D. L. Shumaker, Eds., “Propagation Throught Atmospheric Optical Turbulence,” in The infrared & electro-optical systems handbook, vol. 2: atmospheric propagation of radiation, in PM / SPIE, no. 1002. , Ann Arbor, Michigan: Infrared Information Analysis Center, 2023. doi: 10.1117/3.2543821.
        [3] L. B. Stotts and L. C. Andrews, “Optical communications in turbulence: a tutorial,” Opt. Eng., vol. 63, no. 04, Dec. 2023, doi: 10.1117/1.OE.63.4.041207.
        [4] L. B. Stotts, M. Toyoshima, and L. C. Andrews, “Effect of satellite slew rate on bit error rate model under atmospheric turbulence,” Opt. Eng., vol. 64, no. 05, May 2025, doi: 10.1117/1.OE.64.5.058104.
    """
    # pre-calculate
    lct_altitude = np.min(altitude)
    sat_altitude = np.max(altitude)
    link_len = sat_altitude - lct_altitude

    # divide
    left = 0.42 * np.pow(2 * np.pi / wavelength, 2) / np.cos(zenith_angle)
    right = integrate.simpson( ris_model(altitude) * np.pow( (altitude - lct_altitude) / link_len, 5/3), altitude )

    return np.pow(left * right, -3/5)


@warn_not_tested
def rytov_variance_UL_spherical(wavelength: real_t, \
                            zenith_angle: real_t, \
                            altitude: real_array_t, \
                            ris_model: Callable[[real_array_t], real_array_t],
                            **_) \
                            -> np.float64:
    """Calculate the Rytov variance for the case of an **uplink spherical wave**.

    Note:
        This function is for a flat-earth model
        Also known as $\\sigma_{Bu}$

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Returns:
        out (np.float64): the Rytov variance for an spherical wave [unitless].

    Source
        L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 52

    """

    # to not re-calculate
    max_altitude = np.max(altitude)
    lct_altitude = np.min(altitude)
    wavenum = 2 * np.pi / wavelength
    xi = 1 - (altitude - lct_altitude) / (max_altitude - lct_altitude)

    # we divide into left and right to calculate
    left = 2.25 * np.pow(wavenum, 7/6) * np.pow(max_altitude - lct_altitude, 5/6) / np.pow(np.cos(zenith_angle), 11/6)
    inner_right = ris_model(altitude) * np.pow(xi, 5/6) * np.pow(1 - xi, 5/6)
    right = integrate.simpson(inner_right, altitude)

    return left * right


@warn_not_tested
def scint_index_UL_spherical(wavelength: real_t, \
                            zenith_angle: real_t, \
                            altitude: real_array_t, \
                            ris_model: Callable[[real_array_t], real_array_t],
                            **_) \
                            -> np.float64:
    """Calculate the scintillation index for the case of \
    a tracked **uplink** spherical wave being captured by a **point receiver** \
    under **all turbulence conditions** weak, moderate, or strong.

    Note:
        This function is for a flat-earth model
        Also known as $\\sigma_{Bu}$

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 52

    """

    rytov_var = rytov_variance_UL_spherical(wavelength, zenith_angle, altitude, ris_model)

    in_exp_1 = 0.49 * rytov_var / np.pow(1 + 0.56 * np.pow(rytov_var, 6/5), 7/6)
    in_exp_2 = 0.51 * rytov_var / np.pow(1 + 0.69 * np.pow(rytov_var, 6/5), 5/6)
    return np.exp(in_exp_1 + in_exp_2) - 1


def rytov_variance_UL_gaussian(wavelength: real_t, \
                            zenith_angle: real_t, \
                            altitude: real_array_t, \
                            ris_model: Callable[[real_array_t], real_array_t],
                            Lambda: real_t,
                            Theta: real_t,
                            **_) \
                            -> np.float64:
    """Calculate the Rytov variance for the case of an **uplink collimated gaussian-beam wave**.

    Note:
        This function is for a flat-earth model
        Also known as $\\sigma_{Bu}$

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
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
    sat_altitude = np.max(altitude)
    lct_altitude = np.min(altitude)
    wavenum = 2 * np.pi / wavelength
    xi = 1 - (altitude - lct_altitude) / (sat_altitude - lct_altitude)
    neg_Theta = 1 - Theta
    
    # divide by sectors
    left = 8.70 * np.pow(wavenum, 7/6) / np.pow(np.cos(zenith_angle), 11/6) * np.pow(sat_altitude - lct_altitude, 5/6)

    right_in = np.pow(xi, 5/6) * np.pow(Lambda * xi + 1j * (1 - neg_Theta * xi), 5/6) - np.pow(Lambda, 5/6) * np.pow(xi, 5/3)
    right = integrate.simpson(ris_model(altitude) * right_in, altitude)
    return left * np.real(right)


def _total_beam_wander_variance_UL_gaussian(zenith_angle: real_t, \
                                            altitude: real_array_t, \
                                            ris_model: Callable[[real_array_t], real_array_t],
                                            beam_radius: real_t,
                                            pfront_radius: real_t) \
                                            -> np.float64:
    """Calculate the uplink total-beam-wander variance ($\\langle r_c^2\rangle$) for a collimated or convergent beam

    Note:
        flat-earth model
        Adapted by me for flat-earth

    Args:
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
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
    lct_altitude = np.min(altitude)
    sat_altitude = np.max(altitude)
    cos_z = np.cos(zenith_angle)
    effective_altitude = sat_altitude - lct_altitude

    # divide in sides
    left = 7.25 * np.pow(effective_altitude, 2) / ( np.pow(beam_radius, 1/3) * np.pow(cos_z, 3) )

    right_in1 = np.pow(1 - (altitude - lct_altitude) / effective_altitude , 2)
    right_in2 = np.pow(np.abs( 1 - (altitude - lct_altitude) / (pfront_radius * cos_z) ), 1/3)

    right = integrate.simpson(ris_model(altitude) * right_in1 / right_in2, altitude)

    return left * right


def _bw_pointing_error_var_UL_gaussian(wavelength: real_t, \
                                        zenith_angle: real_t, \
                                        altitude: real_array_t, \
                                        ris_model: Callable[[real_array_t], real_array_t],
                                        beam_radius: real_t,
                                        pfront_radius: real_t) \
                                        -> np.float64:
    """Calculate the beam-wander induced pointing-error variance for a collimated or convergent beam for \
        the uplink case. This pointing-error variance is NOT tracked/corrected and applies uniquely to LEO.

    Note:
        flat-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
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
                                                                   altitude=altitude,
                                                                   ris_model=ris_model,
                                                                   beam_radius=beam_radius,
                                                                   pfront_radius=pfront_radius)
    
    tx_fried_param = fried_parameter_UL_TX(wavelength=wavelength, zenith_angle=zenith_angle,
                                           altitude=altitude, ris_model=ris_model)

    # pre-calculate
    radius_to_fried_sq_LEO = np.pow( np.pi * beam_radius / tx_fried_param, 2)

    second_term = 1 - np.pow( radius_to_fried_sq_LEO / (1 + radius_to_fried_sq_LEO), 1/6 )

    return beam_wander_variance * second_term


def _bw_pointing_error_var_UL_gaussian_TT(wavelength: real_t, \
                                            zenith_angle: real_t, \
                                            altitude: real_array_t, \
                                            ris_model: Callable[[real_array_t], real_array_t],
                                            beam_radius: real_t,
                                            pfront_radius: real_t) \
                                            -> np.float64:
    """Calculate the beam-wander induced pointing-error variance for a collimated or convergent beam for \
        the tilt-corrected uplink case. This pointing-error variance IS tilt-corrected and applies uniquely to LEO.

    Note:
        flat-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
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
                                                                   altitude=altitude,
                                                                   ris_model=ris_model,
                                                                   beam_radius=beam_radius,
                                                                   pfront_radius=pfront_radius)
    
    tx_fried_param = fried_parameter_UL_TX(wavelength=wavelength, zenith_angle=zenith_angle,
                                           altitude=altitude, ris_model=ris_model)

    # pre-calculate
    radius_to_fried_sq_LEO = np.pow( np.pi * beam_radius / tx_fried_param, 2)

    # Calculate zernike tilt variance and the two terms separately
    zernike_tilt_variance = 0.57 * ( wavelength / (2 * beam_radius) ) * np.pow(2 * beam_radius / tx_fried_param, 5/6)
    first_term = np.pow(np.sqrt(beam_wander_variance) - zernike_tilt_variance * ( np.max(altitude) - np.min(altitude) ) / np.cos(zenith_angle), 2)
    second_term = 1 - np.pow( radius_to_fried_sq_LEO / (1 + radius_to_fried_sq_LEO), 1/6 )

    return first_term * second_term


def scint_index_UL_untracked_gaussian(wavelength: real_t, \
                                        zenith_angle: real_t,
                                        altitude: real_array_t,
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
        This function is for a flat-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
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
    tx_fried_radius = fried_parameter_UL_TX(wavelength=wavelength, zenith_angle=zenith_angle, altitude=altitude, ris_model=ris_model)
    perror_variance = _bw_pointing_error_var_UL_gaussian(wavelength=wavelength, zenith_angle=zenith_angle, altitude=altitude, ris_model=ris_model,\
                                                         beam_radius=beam_radius, pfront_radius=pfront_radius)
    rytov_var = rytov_variance_UL_gaussian(wavelength=wavelength, zenith_angle=zenith_angle, altitude=altitude, ris_model=ris_model,
                                           Lambda=Lambda, Theta=Theta)
    wavenum = 2 * np.pi / wavelength
    link_len = (np.max(altitude) - np.min(altitude)) / np.cos(zenith_angle)

    # separate in the sum
    left = 34.29 * np.pow( Lambda * link_len / ( wavenum * np.pow(tx_fried_radius, 2) ), 5/6 ) * ( perror_variance / np.pow(rx_spot_size, 2) )
    right_1 = 0.49 * rytov_var / np.pow( 1 + (1 + Theta) * 0.56 * np.pow(rytov_var, 6/5), 7/6 )
    right_2 = 0.51 * rytov_var / np.pow( 1 + 0.69 * np.pow(rytov_var, 6/5), 5/6 )
    right = np.exp(right_1 + right_2) - 1

    return left + right


def scint_index_UL_tracked_gaussian(wavelength: real_t, \
                                        zenith_angle: real_t,
                                        altitude: real_array_t,
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
        This function is for a flat-earth model

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        zenith_angle (real_t): zenith angle of the link [rad]
        altitude (real_array_t): array with altitudes that cover the SATCOM link [m]
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
    tx_fried_radius = fried_parameter_UL_TX(wavelength=wavelength, zenith_angle=zenith_angle, altitude=altitude, ris_model=ris_model)
    perror_variance = _bw_pointing_error_var_UL_gaussian_TT(wavelength=wavelength, zenith_angle=zenith_angle, altitude=altitude, ris_model=ris_model,\
                                                         beam_radius=beam_radius, pfront_radius=pfront_radius)
    rytov_var = rytov_variance_UL_gaussian(wavelength=wavelength, zenith_angle=zenith_angle, altitude=altitude, ris_model=ris_model,
                                           Lambda=Lambda, Theta=Theta)
    wavenum = 2 * np.pi / wavelength
    link_length = (np.max(altitude) - np.min(altitude)) / np.cos(zenith_angle)

    # Long-term spot/beam radius for H >> 20 km ([2] p. 180)
    lt_spot_radius = rx_spot_size * np.pow( 1 + np.pow( 2 * np.sqrt(2) * beam_radius / tx_fried_radius, 5/3) , 3/5 )

    # Long-term diffractive parameter ([2] p. 182)
    lt_Lambda = 2 * link_length / ( wavenum * np.pow(lt_spot_radius, 2) )

    # separate in the sum
    left = 34.29 * np.pow( lt_Lambda * link_length / ( wavenum * np.pow(tx_fried_radius, 2) ), 5/6 ) * ( perror_variance / np.pow(lt_spot_radius, 2) )
    right_1 = 0.49 * rytov_var / np.pow( 1 + (1 + Theta) * 0.56 * np.pow(rytov_var, 6/5), 7/6 )
    right_2 = 0.51 * rytov_var / np.pow( 1 + 0.69 * np.pow(rytov_var, 6/5), 5/6 )
    right = np.exp(right_1 + right_2) - 1

    return left + right


#endregion UPLINK