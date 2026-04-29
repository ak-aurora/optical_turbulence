"""
Scintillation index, Rytov variance, and Fried parameter following the "link distance" model. Downlink and Uplink.
The implementations focus on link distance and are earth-model agnostic \
"""

__author__ = "Agustín González Uriarte"

# ----------------- IMPORTS ----------------- #

from typing import Callable

import numpy as np
from scipy import integrate

from .__decorators import warn_not_tested
from .typing import real_array_t, real_t

# ----------------- CONSTANTS ----------------- #

EARTH_RADIUS = 6371e3 # [m]

# ----------------- Functions ----------------- #

#region DOWNLINK


def scint_index_DL_PR_weak(
        wavelength: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the scintillation index (in this case equivalent to rytov variance) for the case of \
    a **downlink** wave being captured by a **point receiver under weak turbulence**.

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, \
            no. v. PM152. Bellingham: SPIE, 2005. p. 495
        
    """
    
    left_side = 2.25 * np.pow( (2 * np.pi / wavelength), 7/6 )
    rs_inner = ris_model(altitude_array) * np.pow((link_array), 5/6)
    right_side = integrate.simpson(rs_inner, link_array)
    
    return left_side * right_side


def scint_index_DL_PR_general(
        wavelength: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the scintillation index for the case of \
    a **downlink** wave being captured by a **point receiver** under **all turbulence conditions** \
    weak, moderate, or strong.

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, \
            no. v. PM152. Bellingham: SPIE, 2005. p. 497
        
    """

    rytov_var = scint_index_DL_PR_weak(
        wavelength=wavelength,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )

    rvar_exp = np.pow(rytov_var, 6/5) # (sigma^2)^(6/5) = sigma^(12/5)
    in_exp_l = (0.49 * rytov_var) / np.pow(1 + 1.11 * rvar_exp, 7/6)
    in_exp_r = (0.51 * rytov_var) / np.pow(1 + 0.69 * rvar_exp, 5/6)

    return np.exp(in_exp_l + in_exp_r) - 1


def scint_index_DL_AA_general(
        aperture_diameter: real_t,
        wavelength: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the scintillation index for the case of \
    a **downlink** wave being captured by an aperture with diameter such that **aperture averaging** \
    occurs under **all turbulence conditions** weak, moderate, or strong.


    Args:
        aperture_diameter (real_t): diameter of the receiver aperture [m].
        wavelength (real_t): wavelength of the beam sent [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, \
            no. v. PM152. Bellingham: SPIE, 2005. p. 497
    """
    # To not re-calculate
    wavenum = 2 * np.pi / wavelength

    # We divide into left (out of Re function) and right (inside Re function)
    left = 8.70 * np.pow(wavenum, 7/6)
    # We calculate terms inside the integral
    in_integ_r1 = np.float_power( wavenum * aperture_diameter ** 2 / 16 + 1j * link_array, 5/6, dtype=complex)
    in_integ_r2 = np.pow(wavenum * aperture_diameter ** 2 / 16, 5/6)
    in_integral = ris_model(altitude_array) * (in_integ_r1 - in_integ_r2)
    # We calculate the integral and keep its real part
    right = np.real(integrate.simpson(in_integral, link_array))

    return left * right


@warn_not_tested
def fried_parameter_DL(
        wavelength: real_t, 
        ris_model: Callable[[real_array_t], real_array_t], 
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the Fried paramater ($r_0$) for a downlink wave, assuming plane wave.

    Args:
        wavelength (real_t): wavelenght of the wave [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Note:
        This function is for a round-earth model

    Returns:
        np.float64: the fried parameter for the specified SATCOM system [m].

    Source:
        L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, \
            no. v. PM152. Bellingham: SPIE, 2005. p. 492

    """

    wavenum_sq = np.pow( 2 * np.pi / wavelength, 2)
    base_val = 0.42 * wavenum_sq * integrate.simpson(ris_model(altitude_array), link_array)
    return np.pow(base_val, -3/5)


#endregion DOWNLINK

#region UPLINK


@warn_not_tested
def isonoplanatic_angle_UL(
        wavelength: real_t, \
        Lambda: real_t,
        Theta: real_t,
        neg_Theta: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the isoplanatic angle of an **uplink gaussian-beam wave**.

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        Lambda (real_t): diffractive parameter of the beam at the receiver plane [unitless]
        Theta (real_t): refractive parameter of the beam at the receiver plane [unitless]
        neg_Theta (real_t): overbar refractive parameter of the beam at the receiver plane [unitless] (1 - Theta)
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        out (np.float64): the isoplanatic angle for the uplink gaussian beam wave [rad].

    Source
        Andrews, Larry C. Laser Beam Propagation Through Random Media, \
            2nd ed. Press Monograph Series, v. PM152. SPIE, 2005. p. 493

    """
    # Since the link array goes from 0 (at OGS) to L (at satellite)
    link_distance = link_array[-1]

    wavenumber_sq = np.pow( 2 * np.pi / wavelength, 2)

    mu_frac = link_array / link_distance

    mu1u = integrate.simpson(ris_model(altitude_array) * 
        np.pow( Theta + neg_Theta * mu_frac , 5/3 ),
        link_array
    )

    mu2u = integrate.simpson(ris_model(altitude_array) * 
        np.pow( 1 - mu_frac, 5/3 ),
        link_array
    )

    outside_pow = np.pow( link_distance, -1 )
    inside_pow = 2.91 * wavenumber_sq * (mu1u + 0.62 * mu2u * np.pow( Lambda, 11/6 ))
    pow_term = np.pow( inside_pow, -3/5 )

    return outside_pow * pow_term


def fried_parameter_UL_TX(
        wavelength: real_t, 
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the Fried paramater ($r_{0T}$) for an uplink beam, assuming an spherical wave \
        as seen from the transmitter.

    Args:
        wavelength (real_t): wavelenght of the wave [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        np.float64: the fried parameter for the specified SATCOM system [m].

    Source:
        [1] (p. 492) L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
        [2] (p. 189) F. G. Smith, J. S. Accetta, and D. L. Shumaker, Eds., “Propagation Throught Atmospheric Optical Turbulence,” in The infrared & electro-optical systems handbook, vol. 2: atmospheric propagation of radiation, in PM / SPIE, no. 1002. , Ann Arbor, Michigan: Infrared Information Analysis Center, 2023. doi: 10.1117/3.2543821.
        [3] L. B. Stotts and L. C. Andrews, “Optical communications in turbulence: a tutorial,” Opt. Eng., vol. 63, no. 04, Dec. 2023, doi: 10.1117/1.OE.63.4.041207.
        [4] L. B. Stotts, M. Toyoshima, and L. C. Andrews, “Effect of satellite slew rate on bit error rate model under atmospheric turbulence,” Opt. Eng., vol. 64, no. 05, May 2025, doi: 10.1117/1.OE.64.5.058104.
    """
    # Since the link array goes from 0 (at OGS) to L (at satellite)
    link_distance = link_array[-1]

    # divide
    left = 0.42 * np.pow(2 * np.pi / wavelength, 2)
    
    right = integrate.simpson( 
        ris_model(altitude_array) * np.pow(1 - link_array / link_distance, 5/3),
        link_array
    )

    return np.pow(left * right, -3/5)


@warn_not_tested
def fried_parameter_UL_RX(
        wavelength: real_t,  
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the Fried paramater ($r_{0R}$) for an uplink beam, assuming an spherical wave \
        as seen from the receiver.

    Args:
        wavelength (real_t): wavelenght of the wave [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        np.float64: the fried parameter for the specified SATCOM system [m].

    Source:
        [1] (p. 492) L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
        [2] (p. 189) F. G. Smith, J. S. Accetta, and D. L. Shumaker, Eds., “Propagation Throught Atmospheric Optical Turbulence,” in The infrared & electro-optical systems handbook, vol. 2: atmospheric propagation of radiation, in PM / SPIE, no. 1002. , Ann Arbor, Michigan: Infrared Information Analysis Center, 2023. doi: 10.1117/3.2543821.
        [3] L. B. Stotts and L. C. Andrews, “Optical communications in turbulence: a tutorial,” Opt. Eng., vol. 63, no. 04, Dec. 2023, doi: 10.1117/1.OE.63.4.041207.
        [4] L. B. Stotts, M. Toyoshima, and L. C. Andrews, “Effect of satellite slew rate on bit error rate model under atmospheric turbulence,” Opt. Eng., vol. 64, no. 05, May 2025, doi: 10.1117/1.OE.64.5.058104.
    """
    # Since the link array goes from 0 (at OGS) to L (at satellite)
    link_distance = link_array[-1]

    # divide
    left = 0.42 * np.pow(2 * np.pi / wavelength, 2)

    right = integrate.simpson( 
        ris_model(altitude_array) * np.pow( link_array / link_distance, 5/3),
        link_array
    )

    return np.pow(left * right, -3/5)


@warn_not_tested
def rytov_variance_UL_spherical(
        wavelength: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the Rytov variance for the case of an **uplink spherical wave**.

    Note:
        Also known as $\\sigma_{Bu}$

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        out (np.float64): the Rytov variance for an spherical wave [unitless].

    Source
        L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 52

    """
    # Since the link array goes from 0 (at OGS) to L (at satellite)
    link_distance = link_array[-1]
    wavenum = 2 * np.pi / wavelength

    # we divide into left and right to calculate
    left = 2.25 * np.pow(wavenum, 7/6)
    inner_right = ris_model(altitude_array) * np.pow(link_array, 5/6) * np.pow(1 - link_array / link_distance, 5/6)
    right = integrate.simpson(inner_right, link_array)

    return left * right


@warn_not_tested
def scint_index_UL_spherical(
        wavelength: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the scintillation index for the case of \
    a tracked **uplink** spherical wave being captured by a **point receiver** \
    under **all turbulence conditions** weak, moderate, or strong.

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 52

    """

    rytov_var = rytov_variance_UL_spherical(
        wavelength=wavelength,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )

    in_exp_1 = 0.49 * rytov_var / np.pow(1 + 0.56 * np.pow(rytov_var, 6/5), 7/6)
    in_exp_2 = 0.51 * rytov_var / np.pow(1 + 0.69 * np.pow(rytov_var, 6/5), 5/6)
    return np.exp(in_exp_1 + in_exp_2) - 1


def rytov_variance_UL_gaussian(
        wavelength: real_t,
        Lambda: real_t,
        Theta: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the Rytov variance for the case of an **uplink collimated gaussian-beam wave**.

    Note:
        Also known as $\\sigma_{Bu}$

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        Lambda (real_t): diffractive parameter of the beam at the receiver plane [unitless]
        Theta (real_t): refractive parameter of the beam at the receiver plane [unitless]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        out (np.float64): the Rytov variance for a collimated gaussian-beam wave [unitless].

    Source
        L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 110

    """
    # to not re-calculate
    # Since the link array goes from 0 (at OGS) to L (at satellite)
    link_distance = link_array[-1]

    wavenum = 2 * np.pi / wavelength
    xi = 1 - ( link_array / link_distance )
    neg_Theta = 1 - Theta
    
    # divide by sectors
    left = 8.70 * np.pow(wavenum, 7/6) * np.pow(link_distance, 5/6)

    right_in = np.pow(xi, 5/6) * np.pow(Lambda * xi + 1j * (1 - neg_Theta * xi), 5/6) - np.pow(Lambda, 5/6) * np.pow(xi, 5/3)
    right = integrate.simpson(ris_model(altitude_array) * right_in, link_array)
    return left * np.real(right)


@warn_not_tested
def scint_index_UL_gaussian(
        wavelength: real_t,
        Lambda: real_t,
        Theta: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the scintillation index for the case of \
    a perfectly-tracked **uplink** gaussian wave being captured by a **point receiver** \
    under **all turbulence conditions** weak, moderate, or strong.

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        Lambda (real_t): diffractive parameter of the beam at the receiver plane [unitless]
        Theta (real_t): refractive parameter of the beam at the receiver plane [unitless]
        ris_model (callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 52

    """

    rytov_var = rytov_variance_UL_gaussian(
        wavelength=wavelength,
        Lambda=Lambda,
        Theta=Theta,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )

    in_exp_1 = 0.49 * rytov_var / np.pow(1 + ( 1 + Theta ) * 0.56 * np.pow(rytov_var, 6/5), 7/6)
    in_exp_2 = 0.51 * rytov_var / np.pow(1 + 0.69 * np.pow(rytov_var, 6/5), 5/6)
    return np.exp(in_exp_1 + in_exp_2) - 1


def _total_beam_wander_variance_UL_gaussian(
        beam_radius: real_t,
        pfront_radius: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the uplink total-beam-wander variance ($\\langle r_c^2\rangle$) for a collimated or convergent beam

    Args:
        beam_radius (real_t): $W_0$ effective beam radius (spot size) [m]
        pfront_radius (real_t): phase front radius of curvature at the transmitter [m].
        ris_model (Callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        np.float64: total beam wander variance for the given beam type [m^2]

    Source:
        L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. \
        Bellingham, Washington, USA: SPIE Press, 2023. p. 180

    """
    # Since the link array goes from 0 (at OGS) to L (at satellite)
    link_distance = link_array[-1]

    # divide in sides
    left = 7.25 * np.pow(link_distance, 2) / ( np.pow(beam_radius, 1/3) )

    right_in1 = np.pow(1 - link_array / link_distance , 2)
    right_in2 = np.pow(np.abs( 1 - link_array / (pfront_radius) ), 1/3)

    right = integrate.simpson(
        ris_model(altitude_array) * right_in1 / right_in2,
        link_array
    )

    return left * right


def _bw_pointing_error_var_UL_gaussian(
        wavelength: real_t,
        beam_radius: real_t,
        pfront_radius: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the beam-wander induced pointing-error variance for a collimated or convergent beam for \
        the uplink case. This pointing-error variance is NOT tracked/corrected and applies uniquely to LEO.

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        beam_radius (real_t): $W_0$ effective beam radius (spot size) [m]
        pfront_radius (real_t): phase front radius of curvature at the transmitter [m].
        ris_model (Callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        np.float64: total beam wander variance for the given beam type [m^2]

    Source:
        L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. \
        Bellingham, Washington, USA: SPIE Press, 2023. p. 181

    """
    beam_wander_variance = _total_beam_wander_variance_UL_gaussian(
        beam_radius=beam_radius,
        pfront_radius=pfront_radius,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )
    
    tx_fried_param = fried_parameter_UL_TX(
        wavelength=wavelength,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )

    # pre-calculate
    radius_to_fried_sq_LEO = np.pow( np.pi * beam_radius / tx_fried_param, 2)

    second_term = 1 - np.pow( radius_to_fried_sq_LEO / (1 + radius_to_fried_sq_LEO), 1/6 )

    return beam_wander_variance * second_term


def _bw_pointing_error_var_UL_gaussian_TT(
        wavelength: real_t,
        beam_radius: real_t,
        pfront_radius: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the beam-wander induced pointing-error variance for a collimated or convergent beam for \
        the tilt-corrected uplink case. This pointing-error variance IS tilt-corrected and applies uniquely to LEO.

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        beam_radius (real_t): $W_0$ effective beam radius (spot size) [m]
        pfront_radius (real_t): phase front radius of curvature at the transmitter [m].
        ris_model (Callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        np.float64: total beam wander variance for the given beam type [m^2]

    Source:
        L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. \
        Bellingham, Washington, USA: SPIE Press, 2023. p. 181

    """
    beam_wander_variance = _total_beam_wander_variance_UL_gaussian(
        beam_radius=beam_radius,
        pfront_radius=pfront_radius,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )
    
    tx_fried_param = fried_parameter_UL_TX(
        wavelength=wavelength,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )

    # pre-calculate
    radius_to_fried_sq_LEO = np.pow( np.pi * beam_radius / tx_fried_param, 2)
    link_distance = link_array[-1]

    # Calculate zernike tilt variance and the two terms separately
    zernike_tilt_variance = 0.57 * ( wavelength / (2 * beam_radius) ) * np.pow(2 * beam_radius / tx_fried_param, 5/6)
    first_term = np.pow(np.sqrt(beam_wander_variance) - zernike_tilt_variance * link_distance, 2)
    second_term = 1 - np.pow( radius_to_fried_sq_LEO / (1 + radius_to_fried_sq_LEO), 1/6 )

    return first_term * second_term


def scint_index_UL_untracked_gaussian(
        wavelength: real_t,
        beam_radius: real_t,
        pfront_radius: real_t,
        Lambda: real_t,
        Theta: real_t,
        rx_spot_size: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the scintillation index for the case of \
    an **untracked uplink** collimated or convergent wave being captured by a **point receiver** \
    under **all turbulence conditions** weak, moderate, or strong.

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        beam_radius (real_t): $W_0$ effective beam radius (spot size) [m]
        pfront_radius (real_t): $F_0$ phase front radius of curvature at the transmitter [m].
        Lambda (real_t): diffractive parameter of the beam at the receiver plane [unitless]
        Theta (real_t): refractive parameter of the beam at the receiver plane [unitless]
        rx_spot_size (real_t): $W$ spot size (effective beam radius) of the beam at the receiver plane [m]
        ris_model (Callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        [1] L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 110
        [2] L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. \
        Bellingham, Washington, USA: SPIE Press, 2023.

    """
    # pre-calculate some values
    tx_fried_radius = fried_parameter_UL_TX(
        wavelength=wavelength,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )

    perror_variance = _bw_pointing_error_var_UL_gaussian(
        wavelength=wavelength,
        beam_radius=beam_radius,
        pfront_radius=pfront_radius,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )

    rytov_var = rytov_variance_UL_gaussian(
        wavelength=wavelength,
        Lambda=Lambda,
        Theta=Theta,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )

    # link array goes from 0 to L
    link_distance = link_array[-1]
    wavenum = 2 * np.pi / wavelength

    # separate in the sum
    left = 34.29 * np.pow( Lambda * link_distance / ( wavenum * np.pow(tx_fried_radius, 2) ), 5/6 ) * ( perror_variance / np.pow(rx_spot_size, 2) )
    
    right_1 = 0.49 * rytov_var / np.pow( 1 + (1 + Theta) * 0.56 * np.pow(rytov_var, 6/5), 7/6 )
    right_2 = 0.51 * rytov_var / np.pow( 1 + 0.69 * np.pow(rytov_var, 6/5), 5/6 )
    right = np.exp(right_1 + right_2) - 1

    return left + right


def scint_index_UL_tracked_gaussian(
        wavelength: real_t, \
        beam_radius: real_t,
        pfront_radius: real_t,
        Lambda: real_t,
        Theta: real_t,
        rx_spot_size: real_t,
        ris_model: Callable[[real_array_t], real_array_t],
        altitude_array: real_array_t,
        link_array: real_array_t,
        **_) -> np.float64:
    """Calculate the scintillation index for the case of \
    a **tracked uplink** collimated or convergent wave being captured by a **point receiver** \
    under **all turbulence conditions** weak, moderate, or strong.

    Args:
        wavelength (real_t): wavelength of the beam sent [m]
        beam_radius (real_t): $W_0$ effective beam radius (spot size) [m]
        pfront_radius (real_t): $F_0$ phase front radius of curvature at the transmitter [m].
        Lambda (real_t): diffractive parameter of the beam at the receiver plane [unitless]
        Theta (real_t): refractive parameter of the beam at the receiver plane [unitless]
        rx_spot_size (real_t): $W$ spot size (effective beam radius) of the beam at the receiver plane [m]
        ris_model (Callable[[real_array_t], real_array_t]): callable of the refractive-index \
            structure model that only has one parameter: an array of altitudes [m^{-2/3}].
        altitude_array (real_array_t): array with altitudes above sea level
            corresponding to the different link values [m].
        link_array (real_array_t): distances following the on-axis line from \
            OGS to satellite. The indices match with the altitude array indices [m].
        _ (Any): consume all the extra keyword arguments [n/a].

    Returns:
        out (np.float64): the scintillation index for the specific scenario [unitless].

    Source
        [1] L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. \
            v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019. p. 110
        [2] L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. \
        Bellingham, Washington, USA: SPIE Press, 2023.

    """
    # pre-calculate some values
    tx_fried_radius = fried_parameter_UL_TX(
        wavelength=wavelength,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )

    perror_variance = _bw_pointing_error_var_UL_gaussian_TT(
        wavelength=wavelength,
        beam_radius=beam_radius,
        pfront_radius=pfront_radius,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )

    rytov_var = rytov_variance_UL_gaussian(
        wavelength=wavelength,
        Lambda=Lambda,
        Theta=Theta,
        ris_model=ris_model,
        altitude_array=altitude_array,
        link_array=link_array
    )

    wavenum = 2 * np.pi / wavelength
    link_distance = link_array[-1]

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
