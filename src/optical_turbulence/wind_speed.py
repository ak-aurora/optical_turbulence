"""
Wind speed models for the RIS models.
"""

import numpy as np
from scipy import integrate

from .typing import real_array_t, real_t


def bufton_model(height: real_array_t | real_t, slew_rate, ground_speed) -> real_array_t | real_t:
    """Calculate the wind speed using the Bufton wind model.

    Args:
        height (real_array_t or real_t): altitude at which to calculate the wind speed [m]
        slew_rate (real_t): slew rate of the satellite connected via the link [rad/s]
        ground_speed (real_t): ground wind speed [m/s]

    Returns:
        numeric_t: wind speed for the given parameters

    Sources:
        [1] M. Toyoshima, H. Takenaka, and Y. Takayama, “Atmospheric\
            turbulence-induced fading channel model for space-to-ground\
            laser communications links,” Opt. Express, vol. 19,\
            no. 17, p. 15965, Aug. 2011, doi: 10.1364/OE.19.015965.
        [2] L. C. Andrews, Field Guide to Atmospheric Optics, \
            Second Edition, 2nd ed. Bellingham, WA: SPIE, 2019.


    """

    return slew_rate * height + ground_speed + 30 * np.exp( -1 * ( (height - 9400) / 4800 ) ** 2 )

def rms_windspeed_bufton(slew_rate: real_t, ground_speed: real_t = 21) -> np.float64:
    """Calculate the RMS windspeed to be used in the HV model. It internally uses the \
    bufton wind model to estimate the wind speed at different heights.

    Args:
        slew_rate (real_t): slew rate of the satellite [rad/s]
        ground_speed (real_t, optional): ground windspeed [m/s]. Defaults to 21 m/s.

    Returns:
        windspeed (real_t): rms windspeed.

    Source:
        [1] Laser beam propagation through random media 2 ed.
        [2] L. C. Andrews, Field Guide to Atmospheric Optics, \
            Second Edition, 2nd ed. Bellingham, WA: SPIE, 2019.

    """
    def _bufton(h):
        return np.pow(bufton_model(h, slew_rate, ground_speed), 2)
    integral = integrate.quad(_bufton, 5e3, 20e3)
    return np.sqrt( integral[0] / ( 15e3 ) )