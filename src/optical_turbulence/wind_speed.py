"""
Wind speed models for the RIS models.
"""

import numpy as np
from scipy import integrate
from .typing import *
from .__decorators import warn_not_tested

@warn_not_tested # No results identified in literature
def bufton_model(height: real_array_t | real_t, slew_rate, ground_speed) -> real_array_t | real_t:
    """Calculate the wind speed using the Bufton wind model.

    Args:
        height (real_array_t or real_t): altitude at which to calculate the wind speed [m]
        slew_rate (real_t): slew rate of the satellite connected via the link [rad/s]
        ground_speed (real_t): ground wind speed [m/s]

    Returns:
        numeric_t: wind speed for the given parameters

    Source:
        Laser propagation through random media, 2 ed. p. 481
        D. P. Greenwood, “Bandwidth specification for adaptive optics systems*,” \
            Journal of the Optical Society of America, vol. 67, no. 3, p. 390, \
            Mar. 1977, doi: 10.1364/josa.67.000390. (it includes the 3048 m site altitude).
    """

    return slew_rate * height + ground_speed + 30 * np.exp( -1 * ( (height - 12448) / 4800 ) ** 2 )

@warn_not_tested # No results identified in literature
def rms_windspeed_bufton(slew_rate: real_t, ground_speed: real_t) -> np.float64:
    """Calculate the RMS windspeed to be used in the HV model. It internally uses the \
    bufton wind model to estimate the wind speed at different heights.

    Args:
        slew_rate (real_t): slew rate of the satellite [rad/s]
        ground_speed (real_t): ground windspeed [m/s]

    Returns:
        windspeed (real_t): rms windspeed.

    Source:
        Laser beam propagation through random media 2 ed.

    """
    _bufton = lambda h: bufton_model(h, slew_rate, ground_speed)
    integral = integrate.quad(_bufton, 5e3, 20e3)
    return np.sqrt( integral[0] / ( 15e3 ) )