"""Some useful functions
"""

from .typing import real_t  
import numpy as np


def FWHM_angle_to_beam_radius(wavelength: real_t, FWHM_angle: real_t):
    """Calculate the beam radius at the transmitter (1/e^2 point of irradiance)\
    using the full-width half-maximum angle (FWHM). 

    Args:
        wavelength (real_t): wavelength of the laser [m]
        FWHM_angle (real_t): full-width half-maximum angle [rad]

    Source:
        [1] D. Giggenbach, M. T. Knopp, and C. Fuchs, “Link budget calculation in optical LEO satellite downlinks with on/off-keying\
            and large signal divergence: A simplified methodology,” Satell Commun Network, vol. 41, no. 5, pp. 460-476,\
            Sep. 2023, doi: 10.1002/sat.1478.
        
    """

    full_divergence_angle = FWHM_angle / np.sqrt( np.log(2) / 2 )
    radius = 2 * wavelength / ( np.pi * full_divergence_angle )

    return radius