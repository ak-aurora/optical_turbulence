"""Some useful functions"""

from typing import Literal

import numpy as np
import numpy.typing as npt

from .typing import real_t


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

    full_divergence_angle = FWHM_angle / np.sqrt(np.log(2) / 2)
    radius = 2 * wavelength / (np.pi * full_divergence_angle)

    return radius

METHODS = Literal["linspace", "ITU-R P.1621-2"]
def create_altitude_array(
        sat_altitude: real_t,
        lct_altitude: real_t = 0,
        method: METHODS = "linspace",
        **_) -> npt.NDArray[np.float64]:
    """Create an array for altitudes above the LCT/OGS altitude until a maximum
    value.

    Args:
        sat_altitude (real_t): Satellite altitude from sea level [m]
        lct_altitude (real_t, optional): Laser Communication Terminal altitude from sea level [m]. Default value: 0 m.
        method: (str, optional): the method to generate the altitudes above the
            LCT/OGS. The method defines how the altitudes are sliced and the subsequent
            definition of the slicing. Options:
            - linspace: the altitude between LCT and SAT is sliced into 1 m intervals.
            - ITU-R P.1621-2: the altitude follows the recommendation of ITU-R P.1621-2 (available online)

    Returns:
        npt.NDArray[np.float64]: array with distances

    TODO:
        Implement more ways to generate the points (so its not necessarily always linear)
    """
    
    match method:
        case "linspace":
            altitude_array = np.linspace(lct_altitude, sat_altitude, int(sat_altitude - lct_altitude) + 1, endpoint=True)

        case "ITU-R P.1621-2":
            """We follow ITU's recommendation. Slicing the atmosphere from
            1 m to 20 km, and then we only consider values equal to or greater 
            than the LCT altitude.
            """

            indices = np.arange(1, 139 + 1)
            slices = np.exp( ( indices - 1 ) / 20 )
            altitude_array = np.cumsum(slices)
            altitude_array = altitude_array[altitude_array >= lct_altitude]

    return altitude_array
