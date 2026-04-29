"""
Refractive-index structure (RIS) models for varying altitude.
"""

import numpy as np

from .typing import real_array_t, real_t

# ---------

HAP_GND_HEIGHT = 5 # [m]
HAP_REF_HEIGHT = 2 # [m]  


# ---------------------------------------
#   Hufnagel Valley model
#


def hufnagel_valley_model[T: real_t | real_array_t](
        altitude: T, 
        rms_wind_speed: real_t = np.float64(21), 
        A: real_t = np.float64(1.7e-14)) -> T:
    """Calculate the refractive-index structure ($C_n^2$) using the Hufnagel-Valley\
    boundary model. If the wind and A parameters are not given, the HV5/7 model is used.

    Args:
        altitude (real_array_t or real_t): altitude above ground [m]. If an array is given, an array is returned.
        rms_wind_speed (real_t): root-mean-square wind speed [m/s].
        A (real_t): nominal value of the refractive-index structure at the ground [m^(-2/3)]

    Returns:
        out (np.float64 or np.array[np.float64]): The refractive-index structure for the given input [m^(-2/3)].

    Source:
        Laser Beam Propagation through Random Media, 2 ed. Larry Andrews
        
    See Also:
        - Plotter functions in tools_plotter.py to visualize the model and compare against literature.
        
        - ITU-R P.1621-2 (07/2015): "to ensure an accurate estimate of the atmospheric turbulence profile, layer \
            thickness or integration step size in height should increase exponentially, from 0.001 km at the \
            lowest layer (ground level) to 1 km at an altitude of 20 km following $h_i = exp[(i-1)/20]$ [m] \
            from i = 1 to i = 139.
    """

    a = 0.00594 * (rms_wind_speed / 27)**2 * (10**(-5) * altitude) ** 10 * np.exp(-altitude / 1000)  # ty:ignore[unsupported-operator]
    b = 2.7 * 10 ** (-16) * np.exp(-altitude / 1500)
    c = A * np.exp(-altitude / 100)

    return a + b + c # pyright: ignore[reportReturnType]

# ---------------------------------------
#   Hufnagel-Andrews-Phillips Model
#

def hap_model_daytime[T: real_t | real_array_t](
        altitude: T, 
        rms_wind_speed: real_t = np.float64(21), 
        A: real_t = np.float64(1.7e-14),
        M: real_t = np.float64(1), **_) -> T:
    """Calculate the refractive-index structure ($C_n^2$) using the Hufnagel-Andrews-Phillips\
    boundary model for **daytime**. If the wind and A parameters are not given, the HV5/7 model parameters \
        are used.

    Args:
        altitude (real_array_t or real_t): altitude above ground [m]. If an array is given, an array is returned.
        rms_wind_speed (real_t): root-mean-square wind speed [m/s].
        A (real_t): nominal value of the refractive-index structure at the LCT altitude [m^(-2/3)]
        M (real_t): multiplicative factor [unitless]. Defaults to 1.

    Returns:
        out (np.float64 or np.array[np.float64]): The refractive-index structure for the given input [m^(-2/3)].

    Source:
        [1] L. B. Stotts, M. Toyoshima, and L. C. Andrews, “Effect of satellite slew rate on bit error rate model under \
            atmospheric turbulence,” Opt. Eng., vol. 64, no. 05, May 2025, doi: 10.1117/1.OE.64.5.058104.
        [2] L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. v. FG41. \
            Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019.
        [3] L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics.\
            Bellingham, Washington, USA: SPIE Press, 2023.

        
    See Also:
        - Plotter functions in tools_plotter.py to visualize the model and compare against literature.
        
        - ITU-R P.1621-2 (07/2015): "to ensure an accurate estimate of the atmospheric turbulence profile, layer \
            thickness or integration step size in height should increase exponentially, from 0.001 km at the \
            lowest layer (ground level) to 1 km at an altitude of 20 km following $h_i = exp[(i-1)/20]$ [m] \
            from i = 1 to i = 139.

    More Info:
        - As stated in [2], the reference height above ground $h_0$ is set to 2 m.
        - As stated in [3], the ground height is set to 5 m.

    """
    if np.min(altitude) <= 0:
        raise ValueError("altitude has to be greater than zero.")
    
    a = 1.04 * 1e-3 * ( (rms_wind_speed / 27) ** 2 ) * ( (( altitude + HAP_GND_HEIGHT ) * 1e-5) ** 10 ) * np.exp(- ((altitude + HAP_GND_HEIGHT) / 1200.0) )  # ty:ignore[unsupported-operator]
    b = 2.7 * 1e-16 * np.exp( -(( altitude + HAP_GND_HEIGHT ) / 1700.0) )  # ty:ignore[unsupported-operator]
    c = A * np.pow( ( HAP_REF_HEIGHT / altitude), 4/3 )  # ty:ignore[unsupported-operator]
    return M * (a + b) + c # pyright: ignore[reportReturnType]

# @warn_not_tested # missing a plot to compare with
def hap_model_nightime[T: real_t | real_array_t](
        altitude: T, 
        rms_wind_speed: real_t = np.float64(21), 
        A: real_t = np.float64(1.7e-14),
        M: real_t = np.float64(1), **_) -> T:
    """Calculate the refractive-index structure ($C_n^2$) using the Hufnagel-Andrews-Phillips\
    boundary model for **nighttime**. If the wind and A parameters are not given, the HV5/7 model parameters \
        are used.

    Args:
        altitude (real_array_t or real_t): altitude above ground [m]. If an array is given, an array is returned.
        rms_wind_speed (real_t): root-mean-square wind speed [m/s].
        A (real_t): nominal value of the refractive-index structure at the LCT altitude [m^(-2/3)]
        lct_altitude (real_t): altitude of the Laser Communication Terminal [m] (if altitude is an array, \
            the default value is min(altitude), if it is a scalar, the default value is altitude / 2) 
        M (real_t): multiplicative factor [unitless]. Defaults to 1.

    Returns:
        out (np.float64 or np.array[np.float64]): The refractive-index structure for the given input [m^(-2/3)].

    Source:
        [1] L. B. Stotts, M. Toyoshima, and L. C. Andrews, “Effect of satellite slew rate on bit error rate model under \
            atmospheric turbulence,” Opt. Eng., vol. 64, no. 05, May 2025, doi: 10.1117/1.OE.64.5.058104.
        [2] L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. v. FG41. \
            Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019.

        
    See Also:
        - Plotter functions in tools_plotter.py to visualize the model and compare against literature.
        
        - ITU-R P.1621-2 (07/2015): "to ensure an accurate estimate of the atmospheric turbulence profile, layer \
            thickness or integration step size in height should increase exponentially, from 0.001 km at the \
            lowest layer (ground level) to 1 km at an altitude of 20 km following $h_i = exp[(i-1)/20]$ [m] \
            from i = 1 to i = 139.

    More Info:
        - As stated in [2], the reference height above ground $h_0$ is set to 2 m.
        - As stated in [3], the ground height is set to 5 m.

    """
    if np.min(altitude) <= 0:
        raise ValueError("altitude has to be greater than zero.")

    a = 1.04 * 1e-3 * ( (rms_wind_speed / 27) ** 2 ) * ( (( altitude + HAP_GND_HEIGHT ) * 1e-5) ** 10 ) * np.exp(- (( altitude + HAP_GND_HEIGHT ) / 1200.0))  # ty:ignore[unsupported-operator]
    b = 2.7 * 1e-16 * np.exp( -(( altitude + HAP_GND_HEIGHT ) / 1700.0) )  # ty:ignore[unsupported-operator]
    c = A * np.exp( -(altitude / 1000.0)) * np.pow(( HAP_REF_HEIGHT / altitude), 2/3)  # ty:ignore[unsupported-operator]

    return M * (a + b) + c # pyright: ignore[reportReturnType]