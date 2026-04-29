from abc import ABC, abstractmethod

import numpy as np

from .typing import real_t, real_array_t

# ----------------- CONSTANTS ----------------- #

EARTH_RADIUS = 6371e3 # [m]

# ----------------- GENERIC MODEL ----------------- #
 
class EarthModelGeneric(ABC):

    @staticmethod
    @abstractmethod
    def altitude_to_link[T: real_t | real_array_t](
                         altitude_array: T,
                         sat_altitude: real_t, 
                         lct_altitude: real_t, 
                         zenith_angle: real_t,
                         **_) -> T:
        """From an array with altitudes from sea level calculate the corresponding array with \
            link distances (distance from the OGS to that point for each altitude). 

        Args:
            altitude_array (real_array_t): array with altitudes from lct_altitude to sat_altitude [m].
            sat_altitude (real_t): satellite altitude [m].
            lct_altitude (real_t): laser communication terminal/optical ground station altitude [m].
            zenith_angle (real_t): zenith angle of the link [rad].

        Returns:
            real_array_t: array with the corresponding link distances for the given altitudes [m].
        """
        
        pass


    @staticmethod
    @abstractmethod
    def link_to_altitude[T: real_t | real_array_t](
                         link_array: T,
                         sat_altitude: real_t, 
                         lct_altitude: real_t, 
                         zenith_angle: real_t,
                         **_) -> T:
        """From an array with link distances calculate the corresponding array with altitudes \
        (altitude from sea level for each corresponding link distance). 

        Args:
            link_array (real_array_t): array with link distances from the LCT to the satellite [m].
            sat_altitude (real_t): satellite altitude [m].
            lct_altitude (real_t): laser communication terminal/optical ground station altitude [m].
            zenith_angle (real_t): zenith angle of the link [rad].

        Returns:
            real_array_t: array with the corresponding altitudes for the given link distances [m].
        """

        pass

    
    @staticmethod
    @abstractmethod
    def calculate_total_link_distance(sat_altitude: real_t,
                                      lct_altitude: real_t,
                                      zenith_angle: real_t,
                                      **_) -> real_t:
        """Calculate the total link distance between LCT/OGS and satellite.

        Args:
            sat_altitude (real_t): altitude to the satellite from sea level [m].
            lct_altitude (real_t): altitude of the laser communication terminal/optical ground station above sea level [m].
            zenith_angle (real_t): zenith angle of the link [rad].

        Returns:
            real_t: distance from the LCT/OGS to the satellite [m].
        """
        
        pass


# ----------------- MODEL IMPLEMENTATIONS ----------------- #

class FlatEarthModel(EarthModelGeneric):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def altitude_to_link[T: real_t | real_array_t](
                         altitude_array: T,
                         sat_altitude: real_t, 
                         lct_altitude: real_t, 
                         zenith_angle: real_t,
                         **_) -> T:
        """From an array with altitudes from sea level calculate the corresponding array with \
            link distances (distance from the OGS to that point for each altitude). 

        Args:
            altitude_array (real_array_t): array with altitudes from lct_altitude to sat_altitude [m].
            sat_altitude (real_t): satellite altitude [m].
            lct_altitude (real_t): laser communication terminal/optical ground station altitude [m].
            zenith_angle (real_t): zenith angle of the link [rad].

        Returns:
            real_array_t: array with the corresponding link distances for the given altitudes [m].
        """
        
        # ensure we are below the satellite altitude
        if np.max(altitude_array) > sat_altitude:
            raise ValueError(f"Altitude {np.max(altitude_array):.2e} is greater than the satellite altitude ({sat_altitude:.2e})")
        
        if np.min(altitude_array) < lct_altitude:
            raise ValueError(f"Altitude {np.min(altitude_array):.2e} is lower than LCT altitude ({lct_altitude:.2e})")

        link_array = (altitude_array - lct_altitude) / np.cos(zenith_angle)  # ty:ignore[unsupported-operator]

        return link_array

    @staticmethod
    def link_to_altitude[T: real_t | real_array_t](
                         link_array: T,
                         sat_altitude: real_t, 
                         lct_altitude: real_t, 
                         zenith_angle: real_t,
                         **_) -> T:
        """From an array with link distances calculate the corresponding array with altitudes \
        (altitude from sea level for each corresponding link distance). 

        Args:
            link_array (real_array_t): array with link distances from the LCT to the satellite [m].
            sat_altitude (real_t): satellite altitude [m].
            lct_altitude (real_t): laser communication terminal/optical ground station altitude [m].
            zenith_angle (real_t): zenith angle of the link [rad].

        Returns:
            real_array_t: array with the corresponding altitudes for the given link distances [m].
        """

        altitude_array = link_array * np.cos(zenith_angle) + lct_altitude

        if (np.max(altitude_array) > sat_altitude):
            raise ValueError(f"Altitude {np.max(altitude_array):.2e} is greater than the satellite altitude ({sat_altitude:.2e})")
        
        if np.min(altitude_array) < lct_altitude:
            raise ValueError(f"Altitude {np.min(altitude_array):.2e} is lower than LCT altitude ({lct_altitude:.2e})")

        return altitude_array
    

    @staticmethod
    def calculate_total_link_distance(sat_altitude: real_t,
                                      lct_altitude: real_t,
                                      zenith_angle: real_t,
                                      **_) -> real_t:
        """Calculate the total link distance between LCT/OGS and satellite.

        Args:
            sat_altitude (real_t): altitude to the satellite from sea level [m].
            lct_altitude (real_t): altitude of the laser communication terminal/optical ground station above sea level [m].
            zenith_angle (real_t): zenith angle of the link [rad].

        Returns:
            real_t: distance from the LCT/OGS to the satellite [m].
        """

        if sat_altitude < lct_altitude:
            raise ValueError("Satellite altitude is less than LCT altitude.")
        
        link_distance = (sat_altitude - lct_altitude) / np.cos(zenith_angle)

        return link_distance

class RoundEarthModel(EarthModelGeneric):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def altitude_to_link[T: real_t | real_array_t](
                         altitude_array: T,
                         sat_altitude: real_t, 
                         lct_altitude: real_t, 
                         zenith_angle: real_t,
                         **_) -> T:
        """From an array with altitudes from sea level calculate the corresponding array with \
            link distances (distance from the OGS to that point for each altitude). 

        Args:
            altitude_array (real_array_t): array with altitudes from lct_altitude to sat_altitude [m].
            sat_altitude (real_t): satellite altitude [m].
            lct_altitude (real_t): laser communication terminal/optical ground station altitude [m].
            zenith_angle (real_t): zenith angle of the link [rad].

        Returns:
            real_array_t: array with the corresponding link distances for the given altitudes [m].
        """
        
        if np.max(altitude_array) > sat_altitude:
            raise ValueError(f"Altitude {np.max(altitude_array):.2e} is greater than the satellite altitude ({sat_altitude:.2e})")
        
        if np.min(altitude_array) < lct_altitude:
            raise ValueError(f"Altitude {np.min(altitude_array):.2e} is lower than LCT altitude ({lct_altitude:.2e})")

        left_term = - np.cos(zenith_angle) * ( EARTH_RADIUS + lct_altitude )
        in_sqrt1 = np.pow( np.cos(zenith_angle) * ( EARTH_RADIUS + lct_altitude ) , 2 )
        in_sqrt2 = (altitude_array - lct_altitude) * ( 2 * EARTH_RADIUS + altitude_array + lct_altitude)  # ty:ignore[unsupported-operator]

        link_array = left_term + np.sqrt( in_sqrt1 + in_sqrt2 )

        return link_array
    
    @staticmethod
    def link_to_altitude[T: real_t | real_array_t](
                         link_array: T,
                         sat_altitude: real_t, 
                         lct_altitude: real_t, 
                         zenith_angle: real_t,
                         **_) -> T:
        """From an array with link distances calculate the corresponding array with altitudes \
        (altitude from sea level for each corresponding link distance). 

        Args:
            link_array (real_array_t): array with link distances from the LCT to the satellite [m].
            sat_altitude (real_t): satellite altitude [m].
            lct_altitude (real_t): laser communication terminal/optical ground station altitude [m].
            zenith_angle (real_t): zenith angle of the link [rad].

        Returns:
            real_array_t: array with the corresponding altitudes for the given link distances [m].
        """

        in_sqrt1 = np.pow( lct_altitude + EARTH_RADIUS, 2 )
        in_sqrt2 = 2 * link_array * (lct_altitude + EARTH_RADIUS) * np.cos(zenith_angle)  # ty:ignore[unsupported-operator]
        in_sqrt3 = np.pow( link_array, 2 )
        sqrt_term = np.sqrt( in_sqrt1 + in_sqrt2 + in_sqrt3 )

        altitude_array = sqrt_term - EARTH_RADIUS

        if (np.max(altitude_array) > sat_altitude):
            raise ValueError(f"Altitude {np.max(altitude_array):.2e} is greater than the satellite altitude ({sat_altitude:.2e})")
        
        if np.min(altitude_array) < lct_altitude:
            raise ValueError(f"Altitude {np.min(altitude_array):.2e} is lower than LCT altitude ({lct_altitude:.2e})")
        
        return altitude_array
    

    @staticmethod
    def calculate_total_link_distance(sat_altitude: real_t,
                                      lct_altitude: real_t,
                                      zenith_angle: real_t,
                                      **_) -> real_t:
        """Calculate the total link distance between LCT/OGS and satellite.

        Args:
            sat_altitude (real_t): altitude to the satellite from sea level [m].
            lct_altitude (real_t): altitude of the laser communication terminal/optical ground station above sea level [m].
            zenith_angle (real_t): zenith angle of the link [rad].

        Returns:
            real_t: distance from the LCT/OGS to the satellite [m].
        """

        if sat_altitude < lct_altitude:
            raise ValueError("Satellite altitude is less than LCT altitude.")
        
        left_term = - np.cos(zenith_angle) * ( EARTH_RADIUS + lct_altitude )
        in_sqrt1 = np.pow( np.cos(zenith_angle) * ( EARTH_RADIUS + lct_altitude ) , 2 )
        in_sqrt2 = (sat_altitude - lct_altitude) * ( 2 * EARTH_RADIUS + sat_altitude + lct_altitude)

        link_distance = left_term + np.sqrt( in_sqrt1 + in_sqrt2 )

        return link_distance