from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .typing import real_t, real_array_t

# - - - - - - - - - - - - -
 
class EarthModel(ABC):

    @staticmethod
    @abstractmethod
    def altitude_to_link(altitude_array: real_array_t,
                         sat_altitude: real_t, 
                         lct_altitude: real_t, 
                         zenith_angle: real_t
                         ) -> real_array_t:
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
    def link_to_altitude(link_array: real_array_t,
                         sat_altitude: real_t, 
                         lct_altitude: real_t, 
                         zenith_angle: real_t
                         ) -> real_array_t:
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
                                      zenith_angle: real_t) -> real_t:
        """Calculate the total link distance between LCT/OGS and satellite.

        Args:
            sat_altitude (real_t): altitude to the satellite from sea level [m].
            lct_altitude (real_t): altitude of the laser communication terminal/optical ground station above sea level [m].
            zenith_angle (real_t): zenith angle of the link [rad].

        Returns:
            real_t: distance from the LCT/OGS to the satellite [m].
        """
        
        pass



class FlatEarthModel(EarthModel):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def altitude_to_link(altitude_array: real_array_t,
                         sat_altitude: real_t,
                         lct_altitude: real_t,
                         zenith_angle: real_t) -> real_array_t:
        
        
        return np.array(2)



class RoundEarthModel(EarthModel):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def altitude_to_link(altitude_array: real_array_t,
                         sat_altitude: real_t,
                         lct_altitude: real_t,
                         zenith_angle: real_t) -> real_array_t:
        
        return np.array(2)