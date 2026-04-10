"""
Classes to facilitate the use of the library
"""

from dataclasses import dataclass
import warnings
from .typing import real_t
from .earth_models import EarthModel

from typing import Dict

import numpy as np


@dataclass
class OpticalBeam:
    """Based on the Gaussian-Beam Wave Model detailed in [1]
    
    [1] L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. Bellingham, Washington, USA: SPIE Press, 2023.
    """

    # ----------------- Fixed attributes ----------------- #

    beam_radius : real_t
    """[m] : Effective beam radius (spot size) at which the field amplitude falls to 1/e"""

    pfront_radius: real_t
    """[m] : Phase front radius of curvature. Infinite for a collimated beam."""

    wavelength: real_t
    """[m] : Wavelength of the optical beam."""

    link_distance: real_t = 0
    """[m] : Link distance"""

    amplitude: real_t = 1
    """[a.u.] : Amplitude of the beam at the output aperture of the transmitter."""

    # ----------------- Dynamic attributes ----------------- #

    _params_changed: bool = True
    _dic : dict | None = None

    # ----------------- At-run-time-calculated properties ----------------- #

    @property
    def wavenumber(self):
        """[1/m] : 2 * pi / wavelength"""

        return 2 * np.pi / self.wavelength


    @property
    def refractive_param_transmitter(self):
        """[unitless] : (Theta_0) Non-dimensional refractive beam parameter at the transmitter."""

        value = 1 - self.link_distance / self.pfront_radius
        return value


    @property
    def diffractive_param_transmitter(self):
        """[unitless] : (Lambda_0) Non-dimensional diffractive beam parameter at the transmitter."""

        value = 2 * self.link_distance / ( self.wavenumber * np.pow(self.beam_radius, 2) )
        return value


    @property
    def refractive_param_receiver(self):
        """[unitless] : (Theta) Non-dimensional refractive beam parameter at the receiver."""
        
        denominator = np.pow(self.refractive_param_transmitter, 2) + np.pow(self.diffractive_param_transmitter, 2)

        value = self.refractive_param_transmitter / denominator
        return value


    @property
    def diffractive_param_receiver(self):
        """[unitless] : (Lambda) Non-dimensional diffractive beam parameter at the receiver."""

        denominator = np.pow(self.refractive_param_transmitter, 2) + np.pow(self.diffractive_param_transmitter, 2)

        value = self.diffractive_param_transmitter / denominator
        return value
    
    @property
    def Lambda(self):
        """[unitless] : (Lambda) Non-dimensional diffractive beam parameter at the receiver."""

        return self.diffractive_param_receiver


    @property
    def Theta(self):
        """[unitless] : (Theta) Non-dimensional refractive beam parameter at the receiver."""

        return self.refractive_param_receiver


    @property
    def rx_spot_size(self):
        """[m] : $W$ spot size (effective beam radius) of the beam at the receiver plane"""

        in_sqrt = np.pow(self.refractive_param_transmitter, 2) + np.pow(self.diffractive_param_transmitter, 2)

        value = self.beam_radius * np.sqrt(in_sqrt)
        return value

    @property
    def rx_pfront_radius(self):
        """[m] : Phase front radius of curvature at the receiver."""

        value = self.link_distance / (self.Theta - 1)
        return value
    
    @property
    def neg_Theta(self):
        """[unitless] : (overbar Theta) **Negated** non-dimensional refractive beam parameter at the receiver."""
        return 1 - self.Theta


    # ----------------- Methods ----------------- #

    def __setattr__(self, name, value):
        # We want to cache the "dictionary of params"
        if not name.startswith("_"):
            super().__setattr__("_params_changed", True)
            value = np.float64(value)

        super().__setattr__(name, value)

    
    def as_dict(self) -> Dict:
        """ Return all properties and attributes of this class a dictionary
        to be used in **kwargs.

        Returns
            dict: the dictionary with keys and values.
        """

        if self.link_distance <= 0:
            warnings.warn(f"OpticalBeam instance has an invalid link distance.\
                Link distance: {self.link_distance}")

        if self._params_changed or self._dic is None:
            self._dic = {prop: getattr(self, prop) for prop in dir(self) if not (prop.startswith('_') or callable(getattr(self, prop, None)))}
            self._params_changed = False

        return self._dic


@dataclass
class LinkDescriptor:
    """Descriptor for a sat-Earth link."""
    # TODO: Altitudes w.r.t ground/sea level

    satellite_altitude: real_t
    """[m] : Altitude of the satellite."""

    lct_altitude: real_t
    """[m] : Altitude of the Laser Communication Terminal/Optical Ground Station"""

    zenith_angle: real_t
    """[rad] : Zenith angle of the link."""

    earth_model: EarthModel
    """[n/a] : Earth model to use"""


if __name__ == "__main__":
    a = OpticalBeam.rx_spot_size