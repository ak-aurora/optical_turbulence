"""
Classes to facilitate the use of the library
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, TypedDict

import numpy as np
import numpy.typing as npt

from .earth_models import EarthModelGeneric, RoundEarthModel
from .typing import real_array_t, real_t
from .utils import create_altitude_array


class OpticalBeamParameters(TypedDict):
    beam_radius: real_t
    pfront_radius: real_t
    wavelength: real_t
    link_distance: real_t
    amplitude: real_t
    wavenumber: real_t
    refractive_param_transmitter: real_t
    diffractive_param_transmitter: real_t
    refractive_param_receiver: real_t
    diffractive_param_receiver: real_t
    Lambda: real_t
    Theta: real_t
    rx_spot_size: real_t
    rx_pfront_radius: real_t
    neg_Theta: real_t

@dataclass
class OpticalBeam:
    """Based on the Gaussian-Beam Wave Model detailed in [1]

    [1] L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. Bellingham, Washington, USA: SPIE Press, 2023.
    """

    # ----------------- Fixed attributes ----------------- #

    beam_radius: real_t
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

    _params_changed: bool = field(init=False, default=True)
    _dic: OpticalBeamParameters | None = field(init=False, default=None)

    # ----------------- At-run-time-calculated properties ----------------- #

    @property
    def wavenumber(self) -> real_t:
        """[1/m] : 2 * pi / wavelength"""

        return 2 * np.pi / self.wavelength

    @property
    def refractive_param_transmitter(self) -> real_t:
        """[unitless] : (Theta_0) Non-dimensional refractive beam parameter at the transmitter."""

        value = 1 - self.link_distance / self.pfront_radius
        return value

    @property
    def diffractive_param_transmitter(self) -> real_t:
        """[unitless] : (Lambda_0) Non-dimensional diffractive beam parameter at the transmitter."""

        value = 2 * self.link_distance / (self.wavenumber * np.pow(self.beam_radius, 2))
        return value

    @property
    def refractive_param_receiver(self) -> real_t:
        """[unitless] : (Theta) Non-dimensional refractive beam parameter at the receiver."""

        denominator = np.pow(self.refractive_param_transmitter, 2) + np.pow(
            self.diffractive_param_transmitter, 2
        )

        value = self.refractive_param_transmitter / denominator
        return value

    @property
    def diffractive_param_receiver(self) -> real_t:
        """[unitless] : (Lambda) Non-dimensional diffractive beam parameter at the receiver."""

        denominator = np.pow(self.refractive_param_transmitter, 2) + np.pow(
            self.diffractive_param_transmitter, 2
        )

        value = self.diffractive_param_transmitter / denominator
        return value

    @property
    def Lambda(self) -> real_t:
        """[unitless] : (Lambda) Non-dimensional diffractive beam parameter at the receiver."""

        return self.diffractive_param_receiver

    @property
    def Theta(self) -> real_t:
        """[unitless] : (Theta) Non-dimensional refractive beam parameter at the receiver."""

        return self.refractive_param_receiver

    @property
    def rx_spot_size(self) -> real_t:
        """[m] : $W$ spot size (effective beam radius) of the beam at the receiver plane"""

        in_sqrt = np.pow(self.refractive_param_transmitter, 2) + np.pow(
            self.diffractive_param_transmitter, 2
        )

        value = self.beam_radius * np.sqrt(in_sqrt)
        return value

    @property
    def rx_pfront_radius(self) -> real_t:
        """[m] : Phase front radius of curvature at the receiver."""

        value = self.link_distance / (self.Theta - 1)
        return value

    @property
    def neg_Theta(self) -> real_t:
        """[unitless] : (overbar Theta) **Negated** non-dimensional refractive beam parameter at the receiver."""
        return 1 - self.Theta

    # ----------------- Methods ----------------- #

    def __setattr__(self, name, value):
        # We want to cache the "dictionary of params"
        if not name.startswith("_"):
            super().__setattr__("_params_changed", True)
            value = np.float64(value)

        super().__setattr__(name, value)

    def as_dict(self) -> OpticalBeamParameters:
        """Return all properties and attributes of this class a dictionary
        to be used in **kwargs.

        Returns
            dict: the dictionary with keys and values.
        """

        if self.link_distance <= 0:
            warnings.warn(
                f"OpticalBeam instance has an invalid link distance.\
                Link distance: {self.link_distance}"
            )

        if self._params_changed or self._dic is None:
            self._dic = OpticalBeamParameters(**{  # ty:ignore[missing-typed-dict-key]
                prop: getattr(self, prop)
                for prop in dir(self)
                if not (prop.startswith("_") or callable(getattr(self, prop, None)))
            })
            self._params_changed = False

        return self._dic



class LinkDescriptionArrays(TypedDict):
    altitude_array: npt.NDArray[np.float64]
    link_array: npt.NDArray[np.float64]

class LinkDescriptionManager():
    """Manage the link, auto calculating relevant values in case of an update"""

    def __init__(
            self,
            satellite_altitude: real_t,
            lct_altitude: real_t,
            zenith_angle: real_t,
            earth_model: EarthModelGeneric | None = None,
            altitude_array_factory: Callable[[real_t, real_t], npt.NDArray[np.float64]] | None = None
    ):
        """Constructor

        Args:
            satellite_altitude (real_t): satellite altitude [m]
            lct_altitude (real_t): LCT/OGS altitude [m]
            zenith_angle (real_t): zenith angle of the link [rad]
            earth_model (EarthModelGeneric, optional): earth model [n/a]. Defaults to RoundEarthModel.
            altitude_array_factory (Callable[[sat_altitude: real_t, \
                lct_altitude: real_t], altitude_array: npt.NDArray[np.float64]], optional): \
                callable to slice the atmosphere all values have to be in meters. Defaults to slicing every 1 m.
        """

        self.sat_altitude = satellite_altitude
        self.lct_altitude = lct_altitude
        self.zenith_angle = zenith_angle

        if earth_model is None:
            self.earth_model = RoundEarthModel()
        else:
            self.earth_model = earth_model

        if altitude_array_factory is None:
            self.altitude_array_factory = create_altitude_array
        else:
            self.altitude_array_factory = altitude_array_factory
        
        self._update_altitude_array()
        self._update_link_array()

        self._update_arrays = False

    def __setattr__(self, name, value):
        # We want to update the arrays if the link description changes
        super().__setattr__(name, value)

        if "link_array" not in name \
            and "altitude_array" not in name\
            and "update_arrays" not in name:
            self._update_arrays = True

    @property
    def altitude_array(self) -> npt.NDArray[np.float64]:

        if self._update_arrays:
            self._update_altitude_array()
            self._update_link_array()
            self._update_arrays = False

        return self._altitude_array

    @altitude_array.setter
    def altitude_array(self, _) -> None:
        raise AttributeError("Do not set the value of the altitude array manually, update an attribute.")

    @property
    def link_array(self) -> npt.NDArray[np.float64]:

        if self._update_arrays:
            self._update_altitude_array()
            self._update_link_array()
            self._update_arrays = False

        return self._link_array
    
    @link_array.setter
    def link_array(self, _) -> None:
        raise AttributeError("Do not set the value of the link array manually, update an attribute.")

    def _update_altitude_array(self):
        self._altitude_array = self.altitude_array_factory(
            self.sat_altitude,
            self.lct_altitude
        )

    def _update_link_array(self):
        self._link_array = self.earth_model.altitude_to_link(
            altitude_array=self._altitude_array,
            sat_altitude=self.sat_altitude,
            lct_altitude=self.lct_altitude,
            zenith_angle=self.zenith_angle
        )

    def arrays_as_dict(self) -> LinkDescriptionArrays:
        """Return the altitude and link arrays as a dictionary to be used in
        kwargs.

        Returns:
            LinkDescriptionArrays: keys "altitude_array" and "link_array"
        """

        if self._update_arrays:
            self._update_altitude_array()
            self._update_link_array()
            self._update_arrays = False

        ret: LinkDescriptionArrays = {
            "altitude_array": self._altitude_array,
            "link_array": self._link_array
        }

        return ret
    
    def link_distance(self) -> real_t:
        """Calculate the total link distance between the OGS and satellite.

        Returns:
            real_t: the link distance
        """

        distance = self.earth_model.calculate_total_link_distance(
            sat_altitude=self.sat_altitude,
            lct_altitude=self.lct_altitude,
            zenith_angle=self.zenith_angle
        )

        return distance

    def link2altitude[T: real_t | real_array_t](self, link_distance: T) -> T:
        """Given a link distance, calculate the corresponding altitude for the current \
            link configuration.

        Args:
            link_value (real_t or real_array_t): link distance to convert to altitude [m].

        Returns:
            real_t or real_array_t: corresponding altitude [m].
        """

        _altitude = self.earth_model.link_to_altitude(
            link_array=link_distance,
            sat_altitude=self.sat_altitude,
            lct_altitude=self.lct_altitude,
            zenith_angle=self.zenith_angle
        )

        return _altitude
    
    def altitude2link[T: real_t | real_array_t](self, altitude: T) -> T:
        """Given an altitude, calculate the corresponding link distance for the current \
            link configuration.

        Args:
            altitude (real_t or real_array_t): altitude to convert to link distance [m].

        Returns:
            real_t or real_array_t: corresponding link distance [m].
        """

        _link_distance = self.earth_model.altitude_to_link(
            altitude_array=altitude,
            sat_altitude=self.sat_altitude,
            lct_altitude=self.lct_altitude,
            zenith_angle=self.zenith_angle
        )

        return _link_distance

if __name__ == "__main__":
    a = OpticalBeam.rx_spot_size
