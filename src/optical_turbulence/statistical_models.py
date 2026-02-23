""" Statistical PDFs models for downlink (DL) and uplink (UL).
"""

# ----------------- IMPORTS ----------------- #

import numpy as np
import scipy as sp
from scipy.stats.sampling import NumericalInversePolynomial
import mpmath
from ._definitions import *
from collections.abc import Callable
from abc import ABC, abstractmethod

# ----------------- TEMPLATES AND CONSTANTS ----------------- #

# Template for distribution class
class Distribution(ABC):

    @abstractmethod
    def pdf(self, x: real_t) -> real_t:
        pass

    @abstractmethod
    def cdf(self, thrsh: real_t) -> real_t:
        pass

    @abstractmethod
    def support(self) -> tuple[real_t, real_t]:
        """Return the support of the function.

        Returns:
            tuple[real_t, real_t]: minimum, maximum value
        """
        pass


SAMPLER_SIGNATURE = Callable[[int| tuple[int, ...]], float | np.typing.NDArray[np.float64]]

# ----------------- FUNCTION DEFINITIONS ----------------- #

#region DOWNLINK


def lognormal_DL_AA(scintillation_index: real_t, gen: None | np.random.Generator = None) -> SAMPLER_SIGNATURE:
    """Create a lognormal generator given the scintillation index of the system. This \
    distribution applies to **downlink** channels with an **aperture-averaged scintillation index** \
    that may come from a flat-earth or round-earth model.

    Args:
        scintillation_index (real_t): scintillation index to generate the distribution
        gen (np.random.Generator, optional): the generator object to which the lognormal will be sampled from. If not given \
            default_rng is used.

    Returns:
        SAMPLER_SIGNATURE: returns a lognormal generator configured \
            with the specified scintillation index. The generator takes as an optional argument the size of the sample to \
            generate.

    See also:
        Numpy random.Generator.lognormal

    Sources:
        [1] L. B. Stotts, “Some new insights into aperture averaging and probability density function selection,” Opt. Eng., vol. 64, no. 11, Nov. 2025, doi: 10.1117/1.OE.64.11.110801.
        [2] L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
        [3] D. R. Kolev and M. Toyoshima, “Received-Power Fluctuation Analysis for LEO Satellite-to-Ground Laser Links,” J. Lightwave Technol., vol. 35, no. 1, pp. 103-112, Jan. 2017, doi: 10.1109/JLT.2016.2627038.
        [4] R. Barrios Porras and V. F. Dios Otín, “Exponentiated weibull fading channel model in free-space optical communications under atmospheric turbulence,” Universitat Politècnica de Catalunya, 2013. doi: 10.5821/dissertation-2117-94937.
        [5] D. Giggenbach, S. Parthasarathy, A. Shrestha, F. Moll, and R. Mata-Calvo, “Power vector generation tool for free-space optical links — PVGeT,” in 2017 IEEE International Conference on Space Optical Systems and Applications (ICSOS), Naha: IEEE, Nov. 2017, pp. 160-165. doi: 10.1109/ICSOS.2017.8357228.

    """

    if gen is None:
        rng = np.random.default_rng()
    else:
        rng = gen
    
    ln_scintillation: np.float64 = np.log(1 + scintillation_index)
    _scale = np.exp(-0.5 * ln_scintillation) # mean
    _sparam = np.sqrt(ln_scintillation) # standard deviation

    def _lognormal_callable(size: int| tuple[int, ...] = 1) -> np.typing.NDArray[np.float64] | float:
        return sp.stats.lognorm.rvs(s=_sparam, scale=_scale, size=size, random_state=rng)

    return _lognormal_callable

def _exponweib_g_function(order: int, alpha: real_t, beta: real_t) -> np.float64:
    """Calculate the nth-order _g_ function from Barrios and Dios. The function is used in \
    the calculus of parameters related to the exponentiated Weibull PDF.

    Args:
        order (int): order of the function [int]
        alpha (real_t): first term of the function [real]
        beta (real_t): second term of the function [real]

    Raises:
        ArithmeticError: in case the function does not converge.

    Returns:
        np.float64: the value of the g function for the given order and parameters.

    Sources:
        [1] R. Barrios and F. Dios, “Exponentiated Weibull model for the irradiance probability density function of a laser beam propagating through atmospheric turbulence,” Optics & Laser Technology, vol. 45, pp. 13–20, Feb. 2013, doi: 10.1016/j.optlastec.2012.08.004.
        
    """

    # Inner part of the infinite sum
    def _internal_sum(terms_array, *args):
        
        numerator = np.pow(-1, terms_array) * sp.special.gamma(alpha)
        denominator = sp.special.factorial(terms_array) * np.pow(terms_array + 1, 1 + order / beta) * sp.special.gamma(alpha - terms_array)
        value = numerator / denominator
        return value

    # parameters for the nsum function
    lower_limit = 0
    upper_limit = 30 # according to the source paper, 10 terms are more than enough, but we have computer power, so lets calculate 30
    step = 1

    res = sp.integrate.nsum(f=_internal_sum, a=lower_limit, b=upper_limit, step=step)

    # the g function has to converge (it does according to the paper)
    if not res.success:
        raise ArithmeticError("g function from exponentiated weibull did not converge.")

    return res.sum


def exponweib_DL_AA(scintillation_index: real_t, gen: None | np.random.Generator = None) -> SAMPLER_SIGNATURE:
    """Create an Exponential Weibull generator given the scintillation index of the system. This \
    distribution applies to **downlink** channels with an **aperture-averaged scintillation index** \
    that may come from a flat-earth or round-earth model.

    Args:
        scintillation_index (real_t): scintillation index to generate the distribution
        gen (np.random.Generator, optional): the generator object to which the exponentiated weibull will be sampled from. If not given \
            default_rng is used.

    Returns:
        SAMPLER_SIGNATURE: returns an exponentiated weibull generator configured \
            with the specified scintillation index. The generator takes as an optional argument the size of the sample to \
            generate.

    See also:
        SciPy scipy.stats.exponweib

    Sources:
        [1] D. R. Kolev and M. Toyoshima, “Received-Power Fluctuation Analysis for LEO Satellite-to-Ground Laser Links,” J. Lightwave Technol., vol. 35, no. 1, pp. 103–112, Jan. 2017, doi: 10.1109/JLT.2016.2627038.
        [2] L. B. Stotts, “Some new insights into aperture averaging and probability density function selection,” Opt. Eng., vol. 64, no. 11, Nov. 2025, doi: 10.1117/1.OE.64.11.110801.
        [3] R. Barrios and F. Dios, “Exponentiated Weibull model for the irradiance probability density function of a laser beam propagating through atmospheric turbulence,” Optics & Laser Technology, vol. 45, pp. 13–20, Feb. 2013, doi: 10.1016/j.optlastec.2012.08.004.

    """

    if gen is None:
        rng = np.random.default_rng()
    else:
        rng = gen

    _alpha = 7.220 * np.pow(scintillation_index, 1/3) / sp.special.gamma( 2.487 * np.pow(scintillation_index, 1/6) - 0.104 )
    _beta = 1.012 * np.pow(_alpha * scintillation_index, -13/25) + 0.142

    # "eta"
    _scale = 1 / ( _alpha * sp.special.gamma(1 + 1/_beta) * _exponweib_g_function(order=1, alpha=_alpha, beta=_beta)) 

    def _exponweib_callable(size: int| tuple[int, ...] = 1) -> np.typing.NDArray[np.float64] | float:
        return sp.stats.exponweib.rvs(a=_alpha, c=_beta, scale=_scale, size=size, random_state=rng)

    return _exponweib_callable

#endregion DOWNLINK



#region UPLINK

class __GammaGammaDist(Distribution):
    """Normalized GammaGamma distribution
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.lower_bound = 1e-12
        self.upper_bound = np.inf

    def pdf(self, x) -> real_t:
        """Obtain the value of the PDF of the GammaGamma distribution for a given value.

        Args:
            thrsh (real_t): value to calculate the PDF for, has to be greater than zero.

        Returns:
            np.float64: the PDF value for that value
        """

        # This is not mathematically proven, just assume if x -> inf, then x ^{c} * K_v() -> 0
        if x == np.inf:
            return 0

        # Fraction
        left_frac_num = 2 * np.pow( self.alpha * self.beta, (self.alpha + self.beta) / 2 )
        left_frac_den = sp.special.rgamma(self.alpha) * sp.special.rgamma(self.beta) # reciprocal of gamma function
        first_term = left_frac_num * left_frac_den # we used the reciprocal in the denominator, so now we multiply

        # Irradiance to the power of ...
        second_term = np.pow(x, (self.alpha + self.beta) / 2 - 1)

        # Modified bessel function of second kind
        third_term =  sp.special.kv(self.alpha - self.beta, 2 * np.sqrt(self.alpha * self.beta * x))

        return first_term * second_term * third_term

    @warn_custom(f"[GammaGammaDist] this function may not be stable after a given value.")
    def cdf(self, thrsh) -> real_t:
        """Obtain the value of the CDF of the GammaGamma distribution for a given threshold.

        Args:
            thrsh (real_t): threshold to obtain the CDF of, has to be greater than zero.

        Returns:
            np.float64: the CDF value for that threshold
        """

        # From tests, this CDF destabilizes depending on the values

        # Taken from [1] L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
        # page 371

        # we use rgamma because it is made as the reciprocate
        first_term = np.pi / np.sin( np.pi * (self.alpha - self.beta) ) * sp.special.rgamma(self.alpha) * sp.special.rgamma(self.beta)        
        
        # Divide in frac and hyperfunc to ease implementation
        second_term_frac = np.pow( self.alpha * self.beta * thrsh, self.beta ) / ( self.beta * sp.special.gamma(self.beta - self.alpha + 1) )
        second_term_hyper = mpmath.hyp1f2(self.beta, self.beta + 1, self.beta - self.alpha + 1, self.alpha * self.beta * thrsh )
        second_term = second_term_frac * second_term_hyper

        third_term_frac = np.pow( self.alpha * self.beta * thrsh, self.alpha ) / ( self.alpha * sp.special.gamma(self.alpha - self.beta + 1) )
        third_term_hyper = mpmath.hyp1f2(self.alpha, self.alpha + 1, self.alpha - self.beta + 1, self.alpha * self.beta * thrsh )
        third_term = third_term_frac * np.float128(third_term_hyper) # mpmath works with mpfloats

        return first_term * ( second_term - third_term )

    def support(self):
        return (self.lower_bound, self.upper_bound)


def gamma_gamma_UL_PR(rytov_variance: real_t, gen: None | np.random.Generator = None, Theta: real_t = 0) -> SAMPLER_SIGNATURE:
    """Implements the Gamma-Gamma probability distribution which is valid for a point receiver in the uplink direction. The \
    sample generator is implemented using the CDF Inversion method (more specifically, NumericalInversePolynomial from scipy).

    Args:
        rytov_variance (real_t): the rytov variance (weak-turbulence scintillation index) of an uplink beam
        gen (None | np.random.Generator, optional): the generator object to which the gammagamma will be sampled from. If not given \
            default_rng is used.
        Theta (real_t, optional):Theta (real_t): refractive parameter of the beam at the receiver plane [unitless]. Defaults to 0 (spherical wave).

    Returns:
        SAMPLER_SIGNATURE: returns a gammagamma sample generator configured \
            with the specified scintillation index. The generator takes as an optional argument the size of the sample to \
            generate.
    """

    # we can set our own RNG generator
    if gen is None:
        rng = np.random.default_rng()
    else:
        rng = gen

    # Calculate for the distribution
    _pre_alpha = 0.49 * rytov_variance / np.pow( 1 + (1 + Theta) * 0.56 * np.pow(rytov_variance, 6/5), 7/6 )
    _alpha = 1 / ( np.exp(_pre_alpha) - 1)
    
    _pre_beta = 0.51 * rytov_variance / np.pow( 1 + 0.69 * np.pow(rytov_variance, 6/5), 5/6 )
    _beta = 1 / ( np.exp(_pre_beta) - 1 )

    # Create the distribution and obtain the inverse
    dist = __GammaGammaDist(alpha=_alpha, beta=_beta)
    sampler = NumericalInversePolynomial(dist, random_state=rng, center=1) # type: ignore

    # this way the call is just the amount of samples required
    def _gg_callable(size: int| tuple[int, ...] = 1) -> np.typing.NDArray[np.float64] | float:
        return sampler.rvs(size)
    
    return _gg_callable


def gamma_UL_PR_tracked(scintillation_index: real_t, gen: None | np.random.Generator = None) -> SAMPLER_SIGNATURE:
    """Implements the Gamma PDF which is valid for a point receiver in the uplink direction for a tracked beam.

    Args:
        scintillation_index (real_t): scintillation index of the tracked beam.
        gen (None | np.random.Generator, optional): the generator object to which the gamma will be sampled from. If not given \
            default_rng is used.

    Returns:
        Callable[[], np.typing.NDArray[np.float64] | float]: returns a gamma generator configured \
            with the specified scintillation index. The generator takes as an optional argument the size of the sample to \
            generate.
    """

    # we can set our own RNG generator
    if gen is None:
        rng = np.random.default_rng()
    else:
        rng = gen

    _scale = scintillation_index
    _shape = 1 / scintillation_index

    def _gamma_callable(size: int| tuple[int, ...] = 1) -> np.typing.NDArray[np.float64] | float:
        return sp.stats.gamma.rvs(a=_shape, scale=_scale, size=size, random_state=rng)
    
    return _gamma_callable


class __ModulatedGammaDist(Distribution):
    """Normalized Modulated Gamma distribution
    """

    def __init__(self, wander_param, shape_param) -> None:
        
        self.shape = shape_param # also known as m1 or m
        self.wparam = wander_param
        self.lower_bound = 1e-9
        self.upper_bound = np.inf

    def pdf(self, x):

        if x == np.inf:
            return 0

        # divide into three terms to multiply
        first_term = self.wparam * sp.special.rgamma(self.shape) / x

        interim_division = self.shape * x / ( 1 + 1 / self.wparam )
        second_term = np.pow(interim_division, self.wparam)

        # The function to use is the complementary incomplete because of an identity that relates them
        # we need to multiply by gamma because the incomplete version is regularized in scipy
        third_term = sp.special.gammaincc( self.shape - self.wparam, interim_division ) * sp.special.gamma(self.shape - self.wparam)

        return first_term * second_term * third_term
    
    def pdf_full(self, x):
        
        if x == np.inf:
            return 0
        
        # separate into terms
        first_term = self.wparam * sp.special.rgamma(self.shape) / x

        interim_division = self.shape * x / ( 1 + 1 / self.wparam )
        shape_minus_wp = self.shape - self.wparam

        first_pterm = sp.special.gamma(shape_minus_wp) * np.pow( interim_division, self.wparam )
        second_pterm_1 = np.pow( interim_division, self.shape ) / ( shape_minus_wp )
        second_pterm_2 = sp.special.hyp1f1(shape_minus_wp, 1 + shape_minus_wp, -1 * interim_division)

        return first_term * ( first_pterm - second_pterm_1 * second_pterm_2 )

    # TODO: maybe complementary incomplete Gamma??
    def cdf(self, thrsh):
        
        # From andrews2023 p. 193
        interim_division = self.shape * thrsh / ( 1 + 1 / self.wparam )

        # The incomplete function is normalized in the CDF expression
        first_term = sp.special.gammainc(self.shape, interim_division)

        second_term_1 = sp.special.rgamma(self.shape) * np.pow(interim_division, self.wparam)
        # The incomplete function is normalized so we "un-normalize" it
        second_term_2 = sp.special.gammainc(self.shape - self.wparam, interim_division) * sp.special.gamma(self.shape - self.wparam) 
        second_term = second_term_1 * second_term_2

        return 1 - first_term + second_term

    def support(self):
        return (self.lower_bound, self.upper_bound)


@warn_custom("There are issues with the modulated Gamma PDF model (cannot replicate literature)")
@warn_not_tested
def modulated_gamma_UL_PR_untracked(scintillation_index: real_t,  
                                    link_length: real_t, 
                                    fried_parameter_T: real_t, 
                                    wavenum: real_t, 
                                    pointing_error: real_t, 
                                    rx_spot_size: real_t,
                                    Lambda: real_t, 
                                    gen: None | np.random.Generator = None,
                                    **_) -> SAMPLER_SIGNATURE:
    """Implements the Modulated Gamma PDF which is valid for a point receiver in the **uplink direction for an untracked beam**.

    Args:
        scintillation_index (real_t): untracked scintillation index [unitless].
        link_length (real_t): length of the link between the transmitter and receiver [m]. Greater than zero.
        fried_parameter_T (real_t): Fried parameter as seen near the OGS (uplink-transmitter) [m]
        wavenum (real_t): wavenumber $2pi / wavelength$ [1/m] 
        pointing_error (real_t): Pointing error variance [m^2]
        rx_spot_size (real_t): $W$ spot size (effective beam radius) of the beam at the receiver plane [m]
        Lambda (real_t): diffractive parameter of the beam at the receiver plane [unitless]
        gen (None | np.random.Generator, optional): the generator object to which the gamma will be sampled from. If not given \
            default_rng is used.

    Returns:
        SAMPLER_SIGNATURE: returns a modulated gamma generator configured \
            with the specified scintillation index. The generator takes as an optional argument the size of the sample to \
            generate.

    Sources:
        [1] L. C. Andrews, Field Guide to Atmospheric Optics, Second Edition. in Field Guide Ser, no. v. FG41. Bellingham: Society of Photo-Optical Instrumentation Engineers, 2019.
        [2] L. B. Stotts, “Some new insights into aperture averaging and probability density function selection,” Opt. Eng., vol. 64, no. 11, Nov. 2025, doi: 10.1117/1.OE.64.11.110801.
    """

    # lets check the necessary params are positive

    # we can set our own RNG generator
    if gen is None:
        rng = np.random.default_rng()
    else:
        rng = gen

    # we need to calculate the parameters for the distribution
    # untracked scintillation
    shape_parameter  = 1 / scintillation_index

    wparam_denominator = 34.29 * np.pow( Lambda * link_length / (wavenum * np.pow(fried_parameter_T, 2)), 5/6 ) * pointing_error / np.pow(rx_spot_size, 2)
    wparam_numerator = 1 + scintillation_index

    wander_parameter = np.sqrt(1 + wparam_numerator / wparam_denominator) - 1

    if (shape_parameter - wander_parameter) <= 0:
        raise ValueError(f"the values entered for the system are not valid! The following relation must happen " + r"$m - \vartheta > 0$ (currently not happening)")

    # Create the distribution and obtain the inverse to be able to sample
    dist = __ModulatedGammaDist(wander_param=wander_parameter, shape_param=shape_parameter)
    sampler = NumericalInversePolynomial(dist, random_state=rng, center=1) # type: ignore

    # this way the call is just the amount of samples required
    def _mg_callable(size: int| tuple[int, ...] = 1) -> np.typing.NDArray[np.float64] | float:
        return sampler.rvs(size)
    
    return _mg_callable


#endregion UPLINK