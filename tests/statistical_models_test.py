"""
To compare the statistical models with results from the literature (books or papers).

Types of tests:
    1. Validate PDF/CDF behaviour (for the PDFS implemented by me)
    2. Compare with literature

"""

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
from colorama import Fore, just_fix_windows_console

from src.optical_turbulence.parameters import _bw_pointing_error_var_UL_gaussian
from src.optical_turbulence.ris_models import hufnagel_valley_model
from src.optical_turbulence.statistical_models import (
    Distribution,
    __GammaGammaDist,
    __ModulatedGammaDist,
    exponweib_DL_AA,
    gamma_gamma_UL_PR,
    gamma_UL_PR_tracked,
    lognormal_DL_AA,
    modulated_gamma_UL_PR_untracked,
)
from src.optical_turbulence.utils import create_altitude_array

from ._test_utilities import test_discussion

just_fix_windows_console()

TEST_STR = Fore.MAGENTA + "[TEST INFO] " + Fore.RESET  # ty:ignore[unsupported-operator]

# --------- PDF DEFINITIONS TEST ---------

def _mean_auc_helper(distribution: Distribution) -> None:

    min_val, max_val = distribution.support()

    mean_val, error = sp.integrate.quad(lambda x: x * distribution.pdf(x), min_val, max_val)
    print(TEST_STR + f"Expected value: {mean_val:.2f}. Error {error:.2e}.")

    aoc, err = sp.integrate.quad(distribution.pdf, min_val, max_val)
    print(TEST_STR + f"AUC of PDF: {aoc:.2f}. Error {err:.2e}.")


@test_discussion(__name__, "The GammaGamma PDF seems to fulfill the requirements (mean = 1; AUC = 1)")
def GammaGamma():
    """Lets check that:
        - the under area the curve and the mean value of the PDF is 1
        - the sampler follows the PDF
    """

    # Analyze the PDF
    Theta = 0

    def get_pdf_vals(rytov_variance):

        # Calculate for the distribution
        _pre_alpha = 0.49 * rytov_variance / np.pow( 1 + (1 + Theta) * 0.56 * np.pow(rytov_variance, 6/5), 7/6 )
        _alpha = 1 / ( np.exp(_pre_alpha) - 1)
        
        _pre_beta = 0.51 * rytov_variance / np.pow( 1 + 0.69 * np.pow(rytov_variance, 6/5), 5/6 )
        _beta = 1 / ( np.exp(_pre_beta) - 1 )

        scint = 1/_alpha + 1/_beta + 1 / (_alpha * _beta)
        # print(f"scint: {scint}")
        # print(f"alph: {_alpha:.2f}, beta: {_beta:.2f}")

        dist = __GammaGammaDist(alpha=_alpha, beta=_beta)
        vals = np.linspace(1e-8, 10, 10000)
        pdfs = np.array([dist.pdf(x) for x in vals])
        _mean_auc_helper(dist)

        return pdfs, vals, scint

    parameters = (0.1, 0.4, 0.8, 1.4)
    fig, axs = plt.subplots(2,2)

    for i, rvar in enumerate(parameters):
        ax = axs[i // 2][i % 2]

        sampler = gamma_gamma_UL_PR(rytov_variance=rvar, Theta=Theta)
        samples = sampler(100000)
        _y, _x, scint = get_pdf_vals(rvar)

        ax.set_title(f"Scintillation index: {scint:.2f}")
        sns.histplot(x=samples, stat="density", label="sampler values", ax=ax, color="green")
        sns.lineplot(y=_y, x=_x, label="pdf", ax=ax, color="orange")
        ax.set_xlim([0, 4])
        ax.set_ylim([0, 2])
        
    fig.suptitle("GammaGamma: Comparison of PDF and histogram of samples.")
    fig.tight_layout()
    plt.show()


@test_discussion(__name__, "The Modulated Gamma PDF seems to fulfill the requirements (mean = 1; AUC = 1)")
def ModulatedGamma():
    """Lets check that:
        - the under area the curve and the mean value of the PDF is 1
        - the sampler follows the PDF
    """

    # Analyze the PDF
    # just random values
    def get_dict(radius):
        link_length = 500e3
        wavenum = 2 * np.pi / (1550e-9)
        fried_parameter_T = 0.19

        # radius = 0.707
        theta_zero = 1
        lambda_zero = 2 * link_length / (wavenum * np.pow(radius, 2))
        _denominator = ( np.pow(theta_zero, 2) + np.pow(lambda_zero, 2) )
        Lambda = lambda_zero / _denominator
        rx_spot_size = radius * np.sqrt(_denominator)

        altitude_array = create_altitude_array(
            sat_altitude=link_length,
            lct_altitude=0,
        )

        pointing_error = _bw_pointing_error_var_UL_gaussian(
            wavelength=1.55e-6,
            beam_radius=radius,
            pfront_radius=np.inf,
            ris_model=hufnagel_valley_model,
            altitude_array=altitude_array,
            link_array=altitude_array# They are the same since zenith = 0
        )

        param_dict = {
            "Lambda": Lambda,
            "link_length": link_length,
            "wavenum": wavenum,
            "fried_parameter_T": fried_parameter_T,
            "pointing_error": pointing_error,
            "rx_spot_size": rx_spot_size
        }

        return param_dict

    def get_pdf_vals(scintillation_index, Lambda, link_length, wavenum, fried_parameter_T, pointing_error, rx_spot_size):

        # Calculate for the distribution
        shape_parameter  = 1 / scintillation_index

        wparam_denominator = 34.29 * np.pow( Lambda * link_length / (wavenum * np.pow(fried_parameter_T, 2)), 5/6 ) * pointing_error / np.pow(rx_spot_size, 2)
        wparam_numerator = 1 + scintillation_index

        wander_parameter = np.sqrt(1 + wparam_numerator / wparam_denominator) - 1

        # print(f"wp: {wander_parameter:.2e} | shape: {shape_parameter:.2e}")

        dist = __ModulatedGammaDist(wander_param=wander_parameter, shape_param=shape_parameter)
        vals = np.linspace(1e-8, 10, 10000)
        pdfs = np.array([dist.pdf(x) for x in vals])
        _mean_auc_helper(dist)

        return pdfs, vals, scintillation_index

    parameters = (
        (0.084, 3.54 / 100, 0.75),
        (0.268, 7.07 / 100, 0.486)
    )
    fig, axs = plt.subplots(1,2)

    for i, params in enumerate(parameters):
        ax = axs[i % 2]

        scintillation, radius, strehlr = params

        param_dict = get_dict(radius=radius)
        sampler = modulated_gamma_UL_PR_untracked(scintillation_index=scintillation, **param_dict)
        samples = sampler(100000)
        _y, _x, scint = get_pdf_vals(scintillation_index=scintillation, **param_dict)

        ax.set_title(f"Scintillation index: {scint:.2f}")
        sns.histplot(x=samples, stat="density", label="sampler values", ax=ax, color="green")
        sns.lineplot(y=_y, x=_x, label="pdf", ax=ax, color="orange")
        ax.set_xlim([0, 4.6])
        ax.set_ylim([1e-3, 8])
        ax.set_yscale("log")
        
    fig.suptitle("ModulatedGamma: Comparison of PDF and histogram of samples.")
    fig.tight_layout()
    plt.show()


# --------- LITERATURE TESTS ---------

@test_discussion(__name__, "It seems to follow the reference figures correctly.")
def GammaGammaLiterature():
    """To compare with [1, Fig. 7]

    Sources:
        [1] A. Carrillo-Flores, D. Giggenbach, and M. T. Knopp, “Statistics of Received Power Time Series for Optical LEO Satellite Uplinks,” Satell Commun Network, vol. 43, no. 3, pp. 251-263, Jun. 2025, doi: 10.1002/sat.1554.

    """
    Theta = 0

    def get_pdf_vals(rytov_variance):

        # Calculate for the distribution
        _pre_alpha = 0.49 * rytov_variance / np.pow( 1 + (1 + Theta) * 0.56 * np.pow(rytov_variance, 6/5), 7/6 )
        _alpha = 1 / ( np.exp(_pre_alpha) - 1)
        
        _pre_beta = 0.51 * rytov_variance / np.pow( 1 + 0.69 * np.pow(rytov_variance, 6/5), 5/6 )
        _beta = 1 / ( np.exp(_pre_beta) - 1 )

        scint = 1/_alpha + 1/_beta + 1 / (_alpha * _beta)
        # print(f"scint: {scint}")
        # print(f"alph: {_alpha:.2f}, beta: {_beta:.2f}")

        dist = __GammaGammaDist(alpha=_alpha, beta=_beta)
        vals = np.linspace(1e-8, 10, 10000)
        pdfs = np.array([dist.pdf(x) for x in vals])

        return pdfs, vals, scint

    parameters = (0.2, 0.12)
    fig, axs = plt.subplots(1,2)

    for i, rvar in enumerate(parameters):
        ax = axs[i]

        sampler = gamma_gamma_UL_PR(rytov_variance=rvar, Theta=Theta)
        samples = sampler(100000)
        _y, _x, scint = get_pdf_vals(rvar)

        ax.set_title(f"Scintillation index: {scint:.2f}")
        sns.histplot(x=samples, stat="density", label="Sampled values", ax=ax, color="green")
        sns.lineplot(y=_y, x=_x, label="PDF", ax=ax, color="orange")
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_xlim([0, 2])
        ax.set_yticks([0, 0.5, 1, 1.5])
        ax.set_ylim([0, 1.5])
        
    fig.suptitle("GammaGamma: Comparison of PDF and histogram of samples.")
    fig.tight_layout()
    plt.show()


@test_discussion(__name__, "The plot is very close to the one in the book. The left tail in the second test is a bit too far to the left.")
def GammaLiterature():
    """Gamma PDF for tracked case

    To compare with [1, Fig. 6.6-6.7] p. 191

    [1] L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. Bellingham, Washington, USA: SPIE Press, 2023.

    """

    parameters = (
        (0.028, 0.944, [0, 2], [3e-3, 4]), 
        (0.025, 0.837, [0, 1.8], [3e-4, 10])
        )
    fig, axs = plt.subplots(1,2)

    for i, params in enumerate(parameters):
        ax = axs[i]
        scint, strehl, xlim, ylim = params

        _scale = scint
        _shape = 1 / scint
        def pdf(x):
            return sp.stats.gamma.pdf(x, a=_shape, scale=_scale)
        _x = np.linspace(1e-9, 10, 10000)
        _y = pdf(_x)
        _Sy = _y / strehl
        _S = _x * strehl

        sampler = gamma_UL_PR_tracked(scintillation_index=scint)
        samples = sampler(100000)
        samples_S = samples * strehl # We need to transform from irradiance to strehl

        ax.set_title(f"Scintillation index: {scint:.3f} | SR: {strehl:.3f}")
        sns.histplot(x=samples_S, stat="density", label="Sampled values", ax=ax, color="blue", alpha=0.6, kde=True)
        sns.lineplot(y=_Sy, x=_S, label="PDF", ax=ax, color="orange")
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("Strehl")
        ax.set_ylabel("PDF")
        ax.set_yscale("log")
        
    fig.suptitle("Gamma: To compare with literature (line is KDE).")
    fig.tight_layout()
    plt.show()


def pool_func_ln(scint):
    num_samples = int(1e7)

    print(f"Starting scint: {scint}")
    sampler = lognormal_DL_AA(scintillation_index=scint)
    return sampler(num_samples)

@test_discussion(__name__, "The results seem coherent with the ones obtained in the reference. There is something weird that the LN PDF from scipy does not match the KDE obtained from the samples, using the same parameters and same object.")
def LognormalLiterature():
    """Compare with figures in [1]

    [1] R. Barrios and F. Dios, “Exponentiated Weibull model for the irradiance probability density function of a laser beam propagating through atmospheric turbulence,” Optics & Laser Technology, vol. 45, pp. 13–20, Feb. 2013, doi: 10.1016/j.optlastec.2012.08.004.


    Raises:
        NotImplementedError: just that
    """

    scints = (1.23, 3.55, 1.19, 3.15, 1.01, 2.16)

    with Pool(None) as pool:
        results = pool.map(pool_func_ln, scints)

    fig, axs = plt.subplots(3,2)
    for i, samples in enumerate(results):
        ax = axs[i // 2][i % 2]

        samples = np.log(samples)

        ax.set_title(f"Scintillation index: {scints[i]:.2f}")
        sns.histplot(x=samples, stat="density", label="Sampled values", ax=ax, color="blue", alpha=0.6, kde=True)
        ax.set_xticks(range(-10, 7, 2))
        ax.set_xlim(-10, 5)
        ax.set_ylim(bottom=1e-5, top=1e1)
        ax.set_xlabel(r"$\ln(I)$")
        ax.set_ylabel("PDF")
        ax.set_yscale("log")
        
    fig.suptitle("LN: To compare with literature (line is KDE).")
    fig.tight_layout()
    plt.show()



def pool_func_ew(scint):
    num_samples = int(2e6)

    print(f"Starting scint: {scint}")
    sampler = exponweib_DL_AA(scintillation_index=scint)
    return sampler(num_samples)

@test_discussion(__name__, "The results seem coherent with the ones obtained in the reference. There is something weird that the EW PDF from scipy does not match the KDE obtained from the samples, using the same parameters and same object.")
def ExponentiatedWeibullLiterature():
    """Compare with figures in [1]

    [1] R. Barrios and F. Dios, “Exponentiated Weibull model for the irradiance probability density function of a laser beam propagating through atmospheric turbulence,” Optics & Laser Technology, vol. 45, pp. 13–20, Feb. 2013, doi: 10.1016/j.optlastec.2012.08.004.


    Raises:
        NotImplementedError: just that
    """

    # ---- This function is not showing the same PDF shape as the samples, for some reason
    # def recreate_pdf(scintillation_index, vals):
    #     _alpha = 7.220 * np.pow(scintillation_index, 1/3) / sp.special.gamma( 2.487 * np.pow(scintillation_index, 1/6) - 0.104 )
    #     _beta = 1.012 * np.pow(_alpha * scintillation_index, -13/25) + 0.142

    #     # "eta"
    #     _scale = 1 / ( _alpha * sp.special.gamma(1 + 1/_beta) * _exponweib_g_function(order=1, alpha=_alpha, beta=_beta)) 

    #     return sp.stats.exponweib.pdf(x=vals, a=_alpha, c=_beta, scale=_scale)

    scints = (1.23, 3.55, 1.19, 3.15, 1.01, 2.16)

    with Pool(None) as pool:
        results = pool.map(pool_func_ew, scints)

    fig, axs = plt.subplots(3,2)
    for i, samples in enumerate(results):
        ax = axs[i // 2][i % 2]

        samples = np.log(samples)

        ax.set_title(f"Scintillation index: {scints[i]:.2f}")
        sns.histplot(x=samples, stat="density", label="Sampled values", ax=ax, color="blue", alpha=0.6, kde=True)
        ax.set_xticks(range(-10, 7, 2))
        ax.set_xlim(-10, 5)
        ax.set_ylim(bottom=1e-5, top=1e1)
        ax.set_xlabel(r"$\ln(I)$")
        ax.set_ylabel("PDF")
        ax.set_yscale("log")
        
    fig.suptitle("EW: To compare with literature (line is KDE).")
    fig.tight_layout()
    plt.show()


def ModulatedGammaLiterature():
    
    raise NotImplementedError



if __name__ == "__main__":
    # GammaGamma()
    # ModulatedGamma()
    # GammaGammaLiterature()
    # GammaLiterature()
    # ExponentiatedWeibullLiterature()
    # LognormalLiterature()

    print(Fore.CYAN + "[INFO]" + Fore.MAGENTA + " EW and LN distributions and PDFs dont match for same parameters (idk why)" + Fore.RESET)  # ty:ignore[unsupported-operator]
    print("No more tests...")