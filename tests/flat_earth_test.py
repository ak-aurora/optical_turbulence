import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from optical_turbulence.typing import real_array_t
from src.optical_turbulence.classes import (
    OpticalBeam,
    LinkDescriptionManager
)
from src.optical_turbulence.earth_models import FlatEarthModel
from src.optical_turbulence.parameters import (
    scint_index_DL_AA_general,
    scint_index_DL_PR_general,
    scint_index_DL_PR_weak,
    scint_index_UL_tracked_gaussian,
    scint_index_UL_untracked_gaussian,
    _total_beam_wander_variance_UL_gaussian
)
from src.optical_turbulence.ris_models import hufnagel_valley_model
from src.optical_turbulence.utils import create_altitude_array

from . import _test_utilities


@_test_utilities.test_discussion(__name__, "It looks to be coherent; although, I have my doubts with the difference of weak and strong for the stronger ground RIS value.")
def test1():
    """Compare the expressions to the Fig. 12.7 p.497 from
    L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
    
    Tested function:
        - scint_index_DL_PR_weak
        - scint_index_DL_PR_general
    """

    sat_altitude = 300e3
    # altitude_array = np.linspace(0, sat_altitude, int(sat_altitude)) # [m] 300 km altitude
    wavelen = 1.06 * 1e-6 # [m] 1.06 um
    A1 = 1.7e-14 # "weak ground RIS"
    A2 = 3e-13 # "moderate-strong ground RIS"

    link_info = LinkDescriptionManager(
        satellite_altitude=sat_altitude,
        lct_altitude=0,
        zenith_angle=0,
        earth_model=FlatEarthModel(),
        altitude_array_factory=create_altitude_array
    )

    zeniths = np.linspace(0, 85, 85)
    zeniths_rad = zeniths * (np.pi / 180)

    def ris_model_1(h: real_array_t):
        return hufnagel_valley_model(h, A=A1)
    
    def ris_model_2(h: real_array_t):
        return hufnagel_valley_model(h, A=A2)

    scints_1_weak = np.empty(shape=zeniths_rad.shape)
    scints_2_weak = np.empty(shape=zeniths_rad.shape)
    scints_1_modstr = np.empty(shape=zeniths_rad.shape)
    scints_2_modstr = np.empty(shape=zeniths_rad.shape)

    for i, zenang in enumerate(zeniths_rad):
        link_info.zenith_angle = zenang

        scints_1_weak[i] = scint_index_DL_PR_weak(
            wavelength=wavelen,
            ris_model=ris_model_1,
            **link_info.arrays_as_dict()
        )

        scints_2_weak[i] = scint_index_DL_PR_weak(
            wavelength=wavelen,
            ris_model=ris_model_2,
            **link_info.arrays_as_dict()
        )

        scints_1_modstr[i] = scint_index_DL_PR_general(
            wavelength=wavelen,
            ris_model=ris_model_1,
            **link_info.arrays_as_dict()
        )

        scints_2_modstr[i] = scint_index_DL_PR_general(
            wavelength=wavelen,
            ris_model=ris_model_2,
            **link_info.arrays_as_dict()
        )

    _fig, ax = plt.figure(), plt.axes()
    sns.lineplot(y=scints_1_weak, x=zeniths, label="Weak turbulence regime", ax=ax, linestyle="dotted", color="blue")
    sns.lineplot(y=scints_2_weak, x=zeniths, label=None, ax=ax, linestyle="dotted", color="blue")
    sns.lineplot(y=scints_1_modstr, x=zeniths, label="General regime", ax=ax, color="red")
    sns.lineplot(y=scints_2_modstr, x=zeniths, label=None, ax=ax, color="red")

    ax.set_xlabel("Zenith Angle (degrees)")
    ax.set_ylabel("Scintillation Index")
    ax.set_title("Scintillation index vs Zenith angle")
    ax.set_ylim(bottom=-0.11, top=1.4)
    ax.set_xlim(left=-7, right=97)
    plt.suptitle("Values to compare with \"LBPTRM 2nd Ed Fig 12.7 p. 497\"")
    plt.show()
    plt.close()

@_test_utilities.test_discussion(__name__, "ok.")
def test2():
    """Compare the expressions to the Fig. 12.6 p.496 from
    L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
    
    Tested function:
        - scint_index_DL_AA_general
    """

    sat_altitude = 300e3
    wavelen = 1.55 * 1e-6 # [m] 1.55 um
    zangle1 = 45 * (np.pi / 180)
    zangle2 = 60 * (np.pi / 180)

    link_zang1 = LinkDescriptionManager(
        satellite_altitude=sat_altitude,
        lct_altitude=0,
        zenith_angle=zangle1,
        earth_model=FlatEarthModel(),
        altitude_array_factory=create_altitude_array
    )

    link_zang2 = copy.deepcopy(link_zang1)
    link_zang2.zenith_angle = zangle2

    diameters_cm = np.linspace(0, 50, 50) # [cm]
    diameters = diameters_cm / 100 # [m]

    scints_za1 = np.empty(shape=diameters.shape)
    scints_za2 = np.empty(shape=diameters.shape)

    for i, diam in enumerate(diameters):
        
        scints_za1[i] = scint_index_DL_AA_general(
            aperture_diameter=diam,
            wavelength=wavelen,
            ris_model=hufnagel_valley_model,
            **link_zang1.arrays_as_dict()
        )

        scints_za2[i] = scint_index_DL_AA_general(
            aperture_diameter=diam,
            wavelength=wavelen,
            ris_model=hufnagel_valley_model,
            **link_zang2.arrays_as_dict()
        )

    _fig, ax = plt.figure(), plt.axes()
    sns.lineplot(y=scints_za1, x=diameters_cm, label=r"$\zeta = 45^\circ$", ax=ax, color="red")
    sns.lineplot(y=scints_za2, x=diameters_cm, label=r"$\zeta = 60^\circ$", ax=ax, linestyle="dotted", color="blue")

    ax.set_xlabel("Aperture Diameter (cm)")
    ax.set_ylabel(r"$\sigma_I^2(D_G)$")
    ax.set_title("Scintillation index vs Aperture diameter")
    ax.set_ylim(bottom=-0.019, top=0.26)
    ax.set_yticks(np.arange(0, 0.26, 0.02))
    # ax.set_xlim(left=-7, right=55)
    plt.suptitle("Values to compare with \"LBPTRM 2nd Ed Fig 12.6 p. 496\"")
    plt.show()
    plt.close()



@_test_utilities.test_discussion(__name__, "ok.")
def test3():
    """Compare the expressions to the Fig. 12.11 p. 503 from
    [1] L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
    
    Compare to Fig. 2 from
    [2] R. L. Phillips, “Strehl ratio and scintillation theory for uplink Gaussian-beam waves: beam wander effects,” Opt. Eng, vol. 45, no. 7, p. 076001, Jul. 2006, doi: 10.1117/1.2219470.

    Tested function:
        - _total_beam_wander_variance_UL_gaussian

    """
    sat_altitude = 300e3
    # altitude = np.linspace(0, 100e3, 100000) # [m] 300 km altitude
    # link_distance = np.max(altitude) - np.min(altitude)

    zangle1 = 0 * (np.pi / 180)
    zangle2 = 60 * (np.pi / 180)

    link_zang1 = LinkDescriptionManager(
        satellite_altitude=sat_altitude,
        lct_altitude=0,
        zenith_angle=zangle1,
        earth_model=FlatEarthModel(),
        altitude_array_factory=create_altitude_array
    )

    link_zang2 = copy.deepcopy(link_zang1)
    link_zang2.zenith_angle = zangle2

    pfront_radius = np.inf

    beam_radii_cm = np.linspace(1, 100, 99) # [cm]
    beam_radii = beam_radii_cm / 100 # [m]

    ris_model = hufnagel_valley_model
    rms_angular_1 = np.zeros(np.size(beam_radii_cm))
    rms_angular_2 = np.zeros(np.size(beam_radii_cm))

    common_parameters = {
        "ris_model": ris_model,
        "pfront_radius": pfront_radius,
    }

    for i, bradius in enumerate(beam_radii):

        rms_angular_1[i] = _total_beam_wander_variance_UL_gaussian(
            beam_radius=bradius,
            **common_parameters,  # ty:ignore[invalid-argument-type]
            **link_zang1.arrays_as_dict()  # ty:ignore[parameter-already-assigned]
        )

        rms_angular_2[i] = _total_beam_wander_variance_UL_gaussian(
            beam_radius=bradius,
            **common_parameters,  # ty:ignore[invalid-argument-type]
            **link_zang2.arrays_as_dict()  # ty:ignore[parameter-already-assigned]
        )

    rms_angular_1 = np.sqrt(rms_angular_1) / link_zang1.link_distance() * 1e6
    rms_angular_2 = np.sqrt(rms_angular_2) / link_zang2.link_distance() * 1e6

    _fig, ax = plt.figure(), plt.axes()
    sns.lineplot(y=rms_angular_1, x=beam_radii_cm, label=r"Zenith = $0^\circ$", ax=ax, color="blue")
    sns.lineplot(y=rms_angular_2, x=beam_radii_cm, label=r"Zenith = $60^\circ$", ax=ax, color="red", linestyle="dashed")

    ax.set_xscale("log")
    ax.set_xlabel("Beam radius $W_0$ (cm)")
    ax.set_ylabel(r"RMS Angular Displacement ($\mu rad$)")
    ax.set_title("Displacement vs beam radius - Uplink - Collimated")
    ax.set_ylim(bottom=3.3, top=13.7)
    ax.set_yticks(list(range(4,14)))
    ax.set_xlim(left=0.8, right=200)
    plt.suptitle(r"Values to compare with Fig 12.11 p. 503 [1] ($L_0 = \infty$) and Fig. 2 [2]")
    plt.show()
    plt.close()

@_test_utilities.test_discussion(__name__, "The plot seems equal to the one shown in the book.")
def test4():
    """Test for the uplink scintillation index to compare with fig. 6.4 p. 185 from
    L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. Bellingham, Washington, USA: SPIE Press, 2023.
    
    Tested functions:
        - scint_index_UL_tracked_gaussian
        - scint_index_UL_untracked_gaussian
        - rytov_variance_UL_gaussian
        - _bw_pointing_error_var_UL_gaussian
        - _bw_pointing_error_var_UL_gaussian_TT
        - fried_parameter_UL_TX
    """

    sat_altitude = 300e3 # [m]

    link_info = LinkDescriptionManager(
        satellite_altitude=sat_altitude,
        lct_altitude=0,
        zenith_angle=0,
        earth_model=FlatEarthModel(),
        altitude_array_factory=create_altitude_array
    )

    beam = OpticalBeam(
        beam_radius= 0.5, # temp value
        pfront_radius=np.inf,
        wavelength=1.6 * 1e-6, # [m] 1.6 um,
        link_distance=link_info.link_distance()
    )

    beam_radii_cm = np.linspace(0.5, 200, 300) # [cm]
    beam_radii = beam_radii_cm / 100 # [m]

    # This info is not given in the book
    ris_model = hufnagel_valley_model # HV5/7

    # where to save the data
    scint_index_tr = np.zeros(np.size(beam_radii_cm))
    scint_index_utr = np.zeros(np.size(beam_radii_cm))

    num_iterations = np.size(beam_radii_cm)
    for i, radius in enumerate(beam_radii):
        print(f"Running test... {i / num_iterations * 100: .1f}%", end="\r")
        
        beam.beam_radius = radius

        scint_index_tr[i] = scint_index_UL_tracked_gaussian(
            ris_model=ris_model,
            **beam.as_dict(),
            **link_info.arrays_as_dict()
        )

        scint_index_utr[i] = scint_index_UL_untracked_gaussian(
            ris_model=ris_model,
            **beam.as_dict(),
            **link_info.arrays_as_dict()
        )

    print("Running test... finished. Showing results.", end="\r")

    _fig, ax = plt.figure(), plt.axes()
    sns.lineplot(y=scint_index_tr, x=beam_radii_cm, label="Tracked", ax=ax, color="blue", linestyle="dashed")
    sns.lineplot(y=scint_index_utr, x=beam_radii_cm, label="Untracked", ax=ax, color="red")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Beam radius $W_0$ (cm)")
    ax.set_ylabel("On-Axis Scintillation Index")
    ax.set_title("Scintillation index vs beam radius")
    ax.set_xlim(left=0.3, right=300)
    ax.set_xticks([1, 10, 100])
    ax.set_ylim(bottom=7e-4, top=20)
    ax.set_yticks([10.0, 1.0, 0.1, 0.01, 0.001])
    plt.suptitle("Values to compare with \"LBPTRM: N&AT Fig 6.4 p. 185\"")
    plt.show()
    plt.close()

    print("\033[K", end="\r")


if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()
    print("No more tests...")