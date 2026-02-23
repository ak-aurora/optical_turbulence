import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.optical_turbulence.ris_models import hufnagel_valley_model
from ._test_utilities import test_discussion
from src.optical_turbulence.round_earth import *

@test_discussion(__name__, "It seems to follow the behaviour of the plot in the book. There is a small difference with flat-earth for big zenith")
def test1():
    """Compare the expressions to the Fig. 12.7 p.497 from
    L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
    
    Tested function:
        - scint_index_DL_PR_weak
        - scint_index_DL_PR_general
    """

    altitude = np.linspace(0, 300e3, 300000) # [m] 300 km altitude
    wavelen = 1.06 * 1e-6 # [m] 1.06 um
    A1 = 1.7e-14 # "weak ground RIS"
    A2 = 3e-13 # "moderate-strong ground RIS"

    zeniths = np.linspace(0, 85, 85)
    zeniths_rad = zeniths * (np.pi / 180)

    distances = [get_distance_from_altitude(altitude=altitude, zenith_angle=zen) for zen in zeniths_rad]
    zen_dis = list(zip(zeniths_rad, distances))

    ris_model_1 = lambda h: hufnagel_valley_model(h, A=A1)
    ris_model_2 = lambda h: hufnagel_valley_model(h, A=A2)

    scints_1_weak = np.array([ scint_index_DL_PR_weak(wavelen, zenang, dist, ris_model_1) for zenang, dist in zen_dis ]) # type: ignore
    scints_2_weak = np.array([ scint_index_DL_PR_weak(wavelen, zenang, dist, ris_model_2) for zenang, dist in zen_dis ]) # type: ignore
    scints_1_modstr = np.array([ scint_index_DL_PR_general(wavelen, zenang, dist, ris_model_1) for zenang, dist in zen_dis ]) # type: ignore
    scints_2_modstr = np.array([ scint_index_DL_PR_general(wavelen, zenang, dist, ris_model_2) for zenang, dist in zen_dis ]) # type: ignore

    fig, ax = plt.figure(), plt.axes()
    sns.lineplot(y=scints_1_weak, x=zeniths, label="Weak turbulence regime", ax=ax, linestyle="dotted", color="blue")
    sns.lineplot(y=scints_2_weak, x=zeniths, label=None, ax=ax, linestyle="dotted", color="blue")
    sns.lineplot(y=scints_1_modstr, x=zeniths, label="General regime", ax=ax, color="red")
    sns.lineplot(y=scints_2_modstr, x=zeniths, label=None, ax=ax, color="red")

    ax.set_xlabel("Zenith Angle (degrees)")
    ax.set_ylabel("Scintillation Index")
    ax.set_title("Scintillation index vs Zenith angle (Round Earth)")
    ax.set_ylim(bottom=-0.11, top=1.4)
    ax.set_xlim(left=-7, right=97)
    plt.suptitle("Values to compare with \"LBPTRM 2nd Ed Fig 12.7 p. 497\"")
    plt.show()
    plt.close()


@test_discussion(__name__, "ok. No much difference with flat-earth")
def test2():
    """Compare the expressions to the Fig. 12.6 p.496 from
    L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
    
    Tested function:
        - scint_index_DL_AA_general
    """

    altitude = np.linspace(0, 300e3, 300000) # [m] 300 km altitude
    wavelen = 1.55 * 1e-6 # [m] 1.55 um
    zangle1 = 45 * (np.pi / 180)
    zangle2 = 60 * (np.pi / 180)

    diameters_cm = np.linspace(0, 50, 50) # [cm]
    diameters = diameters_cm / 100 # [m]

    distance1 = get_distance_from_altitude(altitude=altitude, zenith_angle=zangle1)
    distance2 = get_distance_from_altitude(altitude=altitude, zenith_angle=zangle2)

    scints_za1 = [scint_index_DL_AA_general(wavelen, zangle1, distance1, hufnagel_valley_model, diam) for diam in diameters] # type: ignore
    scints_za2 = [scint_index_DL_AA_general(wavelen, zangle2, distance2, hufnagel_valley_model, diam) for diam in diameters] # type: ignore

    fig, ax = plt.figure(), plt.axes()
    sns.lineplot(y=scints_za1, x=diameters_cm, label=r"$\zeta = 45^\circ$", ax=ax, color="red")
    sns.lineplot(y=scints_za2, x=diameters_cm, label=r"$\zeta = 60^\circ$", ax=ax, linestyle="dotted", color="blue")

    ax.set_xlabel("Aperture Diameter (cm)")
    ax.set_ylabel(r"$\sigma_I^2(D_G)$")
    ax.set_title("Scintillation index vs Aperture diameter (Round earth)")
    ax.set_ylim(bottom=-0.02, top=0.26)
    ax.set_yticks(np.arange(-0.02, 0.27, 0.02))
    # ax.set_xlim(left=-7, right=55)
    plt.suptitle("Values to compare with \"LBPTRM 2nd Ed Fig 12.6 p. 496\"")
    plt.show()
    plt.close()


from src.optical_turbulence.round_earth import _total_beam_wander_variance_UL_gaussian


@test_discussion(__name__, "ok")
def test3():
    """Compare the expressions to the Fig. 12.11 p. 503 from ((Only the infinite outer scale one))
    [1] L. C. Andrews, Laser Beam Propagation Through Random Media, 2nd Ed, 2nd ed. in Press Monograph Series, no. v. PM152. Bellingham: SPIE, 2005.
    
    Compare to Fig. 2 from
    [2] R. L. Phillips, “Strehl ratio and scintillation theory for uplink Gaussian-beam waves: beam wander effects,” Opt. Eng, vol. 45, no. 7, p. 076001, Jul. 2006, doi: 10.1117/1.2219470.

    Tested function:
        - _total_beam_wander_variance_UL_gaussian

    """
    altitude = np.linspace(0, 100e3, 100000) # [m] 300 km altitude

    zangle1 = 0 * (np.pi / 180)
    zangle2 = 60 * (np.pi / 180)

    distance1 = get_distance_from_altitude(altitude=altitude, zenith_angle=zangle1)
    distance2 = get_distance_from_altitude(altitude=altitude, zenith_angle=zangle2)

    link_len_1 = np.max(distance1) - np.min(distance1)
    link_len_2 = np.max(distance2) - np.min(distance2)

    pfront_radius = np.inf

    beam_radii_cm = np.linspace(1, 100, 99) # [cm]
    beam_radii = beam_radii_cm / 100 # [m]

    ris_model = hufnagel_valley_model
    rms_angular_1 = np.zeros(np.size(beam_radii_cm))
    rms_angular_2 = np.zeros(np.size(beam_radii_cm))


    common_parameters = {
        "ris_model": ris_model,
        "pfront_radius": pfront_radius
    }

    param_1 = {
        **common_parameters,
        "distance": distance1,
        "zenith_angle": zangle1
    }

    param_2 = {
        **common_parameters,
        "distance": distance2,
        "zenith_angle": zangle2
    }

    for i, bradius in enumerate(beam_radii):
        rms_angular_1[i] = _total_beam_wander_variance_UL_gaussian(beam_radius=bradius, **param_1)
        rms_angular_2[i] = _total_beam_wander_variance_UL_gaussian(beam_radius=bradius, **param_2)

    rms_angular_1 = np.sqrt(rms_angular_1) / link_len_1 * 1e6
    rms_angular_2 = np.sqrt(rms_angular_2) / link_len_2 * 1e6

    fig, ax = plt.figure(), plt.axes()
    sns.lineplot(y=rms_angular_1, x=beam_radii_cm, label=r"Zenith = $0^\circ$", ax=ax, color="blue")
    sns.lineplot(y=rms_angular_2, x=beam_radii_cm, label=r"Zenith = $60^\circ$", ax=ax, color="red", linestyle="dashed")

    ax.set_xscale("log")
    ax.set_xlabel("Beam radius $W_0$ (cm)")
    ax.set_ylabel(r"RMS Angular Displacement ($\mu rad$)")
    ax.set_title("Displacement vs beam radius - Uplink - Collimated - Round Earth")
    ax.set_ylim(bottom=3.3, top=13.7)
    ax.set_yticks(list(range(4,14)))
    ax.set_xlim(left=0.8, right=200)
    plt.suptitle("Values to compare with Fig 12.11 p. 503 [1] and Fig. 2 [2]\"")
    plt.show()
    plt.close()


@test_discussion(__name__, "ok, equal to flat-earth (zenith = 0). It takes more time than flat-earth to calculate though.")
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

    altitude = np.linspace(0, 300e3, 300000) # [m] 300 km altitude
    wavelen = 1.6 * 1e-6 # [m] 1.6 um
    zangle = 0 * (np.pi / 180)
    pfront_radius = np.inf
    distance = get_distance_from_altitude(altitude=altitude, zenith_angle=zangle)
    link_length = np.max(distance) - np.min(distance)

    wavenum = 2 * np.pi / wavelen

    beam_radii_cm = np.linspace(0.5, 200, 300) # [cm]
    beam_radii = beam_radii_cm / 100 # [m]

    theta_zero = 1 - link_length / pfront_radius


    # This info is not given in the book
    ris_model = hufnagel_valley_model # HV5/7

    # Function params
    fixed_params = {
        "wavelength": wavelen,
        "zenith_angle": zangle,
        "distance": distance,
        "ris_model": ris_model,
        "pfront_radius": pfront_radius
    }

    # where to save the data
    scint_index_tr = np.zeros(np.size(beam_radii_cm))
    scint_index_utr = np.zeros(np.size(beam_radii_cm))

    num_iterations = np.size(beam_radii_cm)
    for i, radius in enumerate(beam_radii):
        print(f"Running test... {i / num_iterations * 100: .1f}%", end="\r")
        lambda_zero = 2 * link_length / (wavenum * np.pow(radius, 2))
        _denominator = ( np.pow(theta_zero, 2) + np.pow(lambda_zero, 2) )
        _Lambda = lambda_zero / _denominator
        _Theta = theta_zero / _denominator
        rx_spot_radius = radius * np.sqrt(_denominator)

        varying_params = {
            "beam_radius": radius,
            "Lambda": _Lambda,
            "Theta": _Theta,
            "rx_spot_size": rx_spot_radius
        }

        scint_index_tr[i] = scint_index_UL_tracked_gaussian(**fixed_params, **varying_params)
        scint_index_utr[i] = scint_index_UL_untracked_gaussian(**fixed_params, **varying_params)
    print(f"Running test... finished. Showing results.", end="\r")

    fig, ax = plt.figure(), plt.axes()
    sns.lineplot(y=scint_index_tr, x=beam_radii_cm, label="Tracked", ax=ax, color="blue", linestyle="dashed")
    sns.lineplot(y=scint_index_utr, x=beam_radii_cm, label="Untracked", ax=ax, color="red")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Beam radius $W_0$ (cm)")
    ax.set_ylabel("On-Axis Scintillation Index")
    ax.set_title("Scintillation index vs beam radius (Round earth)")
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