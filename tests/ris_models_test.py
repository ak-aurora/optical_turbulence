"""
To test the RIS models, we plot them and compare to the results shown in \
L. C. Andrews and M. Beason, Laser beam propagation in random media: new and advanced topics. \
    Bellingham, Washington, USA: SPIE Press, 2023.

Tested functions:
    - HV Model
    - HAP Daytime    
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.optical_turbulence.ris_models import hap_model_daytime, hufnagel_valley_model

from . import _test_utilities


@_test_utilities.test_discussion(__name__, \
                                 "The results are consistent; however, the plot in the 2023 book" \
                                 "is not the same as the calculated one because the equation from the" \
                                 "slew rate paper is different. When changing the parameters to the book, the" \
                                 "results are similar enough to the ones in the figure.")
def test1():
    """Test HV Model and HAP Daytime versus plot available in base reference, p. 12, Fig. 1.4
    """

    A = 1.7e-14
    W = 21

    altitudes = np.linspace(1, 5e3, 5000)

    fig, ax = plt.figure(), plt.axes()
    
    sns.lineplot(x=altitudes, y=hufnagel_valley_model(altitudes, W, A), label=f"HV-5/7 Model", ax=ax)
    sns.lineplot(x=altitudes, y=hap_model_daytime(altitudes, W, A, M=1), label=f"HAP Model daytime", ax=ax)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Altitude [m]")
    ax.set_ylabel(r"$C_n^2$ [m$^{-2/3}$] (log scale)")
    ax.set_title("Comparison of HV and HAP model")
    plt.suptitle("Values to compare with \"LBPIRM: N&AT Fig 1.4 p. 12\"")
    plt.show()
    plt.close()

if __name__ == "__main__":
    test1()
