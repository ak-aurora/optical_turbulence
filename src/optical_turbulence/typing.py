import numpy as np
import numpy.typing as npt

# ----------------- TYPE DEFINITIONS ----------------- #
# region Type Definitions

real_t = int | float | np.float64
"""Real scalar values"""

real_array_t = (
    npt.NDArray[np.float64]
)  # | npt.NDArray[np.float128]
"""Array of floats"""


# endregion Type Definitions
