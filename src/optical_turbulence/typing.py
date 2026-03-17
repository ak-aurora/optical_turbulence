import numpy as np
import numpy.typing as npt

# ----------------- TYPE DEFINITIONS ----------------- #
#region Type Definitions

# scalar_t = np.number
"""Numeric values
    """
# num_array_t = npt.NDArray[scalar_t]
"""Arrays of numeric values
    """

real_t = int | float | np.float64 | np.int_
"""Real scalar values"""

real_array_t = npt.NDArray[np.float64] | npt.NDArray[np.int_]
"""Array of floats or integers"""

# array_or_scalar_t = scalar_t | num_array_t

#endregion Type Definitions


