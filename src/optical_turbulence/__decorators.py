import warnings
from typing import Optional, Type, Union

# ----------------- DECORATORS ----------------- #
#region Decorators

def _warning_formatter(
    message: Union[Warning, str],
    category: Type[Warning],
    filename: str,
    lineno: int,
    line: Optional[str] = None,
) -> str:
    return '[\033[33mWARN\033[0m:\033[33m%s\033[0m] %s\n' % (category.__name__, message)

warnings.formatwarning = _warning_formatter # ty:ignore[invalid-assignment]

_untested_funcs = list()
def warn_not_tested(func): 
    """Warn that the function has not been tested against literature when run

    Args:
        func (callable): untested/unverified function
    """

    def inner1(*args, **kwargs):

        if func.__name__ not in _untested_funcs:
            warnings.warn(f"[\033[1;35m{func.__name__}\033[0m]] This function has not been tested against literature results.")
            _untested_funcs.append(func.__name__)

        # calling the actual function now 
        # inside the wrapper function. 

        return func(*args, **kwargs)

    return inner1

_custom_warned_funcs = list()
def warn_custom(message):

    def inner0(func): 
        """Warn that the function has not been tested against literature when run

        Args:
            func (callable): untested/unverified function
        """

        def inner1(*args, **kwargs):

            if func.__name__ not in _custom_warned_funcs:
                warnings.warn("\033[33m" + f"[WARN - {func.__name__}] {message}" + "\033[0m")
                _custom_warned_funcs.append(func.__name__)

            # calling the actual function now 
            # inside the wrapper function. 

            return func(*args, **kwargs)

        return inner1
    
    return inner0

#endregion Decorators