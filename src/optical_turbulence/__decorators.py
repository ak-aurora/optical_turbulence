import warnings

# ----------------- DECORATORS ----------------- #
#region Decorators

def _warning_formatter(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    line: str | None = None,
) -> str:
    return f'[\033[33mWARN\033[0m:\033[33m{category.__name__}\033[0m] {message}\n'

warnings.formatwarning = _warning_formatter

_untested_funcs = []
def warn_not_tested(func): 
    """Warn that the function has not been tested against literature when run

    Args:
        func (callable): untested/unverified function
    """

    def inner1(*args, **kwargs):

        if func.__name__ not in _untested_funcs:
            warnings.warn(f"[\033[1;35m{func.__name__}\033[0m] This function has not been tested against literature results.")
            _untested_funcs.append(func.__name__)

        # calling the actual function now 
        # inside the wrapper function. 

        return func(*args, **kwargs)

    return inner1

_custom_warned_funcs = []
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