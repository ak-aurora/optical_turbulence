# ----------------- DECORATORS ----------------- #
#region Decorators

_untested_funcs = list()
def warn_not_tested(func): 
    """Warn that the function has not been tested against literature when run

    Args:
        func (callable): untested/unverified function
    """

    def inner1(*args, **kwargs):

        if func.__name__ not in _untested_funcs:
            print("\033[33m" + f"[WARN - {func.__name__}] This function has not been tested against literature results." + "\033[0m")
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
                print("\033[33m" + f"[WARN - {func.__name__}] {message}" + "\033[0m")
                _custom_warned_funcs.append(func.__name__)

            # calling the actual function now 
            # inside the wrapper function. 

            return func(*args, **kwargs)

        return inner1
    
    return inner0

#endregion Decorators