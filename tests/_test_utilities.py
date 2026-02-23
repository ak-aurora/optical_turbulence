
def test_discussion(module_name: str, discussion: str):

    def inner0(func):
    
        def inner(*args, **kwargs):
            print(f"\033[36m[Discussion]\033[97m {module_name}.{func.__name__}\033[0m: ", end="")
            print(discussion)

            return func(*args, **kwargs)
    
        return inner

    return inner0