import math as math_native

from uncertainties import umath as math_uncertainties


def math_switcher(math_module):
    if math_module == "math":
        return math_native
    elif math_module == "umath":
        return math_uncertainties
    else:
        raise ValueError("Invalid math module specified")


# Set the initial math module (you can change this later)
math = math_switcher("math")
