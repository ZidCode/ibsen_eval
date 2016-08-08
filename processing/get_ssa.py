import numpy as np
from math import exp


def get_ssa(RH, AM_type=1):
    '''
    This method evaluates the single scattering albedo
    of aerosol due to mass type and relative humidity
    Reference: Greg and Carder
    I still do have no idea from where the function actually comes from
    No paper is using such a formula.
    Args:
        RH (double): rel. humidity [0.0-1.0]
        AM_type (int): Air Mass type [1 - 10]
    Returns:
        ssa: single scattering albedo
    '''
    a = -0.0032  # unknown
    b = 0.972  # unknown
    c = 3.06e-4  # still unknown

    ssa = (a * AM_type + b) * exp(c * RH)
    return ssa


def show_ssa():
    import matplotlib.pyplot as plt
    RH = np.arange(0, 1, 0.1)
    AM_type = np.arange(0, 10, 1)

    v_get_ssa = np.vectorize(get_ssa)
    for AM in AM_type:
        plt.plot(RH, v_get_ssa(RH, AM), '+', label='Air mass type %s' % AM)
    plt.xlabel('rel. humidity')
    plt.ylabel('ssa [Greg. et Carder]')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    show_ssa()
