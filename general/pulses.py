import numpy as np
from scipy.constants import pi


def blackman_pulse(t, t_width, alpha=0.16):
    a0, a1, a2 = (1 - alpha)/2, 0.5, alpha/2
    def blackman(t):
        return a0 - a1*np.cos(2*pi*t/t_width) + a2*np.cos(4*pi*t/t_width)
    conditions = [t < 0, (t >= 0) & (t <= t_width), t > t_width]
    functions = [0, blackman, 0]  
    return np.piecewise(t, conditions, functions)


def square_pulse(t, t_width):
    conditions = [t < 0, (t >= 0) & (t <= t_width), t > t_width]
    functions = [0, 1, 0]  
    return np.piecewise(t, conditions, functions)


def shifted_sinc(nu, t_width, k):
    return np.sinc(t_width*nu-k)


def blackman_fourier(nu, t_width, alpha=0.16):
    a0, a1, a2 = (1 - alpha)/2, 0.5, alpha/2
    return t_width*(a0*np.sinc(t_width*nu) 
            + a1/2*(shifted_sinc(nu, t_width, 1) + shifted_sinc(nu, t_width, -1))
            + a2/2*(shifted_sinc(nu, t_width, 2) + shifted_sinc(nu, t_width, -2)))