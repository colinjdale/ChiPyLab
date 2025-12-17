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


def blackman_instrument(nu, t_width):
    """Analytic solution of the Blackman Fourier transform. Obtained from
       https://mathworld.wolfram.com/BlackmanFunction.html."""
    return t_width * (0.42 - 0.18/4 * t_width**2 * nu**2)*np.sinc(t_width * nu) \
            / ((1 - t_width**2 * nu**2/4) * (1 - t_width**2 * nu**2))


def blackman_convolution_correction(detuning, trf, x_a=4e6):
    """Computes the overestimation of HFT transfer for a Blackman pulse of full length trf
       detuned by detuning into the HFT. Detuning must be > 1/3trf."""
    if detuning/(3/trf) <= 0.99:
        raise ValueError("Detuning is too small, should be > 3/trf.")
    
    samples = 1e5
    nu_min = detuning - 2.9/trf
    nu_max = detuning + 2.9/trf
    nu = np.linspace(nu_min, nu_max, int(samples))

    S = nu**(-3/2)/(1 + nu/x_a)
    S_norm = np.trapezoid(S, nu)
    S /= S_norm

    # Infinitely small normalized blackman response
    B_inf_small = (detuning**(-3/2)/(1 + detuning/x_a)/S_norm)

    # Blackman
    def blackman_ft(nu):
        return (blackman_fourier(nu, trf)/trf)**2
    
    B_pulse = blackman_ft(nu - detuning)
    B_norm = np.trapezoid(B_pulse, nu)
    B_pulse /= B_norm

    B = blackman_ft(detuning - nu)/B_norm
    M = np.trapezoid(S * B, nu)/B_inf_small
    return M




# def blackman_instrument(nu, t_width):
#     """Analytic solution of the Blackman Fourier transform. Obtained from
#        https://mathworld.wolfram.com/BlackmanFunction.html."""
#     a = t_width/2
#     return a * (0.84 - 0.36 * a**2 * nu**2)*np.sinc(2 * a * nu) \
#             / ((1 - a**2 * nu**2) * (1 - 4 * a**2 * nu**2))