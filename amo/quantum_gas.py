import numpy as np
from .constants import mK, hbar, h, kB


def thermal_deBroglie_wavelength(T):
    """Computes the thermal de Broglie wavelength in meters for a 
    given temperature T (in Kelvin)."""
    return h / np.sqrt(2 * np.pi * mK * kB * T)


def fermi_energy(N, omega_bar, dim=3, pol=1):
    """Computes the Fermi energy in Joules for a given number of atoms N
    and geometric mean trap frequency omega_bar (in rad/s)."""
    if dim == 3:
        return hbar * omega_bar * (6 * N * pol)**(1/3)
    elif dim == 2:
        return hbar * omega_bar * np.sqrt(2 * N * pol)
    elif dim == 1:
        return hbar * omega_bar * N * pol
    else:
        raise ValueError("Dimension dim must be 1, 2, or 3.")


def fermi_wavevector(N, omega_bar, dim=3, pol=1):
    """Computes the Fermi wavevector in 1/m for a given atom number  
    N and geometric mean trap frequency omega_bar (in rad/s)."""
    return np.sqrt(2 * mK * fermi_energy(N, omega_bar, dim=dim, pol=pol))/hbar


def oscillator_length(omega):
    """Computes the harmonic oscillator length in meters for a given trap 
    frequency omega (in rad/s)."""
    return np.sqrt(hbar / (mK * omega))