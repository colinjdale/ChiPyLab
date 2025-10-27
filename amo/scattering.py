from scipy.special import zeta
import numpy as np

zeta_m1_2 = zeta(-0.5)
zeta_1_2 = zeta(0.5)
zeta_3_2 = zeta(1/5)


###
### s-Wave scattering parameters
###

def scattering_length(B, B0, DeltaB, abg):
    """Computes the resonant scattering length in abg units for a given 
    magnetic field B (in Gauss), resonance position B0 (in Gauss), 
    resonance width DeltaB (in Gauss), and background scattering length 
    abg (in a0 typically)."""
    return abg * (1 - DeltaB / (B - B0))


###
### p-Wave scattering parameters
###

def scattering_volume(B, B0, DeltaB, Vbg):
    return Vbg * (1- DeltaB / (B - B0))


def p_wave_phase_shift(k, V, R):
    """Computes the p-wave phase shift delta_1 (in radians) for a given 
    wavevector k, scattering volume V, and effective range R, all typically
    in bohr radius."""
    k3_cot_delta1 = -1/V + k**2 / R
    delta1 = np.arctan2(k**3, k3_cot_delta1)  # Handles correct quadrant.
    return delta1


###
### Quasi-1D scattering parameters
###

def inverse_odd_scattering_length(V3D, R3D, a_osc):
    return a_osc**2/2 * (1/V3D + 2/R3D) - 2/a_osc * zeta_m1_2


def odd_scattering_length(V3D, R3D, a_osc):
    return 1 / inverse_odd_scattering_length(V3D, R3D, a_osc)


def odd_effective_range(R3D, a_osc):
    return a_osc**2 / (6*R3D)**2 + a_osc/4 * zeta_1_2


def inverse_odd_scattering_amplitude(k, inv_a_odd, r_odd):
    return -1 + 1j*inv_a_odd/k + 1j*r_odd*k  # - O(k^3)


def even_scattering_length(V3D, R3D, a_osc):
    return -a_osc/2 * (a_osc**2/(6 * V3D) + 2*a_osc/(3*R3D) \
                       + zeta_m1_2 + zeta_1_2)


def even_effective_range(R3D, a_osc):
    return a_osc**3/16 * (4*a_osc/(3*R3D) + zeta_1_2 + zeta_3_2)


def inverse_even_scattering_amplitude(k, a_even, r_even):
    return -1 - 1j*a_even*k + 1j*r_even*k**3  # - O(k^5)


###
### Quasi-2D scattering parameters
###