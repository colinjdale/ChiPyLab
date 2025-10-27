import numpy as np

from amo.constants import R_max, V_bar
from scipy.special import zeta
from scipy.optimize import root_scalar

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


def inv_R_vdW(V, R_max=R_max, V_bar=V_bar):
    """Van der Waals limit of effective range.
    From B. Gao, PRA 84, 022706 (2011)."""
    return 1/R_max * (1 + 2*(V_bar/V) + 2*V_bar**2/V**2)


def inverse_effective_range(V, R0, Vbg, R_max=R_max):
    """From D.J.M. Ahmed-Braun et al. PRR 3, 033269 (2021)."""
    width_param = -R0/(R_max - R0)
    return -1/(R_max * width_param) * (1-Vbg/V)**2 + inv_R_vdW(V)


def p_wave_phase_shift(k, V, inv_R):
    """Computes the p-wave phase shift delta_1 (in radians) for a given 
    wavevector k, scattering volume V, and effective range R, all typically
    in bohr radius."""
    k3_cot_delta1 = -1/V + k**2 * inv_R
    delta1 = np.arctan2(k**3, k3_cot_delta1)  # Handles correct quadrant.
    return delta1


###
### Quasi-1D scattering
###

def q1d_inverse_odd_scattering_length(V3D, inv_R3D, a_osc):
    return 1/6 * (a_osc**2/V3D + 2*inv_R3D) - 2/a_osc * zeta_m1_2


def q1d_odd_scattering_length(V3D, inv_R3D, a_osc):
    return 1 / q1d_inverse_odd_scattering_length(V3D, inv_R3D, a_osc)


def q1d_odd_effective_range(inv_R3D, a_osc):
    return a_osc**2 * inv_R3D / 6 + a_osc/4 * zeta_1_2


def q1d_inverse_odd_scattering_amplitude(k, inv_a_odd, r_odd):
    return -1 + 1j*inv_a_odd/k + 1j*r_odd*k  # - O(k^3)


def q1d_even_scattering_length(V3D, inv_R3D, a_osc):
    return -a_osc/2 * (a_osc**3/(6 * V3D) + 2*inv_R3D*a_osc/3 \
                       + zeta_m1_2 + zeta_1_2)


def q1d_even_effective_range(inv_R3D, a_osc):
    return a_osc**3/16 * (4*inv_R3D*a_osc/3 + zeta_1_2 + zeta_3_2)


def q1d_inverse_even_scattering_amplitude(k, a_even, r_even):
    return -1 - 1j*a_even*k + 1j*r_even*k**3  # - O(k^5)


###
### Quasi-2D scattering parameters
###