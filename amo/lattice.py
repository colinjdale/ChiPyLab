import numpy as np
from amo.constants import h, mK


lambda_lattice = 760.6e-9  # Lattice wavelength in m.


def recoil_energy(lambda_=lambda_lattice, m=mK):
    """Computes the recoil energy in Joules for a given wavelength 
    lambda (in meters)."""
    return h**2 / (2 * m * lambda_**2)


def harmonic_frequency(s, lambda_=lambda_lattice, m=mK):
    return recoil_energy(lambda_, m) * 2 * np.sqrt(s) / h


def lattice_hamiltonian(s, q=0.0, FO=18):
    r"""Hamiltonian matrix in the plane wave basis for a particle in a periodic
    potential. The wavefunction in the plane wave basis is expanded in $N$ Fourier modes,
    and is given by
    
    .. math::
        \psi(x) = \sum_{n=-N}^N c_n e^{i(q+2n)x}\,,

    where $x$ is the position, $q$ is the quasi-momentum and $n$ is the mode index with
    coefficient $c_n$. The Hamiltonian is then

    .. math::
        \mathcal{H} = ...

    Arguments:
        s (float) -    the potential depth (in E_R)
        q (float) -     the quasi-momentum (default is 0.0)
        FO (int)  -      number of Fourier orders (default is 18)
    Returns a numpy matrix that is 2FO-1 x 2FO-1. """
    N = 2 * FO - 1
    H = np.zeros((N, N), dtype=float)
    
    for i in range(N):
        for j in range(N):
            diff = i - j
            
            if diff == 0:
                # diagonal element
                H[i, j] = s/2 + (q + 2*(i + 1 - FO))**2
            elif diff == 1 or diff == -1:
                # off-diagonals
                H[i, j] = -s/4
            else:
                H[i, j] = 0.0

    return H


def band_energies(H):
    """Returns a sorted list of eigenvalues for lattice Hamiltonian H."""
    eigvals = np.linalg.eigvalsh(H)   # For symmetric/hermitian matrices.
    eigvals = np.sort(eigvals)
    
    # Set tiny values (|x| < 1e-10) to zero.
    eigvals[np.abs(eigvals) < 1e-10] = 0.0
    return eigvals


class Lattice:
    """Object describing an optical lattice potential of depth s (in ER)
    with wavelength lambda_ (default is lambda_lattice).
    Computes Hamiltonian matrix with FO Fourier orders (default 18) at quasi
    momentum q (detault 0). Lattice.En is then the list of band energies,
    so the ground band energy is En[0], etc. All band energies given in ER."""

    def __init__(self, s, q=0.0, FO=18, lambda_=lambda_lattice):
        self.lambda_ = lambda_
        self.ER = recoil_energy(lambda_=lambda_lattice)

        self.s = s
        self.FO = 18
        self.N = 2 * FO - 1
        self.H = lattice_hamiltonian(s, q, FO)
        self.En = band_energies(self.H)
        self.harmonic_energy_gap = harmonic_frequency(s, lambda_=lambda_lattice)*h/self.ER


    def band_gap(self, i, j):
        """Computes the band gap between bands i < j."""
        if j < i:
            raise ValueError("Band index j must be >= i.")
        return self.En[j] - self.En[i]


    def calculate_En(self, q, FO=18):
        """Calculates H and E_n at a new q."""
        H = lattice_hamiltonian(self.s, q, FO)
        E_n = band_energies(H)
        return E_n

