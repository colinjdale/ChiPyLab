import numpy as np
from amo.constants import h, mK


lambda_lattice = 760.6e-9  # Lattice wavelength in m


def recoil_energy(lambda_=lambda_lattice):
    """Computes the recoil energy in Joules for a given wavelength 
    lambda (in meters)."""
    return h**2 / (2 * mK * lambda_**2)


def lattice_hamiltonian(V0, q, FO=18):
    """Hamiltonian matrix in the plane wave basis for a particle in a periodic
    potential. 
    The wavefunction in the plane wave basis is expanded in Fourier modes
    $$\psi(x) = \sum_{n=-N}^N c_n e^{i(q+2n)x}\,,$$
    where x is the position, q is the quasi-momentum and n is the mode index with
    coefficient c_n. 
    Arguments:
        V0 (float) -    the potential depth
        q (float) -     the quasi-momentum
        FO (int) -      number of Fourier orders
    Returns a numpy matrix that is 2FO-1 x 2FO-1. FO defaults to 18."""
    N = 2 * FO - 1
    H = np.zeros((N, N), dtype=float)
    
    for i in range(N):
        for j in range(N):
            diff = i - j
            
            if diff == 0:
                # diagonal element
                H[i, j] = V0/2 + (q + 2*(i + 1 - FO))**2
            elif diff == 1 or diff == -1:
                # off-diagonals
                H[i, j] = -V0/4
            else:
                H[i, j] = 0.0

    return H


def band_energies(s, q, FO):
    """Returns a sorted list of eigenvalues for lattice Hamiltonian at
    quasi-momentum q, lattice depth s and 2FO-1 matrix elements."""
    H = lattice_hamiltonian(s, q, FO)
    eigvals = np.linalg.eigvalsh(H)   # For symmetric/hermitian matrices.
    eigvals = np.sort(eigvals)
    
    # Set tiny values (|x| < 1e-10) to zero.
    eigvals[np.abs(eigvals) < 1e-10] = 0.0
    
    return eigvals

