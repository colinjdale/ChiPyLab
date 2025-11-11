import os
import pandas as pd
import numpy as np

# Load tabulated data from
# Thermodynamics of unitary Fermi gas
# Haussmann, Rantner, Cerrito, Zwerger 2007; and Enss, Haussmann, Zwerger 2011
# Density:  n=k_F^3/(3\pi^2)
# columns:  T/T_F, mu/E_F, u/(E_F*n), s/(k_B*n), p/(E_F*n), C/k_F^4
df = pd.read_csv(os.path.join('..', 'data', 'luttward-thermodyn.txt'), skiprows=4, sep=' ')
xlabel = 'T/T_F'
ylabel = 'C/(k_F*n)'

# Calculate contact density
df[ylabel] = df['C/k_F^4'] * 3*np.pi**2  # contact density c/(k_F n) = C/k_F^4 * (3 pi^2)

def contact_density(ToTF):
    """Functions that computes conatct density C/(k_F n) using
       tabulated data from Haussmann, Rantner, Cerrito, Zwerger 2007; 
       and Enss, Haussmann, Zwerger 2011. 
       Note data is tabulated as C/k_F^2, but we have used n=k_F^3/(3\pi^2)
       to convert to density."""
    return np.interp(ToTF, df[xlabel], df[ylabel])