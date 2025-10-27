from scipy.constants import c, pi, h, hbar, k as kB, m_e, mu_0, e, epsilon_0
from scipy.constants import physical_constants as pc

from math import gamma


###
### Fundamental constants
###

uatom = pc["atomic mass constant"][0]  # Atomic mass unit (kg)
a_0 = pc["Bohr radius"][0]  # Bohr radius (m)
gS = -pc["electron g factor"][0]  # Electron g-factor, NOTE THE MINUS SIGN
mu_B = pc["Bohr magneton"][0]  # Bohr magneton (J/T)


###
### Potassium-40 specific constants
###

mK = 39.96399848 * uatom  # Mass of potassium-40 (kg)
I = 4  # Nuclear spin of potassium-40

ahf = -h * 285.7308e6  # For groundstate
gI = 0.000176490  # Total nuclear g-factor  
# gJ = 2.00229421  # For groundstate measured value 
gJ = gS  # For theoretical value 

### D1 and D2 line parameters
LambdaD1 = 770.108136507e-9  # (m)
LambdaD2 = 766.700674872e-9  # (m)
NuD1, NuD2 = (c / LambdaD1, c / LambdaD2)
kD1, kD2 = (2 * pi / LambdaD1, 2 * pi / LambdaD2)
GammaD1 = 2 * pi * 6.035e6  # (Hz)
GammaD2 = 2 * pi * 6.035e6  # (Hz)


### Van der Waals parameters
C6 = 3897
r0 = 1/8**(1/2) * gamma(3/4)/gamma(5/4) * ((mK * C6)/hbar**2)**(1/4)
r0 = 60 * a_0


### Feshbach resonance parameters

# s-wave resonance at 202.1 G between |9/2,-9/2> and |9/2,-7/2>
abg_97 = 167 * a_0 
DeltaB_97 = 6.9  # (G)
B0_97 = 202.10  # (G)
Bzero_97 = 209.115  # (G)

# s-wave resonance at 224.2 G between |9/2,-9/2> and |9/2,-5/2>
abg_95 = 174 * a_0
B0_95 = 224.2  # (G)
DeltaB_95 = 7.2  # (G)

# p-wave resonance at 198.3 and 198.8 G between |9/2,-7/2> and |9/2,-7/2>
B0_77_pm1 = 198.3  # (G)
Vbg_77_pm1 = -(107.35)**3 * a_0**3  # (m^3)
DeltaB_77_pm1 = -19.54  # (G)
R0_77_pm1 = 48.9* a_0  # (m)

B0_77_0 = 198.803  # (G)
Vbg_77_0 = -(108.0)**3 * a_0**3  # (m^3)
DeltaB_77_0 = -19.89  # (G)
R0_77_0 = 49.4 * a_0  # (m)



