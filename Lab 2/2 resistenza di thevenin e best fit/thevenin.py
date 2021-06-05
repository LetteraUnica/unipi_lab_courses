import numpy as np
import matplotlib.pyplot as plt

# Metodo thevenin

V0 = 5.0
dV0 = 0.03
R1 = 76.7
R2 = 30.9
dR1 = 0.68
dR2 = 0.4
Rl = 1/R1 + 1/R2
Rl = 1/Rl
dRl = (dR2*R1**2 + dR1*R2**2)/((R1+R2)**2)

Vl = 2.65
dVl = 0.02

Rg = Rl*(V0/Vl-1)
dRg = (V0/Vl-1)*dRl + (Rl/Vl)*dV0 + ((Rl*V0)/Vl**2)*dVl

print(Rg, dRg)