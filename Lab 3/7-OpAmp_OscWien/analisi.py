import numpy as np
import menzalib as mz
import pylab as pl

Vs, dVs = 0.512/2, mz.dVosc(0.512)/2
f, Vapp = np.genfromtxt("dati/1guadagno.txt", unpack=True)
Va, dVa = Vapp/2, mz.dVosc(Vapp)/2
df = f*0.01
Av, dAv = Va/Vs, Va/Vs*0.04
print(dAv/Av)

pl.figure()
pl.subplot(211)
pl.errorbar(f, 20*np.log10(Av), 20*mz.dlog10(Av,dAv), df, fmt='.', label="dati")
pl.xscale("log")
pl.xlabel("Frequenza [Hz]")
pl.ylabel("Guadagno [dB]")
pl.legend()
#pl.show()

f, t = np.genfromtxt("dati/1fase.txt", unpack=True)
t=t*10**-6
df, dt = f*0.01, mz.dtosc(t)
w, dw = f*2*np.pi, df*2*np.pi
fase, dfase = w*t, mz.dprod(w, dw, t, dt)

pl.subplot(212)
pl.errorbar(f, fase*180/np.pi, dfase*180/np.pi, df, fmt='.', label="dati")
pl.xscale("log")
pl.xlabel("Frequenza [Hz]")
pl.ylabel("Fase [gradi]")
pl.legend()
pl.savefig("figure/grafico.png")
#pl.show()
pl.close()