import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.odr import odrpack


# Funzione di fit
def linear(b, x) :
    return b[0]*x + b[1]

def retta(x, a, b) :
    return a*x + b


# Creazione array medie e errori
# Riceve documenti txt nella forma "nome_documento"+i+j
medie = np.array([])
errori = np.array([])
j = 1
while j < 3 :       #Numero cilindro
    i = 1
    while i < 15 :  #Numero foro
        n = str(j) + str(i)
        percorso = "/Users/Lorenzo/Desktop/Relazioni fisica/4) Conducibilità termica/Nuova relazione/Temperature/temp" + n + ".txt"
        array_temperature = np.genfromtxt(percorso, unpack = True, skip_header = 4)
        
        array_temp1 = np.array([array_temperature[0], array_temperature[1]])
        array_temp2 = np.array([array_temperature[2], array_temperature[3]])
        errore_temp1 = np.std(array_temp1[1])
        errore_temp2 = np.std(array_temp2[1])
        media_temp1 = np.average(array_temp1[1])
        media_temp2 = np.average(array_temp2[1])
        medie = np.append(medie, media_temp1)
        errori = np.append(errori, errore_temp1)
        print(np.round(media_temp1, decimals = 2) , "+-" , np.round(errore_temp1, decimals = 2) , " " , np.round(media_temp2, decimals = 2) , "+-" , np.round(errore_temp2, decimals = 2))
        
        i = i + 1
    print("\n")
    j = j + 1

# Analisi dati
n_dati = int(len(medie) / 2)                            #Numero dati analizzati
x = (np.linspace(0, 13, n_dati) * 2.5) + 2.5
media_nonisolato = medie[0:n_dati]                      #medie cilindro non isolato
media_isolato = medie[n_dati:]                          #medie cilindro isolato
errore_nonisolato = errori[0:n_dati]                    #errori cilindro non isolato
errore_isolato = errori[n_dati:]                        #errori cilindro isolato
errorex = np.ones(n_dati) * 0.1                         #errore sulle lunghezze

# Fit cilindro non isolato
ddof = 2
model = odrpack.Model(linear)
data = odrpack.RealData(x, media_nonisolato, sx=errorex, sy=errore_nonisolato)
odr = odrpack.ODR(data, model, beta0=(1., 1.))
out = odr.run()
popt_nonisolato, pcov_nonisolato = out.beta, out.cov_beta
chi2_nonisolato = out.sum_square

print("Parametri ottimali: {}\n".format(popt_nonisolato))
print("Errori parametri: {}\n".format(np.sqrt(np.diagonal(pcov_nonisolato))))
print("Chi2: {}, aspettato {}\n".format(chi2_nonisolato, n_dati-ddof))

# Fit cilindro isolato
ddof = 2
model = odrpack.Model(linear)
data = odrpack.RealData(x, media_isolato, sx=errorex, sy=errore_isolato)
odr = odrpack.ODR(data, model, beta0=(1., 1.))
out = odr.run()
popt_isolato, pcov_isolato = out.beta, out.cov_beta
chi2_isolato = out.sum_square

print("Parametri ottimali: {}\n".format(popt_isolato))
print("Errori parametri: {}\n".format(np.sqrt(np.diagonal(pcov_isolato))))
print("Chi2: {}, aspettato {}\n".format(chi2_isolato, n_dati-ddof))

# Parte grafici
x1 = (np.linspace(-2, 15, 1000) * 2.5) + 2.5   
# Creazione grafico temperature cilindro non isolato
plt.figure()
plt.title("Cilindro non isolato")
plt.errorbar(x, media_nonisolato, errore_nonisolato, errorex, fmt = 'o')
plt.plot(x1, retta(x1, *popt_nonisolato))
plt.grid()
plt.ylabel('Temperatura [°C]')
plt.xlabel('Distanza [cm]')
plt.savefig("Cilindro non isolato.png", bbox_inches='tight')
plt.show()

# Creazione grafico temperature cilindro isolato
plt.figure()
plt.title("Cilindro isolato")
plt.errorbar(x, media_isolato, errore_isolato, errorex, fmt = 'o')
plt.plot(x1, retta(x1, *popt_isolato))
plt.grid()
plt.ylabel('Temperatura [°C]')
plt.xlabel('Distanza [cm]')
plt.savefig("Cilindro isolato.png", bbox_inches='tight')
plt.show()