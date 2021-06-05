import numpy as np

zeri_pendoli = np.genfromtxt("C:/Users/Lorenzo/Desktop/Relazioni fisica/Oscillatore smorzato cavlong/Dati/ZeriPendoli.txt", skip_header = 4, unpack = True)

media_pen1 = np.average(zeri_pendoli[1])
media_pen2 = np.average(zeri_pendoli[3])
errore_pen1 = np.std(zeri_pendoli[1])
errore_pen2 = np.std(zeri_pendoli[3])

print("Media primo pendolo: {} +- {}\n".format(media_pen1, errore_pen1))
print("Media secondo pendolo: {} +- {}\n".format(media_pen2, errore_pen2))