import os
import re
import time
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from PyLTSpice import RawRead


# CHEMINS


LTSPICE = r"C:\Program Files\ADI\LTspice\LTspice.exe"

NET_ORIGINE = (
    r"C:\Users\fyacoubi\OneDrive - Capgemini\LLC_IA\modèle_avec_diodes_1.net"
)

DOSSIER = os.path.dirname(NET_ORIGINE)



# ENTREES UTILISATEUR


Vin = float(input("Vin (V) = "))
Fsw = float(input("Fsw (kHz) = "))

print("\n-------------------------------")
print(f"Vin = {Vin:.2f} V")
print(f"Fsw = {Fsw:.2f} kHz")
print("---------------------------------\n")



# LECTURE DE LA NETLIST

with open(
    NET_ORIGINE,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    texte = f.read()


# MODIFICATION DES PARAMETRES


texte = re.sub(
    r"Vin=[^\s]+",
    f"Vin={Vin}",
    texte
)

texte = re.sub(
    r"frequency=[^\s]+",
    f"frequency={Fsw}k",
    texte
)



# CREATION DE LA NOUVELLE NETLIST


nom_simulation = (f"LLC_Vin_{Vin:.2f}_Fsw_{Fsw:.2f}")

netlist = os.path.join(
    DOSSIER,
    nom_simulation + ".net"
)

raw = os.path.join(
    DOSSIER,
    nom_simulation + ".raw"
)

log = os.path.join(
    DOSSIER,
    nom_simulation + ".log"
)

with open(
    netlist,
    "w",
    encoding="utf-8"
) as f:

    f.write(texte)



# SUPPRESSION ANCIENS RESULTATS


for fichier in [raw, log]:
    if os.path.exists(fichier):
        os.remove(fichier)


# LANCEMENT LTSPICE

print("Simulation LTspice en cours...\n")
process = subprocess.Popen(
    [
        LTSPICE,
        "-b",
        netlist
    ]
)

process.wait()

print("Code retour LTspice :", process.returncode)

if process.returncode != 0:
    print("Erreur LTspice")
    if os.path.exists(log):
        with open(
            log,
            "r",
            errors="ignore"
        ) as f:
            print(f.read())
    raise SystemExit


# ATTENTE DU RAW


timeout = 30

t0 = time.time()
while not os.path.exists(raw):
    if time.time() - t0 > timeout:
        raise Exception("Le fichier RAW n'a pas été créé.")

    time.sleep(0.2)

print("RAW détecté.\n")


# LECTURE DU RAW


lecture = RawRead(raw)
temps = lecture.get_trace("time").get_wave()
vout = lecture.get_trace("V(vout)").get_wave()


# CALCUL VOUT MOYEN


mask = ((temps >= 0.6e-3) & (temps <= 0.7e-3))

vout_steady = vout[mask]
vout_mean = np.mean(vout_steady)
vout_max = np.max(vout_steady)
vout_min = np.min(vout_steady)

ripple = ((vout_max - vout_min)/vout_mean * 100)

print(f"Vout moyen : {vout_mean:.3f} V")
print(f"Vout max   : {vout_max:.3f} V")
print(f"Vout min   : {vout_min:.3f} V")
print(f"Ripple     : {ripple:.3f} %")



# AFFICHAGE DE LA COURBE


plt.figure(figsize=(9, 5))

plt.plot(
    temps * 1e3,
    vout,
    linewidth=2
)

plt.xlabel("Temps (ms)")
plt.ylabel("Vout (V)")

plt.title(f"Vout(t) | Vin={Vin:.2f} V | Fsw={Fsw:.2f} kHz")

plt.grid(True)
plt.tight_layout()
plt.show()