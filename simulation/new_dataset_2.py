import os
import re
import subprocess
import time

import numpy as np
import pandas as pd

from PyLTSpice import RawRead


LTSPICE = r"C:\Program Files\ADI\LTspice\LTspice.exe"

NET_ORIGINE = (r"C:\Users\fyacoubi\LLC_IA\modèle_avec_diodes_1.net")

DATASET = (r"C:\Users\fyacoubi\LLC_IA\time_test_dataset.csv")

N_SIMULATIONS = 100

VIN_MIN = 240.0
VIN_MAX = 300.0

MARGE_FSW = 50.0




Vin_reference = np.array([
    240.0,
    250.0,
    260.0,
    270.0,
    280.0,
    290.0,
    300.0
])

Fsw_reference = np.array([
    147.0,
    154.0,
    160.0,
    179.0,
    191.0,
    203.0,
    214.0
])



colonnes = [
    "Vin_V",
    "Fsw_kHz",
    "Vout_mean_V",
    "Vout_max_steady_V",
    "Vout_min_steady_V",
    "Ripple_percent",
    "Overshoot_percent",
    "Static_error_V",
    "Static_error_percent"
]


if os.path.exists(DATASET):

    df_dataset = pd.read_csv(DATASET,sep=";")

    print(f"Dataset existant : {len(df_dataset)} échantillons")

else:

    df_dataset = pd.DataFrame(columns=colonnes)

    print("Aucun dataset existant.")
    print("Création d'un nouveau dataset.")



def creer_cle(Vin, Fsw): # ça évite les problèmes liés aux nombres flottants par exemple vin = 257.31001 on met vin = 257.31

    return (
        round(float(Vin), 2),
        round(float(Fsw), 2)
    )


points_existants = set()

for _, ligne in df_dataset.iterrows():

    cle = creer_cle(ligne["Vin_V"],ligne["Fsw_kHz"])

    points_existants.add(cle)


print(f"Points déjà présents : {len(points_existants)}")



def generer_point():

   
    # Vin aléatoire

    Vin = np.random.uniform(VIN_MIN,VIN_MAX)
    Fsw = np.random.uniform( 100, 240)

 
    # Arrondi
    Vin = round(Vin, 2)
    Fsw = round(Fsw, 2)
    return Vin, Fsw


# GENERATION DES 100 NOUVEAUX POINTS


nouveaux_points = []
tentatives = 0
MAX_TENTATIVES = N_SIMULATIONS * 20


while len(nouveaux_points) < N_SIMULATIONS:

    tentatives += 1

    if tentatives > MAX_TENTATIVES:

        print(
            "Impossible de générer suffisamment "
            "de nouveaux points."
        )

        break

    Vin, Fsw = generer_point()

    cle = creer_cle(Vin,Fsw)

    # Vérification doublon


    if cle in points_existants:

        continue

    # Ajout du nouveau point

    points_existants.add(cle)

    nouveaux_points.append(
        (Vin, Fsw)
    )


print("\n---------------------")
print("NOUVEAUX POINTS")
print("-----------------------")

print(f"{len(nouveaux_points)} nouveaux points générés.")


# LECTURE DE LA NETLIST ORIGINALE

with open(
    NET_ORIGINE,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    texte_original = f.read()


#calcul de temps 

debut_generation = time.perf_counter()

# BOUCLE DE SIMULATION


for numero, (Vin, Fsw) in enumerate(nouveaux_points,start=1):

    print("\n")
    print("-----------------")
    print(
        f"SIMULATION {numero}/{len(nouveaux_points)}"
    )
    print("----------------")

    print(f"Vin = {Vin:.2f} V")

    print(f"Fsw = {Fsw:.2f} kHz")


   
    # CREATION DE LA NETLIST
  

    texte = texte_original


 
    # Vin
  

    texte = re.sub(
        r"Vin=[^\s]+",
        f"Vin={Vin}",
        texte
    )


  
    # Fsw
  

    texte = re.sub(r"frequency=[^\s]+",f"frequency={Fsw}k",texte)


 
    # noms des fichiers 
  

    dossier = os.path.dirname(NET_ORIGINE)

    nom_simulation = (
        f"LLC_"
        f"Vin_{Vin:.2f}_"
        f"Fsw_{Fsw:.2f}"
    )

    netlist = os.path.join(
        dossier,
        nom_simulation + ".net"
    )

    raw = os.path.join(
        dossier,
        nom_simulation + ".raw"
    )

    log = os.path.join(
        dossier,
        nom_simulation + ".log"
    )


    # ECRITURE NETLIST


    with open(
        netlist,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(texte)


    # SUPPRESSION D'ANCIENS RESULTATS

    for fichier in [raw, log]:

        if os.path.exists(fichier):

            try:
                os.remove(fichier)

            except PermissionError:

                print(
                    "Impossible de supprimer :",
                    fichier
                )

                continue


    # LANCEMENT LTSPICE


    try:

        process = subprocess.Popen([
            LTSPICE,
            "-b",
            netlist
        ])

    except Exception as e:
        print("Erreur lancement LTspice :",e)
        continue


    # ATTENTE DE LA FIN DE SIMULATION

    try:

        process.wait(timeout=60)

    except subprocess.TimeoutExpired:

        print("Simulation trop longue.")

        process.kill()
        
        stdout, stderr = process.communicate()
        
        if os.path.exists(raw):
            os.remove(raw)
        continue


    #Attendre la fin proprement 
    
    # VERIFICATION

    if process.returncode != 0:

        print("Erreur LTspice.")
        
        #nettoyage de sécurité en cas de cash de ltspice
        
        if os.path.exists(raw):
            os.remove(raw)
        continue


    # ATTENTE DU RAW

    timeout = 10

    t0 = time.time()

    while not os.path.exists(raw):

        if time.time() - t0 > timeout:

            print( "RAW non créé.")

            break

        time.sleep(0.2)


    if not os.path.exists(raw):

        continue


    print("RAW détecté.")



    # LECTURE DU RAW


    try:

        lecture = RawRead(raw)

        temps = lecture.get_trace("time").get_wave()

        vout = lecture.get_trace("V(vout)").get_wave()

    except Exception as e:

        print("Erreur lecture RAW :",e)

        continue
    
    finally:
        
        # Cela garantit que le fichier .raw est supprimé avant de passer au tour suivant.
        if os.path.exists(raw):
            try:
                os.remove(raw)
                print("Fichier RAW temporaire supprimé.")
            except Exception as e:
                print("Impossible de supprimer le fichier RAW courant :", e)



    # REGIME PERMANENT


    mask = (
        (temps >= 0.6e-3)
        &
        (temps <= 0.7e-3)
    )

    vout_steady = vout[mask]


    if len(vout_steady) == 0:
        print("Aucune donnée dans la fenêtre.")
        continue



    # CALCUL DES METRIQUES

    vout_mean = np.mean(vout_steady)
    vout_max = np.max(vout_steady)
    vout_min = np.min(vout_steady)


   
    # Ripple
 
    ripple = ((vout_max - vout_min)/vout_mean*100)

    # Erreur statique par rapport à 28 V

    static_error = (vout_mean - 28.0)
    static_error_percent = (abs(static_error)/28.0*100)



    # Overshoot par rapport à 28V


    overshoot = (
        max(0,(vout_max - 28.0))/28.0*100)


    # CREATION DU RESULTAT
   
    resultat = {

        "Vin_V":
            Vin,

        "Fsw_kHz":
            Fsw,

        "Vout_mean_V":
            vout_mean,

        "Vout_max_steady_V":
            vout_max,

        "Vout_min_steady_V":
            vout_min,

        "Ripple_percent":
            ripple,

        "Overshoot_percent":
            overshoot,

        "Static_error_V":
            static_error,

        "Static_error_percent":
            static_error_percent
    }



    # AJOUT AU DATASET
  

    df_nouveau = pd.DataFrame([resultat])
    df_dataset = pd.concat([df_dataset,df_nouveau],ignore_index=True)




    # SAUVEGARDE 
    # on sauvegarde après CHAQUE simulation.
   

    df_dataset.to_csv(DATASET,sep=";",index=False)


  
    # AFFICHAGE


    print(f"Vout moyen = {vout_mean:.3f} V")
    print(f"Ripple = {ripple:.3f} %")

    print(f"Erreur statique = "f"{static_error:.3f} V")

    print(f"Dataset total = "f"{len(df_dataset)} points")


# FIN 
#fin de temps 
fin_generation = time.perf_counter()
temps_total = (fin_generation - debut_generation)

temps_moyen = (temps_total / len(nouveaux_points))

print("\n")
print("------------------")
print("LOT TERMINE")
print("------------------")

print(
    f"Dataset contient maintenant "
    f"{len(df_dataset)} échantillons."
)
print(
    f"Temps total : {temps_total:.2f} s"
)
print(
    f"Temps total : {temps_total/60:.2f} min"
)
print(
    f"Temps moyen par simulation : "
    f"{temps_moyen:.2f} s"
)
print(f"Fichier : {DATASET}")