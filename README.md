# Stage_IA_Pour_Les_Convertisseurs_De_Puissance_YACOUBI_Fatima_Zahra
## **Présentation**
Ce dépôt contient les travaux réalisés dans le cadre de mon stage de fin d’études portant sur le développement d’une stratégie de contrôle basée sur l’intelligence artificielle pour un convertisseur DC-DC résonant LLC destiné à une application aéronautique.

L'objectif est de développer un contrôleur capable d'adapter automatiquement la fréquence de commutation afin de maintenir une tension de sortie de 28 V malgré les variations de la tension d'entrée.

La méthode proposée combine :
* la simulation électrique du convertisseur sous LTspice ;
* l'automatisation des simulations avec Python ;
* le développement d'un modèle de substitution basé sur un réseau de neurones multicouche (MLP) ;
* l'apprentissage par renforcement avec l'algorithme Soft Actor-Critic (SAC).

## **Workflow du projet**
<p align="center">
  <img src="Workflow du projet.png" alt=Schéma du projet" witdth="600">
</p>

## **Conditions de fonctionnement**

| **Paramètre** | **Valeur** |

| :--- | :--- | :--- |

| `Tension d'entrée` | 240 V – 300 V |

| `Tension nominale` | 270 V |

| `Tension de sortie de référence` | 28 V |

| `Puissance nominale` | 28 V |

| `Fréquence de commutation` | 140 – 217 kHz |

## **Structure du dépôt**
.

├── data/

│   └── dataset.csv

│

├── ltspice/

│   └── LLC_converter.asc

│

├── simulation/

│   ├── variation_tension_fréquence.py

│   ├── création_dataset.py

│   └── séparation_bases_train_validation_train.py

│

├── surrogate_model/

│   ├── surrogate_model.py

│   ├── model.pth

│   ├── scaler_x.pkl

│   └── scaler_y.pkl

│

├── reinforcement_learning/

│   ├── SAC.py

└── README.md
## **Modèle de substitution**

L'utilisation directe de LTspice pendant l'apprentissage par renforcement est optimale en temps de calcul.

Un réseau de neurones MLP est donc entraîné pour approximer la relation : 
(Vin , Fsw) → Vout

Architecture utilisée :
2 → 64 → 64 → 32 → 1
Le modèle entraîné est ensuite utilisé comme environnement rapide pour l'entraînement de l'agent SAC.

## **Apprentissage par renforcement**

Le contrôleur repose sur l'algorithme Soft Actor-Critic (SAC).
État observé :

[Vin , Vout , erreur , Fsw]

Action :
Fsw ∈ [140 ; 217] kHz

Objectif :

Maintenir Vout ≈ 28 V

Le contrôleur est entraîné sur 50 000 interactions avec l'environnement.

## **Résultats**
Les tests réalisés montrent que l'agent SAC est capable :

* d'adapter automatiquement la fréquence de commutation ;
* de maintenir la tension de sortie à proximité de 28 V ;
* d'obtenir une erreur maximale inférieure à 0,4 % sur la plage de fonctionnement étudiée.

## **Technologies utilisées**
* Python
* LTspice
* PyTorch
* Gymnasium
* Stable-Baselines3
* SAC
* NumPy
* Pandas
* Matplotlib
  
## **Références**

### **Soft Actor-Critic**
* Article sur SAC : https://proceedings.mlr.press/v80/haarnoja18b.html
* Implémentation : https://github.com/haarnoja/sac
### **Stable-Baselines3**
* Dépôt GitHub officiel : https://github.com/DLR-RM/stable-baselines3
* Implémentation SAC : https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/sac
* Code source  de SAC : https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py 
