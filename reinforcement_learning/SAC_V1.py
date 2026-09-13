import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env




# Tension de référence 
VREF = 28.0       # V

# Plage de tension d'entrée étudiée
VIN_MIN = 240.0   # V
VIN_MAX = 300.0   # V

# Plage de fréquence utilisée par le contrôleur


FSW_MIN = 140.0   # kHz
FSW_MAX = 217.0   # kHz


# Nombre de pas maximum par épisode
MAX_STEPS = 30



# PATHS DES MODELES


MODEL_PATH = "model.pth"
SCALER_X_PATH = "scaler_x.pkl"
SCALER_Y_PATH = "scaler_y.pkl"



# VERIFICATION DES FICHIERS


print("=" * 60)
print("VERIFICATION DES FICHIERS")
print("=" * 60)

print("model.pth :", os.path.exists(MODEL_PATH))
print("scaler_x.pkl :", os.path.exists(SCALER_X_PATH))
print("scaler_y.pkl :", os.path.exists(SCALER_Y_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier {MODEL_PATH} est introuvable.")

if not os.path.exists(SCALER_X_PATH):
    raise FileNotFoundError(f"Le fichier {SCALER_X_PATH} est introuvable.")

if not os.path.exists(SCALER_Y_PATH):
    raise FileNotFoundError(f"Le fichier {SCALER_Y_PATH} est introuvable.")


# DEFINITION DU MLP


class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(2, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)


# CHARGEMENT DU SURROGATE


print("\n" + "=" * 60)
print("CHARGEMENT DU SURROGATE MLP")
print("=" * 60)


device = torch.device("cpu")

surrogate = MLP().to(device)


# Chargement des poids
state_dict = torch.load(MODEL_PATH,map_location=device)

surrogate.load_state_dict(state_dict)

surrogate.eval()


print("MLP chargé avec succès.")




#  CHARGEMENT DES SCALERS

print("\nChargement des scalers...")
scaler_x = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)
print("Scalers chargés avec succès.")



# FONCTION DE PREDICTION DU SURROGATE : Vin et Fsw en entrée , et Vout en sortie 




def predict_vout(vin, fsw):

    # Création du vecteur d'entrée
    X = np.array([[vin, fsw]],dtype=np.float64)

    # Normalisation avec LE MEME scaler utilisé pendant l'entrainement du MLP
    X_scaled = scaler_x.transform(X)

    # Conversion en tenseur PyTorch
    X_tensor = torch.tensor(X_scaled,dtype=torch.float32)

    # Pas de calcul de gradient
    with torch.no_grad():

        y_scaled = surrogate(X_tensor).numpy()

    # Retour à l'échelle réelle
    y = scaler_y.inverse_transform(y_scaled)

    return float(y[0, 0])



#######  TEST DU SURROGATE
"""
print("\n" + "=" * 60)
print("TEST DU SURROGATE")
print("=" * 60)


test_vin = 270.0
test_fsw = 176.0

test_vout = predict_vout(test_vin,test_fsw)

print(f"Vin = {test_vin:.2f} V")

print(f"Fsw = {test_fsw:.2f} kHz")

print(f"Vout prédite = {test_vout:.4f} V")"""


#ENVIRONNEMENT LLC 
class LLCEnv(gym.Env):

    """
    Environnement RL représentant le convertisseur LLC.

    Le convertisseur réel est remplacé par le surrogate MLP.

    Etat :
        [Vin_normalise,
         Vout_normalise,
         erreur_normalisee,
         Fsw_normalise]

    Action :
        Fsw normalisée entre -1 et +1

    Reward :
        récompense élevée lorsque Vout est proche de 28 V
        + pénalisation des variations brutales de fréquence.
    """

    metadata = {"render_modes": []} #ça veut dire que mon environnement ne sait pas produire une fenetre graphique , animation , une image ou un affichage texte particulier 
    
    def __init__(self):

        super().__init__()
        
        #espace d'observation. les variables ( Vin, Vout,erreur , fréquence précédente)
        
        self.observation_space = spaces.Box(low=-1.0,high=1.0,shape=(4,),dtype=np.float32)
        
        #espace d'action . SAC fournit une action entre -1 et 1 
        self.action_space = spaces.Box(low=-1.0,high=1.0,shape=(1,),dtype=np.float32)
        
        #variables internes 
        self.vin = None

        self.fsw = None

        self.previous_fsw = None

        self.vout = None

        self.error = None

        self.step_count = 0
        
        #conversion action -> fréquence 
    def action_to_frequency(self, action):#action SAC
        a = float(action[0])
        
        # sécurité
        
        a = np.clip(a,-1.0,1.0)
        fsw = (FSW_MIN+((a + 1.0) / 2.0)*(FSW_MAX - FSW_MIN))
        return fsw
         
         #normalisations 
         

    def normalize_vin(self, vin):
        return (2.0*(vin - VIN_MIN)/(VIN_MAX - VIN_MIN)- 1.0)


    def normalize_fsw(self, fsw):
        return (2.0*(fsw - FSW_MIN)/(FSW_MAX - FSW_MIN)- 1.0)


    def normalize_vout(self, vout):
        return np.clip((vout - VREF) / 10.0,-1.0,1.0) 

    def normalize_error(self, error):
        return np.clip(error / 10.0,-1.0,1.0)
    
    #OBSERVATION
    
    def get_observation(self):
        
        vin_norm = self.normalize_vin(self.vin)
        vout_norm = self.normalize_vout(self.vout)
        error_norm = self.normalize_error(self.error)
        fsw_norm = self.normalize_fsw(self.previous_fsw)
        observation = np.array([
                vin_norm,
                vout_norm,
                error_norm,
                fsw_norm
            ],

            dtype=np.float32
        )

        return observation

    # RESET


    def reset(self,seed=None,options=None):
        

        super().reset(seed=seed)

        ### Choix aléatoire de Vin
     

        self.vin = self.np_random.uniform(VIN_MIN,VIN_MAX)

        ### Fréquence initiale aléatoire

        self.fsw = self.np_random.uniform(FSW_MIN,FSW_MAX)
        self.previous_fsw = self.fsw

        
        ######################### Calcul Vout avec le surrogate model ###############

        self.vout = predict_vout(self.vin,self.fsw)


        # erreur

        self.error = VREF - self.vout
        self.step_count = 0
        observation = self.get_observation()


        info = {
            "Vin": self.vin,
            "Fsw_kHz": self.fsw,
            "Vout_V": self.vout,
            "error_V": self.error
        }


        return observation, info



    # STEP

    def step(self, action):


        # SAC choisit une nouvelle fréquence

        self.fsw = self.action_to_frequency(action)


        #  Le surrogate prédit Vout

        self.vout = predict_vout(self.vin,self.fsw)



        #  Calcul de l'erreur


        self.error = VREF - self.vout

        # Variation de fréquence
        
        delta_f = abs(self.fsw-self.previous_fsw)

        # REWARD


        # Erreur de tension
        
        voltage_penalty = abs(self.error)


        # Pénalisation des grandes variations de fréquence
        frequency_penalty = delta_f


        # Reward final

        reward = (-(voltage_penalty/28)**2 - 0.01 * (frequency_penalty/(FSW_MAX-FSW_MIN))**2)


        # Mise à jour

        self.previous_fsw = self.fsw
        self.step_count += 1


      
        # FIN DE L'EPISODE
       

        terminated = False
        truncated = (self.step_count>= MAX_STEPS)

        # NOUVEL ETAT

        observation = self.get_observation()


        info = {
            "Vin": self.vin,
            "Fsw_kHz": self.fsw,
            "Vout_V": self.vout,
            "error_V": self.error,
            "reward": reward
        }


        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )



#####CREATION DE L'ENVIRONNEMENT##########



print("CREATION DE L'ENVIRONNEMENT SAC")
env = LLCEnv()
print("Environnement créé")


######VERIFICATION DE L'ENVIRONNEMENT#######

print("\nVérification de l'environnement...")
check_env(env,warn=True)
print("Environnement valide.")



# TEST MANUEL DE L'ENVIRONNEMENT

print("TEST MANUEL")
obs, info = env.reset()
print(f"Vin initial = {info['Vin']:.2f} V")
print(f"Fsw initial = {info['Fsw_kHz']:.2f} kHz")
print(f"Vout initial = {info['Vout_V']:.2f} V")
print(f"Erreur initiale = {info['error_V']:.2f} V")


print("\n10 actions aléatoires :")


for i in range(10):

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(
        f"Step {i+1:02d} | " 
        f"Fsw = {info['Fsw_kHz']:.2f} kHz | "
        f"Vout = {info['Vout_V']:.3f} V | "
        f"Erreur = {info['error_V']:.3f} V | "
        f"Reward = {reward:.3f}"
    )



############# CREATION DU MODELE SAC ###########################


print("\n" + "-" * 60)
print("CREATION DU MODELE SAC")
print("-" * 60)


sac_model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    verbose=1,
    device="cpu"
)


print("Modèle SAC créé.")



##################" ENTRAINEMENT########################

print("\n" + "-" * 60)
print("DEBUT DE L'ENTRAINEMENT SAC")
print("-" * 60)


TOTAL_TIMESTEPS = 50000


sac_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True
)



# SAUVEGARDE

SAC_MODEL_PATH = "SAC_LLC_controller"


sac_model.save(SAC_MODEL_PATH)


print("\n" + "-" * 60)
print("ENTRAINEMENT TERMINE")
print("-" * 60)
print(f"Contrôleur sauvegardé : {SAC_MODEL_PATH}.zip")


# TEST DU CONTROLEUR SAC

print("\n" + "-" * 60)
print("TEST DU CONTROLEUR SAC")
print("-" * 60)

# Plusieurs tensions d'entrée fixes

VIN_TEST = [

    240.0,
    250.0,
    260.0,
    270.0,
    280.0,
    290.0,
    300.0

]

results = []

for vin_test in VIN_TEST:

    # Reset
    obs, info = env.reset()

    # On impose Vin

    env.vin = vin_test


    # Fréquence initiale

    env.fsw = (FSW_MIN + FSW_MAX) / 2.0
    env.previous_fsw = env.fsw


    # Calcul initial

    env.vout = predict_vout(
        env.vin,
        env.fsw
    )

    env.error = VREF - env.vout


    # Observation

    obs = env.get_observation()


    # Quelques étapes

    for step in range(20):
        action, _states = sac_model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        if truncated:
            break


    results.append({
        "Vin_V": vin_test,
        "Fsw_kHz": info["Fsw_kHz"],
        "Vout_V": info["Vout_V"],
        "Error_V": info["error_V"],
        "Abs_Error_V": abs(
            info["error_V"]
        )
    })


    print(
        f"Vin = {vin_test:.0f} V | "
        f"Fsw = {info['Fsw_kHz']:.2f} kHz | "
        f"Vout = {info['Vout_V']:.3f} V | "
        f"Erreur = {info['error_V']:.3f} V"
    )


#####AFFICHAGE DES RESULTATS#######


print("\n" + "=" * 60)
print("RESULTATS FINAUX")
print("=" * 60)


for result in results:
    print(
        f"Vin = {result['Vin_V']:.0f} V | "
        f"Fsw = {result['Fsw_kHz']:.2f} kHz | "
        f"Vout = {result['Vout_V']:.3f} V | "
        f"|Erreur| = {result['Abs_Error_V']:.3f} V"
    )



########GRAPHE VOUT#########

vin_values = [
    r["Vin_V"]
    for r in results
]

vout_values = [
    r["Vout_V"]
    for r in results
]

fsw_values = [
    r["Fsw_kHz"]
    for r in results
]

error_values = [
    r["Error_V"]
    for r in results
]

plt.figure(
    figsize=(10, 5)
)

plt.plot(

    vin_values,
    vout_values,
    marker="o",
    label="SAC + Surrogate"
)

plt.axhline(
    VREF,
    linestyle="--",
    label="Consigne 28 V"
)

plt.xlabel("Vin (V)")
plt.ylabel("Vout (V)")
plt.title("Régulation de Vout par SAC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# GRAPHE FREQUENCE


plt.figure(figsize=(10, 5))
plt.plot(
    vin_values,
    fsw_values,
    marker="o"
)

plt.xlabel("Vin (V)")
plt.ylabel("Fsw (kHz)")
plt.title("Fréquence de commutation choisie par SAC")
plt.grid(True)
plt.tight_layout()
plt.show()



#####GRAPHE ERREUR#########


plt.figure(figsize=(10, 5))

plt.plot(
    vin_values,
    error_values,
    marker="o"
)

plt.axhline(0,linestyle="--")
plt.xlabel("Vin (V)")
plt.ylabel("Erreur Vout (V)")
plt.title("Erreur de régulation")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nProgramme terminé.")
    