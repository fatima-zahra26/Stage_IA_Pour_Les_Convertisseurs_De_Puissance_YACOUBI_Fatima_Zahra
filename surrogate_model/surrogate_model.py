import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import time 


train = pd.read_csv("llc_train.csv", sep=";")
val = pd.read_csv("llc_validation.csv", sep=";")
test = pd.read_csv("llc_test.csv", sep=";")

#il faut séparer les entrées et les sorties 

x_train = train[["Vin_V", "Fsw_kHz"]].values 
y_train = train[["Vout_mean_V"]].values

x_val = val[["Vin_V", "Fsw_kHz"]].values 
y_val = val[["Vout_mean_V"]].values

x_test = test[["Vin_V", "Fsw_kHz"]].values
y_test = test[["Vout_mean_V"]].values

#la normalisation pour éviter que les fortes variables dominent celles faibles + éviter la saturation des activations + convergence rapide de descente du gradient 

scaler_x = StandardScaler()
scaler_y = StandardScaler()

#normalider les entrées 

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)
x_val_scaled = scaler_x.transform(x_val)

#normalider les sorties

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)
y_val_scaled = scaler_y.transform(y_val)

#convertir en tesnor PyTorch

x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32 )
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

#definition de la classe du MLP

class surrogate_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2,64),
            nn.ReLU(),
            
            nn.Linear(64,64),
            nn.ReLU(),
            
            nn.Linear(64,32),
            nn.ReLU(),
            
            nn.Linear(32,1), 
        )
    def forward(self, x):
        return self.network(x)
        
# création du modèle

model = surrogate_model()
print("\narchi du modèle :")
print(model)

#fontion petes

loss_function = nn.MSELoss()

#optimisateur 

opt = torch.optim.Adam(model.parameters(), lr=0.001)

debut = time.perf_counter()

#entainement

epochs = 1000

train_losses = []
val_losses = []


print("\ndébut de l'entraînement--------")

for epoch in range(epochs):
    model.train()
    opt.zero_grad(set_to_none=True)
    prediction = model(x_train_tensor)
    loss = loss_function(prediction, y_train_tensor)
    loss.backward()
    opt.step()
    
    #évaluation du test 
    model.eval()
    with torch.no_grad():
        val_prediction = model(X_val_tensor)
        val_loss = loss_function(val_prediction,y_val_tensor)
    
    
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    
    #affichage des résultats toutes les 100 époques 
    if (epoch + 1)%100 == 0:
        print(
            f"époque {epoch+1}/{epoch}"
            f"| train loss = {loss.item():6f}"
            f"| val loss = {val_loss.item():6f}"
        )


#prédictions

model.eval()

with torch.no_grad():

    y_val_pred_scaled = model(X_val_tensor).numpy()

fin = time.perf_counter()
print(f"temps d'entrainement: {fin-debut: .4f} s")

# retour à l'échelle réelle

y_pred = scaler_y.inverse_transform(
    y_val_pred_scaled
)

#calcul des performances 
rmse = np.sqrt(mean_squared_error(y_val,y_pred))

mae = mean_absolute_error(y_val,y_pred)

r2 = r2_score(y_val,y_pred)

print("\n ------ performances du modèle sur la base de validation-------")

print(f"RMSE : {rmse:.4f} V")
print(f"MAE  : {mae:.4f} V")
print(f"R²   : {r2:.6f}")

#comparaison entre le réel et le prédit pour la base de validation

results_val = pd.DataFrame({

    "Vin_V": val["Vin_V"].values,

    "Fsw_kHz": val["Fsw_kHz"].values,

    "Vout_LTspice_V": y_val.flatten(),

    "Vout_MLP_V": y_pred.flatten()

})

print("\nquelques prédictions sur la base de validation :")

print(
    results_val.head(10)
)

#tester sur la base de test

model.eval()

with torch.no_grad():
    y_test_pred_scaled = model(X_test_tensor).numpy()

y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

rmse_test = np.sqrt(mean_squared_error(y_test,y_test_pred))

mae_test = mean_absolute_error(y_test,y_test_pred)

r2_test = r2_score(y_test,y_test_pred)

print("\n ------ performances du modèle sur la base de test-------")

print(f"RMSE_TEST : {rmse_test:.4f} V")
print(f"MAE_TEST  : {mae_test:.4f} V")
print(f"R²_Test   : {r2_test:.6f}")

#comparaison entre le réel et le prédit pour la base de test

results_test = pd.DataFrame({

    "Vin_V": test["Vin_V"].values,

    "Fsw_kHz": test["Fsw_kHz"].values,

    "Vout_LTspice_V": y_test.flatten(),

    "Vout_MLP_V": y_test_pred.flatten()

})

print("\nquelques prédictions sur la base de test :")

print(results_test.head(10))


#sauvegarder les résultats dans un fichier csv

results_val.to_csv("model_results_val.csv", sep =";", index= False)
results_test.to_csv("model_results_test.csv", sep =";", index= False)

#sauvegarder le modèle
        
torch.save(model.state_dict(),"model.pth")

joblib.dump(scaler_x,"scaler_x.pkl")

joblib.dump(scaler_y,"scaler_y.pkl")


print("\nmodèle sauvegardé : model.pth")   

#courbe de pertes 
     
plt.figure(figsize=(8, 5))

plt.plot(train_losses,label="Train Loss")

plt.plot(val_losses,label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title("Entraînement du modèle")

plt.legend()

plt.grid()

plt.show()


#vout réelle vs vout prédite

plt.figure(figsize=(6, 6))

plt.scatter(y_val, y_pred)

plt.xlabel("Vout LTspice (V)")

plt.ylabel("Vout MLP (V)")

plt.title("LTspice vs Surrogate Model")

# droite parfaite y = x

minimum = min(y_val.min(),y_pred.min())

maximum = max(y_val.max(),y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.grid()

plt.show()

#affichage courbe test



plt.figure(figsize=(6,6))
plt.scatter(y_test,y_test_pred)

plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],linestyle="--")#droite parfaite y = x
plt.xlabel("Vout LTspice (V)")
plt.ylabel("Vout MLP (V)")
plt.title("Test Set")
plt.grid()
plt.show()