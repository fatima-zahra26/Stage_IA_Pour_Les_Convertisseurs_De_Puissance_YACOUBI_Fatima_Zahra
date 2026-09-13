import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("LLC_dataset_filtered_2.csv", sep=";")

print("Nbr de data :", len(dataset))




X = dataset[["Vin_V","Fsw_kHz"]]


y = dataset["Vout_mean_V"]


#séparation des données (base de test qui contient 20% de la base de données)

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.17,
    random_state=42,
)

#les datasets

train_set = X_train.copy()
train_set["Vout_mean_V"] = y_train

val_set = X_val.copy()
val_set["Vout_mean_V"] = y_val

test_set = X_test.copy()
test_set["Vout_mean_V"] = y_test

#enregistrer les fichiers 

train_set.to_csv("llc_train.csv", sep=";", index=False)
val_set.to_csv("llc_validation.csv", sep=";", index=False)
test_set.to_csv("llc_test.csv", sep=";", index=False)

print("\n-------------") 
print("len train_set :", len(train_set))


print("\n------------")
print("len test_set :", len(test_set))

print("\n------------")
print("len val_set :", len(val_set))

#train

print("\n------------")
print("\n min et max Vin dans train :")
print(X_train["Vin_V"].min(), "→", X_train["Vin_V"].max())

print("\n------------")
print("\n min et max Fsw dans train :")
print(X_train["Fsw_kHz"].min(), "→", X_train["Fsw_kHz"].max())

#validation
print("\n------------")
print("\n min et max Vin dans validation :")
print(X_val["Vin_V"].min(), "→", X_val["Vin_V"].max())

print("\n------------")
print("\n min et max Fsw dans validation :")
print(X_val["Fsw_kHz"].min(), "→", X_val["Fsw_kHz"].max())

#test
print("\n------------")
print("\n min et max Vin dans test :")
print(X_test["Vin_V"].min(), "→", X_test["Vin_V"].max())

print("\n------------")
print("\n min et max Fsw dans test :")
print(X_test["Fsw_kHz"].min(), "→", X_test["Fsw_kHz"].max())