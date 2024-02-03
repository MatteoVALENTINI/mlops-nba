import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging

# Configurer le journal
logging.basicConfig(level=logging.INFO)

# Charger les données
data_file = "raw/2023-2024_NBA_Player_Stats_Regular.csv"


try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    logging.error(f"Le fichier '{data_file}' n'a pas été trouvé.")
    exit(1)

# Diviser les données en caractéristiques (X) et cible (y)
X = data.drop("Target", axis=1)
y = data["Target"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle de régression linéaire
model = LinearRegression()

# Fonction pour entraîner un modèle de régression sur les données
def train_model(df):
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédictions
    preds = model.predict(X_test)
    
    # Calcul de la RMSE (racine de l'erreur quadratique moyenne)
    rmse = mean_squared_error(y_test, preds, squared=False)
    logging.info(f"RMSE: {rmse}")
    
    # Sauvegarde du modèle
    joblib.dump(model, "model.joblib")
    return model

# Entraîner le modèle
trained_model = train_model(data)
