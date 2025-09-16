"""
Universidad: UNIR - Curso de Adaptación al Grado de Informática
TFG: Diseño de una RNA para la detección de la influencia en la conducción por el consumo drogas, estupefacientes y psicotrópicos 
Autor: Cristóbal Fernández Romo
Asunto: Análisis de características mediante técnicas de interpretabilidad (XAI) mediante tecnica SHAP
Descripción: .
"""


import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Cargar el dataset
df = pd.read_csv("dataset_acta_extendido_realista_1000.csv", sep=";")

# Separar variables y objetivo
X = df.drop(columns=["influencia"])
y = df["influencia"]

# Escalar la variable continua
scaler = StandardScaler()
X["G_diametro_pupilar_ambos_ojos_presentan"] = scaler.fit_transform(
    X[["G_diametro_pupilar_ambos_ojos_presentan"]]
)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Entrenar red neuronal MLP
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Selección de muestra para SHAP
X_sample = X_train.sample(n=300, random_state=42)

# Crear explainer con SHAP para modelo no lineal
explainer = shap.KernelExplainer(mlp.predict, X_sample)

# Calcular valores SHAP
shap_values = explainer.shap_values(X_sample)

# Crear gráfico de resumen
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot_mlp_300.jpg", format="jpg", dpi=300)
plt.close()
