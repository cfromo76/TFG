"""
Universidad: UNIR - Curso de Adaptación al Grado de Informática
TFG: Diseño de una RNA para la detección de la influencia en la conducción por el consumo drogas, estupefacientes y psicotrópicos 
Autor: Cristóbal Fernández Romo
Asunto: Creación de una curva ROC de una MLP
Descripción: .
"""

# ==============================
# 1. Importación de librerías
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ==============================
# 2. Carga y preprocesamiento del dataset
# ==============================

# Cargar el archivo CSV con los datos sintéticos
df = pd.read_csv("dataset_acta_extendido_realista_1000.csv", sep=";")

# Separar características (X) y variable objetivo (y)
X = df.drop(columns=["influencia"])
y = df["influencia"]

# Normalizar la única variable continua (diámetro pupilar)
scaler = StandardScaler()
X["G_diametro_pupilar_ambos_ojos_presentan"] = scaler.fit_transform(
    X[["G_diametro_pupilar_ambos_ojos_presentan"]]
)

# Dividir el conjunto de datos (80% entrenamiento, 20% prueba, estratificado)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ==============================
# 3. Definición y entrenamiento del modelo MLP
# ==============================

# Arquitectura: dos capas ocultas (64 y 32 neuronas), función ReLU
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=1000,
    random_state=42
)

# Entrenar el modelo
mlp.fit(X_train, y_train)

# ==============================
# 4. Generación de la curva ROC
# ==============================

# Obtener probabilidades predichas para la clase positiva
y_proba = mlp.predict_proba(X_test)[:, 1]

# Calcular los puntos ROC: tasa de verdaderos positivos (TPR) vs. falsos positivos (FPR)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Calcular el área bajo la curva (AUC)
roc_auc = auc(fpr, tpr)

# ==============================
# 5. Visualización de la curva ROC
# ==============================

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f"MLP (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curva ROC - Red Neuronal MLP")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("curva_roc_mlp.jpg", dpi=300)
plt.show()
