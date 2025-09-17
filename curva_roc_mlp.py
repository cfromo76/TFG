"""
Universidad: UNIR - Curso de Adaptación al Grado de Informática
TFG: Diseño de una RNA para la detección de la influencia en la conducción por el consumo drogas, estupefacientes y psicotrópicos 
Autor: Cristóbal Fernández Romo
Asunto: Creación de una curva ROC de una MLP
Descripción: Mediante la implementación de este script en python se pretende generar la curva ROC de una MLP.
El objeto es representar gráficamente la tasa de verdaderos positivos (TPR) frente a la tasa de falsos positivos (FPR)
a distintos umbrales de decisión. Cuanto más se curva hacia la esquina superior izquierda, mejor es el modelo, 
ya que logra maximizar los aciertos sin comprometer la tasa de errores .
"""

# ======================================================================================================
# 1. Importación de librerías
#    Importamos las librerías necesarias para el tratamiento de datos, modelado y evaluación
# ======================================================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ======================================================================================================
# 2. Carga y preparación de datos
#    Cargamos el dataset generado a partir del acta de signos externos.
#    Este archivo debe estar en la misma carpeta que este script.
# ======================================================================================================

df = pd.read_csv("dataset_acta_extendido_realista_1000.csv", sep=";")

# ======================================================================================================
# 3. Separación de variables y preprocesamiento
#    Separamos las variables predictoras y la variable objetivo.    
#    X contendrá las variables predictoras (signos clínicos y comportamentales)
#    y contendrá la variable objetivo: influencia (1 = influenciado, 0 = no influenciado)
# ======================================================================================================

X = df.drop(columns=["influencia"])
y = df["influencia"]

# ======================================================================================================
# 4. Preprocesamiento
#    Normalizamos únicamente la variable continua del diámetro pupilar.
#    El resto de variables son binarias (0/1) y no requieren escalado.
# ======================================================================================================

scaler = StandardScaler()
X["G_diametro_pupilar_ambos_ojos_presentan"] = scaler.fit_transform(
    X[["G_diametro_pupilar_ambos_ojos_presentan"]]
)

# ======================================================================================================
# 5. División del dataset
#    Estratificamos según la variable 'influencia' para mantener la proporción de clases (80/20%)
# ======================================================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ======================================================================================================
# 6. Definición del modelo MLP
#    Creamos y entrenamos un perceptrón multicapa (MLP) con dos capas ocultas:
#       - 64 neuronas en la primera capa
#       - 32 neuronas en la segunda capa
#       - Función de activación es ReLU 
#       - Entrenamiento se limita a 1000 iteraciones.
# ======================================================================================================

mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                    activation='relu',
                    max_iter=1000,
                    random_state=42)

# ======================================================================================================
# 7. Entrenamiento del modelo MLP
# ======================================================================================================

mlp.fit(X_train, y_train)

# ======================================================================================================
# 8. Cálculo de la curva ROC y AUC
# Obtener probabilidades predichas para la clase positiva
# ======================================================================================================
y_proba = mlp.predict_proba(X_test)[:, 1]

# ======================================================================================================
# 9. CCalcular los puntos ROC: tasa de verdaderos positivos (TPR) vs. falsos positivos (FPR)
# ======================================================================================================


fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Calcular el área bajo la curva (AUC)
roc_auc = auc(fpr, tpr)

# ======================================================================================================
# 10. Representar graficamente la curva ROC
# ======================================================================================================   


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
