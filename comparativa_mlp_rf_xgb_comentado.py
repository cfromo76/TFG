"""
Universidad: UNIR - Curso de Adaptación al Grado de Informática
TFG: Diseño de una RNA para la detección de la influencia en la conducción por el consumo drogas, estupefacientes y psicotrópicos 
Autor: Cristóbal Fernández Romo
Asunto: Comparativa de Modelos de Clasificación: MLP, Random Forest y XGBoost
Descripción: Este script compara el rendimiento de tres modelos de clasificación sobre un dataset
que simula signos observacionales recogidos por cuerpos policiales, conforme a la
Instrucción 3/2020 de la Fiscalía para valorar la influencia de drogas en la conducción.
"""

# ======================================================================================================
# 1. Importación de librerías
#    Importamos las librerías necesarias para el tratamiento de datos, modelado y evaluación
# ======================================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns

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
X["G_diametro_pupilar_ambos_ojos_presentan"] = scaler.fit_transform(X[["G_diametro_pupilar_ambos_ojos_presentan"]])

# ======================================================================================================
# 5. División del dataset
#    Estratificamos según la variable 'influencia' para mantener la proporción de clases (80/20%)
# ======================================================================================================

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)

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
# 8. EVALUACIÓN DEL MODELO
#    Evaluamos el modelo con los datos de prueba.
# =====================================================================================================

y_pred_mlp = mlp.predict(X_test)
y_proba_mlp = mlp.predict_proba(X_test)[:, 1]

print("\n=== MLP ===")
print(confusion_matrix(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))

# ======================================================================================================
# 9. Definición del modelo Random Forest
#    Creamos y entrenamos un modelo de Random Forest con 100 árboles.
# ======================================================================================================


rf = RandomForestClassifier(n_estimators=100, random_state=42)

# ======================================================================================================
# 10. Entrenamiento del modelo Random Forest
# ======================================================================================================

rf.fit(X_train, y_train)

# ======================================================================================================
# 11. EVALUACIÓN DEL MODELO Random Forest
#    Evaluamos el modelo con los datos de prueba.
# ======================================================================================================

y_pred_rf = rf.predict(X_test)
print("\n=== Random Forest ===")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ======================================================================================================
# 12. Definición del modelo XGBoost
#    Creamos y entrenamos un modelo de XGBoost con parámetros por defecto.
# ======================================================================================================
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# ======================================================================================================
# 13. Entrenamiento del modelo XGBoost
# ======================================================================================================
xgb.fit(X_train, y_train)

# ======================================================================================================
# 14. EVALUACIÓN DEL MODELO XGBoost
#    Evaluamos el modelo con los datos de prueba.
# ======================================================================================================
y_pred_xgb = xgb.predict(X_test)
print("\n=== XGBoost ===")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# ======================================================================================================
# 15. Curva ROC
#    Mostramos las matrices de confusión para los tres modelos.
# ======================================================================================================
fpr, tpr, _ = roc_curve(y_test, y_proba_mlp)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"MLP (AUC = {roc_auc:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("Curva ROC - MLP")
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
