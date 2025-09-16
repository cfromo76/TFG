"""
Universidad: UNIR - Curso de Adaptación al Grado de Informática
TFG: Diseño de una RNA para la detección de la influencia en la conducción por el consumo drogas, estupefacientes y psicotrópicos 
Autor: Cristóbal Fernández Romo
Asunto: Comparativa modelos
Descripción: El presente código realiza la comparativa entre los modelos Random Forest, XGBoost y MLP

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# === CARGA Y PREPROCESAMIENTO DEL DATASET ===
df = pd.read_csv("dataset_acta_extendido_realista_1000.csv", sep=";")
X = df.drop(columns=["influencia"])
y = df["influencia"]

# Normalización solo de la variable continua
scaler = StandardScaler()
X["G_diametro_pupilar_ambos_ojos_presentan"] = scaler.fit_transform(X[["G_diametro_pupilar_ambos_ojos_presentan"]])

# División estratificada del conjunto
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === MODELO 1: MLP ===
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
y_proba_mlp = mlp.predict_proba(X_test)[:, 1]

print("\n=== MLP ===")
print(confusion_matrix(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))

# === MODELO 2: Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest ===")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# === MODELO 3: XGBoost ===
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n=== XGBoost ===")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# === CURVA ROC MLP ===
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
