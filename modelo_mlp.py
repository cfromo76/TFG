"""
Universidad: UNIR - Curso de Adaptación al Grado de Informática
TFG: Diseño de una RNA para la detección de la influencia en la conducción por el consumo drogas, estupefacientes y psicotrópicos 
Autor: Cristóbal Fernández Romo
Asunto: Implementación del MLP
Descripción: Mediante la implementación del presente script de python se implementa la MLP y se entrena.
"""

# ======================================================================================================
# 1. Importación de librerías
#    Importamos las librerías necesarias para el tratamiento de datos, modelado y evaluación
# ======================================================================================================


import pandas as pd  # Para manejo de datos tabulares (CSV)
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.preprocessing import StandardScaler  # Para normalizar variables numéricas
from sklearn.neural_network import MLPClassifier  # Para construir el perceptrón multicapa (MLP)
from sklearn.metrics import classification_report, confusion_matrix  # Para evaluar el modelo

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

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)

# ======================================================================================================
# 6. Definición, entrenamiento y análisis del modelo MLP
#    Creamos y entrenamos un perceptrón multicapa (MLP) con dos capas ocultas:
#       - 64 neuronas en la primera capa
#       - 32 neuronas en la segunda capa
#       - Función de activación es ReLU 
#       - Entrenamiento se limita a 1000 iteraciones.
# ======================================================================================================
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)


# ======================================================================================================
# 7. Entrenamiento del modelo MLP
# ======================================================================================================

mlp.fit(X_train, y_train)  # Entrenamos el modelo con los datos de entrenamiento


# ======================================================================================================
# 8. EVALUACIÓN DEL MODELO
#    Evaluamos el modelo con los datos de prueba.
# =====================================================================================================

y_pred = mlp.predict(X_test)

# ======================================================================================================
# 9. Resultados de la evaluación
#    Mostramos la matriz de confusión y un informe detallado de precisión, recall y F1-score
# ======================================================================================================

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))
