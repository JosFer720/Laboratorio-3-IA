import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Cargar el dataset
digits = load_digits()
X = digits.data  # 1,797 muestras, 64 atributos
y = digits.target

# Inciso 1: Preprocesamiento de Datos
# Transformaciones y Normalización

# Normalización: los valores originales van de 0 a 16. 
# Se divide por 16 para que los datos estén en el rango [0, 1].
X_normalized = X / 16.0 

# Transformación de etiquetas
y_one_hot = to_categorical(y, num_classes=10)

# Partición de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_one_hot, test_size=0.2, random_state=42
)

# Inciso 2: Diseño de la Arquitectura

model_clf = models.Sequential([
    # Primera capa oculta: 128 neuronas para detectar bordes y formas básicas
    layers.Dense(128, activation='relu', input_shape=(64,)),
    
    # Segunda capa oculta: 64 neuronas para combinar esas formas en patrones
    layers.Dense(64, activation='relu'),
    
    # Capa de salida: 10 neuronas (una por dígito) con Softmax 
    layers.Dense(10, activation='softmax')
])

model_clf.summary()