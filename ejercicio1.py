import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

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

# Inciso 3: Compilación, Función de pérdida e Hiperparámetros
model_clf.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento del modelo
print("\nIniciando entrenamiento...")
history = model_clf.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Métricas de Desempeño
print("\nMétricas de Desempeño")
train_loss, train_accuracy = model_clf.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy = model_clf.evaluate(X_test, y_test, verbose=0)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predicciones
y_pred_probs = model_clf.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nReporte de Clasificación Detallado (Test):")
print(classification_report(y_true_classes, y_pred_classes))

# Inciso 4: Matriz de Confusión con Seaborn
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión - Clasificación de Dígitos')
plt.show()

# Ejemplos bien y mal clasificados
correct_indices = np.where(y_pred_classes == y_true_classes)[0]
incorrect_indices = np.where(y_pred_classes != y_true_classes)[0]

# Visualizar ejemplos correctos
plt.figure(figsize=(15, 3))
for i, idx in enumerate(correct_indices[:5]):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[idx].reshape(8, 8), cmap='gray_r') 
    plt.title(f"Real: {y_true_classes[idx]} | Pred: {y_pred_classes[idx]}")
    plt.axis('off')
plt.suptitle("5 Ejemplos Bien Clasificados", y=1.05, fontsize=14)
plt.show()

# Visualizar ejemplos incorrectos
plt.figure(figsize=(15, 3))
num_errors_to_show = min(5, len(incorrect_indices)) 

if num_errors_to_show > 0:
    for i, idx in enumerate(incorrect_indices[:num_errors_to_show]):
        plt.subplot(1, num_errors_to_show, i + 1)
        plt.imshow(X_test[idx].reshape(8, 8), cmap='gray_r')
        plt.title(f"Real: {y_true_classes[idx]} | Pred: {y_pred_classes[idx]}", color='red')
        plt.axis('off')
    plt.suptitle(f"{num_errors_to_show} Ejemplos Mal Clasificados", y=1.05, fontsize=14)
    plt.show()
else:
    print("El modelo no tuvo errores en el conjunto de prueba...")