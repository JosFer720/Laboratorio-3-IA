from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np

# Cargar el dataset
california = fetch_california_housing()
X, y = california.data, california.target

# Separar datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar los datos porque las variables tienen rangos muy diferentes
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Red neuronal con 3 capas ocultas
# Se mantiene la idea de 64, 32 y 16 neuronas
model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=20
)

# Entrenar el modelo
model.fit(X_train_scaled, y_train)

# Hacer predicciones con train y test
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calcular métricas para entrenamiento
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Calcular métricas para prueba
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nMétricas de entrenamiento")
print(f"MSE: {train_mse:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"MAE: {train_mae:.4f}")
print(f"R²: {train_r2:.4f}")

print("\nMétricas de prueba")
print(f"MSE: {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"R²: {test_r2:.4f}")

# Gráfico para comparar valores reales contra predicciones
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Predicciones vs valores reales")

# Esta línea ayuda a ver qué tan cerca están las predicciones del valor ideal
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.grid(True)
plt.show()

# Tres ejemplos nuevos que no vienen directamente del dataset usado en entrenamiento
# Orden de variables:
# [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
nuevas_observaciones = np.array([
    [5.2, 25.0, 6.0, 1.1, 1200.0, 3.0, 34.05, -118.25],
    [3.1, 18.0, 5.2, 1.0, 900.0, 2.5, 36.77, -119.41],
    [8.0, 35.0, 7.5, 1.2, 2000.0, 4.1, 37.77, -122.42]
])

# Escalar también los nuevos datos antes de predecir
nuevas_observaciones_scaled = scaler.transform(nuevas_observaciones)
predicciones_nuevas = model.predict(nuevas_observaciones_scaled)

print("\nPredicciones para 3 observaciones nuevas")
for i, pred in enumerate(predicciones_nuevas, start=1):
    print(f"Observación {i}: precio predicho = {pred:.4f}")
