from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar el dataset de California Housing
california = fetch_california_housing()
X, y = california.data, california.target

# Partición de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#   Estandarización de los datos   
scaler = StandardScaler()
# Se ajusta y transforma con los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)
# Se transforma con los datos de prueba
X_test_scaled = scaler.transform(X_test)

#   Red más profunda   
model = Sequential([
    # Capa 1: 64 neuronas
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    # Capa 2: 32 neuronas
    Dense(32, activation='relu'),
    # Capa 3: 16 neuronas
    Dense(16, activation='relu'),
    # Capa de Salida: 1 neurona
    Dense(1) 
])

model.summary()