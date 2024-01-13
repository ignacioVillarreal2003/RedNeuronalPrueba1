import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

metros = np.random.uniform(low=500, high=80000, size=300)
kilometros = metros / 1000

# Normalizar los datos
scaler = MinMaxScaler()
metros_scaled = scaler.fit_transform(metros.reshape(-1, 1)).flatten()

modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=8, input_shape=[1], activation="linear"),
    tf.keras.layers.Dense(units=8, activation="linear"),
    tf.keras.layers.Dense(units=1)])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(metros_scaled, kilometros, epochs=300, verbose=False)
print("Modelo entrenado!")

# Normalizar la entrada para hacer la predicción
metros_input = scaler.transform(np.array([[21900]]))

resultado = modelo.predict(metros_input)
resultado_kilometros = resultado[0][0]

print("El resultado es " + str(resultado_kilometros) + " kilómetros")
