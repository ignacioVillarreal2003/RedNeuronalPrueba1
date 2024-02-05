import tensorflow as tf
import numpy as np

# Datos de entrada
altura = np.array([180, 160, 165, 192, 192, 173, 170, 177, 150, 200, 185, 155, 175, 180, 165, 160, 163, 230, 215, 160, 170, 162, 175, 168, 180, 155, 185, 160, 172, 178, 163, 190, 158, 168, 175, 100, 160, 163, 190, 187, 164, 175, 192,199, 176], dtype=float)
peso = np.array([70, 68, 54, 88, 67, 69, 56, 82, 65, 75, 72, 63, 68, 70, 66, 53, 59, 130, 110, 120, 65, 58, 72, 60, 85, 50, 95, 55, 68, 75, 56, 100, 48, 62, 80, 76, 80, 90, 99, 46, 86, 78, 83, 83, 99], dtype=float)
imc = peso / (altura / 100) ** 2

# Normalizar los datos
peso_norm = peso / 100
altura_norm = altura / 200

# Apilar los datos de entrada
datos_entrada = np.column_stack((altura_norm, peso_norm))

# Definir el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, input_shape=[2], activation="relu"),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=16, activation="relu"),
    tf.keras.layers.Dense(units=1)
])

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mean_squared_error'
)

# Entrenar el modelo
print("Comenzando entrenamiento...")
historial = modelo.fit(datos_entrada, imc, epochs=500, verbose=False)
print("¡Modelo entrenado!")

# Hacer una predicción
print("Hagamos una predicción!")
nueva_altura = 184 / 200
nuevo_peso = 74 / 100

resultado = modelo.predict(np.array([[nueva_altura, nuevo_peso]]))
print("El resultado es " + str(resultado[0][0]) + " IMC!")