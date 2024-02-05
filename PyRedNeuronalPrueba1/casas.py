import numpy as np
import tensorflow as tf

habitaciones = np.random.randint(low=2, high=6, size=100)
ubicacion = np.random.uniform(low=1, high=10, size=100)
precio_casa = 50000 + 20000 * habitaciones + 3000 * ubicacion

# Construir el modelo
modelo_precio_casa = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, input_shape=[2], activation="relu"),
    tf.keras.layers.Dense(units=16, activation="relu"),
    tf.keras.layers.Dense(units=1)
])

# Compilar el modelo
modelo_precio_casa.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='mean_squared_error'
)

# Entrenar el modelo
historial_precio_casa = modelo_precio_casa.fit(
    np.column_stack((habitaciones, ubicacion)),
    precio_casa,
    epochs=500,
    verbose=False
)

# Generar datos de prueba
habitaciones_test = np.array([6])
ubicacion_test = np.array([1])
precio_casa_test = 50000 + 20000 * habitaciones_test + 3000 * ubicacion_test

# Evaluar el modelo en el conjunto de prueba
predicciones_precio_casa = modelo_precio_casa.predict(np.column_stack((habitaciones_test, ubicacion_test)))
print(f"Precio esperado: {precio_casa_test}")
print(f"Precio predecido: {predicciones_precio_casa}")

