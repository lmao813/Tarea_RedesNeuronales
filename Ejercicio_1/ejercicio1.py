# Importar librerías necesarias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1. Cargar los datos
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas

# 2. Preprocesamiento de datos
# Normalización de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertir etiquetas a one-hot encoding
y_categorical = tf.keras.utils.to_categorical(y)

# Dividir en conjuntos de entrenamiento y prueba (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42)

# 3. Construir el modelo de red neuronal
model = Sequential([
    Dense(8, activation='relu', input_shape=(4,)),  # Capa oculta con 8 neuronas
    Dense(3, activation='softmax')  # Capa de salida con 3 neuronas (una por clase)
])

# 4. Compilar el modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# 5. Entrenar el modelo
history = model.fit(
    X_train, 
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1)

# 6. Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nPrecisión en datos de prueba: {test_acc:.4f}')

# 7. Visualizar el rendimiento
plt.figure(figsize=(12, 4))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
