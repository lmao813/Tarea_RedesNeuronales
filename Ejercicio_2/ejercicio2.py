import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
# Cargar el dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Definir los nombres de las clases
class_names = ['Camiseta/top', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapatilla deportiva', 'Bolso', 'Botín']

# Explorar los datos
print(f"Dimensiones del conjunto de entrenamiento: {train_images.shape}")
print(f"Número de imágenes de prueba: {test_images.shape[0]}")
print(f"Rango de valores de píxeles: [{train_images.min()}, {train_images.max()}]")

# Visualizar algunas muestras
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
# Normalizar los valores de píxeles al rango [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Redimensionar para agregar el canal de color (necesario para Conv2D)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Convertir las etiquetas a one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model = Sequential([
    # Capa convolucional con 32 filtros de 3x3 y activación ReLU
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # Capa de Max Pooling para reducción dimensional
    MaxPooling2D((2, 2)),
    
    # Segunda capa convolucional con 64 filtros
    Conv2D(64, (3, 3), activation='relu'),
    
    # Segunda capa de Max Pooling
    MaxPooling2D((2, 2)),
    
    # Aplanar la salida para la capa densa
    Flatten(),
    
    # Capa densa con 128 neuronas
    Dense(128, activation='relu'),
    
    # Capa de salida con 10 neuronas (una por clase) y softmax
    Dense(10, activation='softmax')
])

# Resumen de la arquitectura
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels,
                    epochs=10,
                    batch_size=64,
                    validation_data=(test_images, test_labels))
# Evaluación en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'\nPrecisión en datos de prueba: {test_acc:.4f}')

# Gráficos de precisión y pérdida
plt.figure(figsize=(12, 4))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
# Hacer predicciones
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Matriz de confusión
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicciones')
plt.ylabel('Verdaderos valores')
plt.title('Matriz de Confusión')
plt.show()
