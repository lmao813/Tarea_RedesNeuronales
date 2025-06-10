import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
# Cargar dataset Wine
wine = load_wine()
X = wine.data
y = wine.target
features = wine.feature_names
target_names = wine.target_names

# Crear DataFrame para exploración
wine_df = pd.DataFrame(X, columns=features)
wine_df['target'] = y
wine_df['class_name'] = wine_df['target'].map(lambda x: target_names[x])

# Análisis exploratorio
print(f"Número de muestras: {wine_df.shape[0]}")
print(f"Número de características: {wine_df.shape[1]-2}")
print("\nDistribución de clases:")
print(wine_df['class_name'].value_counts())

# Visualización de características
plt.figure(figsize=(12, 8))
sns.boxplot(data=wine_df, x='class_name', y='alcohol')
plt.title('Distribución de alcohol por clase')
plt.show()

# Matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(wine_df.iloc[:, :13].corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación entre Características')
plt.show()
# Dividir datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Normalización de características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir etiquetas a one-hot encoding
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
model = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),  # Capa oculta 1
    Dropout(0.3),  # Regularización
    Dense(32, activation='relu'),  # Capa oculta 2
    Dropout(0.2),  # Regularización
    Dense(3, activation='softmax')  # Capa de salida (3 clases)
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Resumen de la arquitectura
model.summary()
history = model.fit(X_train, y_train_cat,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    verbose=1)
# Evaluación en conjunto de prueba
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"\nPrecisión en datos de prueba: {test_acc:.4f}")

# Gráficos de entrenamiento
plt.figure(figsize=(12, 4))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
# Hacer predicciones
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicciones')
plt.ylabel('Verdaderos valores')
plt.title('Matriz de Confusión')
plt.show()

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_classes, target_names=target_names))
