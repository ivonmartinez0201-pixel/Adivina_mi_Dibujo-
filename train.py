import numpy as np
import urllib.request
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

categories = [
    'cake',          
    'bus',           
    'cell phone',    
    'circle',        
    'crayon',        
    'door',          
    'eye',           
    'fish',          
    'flower',        
    'hexagon',       
    'donut',         
    'triangle'            
]

N_SAMPLES = 8000

print("=" * 60)
print("PROYECTO: ADIVINA MI DIBUJO)")
print("=" * 60)
print(f"\n Categorías seleccionadas ({len(categories)} en total):")
for i, cat in enumerate(categories):
    print(f"   {i+1}. {cat.capitalize()}")

print("\n Descargando datos de Quick, Draw!...")
os.makedirs('data', exist_ok=True)

base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

for cat in categories:
    cat_url = cat.replace(' ', '%20')
    filepath = f'data/{cat.replace(" ", "_")}.npy'
    
    if not os.path.exists(filepath):
        url = f'{base_url}{cat_url}.npy'
        print(f'      Descargando {cat}...')
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f'       {cat} descargado')
        except Exception as e:
            print(f'       Error descargando {cat}: {e}')
    else:
        print(f'      {cat} ya existe, omitiendo descarga.')

print("   Descarga completada!")

print("\n Cargando y preparando datos...")

def load_balanced_data(category, n_samples):
    filename = f'data/{category.replace(" ", "_")}.npy'
    data = np.load(filename)
    data = data[:n_samples]
    images = data.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    return images

X_data = []
y_data = []

for i, cat in enumerate(categories):
    print(f'   Cargando {cat}...')
    try:
        images = load_balanced_data(cat, N_SAMPLES)
        X_data.append(images)
        y_data.append([i] * len(images))
    except Exception as e:
        print(f'     Error cargando {cat}: {e}')

X_data = np.concatenate(X_data, axis=0)
y_data = np.concatenate(y_data, axis=0)

print(f"\n Datos cargados: {X_data.shape[0]:,} imágenes de {len(categories)} categorías")
print(f"   Tamaño de imagen: {X_data.shape[1]}x{X_data.shape[2]} píxeles")

print("\n Dividiendo datos...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, 
    test_size=0.3, 
    stratify=y_data, 
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5, 
    stratify=y_temp, 
    random_state=42
)

print(f"   🟢 Entrenamiento: {X_train.shape[0]:,} imágenes")
print(f"   🟡 Validación:   {X_val.shape[0]:,} imágenes")
print(f"   🔴 Prueba:       {X_test.shape[0]:,} imágenes")

print("\n Construyendo modelo CNN...")

def build_model(num_classes):
    model = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Bloque 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Bloque 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Clasificador
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model(len(categories))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n Entrenando modelo...")

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=8, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=0.0001,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=35,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n" + "=" * 60)
print("EVALUACIÓN DEL MODELO")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n Accuracy en prueba: {test_acc:.4f} ({test_acc*100:.2f}%)")

if test_acc >= 0.75:
    print("  ¡Se superó el 75% de accuracy requerido!")
else:
    print(f" El accuracy ({test_acc*100:.2f}%) está por debajo del 75%.")

model.save('modelo_dibujos.h5')
print("\n Modelo guardado como 'modelo_dibujos.h5'")

print("\n Generando gráficos...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Entrenamiento', color='green')
ax1.plot(history.history['val_accuracy'], label='Validación', color='blue')
ax1.set_title('Precisión durante el entrenamiento')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Precisión')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Entrenamiento', color='green')
ax2.plot(history.history['val_loss'], label='Validación', color='blue')
ax2.set_title('Pérdida durante el entrenamiento')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Pérdida')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("       training_history.png guardado")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[c.capitalize() for c in categories])
fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax_cm, cmap='Greens', values_format='d')
ax_cm.set_title('Matriz de Confusión')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("       confusion_matrix.png guardado")

print("\n" + "=" * 60)
print("¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
print("=" * 60)
print("\nArchivos generados:")
print("   - modelo_dibujos.h5")
print("   - training_history.png")
print("   - confusion_matrix.png")