# 🎨 Adivina mi Dibujo

Proyecto Final de Introducción a la Inteligencia Artificial - Clasificador de dibujos en tiempo real con 12 categorías.

## 📋 Descripción

Sistema que reconoce dibujos en tiempo real mientras se dibujan en un lienzo. El modelo clasifica el dibujo entre 12 categorías diferentes, mostrando las 3 predicciones más probables con su porcentaje de confianza y retroalimentación visual.

## 🎯 Categorías (12)

| Comida 🍩 | Objetos 📱 | Naturaleza 🌸 | Formas 🔷 |
|-----------|------------|---------------|-----------|
| 🎂 Cake | 🚌 Bus | 🐟 Fish | ⭕ Circle |
| 🍩 Donut | 📱 Cell Phone | 🌸 Flower | ⬡ Hexagon |
| | 🖍️ Crayon | 🔺 Triangle | |
| | 🚪 Door | | |
| | 👁️ Eye | | |

## 🛠️ Tecnologías

- **Python 3.11**
- **TensorFlow / Keras** - Deep Learning
- **Streamlit** - Interfaz de usuario
- **NumPy / Pillow** - Procesamiento de imágenes
- **Scikit-learn** - Métricas y evaluación
- **Matplotlib** - Visualización de resultados

## 📊 Resultados

- **Accuracy en prueba:** 92.12%
- **Arquitectura:** CNN con 3 bloques convolucionales
- **Dataset:** Quick, Draw! (96,000 imágenes)
- **Épocas:** 35
- **Batch Size:** 64

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/ivonmartinez0201-pixel/Adivina_mi_Dibujo-.git
cd Adivina_mi_Dibujo-
```

### 2. Crear y activar entorno virtual

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Instalar dependencias

```bash
pip install tensorflow streamlit streamlit-drawable-canvas numpy pillow scikit-learn matplotlib
```

### 🚀 Uso

## 1. Entrenar el modelo

```bash
python train.py
```

## 2. Ejecutar la aplicación

```bash
streamlit run app.py
```

### 📁 Estructura del Proyecto

```
Adivina_mi_Dibujo-/
├── data/                    # Dataset Quick, Draw! (descargado automáticamente)
├── train.py                 # Entrenamiento del modelo
├── app.py                   # Interfaz Streamlit
├── modelo_dibujos.h5        # Modelo entrenado (92.12% accuracy)
├── training_history.png     # Gráficos de entrenamiento
├── confusion_matrix.png     # Matriz de confusión
├── requirements.txt         # Dependencias del proyecto
├── .gitignore               # Archivos ignorados por Git
└── README.md                # Esta documentación
```

### 🎥 Video de Demostración

[Ver video de defensa] (https://ister-my.sharepoint.com/:v:/g/personal/ivon_martinez_ister_edu_ec/IQB3Ujpadw69R59JwR2cL9dCAVijeVml_se3yCAJl0H8Chk?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=73M0Sm)

### 👤 Autor

**Nicole Ivon** - [ivonmartinez0201-pixel](https://github.com/ivonmartinez0201-pixel)

### 📝 Licencia

MIT
