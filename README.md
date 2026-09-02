# Estimación de Edad, Género y Etnicidad a partir de Imágenes 

Este proyecto implementa y compara varios modelos de Machine Learning para **estimar la edad, el género y la etnicidad de una persona a partir de su fotografía facial**. Además, se analiza la presencia de sesgos en los modelos empleados.

---

## 📋 Descripción

A partir de una imagen facial, el sistema es capaz de predecir:

- **Edad** — clasificada en 9 rangos: `0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`
- **Género** — masculino o femenino
- **Etnicidad** — entre 7 grupos: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern y Latino/Hispanic

---

## 📦 Dataset

Se utiliza el dataset **[FairFace](https://huggingface.co/datasets/HuggingFaceM4/FairFace)**, disponible en HuggingFace.

- **108 501 imágenes** faciales anotadas
- **7 grupos étnicos** distintos
- **9 rangos de edad**
- **2 géneros**

---

## Modelos implementados

| Modelo | Tarea | 
|---|---|
| **ViT Age Classifier** (nateraw/HuggingFace) | Edad |
| **YOLOv8 + EfficientNet-B0** | Edad 
| **Red neuronal con Keras** (CNN desde cero) | Edad + Género |
| **ViT-B/32 CLIP** | Edad |
| **FairFace Master** (ResNet34) | Edad + Género + Etnicidad |

---

## ⚙️ Tecnologías utilizadas

- Python
- PyTorch / Keras / TensorFlow
- HuggingFace Transformers
- YOLOv8 
- OpenCV
- scikit-learn

---

## Análisis de sesgo

Se evaluaron métricas de equidad sobre los modelos entrenados, incluyendo **Equalized Odds**, **Disparate Impact**, **Label Bias Multi-Class** y **Test de Chi-cuadrado**, con el objetivo de identificar grupos subrepresentados o discriminados por los modelos.

---

Para más detalles revisar el informe que aparece en el repositorio.
