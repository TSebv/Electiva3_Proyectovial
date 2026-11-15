# ELECTIVA-III
# Clasificador de Daños Viales mediante Deep Learning

Sistema de visión artificial para clasificar fotografías de infraestructura vial en diez categorías relacionadas con daños del pavimento y condiciones de alcantarillas. El proyecto utiliza aprendizaje por transferencia con **MobileNetV2**, ajuste fino (*Fine-Tuning*) y herramientas de TensorFlow para evaluar y preparar el modelo para despliegue.

## 1. Problema

El deterioro de la infraestructura vial —como baches, grietas, hundimientos y daños en alcantarillas— requiere inspección para identificar las condiciones que necesitan atención. La revisión manual de fotografías puede ser lenta y depende de la interpretación de quien realiza la inspección.

Este proyecto propone un modelo de clasificación de imágenes que recibe una fotografía y estima a cuál de diez categorías pertenece. Su propósito es apoyar la identificación y organización de evidencias de daños viales; **no reemplaza una inspección técnica ni determina por sí solo la prioridad de intervención**.

## 2. Objetivo

Desarrollar y evaluar un modelo de aprendizaje profundo capaz de clasificar imágenes de daños y condiciones viales en diez clases, y explorar alternativas de optimización para su posible uso en dispositivos móviles.

## 3. Integrantes

- **Juan Sebastian Gonzalez Cruz**
- **Angel Ricardo Cano Melo**

## 4. Categorías del modelo

El conjunto de datos utilizado en el notebook contiene 3.500 imágenes distribuidas en diez clases, con 350 imágenes por categoría después del aumento de datos:

| N.º | Categoría |
|---:|---|
| 1 | `alcantarilla_abierta` |
| 2 | `alcantarilla_buen_estado` |
| 3 | `alcantarilla_obstruida` |
| 4 | `alcantarilla_tapa_rota` |
| 5 | `asfalto_desprendido` |
| 6 | `bache` |
| 7 | `grieta` |
| 8 | `hundimiento` |
| 9 | `parche_pavimento` |
| 10 | `piel_de_cocodrilo` |

## 5. Tecnologías utilizadas

- Python
- TensorFlow y Keras
- MobileNetV2 (pesos preentrenados en ImageNet)
- TensorFlow Lite
- Scikit-learn
- NumPy y Pandas
- Matplotlib y Seaborn
- Pillow
- Jupyter Notebook

## 6. Instalación

### Requisitos

- Python compatible con la versión de TensorFlow que se vaya a instalar.
- pip.
- Jupyter Notebook o JupyterLab.
- El conjunto de datos organizado en carpetas por categoría.

### Crear y activar un entorno virtual

En Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install tensorflow pillow scikit-learn matplotlib seaborn numpy pandas jupyter icrawler
```

Si se ejecutan las celdas de la API desarrollada con FastAPI, instalar también:

```powershell
pip install fastapi uvicorn python-multipart requests
```

## 7. Estructura del conjunto de datos

El notebook espera una carpeta llamada `dataset_vial_propio`, con una subcarpeta por clase:

```text
proyecto/
├── Untitled.ipynb
└── dataset_vial_propio/
    ├── alcantarilla_abierta/
    ├── alcantarilla_buen_estado/
    ├── alcantarilla_obstruida/
    ├── alcantarilla_tapa_rota/
    ├── asfalto_desprendido/
    ├── bache/
    ├── grieta/
    ├── hundimiento/
    ├── parche_pavimento/
    └── piel_de_cocodrilo/
```

El notebook realiza aumento de datos para completar 350 imágenes por clase. Las transformaciones incluyen volteo horizontal, rotación, cambios de brillo y contraste y desenfoque suave ocasional.

## 8. Cómo ejecutar

1. Clonar o descargar este repositorio.
2. Descargar o ubicar el dataset y organizarlo con la estructura indicada.
3. Crear y activar el entorno virtual.
4. Instalar las dependencias.
5. Abrir una terminal en la carpeta del proyecto.
6. Ejecutar:

   ```powershell
   jupyter notebook
   ```

7. Abrir `Untitled.ipynb`.
8. Revisar las rutas de las carpetas y archivos del notebook. Algunas celdas contienen rutas absolutas de Windows (`C:\Users\...`) que deben modificarse para el equipo local.
9. Ejecutar las celdas en orden. El notebook incluye preparación del dataset, entrenamiento, Fine-Tuning, evaluación, conversión a TensorFlow Lite y experimentos adicionales.

### Configuración principal

- Tamaño de imagen: **224 × 224 píxeles**.
- Batch size: **32**.
- Semilla: **42**.
- División inicial del dataset: **70 % entrenamiento, 15 % validación y 15 % prueba** (la evaluación registrada en el notebook utiliza 512 imágenes de validación y 538 de prueba).
- Arquitectura base: **MobileNetV2**.
- Optimizador: **Adam**.
- Regularización: **Dropout de 0.4**.
- Callbacks: `EarlyStopping` y `ReduceLROnPlateau`.

## 9. Resultados y métricas

Es importante diferenciar los resultados calculados por las celdas de evaluación de los valores escritos como comparación experimental en una tabla del notebook.

### Evaluación de clasificación registrada

El reporte de clasificación generado para el conjunto de validación de 512 imágenes presenta:

| Métrica | Resultado |
|---|---:|
| Accuracy | 75.59 % |
| F1-score macro | **0.7547** |
| F1-score ponderado | 0.7550 |
| Precision macro | 0.7929 |
| Recall macro | 0.7585 |

El F1-score por clase obtenido en ese reporte fue:

| Clase | F1-score |
|---|---:|
| alcantarilla_abierta | 0.7647 |
| alcantarilla_buen_estado | 0.7132 |
| alcantarilla_obstruida | 0.8864 |
| alcantarilla_tapa_rota | 0.6133 |
| asfalto_desprendido | 0.7308 |
| bache | 0.8364 |
| grieta | 0.8687 |
| hundimiento | 0.6897 |
| parche_pavimento | 0.7257 |
| piel_de_cocodrilo | 0.7179 |

### Evaluación regional de 50 fotografías

El notebook también registra una evaluación de 50 fotografías con el modelo ajustado con el conjunto regional:

| Métrica | Resultado |
|---|---:|
| Fotografías procesadas | 50 |
| Predicciones correctas | 10 |
| Predicciones incorrectas | 40 |
| Exactitud | **20.00 %** |
| Latencia promedio de inferencia | **274.29 ms/imagen** |

Esta evaluación regional muestra que el modelo todavía presenta dificultades para generalizar a esas fotografías. Por eso, el resultado de validación y el resultado regional se presentan por separado y no deben confundirse.

### Tamaño del modelo

El resumen de Keras registrado en el notebook indica:

| Dato | Resultado |
|---|---:|
| Parámetros totales | 5.659.488 |
| Tamaño asociado a parámetros reportado por Keras | 21.59 MB |

**21.59 MB corresponde al tamaño de parámetros mostrado por el resumen de Keras, no a una medición verificada del archivo `.keras` ni de los archivos `.tflite`.** El notebook genera variantes Float32, Float16 e Int8, pero el tamaño real de cada archivo debe medirse en disco antes de publicar una comparación de tamaño.

### Comparación de escenarios incluida en el notebook

El notebook contiene una tabla comparativa con los siguientes valores:

| Escenario | Accuracy | F1 macro | Tamaño | Latencia |
|---|---:|---:|---:|---:|
| MobileNetV2 (Base Keras) | 92.75 % | 0.9261 | 28.40 MB | 42.15 ms |
| MobileNetV2 (TFLite optimizado) | 92.40 % | 0.9228 | 8.85 MB | 14.20 ms |
| Ensamble por promedio | 94.10 % | 0.9402 | 126.80 MB | 118.50 ms |
| Ensamble ponderado | 94.85 % | 0.9478 | 142.20 MB | 125.10 ms |
| Híbrido YOLOv8n + MobileNetV2 | 95.60 % | 0.9542 | 38.25 MB | 62.40 ms |

> **Nota metodológica:** estos valores aparecen definidos directamente en un diccionario de datos del notebook para construir una tabla comparativa. No se deben presentar como métricas verificadas por las ejecuciones de entrenamiento y evaluación mostradas. Para un informe definitivo, se deben respaldar con experimentos reproducibles y mediciones de los archivos y equipos utilizados.

## 10. Modelos y archivos generados

Dependiendo de las celdas ejecutadas, se generan modelos Keras y versiones TensorFlow Lite:

```text
modelo_vial_finetuned.keras
modelo_vial_ibague_final.keras
modelo_estudiante_destilado.keras
modelos_moviles/
├── modelo_vial_float32.tflite
├── modelo_vial_float16.tflite
└── modelo_vial_int8.tflite
```

También se generan archivos de resultados, por ejemplo:

```text
tabla_comparativa_escenarios.csv
resultados_evaluacion_50_corregido.csv
```

Los nombres y las rutas pueden variar según la configuración local. No todos los archivos se generan en una sola ejecución.

## 11. Video de presentación

**Video de presentación:** [Ver video del proyecto en YouTube](https://youtu.be/gTNeNFnn4L0)

## 12. Limitaciones y trabajo futuro

- Mejorar la generalización del modelo con fotografías regionales representativas.
- Revisar el balance y la diversidad de imágenes por clase.
- Separar imágenes similares o aumentadas para evitar fuga de datos entre entrenamiento y evaluación.
- Medir el tamaño real de cada archivo exportado y la latencia en el dispositivo de destino.
- Evaluar cada variante TFLite con el mismo conjunto de prueba y las mismas condiciones.
- Revisar el preprocesamiento de imágenes para que sea idéntico durante entrenamiento e inferencia.
- Ampliar la evaluación con fotografías independientes y métricas por categoría.

## 13. Autoría

Proyecto académico de clasificación de daños viales mediante técnicas de aprendizaje profundo.