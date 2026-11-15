import requests
import os
import random

API_URL = "http://127.0.0.1:8000/predict/"
DATASET_DIR = "dataset_vial_propio"

# Seleccionar una imagen real al azar para simular la captura del celular
clases = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
clase_aleatoria = random.choice(clases)
ruta_clase = os.path.join(DATASET_DIR, clase_aleatoria)
foto_prueba = random.choice([f for f in os.listdir(ruta_clase) if f.endswith(('.jpg', '.png'))])
ruta_completa = os.path.join(ruta_clase, foto_prueba)

print(f"-> Simulando envío desde dispositivo móvil...")
print(f"-> Imagen capturada en campo: '{foto_prueba}' (Categoría real: {clase_aleatoria})")

with open(ruta_completa, "rb") as img_file:
    files = {"file": (foto_prueba, img_file, "image/jpeg")}
    try:
        response = requests.post(API_URL, files=files)
        print("\nRespuesta recibida del servidor (Código HTTP", response.status_code, "):")
        print(response.json())
    except requests.exceptions.ConnectionError:
        print("\n[AVISO] No se pudo conectar a la API. Asegúrate de iniciar el servidor con: uvicorn app_vial_api:app --reload")
