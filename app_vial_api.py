import io
import time
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="API de Inspección y Clasificación de Daño Vial",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = [
    "alcantarilla_abierta",
    "alcantarilla_buen_estado",
    "alcantarilla_obstruida",
    "alcantarilla_tapa_rota",
    "asfalto_desprendido",
    "bache",
    "grieta",
    "hundimiento",
    "parche_pavimento",
    "piel_de_cocodrilo"
]

MODEL_PATH = "modelo_vial_finetuned.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print(f"[OK] Modelo cargado: {MODEL_PATH}")

class ImageData(BaseModel):
    image_base64: str

@app.get("/")
def health_check():
    return {"status": "online", "service": "Clasificador Vial Regional"}

@app.post("/predict_base64/")
async def predict_base64(data: ImageData):
    """Recibe la imagen directamente como texto Base64 (100% compatible con móviles)."""
    t_start = time.perf_counter()
    try:
        # Decodificar el texto Base64 a bytes
        raw_b64 = data.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
            
        img_bytes = base64.b64decode(raw_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((224, 224))
        
        img_array = np.array(image, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        # Inferencia
        predictions = model.predict(img_batch, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return JSONResponse(content={
            "success": True,
            "prediction": {
                "class_id": predicted_idx,
                "class_name": CLASS_NAMES[predicted_idx],
                "confidence_score": round(confidence, 4),
                "confidence_percent": f"{round(confidence * 100, 2)}%"
            },
            "metadata": {
                "inference_time_ms": round(latency_ms, 2)
            }
        })
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))