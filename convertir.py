import os
import tensorflow as tf
import keras

print("1. Cargando el modelo Keras...")
model = keras.models.load_model("modelo_vial_finetuned.keras")

print("2. Exportando a formato TensorFlow Lite...")
# En Keras 3 se puede exportar directamente o convertir el modelo base
try:
    # Metodo A: Conversor estandar de TF
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
except Exception as e:
    print(f"Metodo A fallo ({e}), intentando via export...")
    # Metodo B: Exportar arquitectura y pesos
    model.export("saved_model_temp", format="tf_saved_model")
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_temp")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

print("3. Guardando archivo modelo_vial.tflite...")
with open("modelo_vial.tflite", "wb") as f:
    f.write(tflite_model)

peso_mb = os.path.getsize("modelo_vial.tflite") / (1024 * 1024)
print(f"¡CONVERSION EXITOSA! El archivo pesa: {peso_mb:.2f} MB")