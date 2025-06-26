import tensorflow as tf

# Load your trained model (.h5)
model = tf.keras.models.load_model("model.h5")

# Convert it to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite model
with open("plant_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete: plant_disease_model.tflite saved.")
