from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('model.h5')

# Load and preprocess the image
img_path = r'C:\AIML project\project\_leaf.jpg'
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Mapping: update based on your training folder structure
class_labels = {
    0: 'Bacterial_spot',
    1: 'Early_blight',
    2: 'Healthy',
    3: 'Late_blight',
    4: 'Leaf_rust',
    5: 'Mosaic_virus',
    6: 'Leaf_curl',
    7: 'Powdery_mildew'
}

# Print result
print("Predicted class index:", predicted_class)
print("Predicted label:", class_labels[predicted_class])
