from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# 1. Load and preprocess the dataset
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r'C:\AIML project\project\dataset',    # ✅ Make sure this path is correct
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# 2. Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

# 3. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the model
model.fit(train_generator, epochs=10)

# 5. Save the model
model.save(r'C:\AIML project\project\model.h5')  # ✅ Saves it in the correct location
