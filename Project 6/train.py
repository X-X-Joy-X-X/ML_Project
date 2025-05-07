import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import time

# ============================== CONFIG ==============================
no_worms_dir = '/home/rbs/Documents/venv311/worms/0/'
worms_dir   = '/home/rbs/Documents/venv311/worms/1/'
target_size = (32, 32)
batch_size = 64
epochs = 100
learning_rate = 0.001
model_output = 'cnn_worm_classifier.h5'

# ========================== IMAGE LOADING ===========================
def load_and_preprocess_images(directory, label):
    images = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            path = os.path.join(directory, filename)
            img = Image.open(path)
            img = img.resize(target_size, Image.LANCZOS)
            img = img.convert("L")  # Grayscale
            img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

            contrast = ImageEnhance.Contrast(img)
            sharpness = ImageEnhance.Sharpness(img)
            brightness = ImageEnhance.Brightness(img)

            img = contrast.enhance(4)
            img = sharpness.enhance(3)
            img = brightness.enhance(0.6)

            img_arr = np.array(img, dtype=np.float32) / 255.0
            images.append(img_arr)
            labels.append(label)

    return np.array(images), np.array(labels)

# ============================= LOAD DATA ============================
print("Loading data...")
X0, y0 = load_and_preprocess_images(no_worms_dir, 0)
X1, y1 = load_and_preprocess_images(worms_dir, 1)

X = np.concatenate([X0, X1], axis=0)
y = np.concatenate([y0, y1], axis=0)

X = X.reshape(-1, 32, 32, 1)  # add channel dimension
y_cat = tf.keras.utils.to_categorical(y, 2)

# ============================= SPLIT DATA ============================
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 60/20/20

# ============================== MODEL ===============================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D (32, (3,3), activation='relu', 
                                    input_shape=(32,32,1), 
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(2,2),
    
    tf.keras.layers.Conv2D (64, (3,3), activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=100,
    decay_rate=0.9)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================ TRAINING ==============================
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[es],
    verbose=1
)
end_time = time.time()
training_time = end_time - start_time
print(f"Training completed in {training_time:.2f} seconds")

# ============================= EVALUATE =============================
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# ============================== PLOTS ===============================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.tight_layout()
plt.show()

# ========================== SAVE MODEL ==============================
model.save(model_output)
print(f"Model saved to '{model_output}'")
