import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D, Lambda
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam

def load_images_with_time(folder, target_size=(64, 64)):
    """
    Loads images from a folder, converts them to grayscale, and resizes them.
    Returns a numpy array of shape (num_samples, 1, height, width, channels)
    where the time dimension is added as 1.
    """
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
            path = os.path.join(folder, filename)
            img = load_img(path, color_mode='grayscale', target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)
    images = np.array(images).astype('float32') / 255.0
    # Add a time dimension: (samples, time_steps, height, width, channels)
    return np.expand_dims(images, axis=1)

# --- Load Healthy Images (NORMAL) for Training ---
normal_folder = r'C:\Users\ali\Desktop\VIsual_code\Computer vision\Proj\chest_xray\NORMAL'
x_train_convlstm = load_images_with_time(normal_folder, target_size=(64, 64))

# --- Build the ConvLSTM Autoencoder ---
# Input shape: (time_steps, height, width, channels) with time_steps=1.
input_img = Input(shape=(1, 64, 64, 1))

# Encoder: Apply ConvLSTM layers.
x = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
               return_sequences=True)(input_img)
x = BatchNormalization()(x)
# Return a single frame representation.
x = ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same',
               return_sequences=False)(x)
encoded = x  # Shape: (64, 64, 16)

# Decoder: Add back a time dimension using a Lambda layer with an explicit output shape.
x = Lambda(lambda z: tf.expand_dims(z, axis=1),
           output_shape=lambda s: (s[0], 1, s[1], s[2], s[3]))(encoded)  # New shape: (batch, 1, 64, 64, 16)
x = ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same',
               return_sequences=True)(x)
x = BatchNormalization()(x)
x = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
               return_sequences=True)(x)
# Use TimeDistributed to apply Conv2D on the time dimension.
decoded = TimeDistributed(Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same'))(x)

convlstm_autoencoder = Model(input_img, decoded)
convlstm_autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
convlstm_autoencoder.summary()

# --- Train the ConvLSTM Autoencoder on Healthy Images ---
convlstm_autoencoder.fit(x_train_convlstm, x_train_convlstm,
                         epochs=50,
                         batch_size=128,
                         validation_split=0.2)

# --- Save the Trained Model ---
model_save_path = r'C:\Users\ali\Desktop\VIsual_code\Computer vision\Proj\convlstm_autoencoder_model.h5'
convlstm_autoencoder.save(model_save_path)
print("Model saved to:", model_save_path)
