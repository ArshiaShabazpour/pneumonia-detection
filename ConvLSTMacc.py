import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Reuse your image loader that adds a time dimension.
def load_images_with_time(folder, target_size=(64, 64)):
    """
    Loads images from a folder, converts them to grayscale, and resizes them.
    Returns a numpy array of shape (num_samples, 1, height, width, channels)
    where the time dimension is added as 1, along with the filenames.
    """
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            path = os.path.join(folder, filename)
            img = load_img(path, color_mode='grayscale', target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)
            filenames.append(filename)
    images = np.array(images).astype('float32') / 255.0
    # Add a time dimension: (samples, time_steps, height, width, channels)
    return np.expand_dims(images, axis=1), filenames

def load_test_dataset(root_test_folder, target_size=(64, 64)):
    """
    Loads images from two subfolders (NORMAL and PNEUMONIA) and returns:
      - data: all images as a numpy array
      - labels: corresponding labels (0 for NORMAL, 1 for PNEUMONIA)
      - filenames: list of filenames (for debugging/visualization)
    """
    data_list = []
    labels_list = []
    filenames_list = []
    
    # Define your subfolder names and label mapping
    subfolders = {'NORMAL': 0, 'PNEUMONIA': 1}
    
    for subfolder, label in subfolders.items():
        folder_path = os.path.join(root_test_folder, subfolder)
        images, filenames = load_images_with_time(folder_path, target_size=target_size)
        data_list.append(images)
        labels_list.extend([label] * images.shape[0])
        # Prefix filenames with subfolder name for clarity.
        filenames_list.extend([f"{subfolder}/{fname}" for fname in filenames])
    
    data = np.concatenate(data_list, axis=0)
    labels = np.array(labels_list)
    return data, labels, filenames_list

# --- Set the path to your test folder (which contains subfolders NORMAL and PNEUMONIA) ---
root_test_folder = r'chest_xray\test'  # Adjust if needed

# Load the test dataset
x_test, y_test, test_filenames = load_test_dataset(root_test_folder, target_size=(64, 64))
print("Test dataset shape:", x_test.shape)
print("Number of test samples:", x_test.shape[0])

# --- Define the custom function for the Lambda layer (as before) ---
def expand_dims_layer(z):
    import tensorflow as tf  # ensure tf is defined in this scope
    return tf.expand_dims(z, axis=1)

custom_objects = {"<lambda>": expand_dims_layer}

# --- Load the Trained Model with custom_objects ---
model_save_path = r'convlstm_autoencoder_model.h5'
convlstm_autoencoder = load_model(model_save_path, custom_objects=custom_objects)
print("Loaded model from:", model_save_path)

# Patch Lambda layers so that 'tf' is in their globals.
for layer in convlstm_autoencoder.layers:
    if isinstance(layer, tf.keras.layers.Lambda):
        layer.function.__globals__['tf'] = tf

# --- Make Predictions ---
reconstructed = convlstm_autoencoder.predict(x_test)

# --- Calculate Reconstruction Error ---
# Here we use mean absolute error per sample.
reconstruction_errors = np.mean(np.abs(reconstructed - x_test), axis=(1,2,3,4))
print("Reconstruction errors shape:", reconstruction_errors.shape)

# --- Determine Predictions Based on a Threshold ---
# Set your threshold (adjust based on your error distribution).
threshold = 0.0000001  # <-- Example threshold; tune as needed.
y_pred = (reconstruction_errors > threshold).astype(int)

# --- Evaluate Accuracy ---
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['NORMAL', 'PNEUMONIA'])

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# --- (Optional) Plot the Reconstruction Error Distribution ---
plt.figure(figsize=(8, 4))
plt.hist(reconstruction_errors, bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")
plt.title("Reconstruction Error Distribution")
plt.xlabel("Mean Absolute Error")
plt.ylabel("Count")
plt.legend()
plt.show()
