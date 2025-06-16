import os
import time
import glob
import pickle
import random
import numpy as np
import logging
import pytz
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime 
from PIL import Image

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.layers import (
    Input, Conv2D, Concatenate,
    Multiply, Subtract, Add,
    Activation, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping

import cv2
from skimage.metrics import structural_similarity as ssim

IM_SIZE = (720, 1280)
LOAD_MODEL = False

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=IM_SIZE, antialias = True)
    img = img / 255.0
    return img

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def data_path(orig_img_path, hazy_img_path):
    train_img = []
    val_img = []
    test_img = []

    # Get all clear image paths
    clear_imgs = glob.glob(os.path.join(orig_img_path, '*.png'))  # ITS clear images are .png

    # Create a mapping: clear image name (without extension) → all hazy images
    hazy_imgs = glob.glob(os.path.join(hazy_img_path, '*.png'))
    hazy_img_dict = {}

    for hazy_path in hazy_imgs:
        hazy_name = os.path.basename(hazy_path)
        # Extract the clear image ID from hazy image name, e.g., "1_1.png" → "1"
        base_name = hazy_name.split('_')[0]
        hazy_img_dict.setdefault(base_name, []).append(hazy_path)

    random.shuffle(clear_imgs)
    n = len(clear_imgs)

    train_keys = clear_imgs[: int(0.8 * n)]
    val_keys = clear_imgs[int(0.8 * n): int(0.9 * n)]
    test_keys = clear_imgs[int(0.9 * n):]

    split_dict = {img: 'train' for img in train_keys}
    split_dict.update({img: 'val' for img in val_keys})
    split_dict.update({img: 'test' for img in test_keys})

    for clear_path in clear_imgs:
        clear_name = os.path.basename(clear_path)
        base_name = os.path.splitext(clear_name)[0]

        if base_name not in hazy_img_dict:
            print(f"Warning: No hazy images found for {clear_name}")
            continue

        for hazy_path in hazy_img_dict[base_name]:
            pair = [hazy_path, clear_path]
            split = split_dict[clear_path]
            if split == 'train':
                train_img.append(pair)
            elif split == 'val':
                val_img.append(pair)
            else:
                test_img.append(pair)

    return train_img, val_img, test_img

train_images, val_images, test_images = data_path(
    orig_img_path='/content/Reside/Indoor Training Set (ITS)/clear',
    hazy_img_path='/content/Reside/Indoor Training Set (ITS)/hazy'
)

print(f"Total image pairs: {len(train_images + val_images + test_images)}")
print(f"Training pairs: {len(train_images)}")
print(f"Validation pairs: {len(val_images)}")
print(f"Test pairs: {len(test_images)}")

def dataloader(train_data, val_data, test_data, batch_size=1):
    def load_image_pair(hazy_path, clear_path):
        # Read + decode → uint8 tensor
        hazy = tf.io.read_file(hazy_path)
        hazy = tf.image.decode_png(hazy, channels=3)
        # Cast to float32 and normalize in one step:
        hazy = tf.cast(hazy, tf.float32) / 255.0

        clear = tf.io.read_file(clear_path)
        clear = tf.image.decode_png(clear, channels=3)
        clear = tf.cast(clear, tf.float32) / 255.0

        return hazy, clear

    def create_dataset(pairs):
        if not pairs:
            # empty fallback
            return tf.data.Dataset.from_tensor_slices(([], []))

        # unzip into two Python lists of strings
        hazy_paths, clear_paths = zip(*pairs)
        ds = tf.data.Dataset.from_tensor_slices(
            (list(hazy_paths), list(clear_paths))
        )
        ds = ds.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return (
        create_dataset(train_data),
        create_dataset(val_data),
        create_dataset(test_data),
    )

# Get already split data from data_path()
train_data, val_data, test_data = data_path(
    orig_img_path='/content/Reside/Indoor Training Set (ITS)/clear',
    hazy_img_path ='/content/Reside/Indoor Training Set (ITS)/hazy'
)

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

print(f"Training pairs:   {len(train_data)}")
print(f"Validation pairs: {len(val_data)}")
print(f"Test pairs:       {len(test_data)}")

# Load datasets using dataloader
train, val, test = dataloader(train_data, val_data, test_data, batch_size=1)

def preprocess(img):
    return tf.image.convert_image_dtype(img, tf.float32)

# Apply if needed (only once before training/testing)
train = train.map(lambda x, y: (preprocess(x), preprocess(y)))
val = val.map(lambda x, y: (preprocess(x), preprocess(y)))
test = test.map(lambda x, y: (preprocess(x), preprocess(y)))


# Setup logging
logging.basicConfig(level=logging.INFO)

def dehaze_net():
    inputs = Input(shape=(None, None, 3))  # Dynamic input

    # === Feature Extraction Blocks ===
    conv1 = Conv2D(16, 1, padding='same', activation='relu')(inputs)
    conv2 = Conv2D(16, 3, padding='same', activation='relu')(conv1)
    concat1 = Concatenate()([conv1, conv2])

    conv3 = Conv2D(16, 5, padding='same', activation='relu')(concat1)
    concat2 = Concatenate()([conv2, conv3])

    conv4 = Conv2D(16, 7, padding='same', activation='relu')(concat2)
    concat3 = Concatenate()([conv1, conv2, conv3, conv4])

    conv5 = Conv2D(16, 3, padding='same', activation='relu')(concat3)
    K = Conv2D(3, 1, padding='same')(conv5)  # K-map

    # === AOD-Net Dehaze Formula ===
    k_mul_x   = Multiply()([K, inputs])           # K * x
    k_mul_x_m = Subtract()([k_mul_x, K])          # K*x - K

    # Reshape to ensure broadcast compatibility for adding constant
    constant_1 = tf.constant(1., dtype=tf.float32)
    constant_1_reshaped = tf.reshape(constant_1, [1, 1, 1, 1])

    plus_one  = Add()([k_mul_x_m, constant_1_reshaped])  # +1

    # === Refinement Block ===
    x = Conv2D(32, 3, padding='same', activation='relu')(plus_one)
    x = BatchNormalization()(x)
    x = Conv2D(3, 3, padding='same')(x)

    # === Final Activation to clip into [0,1] ===
    output = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    model.summary()
    return model

# === Configurable Parameters & Training Settings ===
EPOCHS_TO_TRAIN = 1  # Update daily as needed
MODEL_PATH = '/content/aod_net_refined.keras'
HISTORY_PATH = '/content/training_history.pkl'

#early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === Data Summary ===
print(f"Train batches:                 {len(train)}")
print(f"Validation batches:            {len(val)}")
print(f"Number of Epochs this session: {EPOCHS_TO_TRAIN}")

# === Track Execution Time ===
start_time = time.time()  # Record the start time

# === Load or Initialize Model ===
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=MeanSquaredError())
    with open(HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)
    print("Loaded existing model and history.")
else:
    model = dehaze_net()
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=MeanSquaredError())
    history = {'loss': [], 'val_loss': []}  # Initialize empty history
    print("New model created.")

# === Always Train for More Epochs ===
try:
    # Use the first code if Early Stopping is used else use the second code!
    #history_obj = model.fit(
        #train,
        #validation_data=val,
        #epochs=EPOCHS_TO_TRAIN,
        #verbose=1,
        #callbacks=[early_stop]
    #)
    history_obj = model.fit(
        train,
        validation_data=val,
        epochs=EPOCHS_TO_TRAIN,
        verbose=1
    )
except Exception as e:
    print(f"Training stopped due to error: {e}")
else:
    # Update and extend history
    new_history = history_obj.history
    for key in new_history:
        if key not in history:
            history[key] = []
        history[key].extend(new_history[key])

    # Save updated history and model
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history, f)
    model.save(MODEL_PATH)
    print("\nModel and training history saved successfully.")

# === Track Total Execution Time ===
end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time  # Calculate elapsed time

hours = elapsed_time // 3600
minutes = (elapsed_time % 3600) // 60
seconds = elapsed_time % 60

print(f"\nTotal epochs trained so far: {len(history['loss'])}")
print(f"Training completed in {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds.")

# === Logging Training Session Details ===
log_path = "/content/training_log.txt"  # Log file path

# Calculate average training and validation loss for this session
avg_train_loss = sum(history_obj.history['loss']) / len(history_obj.history['loss'])
avg_val_loss = sum(history_obj.history['val_loss']) / len(history_obj.history['val_loss'])

india_timezone = pytz.timezone("Asia/Kolkata")
timestamp = datetime.now(india_timezone).strftime("%Y-%m-%d %I:%M:%S %p (%Z)")

log_entry = (
    f"Date & Time:                 {timestamp}\n"
    f"Epochs Trained This Session: {EPOCHS_TO_TRAIN}\n"
    f"Total Epochs Trained So Far: {len(history['loss'])}\n"
    f"Time Taken:                  {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
    f"Average Training Loss:       {avg_train_loss:.4f}\n"
    f"Average Validation Loss:     {avg_val_loss:.4f}\n"
    f"{'-'*40}\n"
)

# Read existing log file and prepend the new entry at the top
if os.path.exists(log_path):
    with open(log_path, "r") as log_file:
        existing_log = log_file.read()
    with open(log_path, "w") as log_file:
        log_file.write(log_entry + existing_log)  # Prepend the new log entry
else:
    # If log file does not exist, create it and write the log entry
    with open(log_path, "w") as log_file:
        log_file.write(log_entry)

print(f"\nTraining log updated at {log_path}")

def display_img(model, hazy, clear, index, results_dir="Results"):
    # Ensure input image has shape (H, W, 3)
    if len(hazy.shape) == 4:
        hazy = tf.squeeze(hazy, axis=0)
    if len(clear.shape) == 4:
        clear = tf.squeeze(clear, axis=0)

    # Predict and squeeze batch dimension
    pred = model.predict(tf.expand_dims(hazy, 0))[0]

    # Clip values for display (ensure they're between 0 and 1)
    hazy = np.clip(hazy, 0, 1)
    clear = np.clip(clear, 0, 1)
    pred = np.clip(pred, 0, 1)

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # === Save output dehazed image ===
    dehazed_filename = os.path.join(results_dir, f"dehazed{index}.png")
    pred_uint8 = (pred * 255).astype(np.uint8)
    im_pred = Image.fromarray(pred_uint8)
    im_pred.save(dehazed_filename)
    print(f"Saved output image to: {dehazed_filename}")

    # === Save real clear image ===
    real_filename = os.path.join(results_dir, f"real{index}.png")
    clear_uint8 = (clear * 255).astype(np.uint8)
    im_clear = Image.fromarray(clear_uint8)
    im_clear.save(real_filename)
    print(f"Saved real clear image to: {real_filename}")

    # === Plot images ===
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(hazy)
    axs[0].set_title('Input Hazy Image')
    axs[0].axis('off')

    axs[1].imshow(clear)
    axs[1].set_title('Real Clear Image')
    axs[1].axis('off')

    axs[2].imshow(pred)
    axs[2].set_title('Output Dehazed Image')
    axs[2].axis('off')

    plt.show()

# === Example Usage ===
for i, (hz, og) in enumerate(test):
    if i == 3:  # Limiting to 3 examples
        break
    print(f"\nExample {i+1}:")
    display_img(model, hz, og, index=i+1)

# MSE Vs Epoch Graph
print("MSE Vs Epoch Graph\n")

plt.title("Plotting Loss (MSE) versus Epoch")
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.001, 0.1])
plt.legend()
# Save the plot before displaying
plot_path = "/content/mse_vs_epoch_graph.png"
plt.savefig(plot_path)
plt.show()

print(f"Plot saved as {plot_path}")

# Training loss and Validation loss at the last epoch of training
train_loss = history['loss'][-1]
val_loss = history['val_loss'][-1]

print(f"Training loss at the last epoch of training: {train_loss}")
print(f"Validation loss at the last epoch of training: {val_loss}")

# Calculate PSNR Value
import math
max_pixel_value = 255
val_MSE = history['val_loss'][-1]
PSNR = 10 * math.log10((max_pixel_value ** 2) / val_MSE)
print(f"PSNR Value: {PSNR}")

# Calculate SSIM Value
# Paths to the images
image1_path = "/content/Results/real1.png"     # Reference clear image
image2_path = "/content/Results/dehazed1.png"  # Generated dehazed image

# Check if both files exist
if not os.path.exists(image1_path):
    raise FileNotFoundError(f"File not found: {image1_path}")
if not os.path.exists(image2_path):
    raise FileNotFoundError(f"File not found: {image2_path}")

# Load the images using OpenCV
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Check if images were loaded successfully
if image1 is None:
    raise ValueError(f"Failed to load image: {image1_path}")
if image2 is None:
    raise ValueError(f"Failed to load image: {image2_path}")

# Resize image1 to match image2's size, if necessary
if image1.shape != image2.shape:
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

# Convert images from BGR (OpenCV default) to RGB
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Compute SSIM (for color images)
ssim_value, _ = ssim(image1_rgb, image2_rgb, full=True, channel_axis=-1)

# Print the SSIM value
print(f"SSIM: {ssim_value:.4f}")
