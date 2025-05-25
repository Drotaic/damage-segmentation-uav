import tensorflow as tf
import numpy as np
import cv2
import sys
import os

# Load saved model
MODEL_PATH = '../models/saved_model'
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(image_path, img_size=256):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0), img

def segment(image_path):
    input_img, original_img = preprocess(image_path)
    mask = model.predict(input_img)[0]
    mask = np.argmax(mask, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = tf.image.resize(mask, (original_img.shape[0], original_img.shape[1]), method='nearest').numpy().astype(np.uint8)

    color_mask = np.zeros_like(original_img)
    color_map = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}  # green, yellow, red
    for k, color in color_map.items():
        color_mask[mask.squeeze() == k] = color

    overlay = cv2.addWeighted(original_img, 0.7, color_mask, 0.3, 0)
    cv2.imshow("Damage Segmentation", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        segment(sys.argv[1])
    else:
        print("Usage: python segment.py <image_path>")
