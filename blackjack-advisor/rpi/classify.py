# classify.py
# CNN card classifier using TensorFlow Lite.
# Loads the trained MobileNetV2 model exported as a .tflite file and runs
# inference on card corner crops from the detection pipeline.
#
# TFLite is a compressed, optimized format for inference on embedded devices
# like the RPi — much faster than running a full TensorFlow model.
#
# The model is a 52-class classifier (13 ranks × 4 suits).
# If confidence is below CV_CONFIDENCE (0.85), the result is flagged as
# uncertain rather than committing a potential misclassification.
#
# Note: AI tools (Claude) were used to assist with code development.

import numpy as np
import tensorflow as tf
tflite = tf.lite

# Ordered list of class labels — must match the order used during training
CARD_LABELS = [
    "2_clubs", "2_diamonds", "2_hearts", "2_spades",
    "3_clubs", "3_diamonds", "3_hearts", "3_spades",
    "4_clubs", "4_diamonds", "4_hearts", "4_spades",
    "5_clubs", "5_diamonds", "5_hearts", "5_spades",
    "6_clubs", "6_diamonds", "6_hearts", "6_spades",
    "7_clubs", "7_diamonds", "7_hearts", "7_spades",
    "8_clubs", "8_diamonds", "8_hearts", "8_spades",
    "9_clubs", "9_diamonds", "9_hearts", "9_spades",
    "10_clubs", "10_diamonds", "10_hearts", "10_spades",
    "J_clubs", "J_diamonds", "J_hearts", "J_spades",
    "Q_clubs", "Q_diamonds", "Q_hearts", "Q_spades",
    "K_clubs", "K_diamonds", "K_hearts", "K_spades",
    "A_clubs", "A_diamonds", "A_hearts", "A_spades",
]


def load_model(model_path):
    """
    Load a TFLite model from disk and allocate tensors.

    Args:
        model_path (str): Path to the .tflite model file.

    Returns:
        tflite.Interpreter: Ready-to-use interpreter.
    """
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print(f"[classify] Model loaded from {model_path}")
    return interpreter


def classify_card(interpreter, corner_img):
    """
    Run inference on a card corner crop and return the rank, suit, and confidence.

    Args:
        interpreter:              Loaded TFLite interpreter.
        corner_img (np.ndarray):  64x64 BGR image of the card corner.

    Returns:
        tuple: (rank: str, suit: str, confidence: float)
               e.g. ("A", "spades", 0.97)
               Returns (None, None, 0.0) if model input shape doesn't match.
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess: normalize to [0, 1] and add batch dimension
    img = corner_img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Shape: (1, 64, 64, 3)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]  # Shape: (52,)

    # Softmax output — highest value is the predicted class
    predicted_idx  = int(np.argmax(output))
    confidence     = float(output[predicted_idx])
    label          = CARD_LABELS[predicted_idx]   # e.g. "A_spades"
    rank, suit     = label.split("_", 1)

    return rank, suit, confidence
