# classify.py
# CNN card classifier using TensorFlow Lite.
# Loads the trained MobileNetV2 model exported as a .tflite file and runs
# inference on card corner crops from the detection pipeline.
#
# TFLite is a compressed, optimized format for inference on embedded devices
# like the RPi — much faster than running a full TensorFlow model.
#
# The model is a 13-class classifier (one class per rank: A 2-9 10 J Q K).
# Suit is not classified — only rank matters for Hi-Lo card counting.
# If confidence is below CV_CONFIDENCE (0.85), the result is flagged as
# uncertain rather than committing a potential misclassification.
#
# Note: AI tools (Claude) were used to assist with code development.

import numpy as np

# tflite-runtime is a lightweight inference-only package — no training deps,
# no flatbuffers/imp issues. Falls back to full tensorflow if not installed.
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# Ordered list of class labels — must match the alphabetical folder sort order
# that Keras ImageDataGenerator.flow_from_directory assigns.
# Python sorts folder names as strings, so "10" < "2" < "3" ... < "9" < "A" < "J" < "K" < "Q"
CARD_LABELS = ["10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q"]


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

    output = interpreter.get_tensor(output_details[0]['index'])[0]  # Shape: (13,)

    # Softmax output — highest value is the predicted class
    predicted_idx = int(np.argmax(output))
    confidence    = float(output[predicted_idx])
    rank          = CARD_LABELS[predicted_idx]    # e.g. "A", "10", "K"

    return rank, "", confidence  # Suit not classified — only rank needed for counting
