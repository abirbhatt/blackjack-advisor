# capture.py
# Handles camera initialization and frame capture using picamera2.
# picamera2 is the modern Python library for the Raspberry Pi Camera Module 3
# (IMX708). It replaces the legacy picamera library and uses libcamera as its
# backend. Frames are returned as numpy arrays, which OpenCV can use directly.
#
# Note: AI tools (Claude) were used to assist with code development.

from picamera2 import Picamera2
import numpy as np

def init_camera():
    """
    Initialize the camera and return a configured Picamera2 instance.
    Captures a baseline (empty table) frame for background subtraction.
    """
    cam = Picamera2()
    config = cam.create_still_configuration(main={"size": (1920, 1080)})
    cam.configure(config)
    cam.start()

    # Allow the sensor to settle before capturing
    import time
    time.sleep(2)

    # Capture and store the baseline empty-table frame for background subtraction
    cam.baseline_frame = cam.capture_array()
    print("[capture] Camera initialized. Baseline frame captured.")
    return cam


def capture_frame(cam):
    """
    Capture and return a single frame as a numpy array (BGR format for OpenCV).
    """
    return cam.capture_array()


def close_camera(cam):
    """
    Cleanly stop and close the camera.
    """
    cam.stop()
    print("[capture] Camera closed.")
