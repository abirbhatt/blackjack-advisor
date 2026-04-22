# detect.py
# OpenCV-based card detection pipeline.
# Takes a raw camera frame and returns bounding boxes for each detected card.
#
# Pipeline:
#   1. Background subtraction   — isolates regions that changed vs. empty table
#   2. Grayscale conversion     — simplifies processing (color not needed for shapes)
#   3. Gaussian blur            — reduces pixel-level noise
#   4. Thresholding             — converts to pure black/white
#   5. Contour detection        — finds outlines of white regions
#   6. Aspect ratio filtering   — keeps only card-shaped contours (~1:1.4)
#   7. Bounding box extraction  — returns cropped regions for the classifier
#
# Note: AI tools (Claude) were used to assist with code development.

import cv2
import numpy as np

# A standard playing card is 63mm x 88mm → aspect ratio ≈ 0.716
# Range is intentionally wide to tolerate perspective distortion and camera angle
CARD_ASPECT_RATIO_MIN = 0.45
CARD_ASPECT_RATIO_MAX = 0.95
CARD_MIN_AREA         = 3000   # Minimum pixel area to ignore small noise


def detect_cards(frame, baseline_frame):
    """
    Detect playing cards in a frame using background subtraction and contour detection.

    Args:
        frame (np.ndarray):          Current camera frame (BGR).
        baseline_frame (np.ndarray): Empty-table reference frame (BGR).

    Returns:
        List of (x, y, w, h) bounding boxes for each detected card.
    """
    # Step 1: Background subtraction
    # Subtract the empty table from the current frame. Pixels that haven't
    # changed (the felt) become near-zero. Pixels where cards appeared are bright.
    diff = cv2.absdiff(frame, baseline_frame)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Step 3: Gaussian blur — smooth out camera noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Threshold — convert to binary (black/white)
    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)

    # Step 5: Find contours — trace outlines of white regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < CARD_MIN_AREA:
            continue  # Skip tiny noise regions

        x, y, w, h = cv2.boundingRect(contour)

        # Step 6: Filter by aspect ratio
        aspect = w / h if h > 0 else 0
        if CARD_ASPECT_RATIO_MIN <= aspect <= CARD_ASPECT_RATIO_MAX:
            boxes.append((x, y, w, h))

    return boxes


def crop_corner(frame, box):
    """
    Crop the top-left corner of a detected card (where rank and suit are printed).
    This is the region fed to the CNN classifier.

    Args:
        frame (np.ndarray): Full camera frame.
        box (tuple):        (x, y, w, h) bounding box of the card.

    Returns:
        np.ndarray: Cropped corner image resized to 64x64.
    """
    x, y, w, h = box
    corner_w = int(w * 0.25)
    corner_h = int(h * 0.30)
    corner = frame[y:y + corner_h, x:x + corner_w]
    return cv2.resize(corner, (64, 64))
