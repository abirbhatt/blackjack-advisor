#!/usr/bin/env python3
"""
collect_training_data.py
Run this on the Raspberry Pi to collect training images for the card classifier CNN.

HOW IT WORKS
------------
The script cycles through all 52 cards (13 ranks × 4 suits).
For each card it captures 8 photos (slight angle/position adjustments between shots).
Images are saved as 64×64 corner crops — the same crop that classify.py sees at
inference time — so training and inference input distributions match exactly.

All 4 suits for the same rank (e.g., 2♥ 2♦ 2♣ 2♠) go into the same folder ("2"),
giving 32 diverse images per rank label (plenty for transfer learning fine-tuning).

OUTPUT FOLDER STRUCTURE  (matches Keras ImageDataGenerator.flow_from_directory)
─────────────────────────
training/
  data/
    2/   00001.jpg  00002.jpg  ...
    3/   ...
    10/  ...
    J/   ...
    Q/   ...
    K/   ...
    A/   ...

HOW TO USE
----------
  1. SSH into the Pi.
  2. cd ~/blackjack-advisor/training
  3. python3 collect_training_data.py
  4. For each card: place the card face-up under the camera with the rank+suit
     corner in the UPPER-LEFT of the frame. Press ENTER to capture. Slightly
     shift/tilt the card between shots for variety.
  5. Press 's' + ENTER to skip a card, 'q' + ENTER to quit (safe to resume later).

RESUMING
--------
Already-captured images are counted at startup. Any card with enough photos
is skipped automatically. Just re-run to pick up where you left off.

Note: AI tools (Claude) were used to assist with code development.
"""

import os
import sys
import time
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
SAVE_DIR        = Path(__file__).parent / "data"
PHOTOS_PER_RANK = 32          # 8 per suit × 4 suits — adjustable
SHOTS_PER_CARD  = 8           # photos per individual card (one suit)
CAPTURE_RES     = (1280, 720) # Pi camera resolution during capture

RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
SUITS = ['Hearts ♥', 'Diamonds ♦', 'Clubs ♣', 'Spades ♠']


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_existing(rank: str) -> int:
    """Count how many images we already have for this rank."""
    folder = SAVE_DIR / rank
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.jpg")))


def progress_bar(n: int, total: int, width: int = 20) -> str:
    filled = int(width * n / total)
    return "█" * filled + "░" * (width - filled)


def print_summary():
    """Print a progress table for all ranks."""
    print("\n  Rank  Progress                    Count")
    print("  " + "─" * 44)
    for rank in RANKS:
        n     = count_existing(rank)
        bar   = progress_bar(n, PHOTOS_PER_RANK)
        done  = "✓ done" if n >= PHOTOS_PER_RANK else f"{n}/{PHOTOS_PER_RANK}"
        print(f"  {rank:>3}   [{bar}]  {done}")
    print()


def find_card_in_frame(frame_bgr, baseline_bgr):
    """
    Use background subtraction to locate the card in the frame.
    Returns a 64×64 BGR crop of the card region, or None if no card found.

    This mirrors the detect.py pipeline so training images look exactly
    like what the classifier will see at inference time (whole card, not
    a corner sliver of the raw camera frame).
    """
    import cv2

    diff     = cv2.absdiff(frame_bgr, baseline_bgr)
    gray     = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour is the card
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 3000:
        return None   # Too small — probably noise

    x, y, w, h = cv2.boundingRect(largest)
    card_crop  = frame_bgr[y:y + h, x:x + w]
    return cv2.resize(card_crop, (64, 64))


def capture_one_card(cam, baseline_bgr, rank: str, suit: str) -> bool:
    """
    Interactively capture SHOTS_PER_CARD images for a single card.
    Returns True on success, False if the user quits.
    """
    import cv2

    folder   = SAVE_DIR / rank
    folder.mkdir(parents=True, exist_ok=True)
    existing = count_existing(rank)
    needed   = max(0, PHOTOS_PER_RANK - existing)

    # How many shots to take for this suit (at most SHOTS_PER_CARD)
    to_shoot = min(SHOTS_PER_CARD, needed)
    if to_shoot == 0:
        return True   # Already enough for this rank

    print(f"\n  ┌─ {rank} of {suit} {'─'*(35 - len(rank) - len(suit))}")
    print(f"  │  Rank folder has {existing}/{PHOTOS_PER_RANK} images. Taking {to_shoot} more.")
    print(f"  │  Place the card ANYWHERE in frame — the script finds it automatically.")
    print(f"  │  ENTER = capture  |  s = skip this card  |  q = quit")
    print(f"  └{'─'*40}")

    captured = 0
    while captured < to_shoot:
        prompt = f"  [{captured + 1}/{to_shoot}]  Ready? > "
        try:
            inp = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted.")
            return False

        if inp == 'q':
            print("  Quitting. Run again to resume — progress is saved.")
            sys.exit(0)
        if inp == 's':
            print(f"  Skipped {rank} of {suit}.")
            return True

        # Capture frame and convert RGB → BGR
        raw       = cam.capture_array()
        frame_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

        # Locate card using background subtraction (same pipeline as inference)
        card_img = find_card_in_frame(frame_bgr, baseline_bgr)
        if card_img is None:
            print("  ⚠ No card detected — make sure the card is visible and the table is clear otherwise.")
            continue

        # Save
        idx      = count_existing(rank) + 1
        filename = folder / f"{idx:05d}.jpg"
        ok = cv2.imwrite(str(filename), card_img)
        if not ok:
            print(f"  ERROR: cv2.imwrite failed — shape={card_img.shape} dtype={card_img.dtype}")
            continue
        captured += 1
        print(f"  Saved → {filename.name}  (rank '{rank}' total: {idx})")

    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python not found. Run:  pip install opencv-python --break-system-packages")
        sys.exit(1)

    try:
        from picamera2 import Picamera2
    except ImportError:
        print("ERROR: picamera2 not found. Run:  sudo apt install python3-picamera2")
        sys.exit(1)

    print("\n" + "═" * 55)
    print("   Blackjack CNN Training Data Collector")
    print("═" * 55)
    print(f"   Save directory : {SAVE_DIR.resolve()}")
    print(f"   Target per rank: {PHOTOS_PER_RANK} images ({SHOTS_PER_CARD} per suit × 4 suits)")
    print(f"   Total target   : {PHOTOS_PER_RANK * len(RANKS)} images across 13 rank folders")

    print_summary()

    # Check if already complete
    complete = [r for r in RANKS if count_existing(r) >= PHOTOS_PER_RANK]
    remaining = [r for r in RANKS if count_existing(r) < PHOTOS_PER_RANK]
    if not remaining:
        print("  ✓ All ranks complete! Nothing to capture.")
        return

    print(f"  Ranks still needed: {', '.join(remaining)}")
    try:
        input("\n  Press ENTER to start the camera, or Ctrl+C to cancel...\n")
    except (EOFError, KeyboardInterrupt):
        print("  Cancelled.")
        return

    # Init camera — preview config keeps the sensor streaming so AE/AWB settle
    print("  Initializing Pi camera...")
    import cv2
    cam    = Picamera2()
    config = cam.create_preview_configuration(main={"size": CAPTURE_RES, "format": "RGB888"})
    cam.configure(config)
    cam.start()
    time.sleep(3)   # Let AE/AWB converge before capturing baseline

    # Capture baseline (empty table, no cards)
    print("  ── Make sure the table is EMPTY, then press ENTER to capture baseline...")
    input("  > ")
    raw_baseline  = cam.capture_array()
    baseline_bgr  = cv2.cvtColor(raw_baseline, cv2.COLOR_RGB2BGR)
    print("  Baseline captured. Camera ready.\n")

    try:
        cards_done = 0
        for rank in RANKS:
            for suit in SUITS:
                if count_existing(rank) >= PHOTOS_PER_RANK:
                    break   # This rank is done — skip remaining suits
                ok = capture_one_card(cam, baseline_bgr, rank, suit)
                if not ok:
                    break
                cards_done += 1
    finally:
        cam.stop()
        print("\n" + "═" * 55)
        print("  Camera closed.")
        print_summary()
        print(f"  Images saved to: {SAVE_DIR.resolve()}")
        print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
