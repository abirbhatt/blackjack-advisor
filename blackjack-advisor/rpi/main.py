# main.py
# Entry point for the Blackjack Card Counting Advisor system.
# Ties together the camera, CV pipeline, card counting, EV calculation,
# MQTT publishing, SMS alerts, Flask UI, and SQLite logging.
#
# Note: AI tools (Claude) were used to assist with code development.
# All design decisions are our own.
#
# Run with: python3 main.py

import threading
import time
import signal
import sys
import cv2

from capture import init_camera, capture_frame, close_camera
from detect import detect_cards
from classify import load_model, classify_card
from deck_manager import DeckManager
from counter import HiLoCounter
from ev_calculator import EVCalculator
from kelly import kelly_bet
from mqtt_publisher import MQTTPublisher
from sms_sender import SMSSender
from logger import Logger
from flask_ui import create_app

# ── Configuration ────────────────────────────────────────────────────────────

PLAYER_PHONE  = "+16502085215"   # TODO: replace with your phone number e.g. "+12135551234"
BASE_BET      = 10               # Minimum bet in dollars
BANKROLL      = 500              # Starting bankroll in dollars
CV_CONFIDENCE = 0.85             # Minimum CNN confidence to commit a classification
CAPTURE_FPS   = 2                # Frames to process per second

# ── Module-level camera reference so shutdown() can access it ────────────────
camera = None

# ── Globals (shared between threads) ─────────────────────────────────────────

game_state = {
    "detected_cards":    [],
    "running_count":     0,
    "true_count":        0.0,
    "deck_state":        {},
    "recommendation":    None,
    "ev_breakdown":      {},
    "bet_recommendation": BASE_BET,
    "player_hand":       [],
    "dealer_upcard":     None,
}

# ── Graceful Shutdown ─────────────────────────────────────────────────────────

def shutdown(sig, frame):
    print("\n[main] Shutting down gracefully...")
    if camera is not None:
        close_camera(camera)
    mqtt.disconnect()
    logger.close()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

# ── Main Processing Loop ──────────────────────────────────────────────────────

def processing_loop():
    global camera
    camera       = init_camera()
    model        = load_model("model.tflite")
    deck_mgr     = DeckManager()
    counter      = HiLoCounter()
    ev_calc      = EVCalculator()
    previous_cards = set()

    while True:
      try:
        frame = capture_frame(camera)

        # Step 1: Detect card bounding boxes with OpenCV
        # Pass the baseline (empty table) frame for background subtraction
        bounding_boxes = detect_cards(frame, camera.baseline_frame)
        print(f"[main] Detected {len(bounding_boxes)} box(es) | cards={current_cards if 'current_cards' in dir() else 'n/a'}", end='\r')

        # Step 2: Classify each detected card with the CNN
        # Crop the full card region and resize to 64x64 — same as training data
        current_cards = set()
        for box in bounding_boxes:
            x, y, w, h = box
            card_img = cv2.resize(frame[y:y + h, x:x + w], (64, 64))
            rank, suit, confidence = classify_card(model, card_img)
            if confidence >= CV_CONFIDENCE:
                current_cards.add((rank, suit))

        # Step 3: Table state reconciliation — only process NEW cards
        new_cards = current_cards - previous_cards
        for rank, suit in new_cards:
            deck_mgr.remove_card(rank)
            counter.update(rank)

        previous_cards = current_cards

        # Step 4: Compute counts AFTER updating for new cards
        running, true = counter.get_counts(deck_mgr.decks_remaining())

        # Log new cards now that we have updated counts
        for rank, suit in new_cards:
            logger.log_card(rank, suit, running, true)

        # Step 5: Update game state
        best_action, ev_breakdown = ev_calc.recommend(
            game_state["player_hand"],
            game_state["dealer_upcard"],
            deck_mgr.deck_state
        )
        bet_rec = kelly_bet(true, BANKROLL, BASE_BET)

        game_state.update({
            "detected_cards":     list(current_cards),
            "running_count":      running,
            "true_count":         true,
            "deck_state":         deck_mgr.deck_state.copy(),
            "recommendation":     best_action,
            "ev_breakdown":       ev_breakdown,
            "bet_recommendation": bet_rec,
        })

        # Step 6: Publish to MQTT (Node 2 — HiveMQ Cloud)
        if new_cards:
            for rank, suit in new_cards:
                mqtt.publish_card(rank, suit)
            mqtt.publish_deck_state(deck_mgr.deck_state)
            mqtt.publish_count(running, true, bet_rec)
        if best_action:
            mqtt.publish_recommendation(best_action, ev_breakdown)

        # Step 7: Log to InfluxDB Cloud
        if new_cards:
            logger.log_to_influx(running, true, bet_rec, best_action)

        # Step 8: Send SMS if it's the player's decision point
        # TODO: Trigger SMS when player_hand is set and it's their turn

        time.sleep(1 / CAPTURE_FPS)

      except Exception as e:
        import traceback
        print(f"\n[main] Processing error: {e}")
        traceback.print_exc()
        time.sleep(1)


if __name__ == "__main__":
    global mqtt, logger, sms

    mqtt   = MQTTPublisher()
    logger = Logger()
    sms    = SMSSender(PLAYER_PHONE)

    # Start Flask UI in a background thread
    app = create_app(game_state)
    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5000, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    print("[main] Flask UI running at http://blackjackpi.local:5000")

    # Start main processing loop (blocking)
    processing_loop()
