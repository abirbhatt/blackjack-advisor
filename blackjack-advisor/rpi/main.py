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

PLAYER_PHONE  = "+1XXXXXXXXXX"   # Player's phone number for SMS
BASE_BET      = 10               # Minimum bet in dollars
BANKROLL      = 500              # Starting bankroll in dollars
CV_CONFIDENCE = 0.85             # Minimum CNN confidence to commit a classification
CAPTURE_FPS   = 2                # Frames to process per second

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
    close_camera()
    mqtt.disconnect()
    logger.close()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

# ── Main Processing Loop ──────────────────────────────────────────────────────

def processing_loop():
    camera       = init_camera()
    model        = load_model("model.tflite")
    deck_mgr     = DeckManager()
    counter      = HiLoCounter()
    ev_calc      = EVCalculator()
    previous_cards = set()

    while True:
        frame = capture_frame(camera)

        # Step 1: Detect card bounding boxes with OpenCV
        bounding_boxes = detect_cards(frame)

        # Step 2: Classify each detected card with the CNN
        current_cards = set()
        for box in bounding_boxes:
            rank, suit, confidence = classify_card(model, frame, box)
            if confidence >= CV_CONFIDENCE:
                current_cards.add((rank, suit))

        # Step 3: Table state reconciliation — only process NEW cards
        new_cards = current_cards - previous_cards
        for rank, suit in new_cards:
            deck_mgr.remove_card(rank)
            counter.update(rank)
            logger.log_card(rank, suit, counter.running_count, counter.true_count)

        previous_cards = current_cards

        # Step 4: Update game state
        running, true = counter.get_counts(deck_mgr.decks_remaining())
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

        # Step 5: Publish to MQTT (Node 2 — HiveMQ Cloud)
        if new_cards:
            for rank, suit in new_cards:
                mqtt.publish_card(rank, suit)
            mqtt.publish_deck_state(deck_mgr.deck_state)
            mqtt.publish_count(running, true, bet_rec)
        if best_action:
            mqtt.publish_recommendation(best_action, ev_breakdown)

        # Step 6: Log to InfluxDB Cloud
        if new_cards:
            logger.log_to_influx(running, true, bet_rec, best_action)

        # Step 7: Send SMS if it's the player's decision point
        # TODO: Trigger SMS when player_hand is set and it's their turn

        time.sleep(1 / CAPTURE_FPS)


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
