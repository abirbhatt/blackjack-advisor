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
    "reset_requested":   False,   # Set True by /reset route to signal the loop
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
    camera   = init_camera()
    model    = load_model("model.tflite")
    deck_mgr = DeckManager()
    counter  = HiLoCounter()
    ev_calc  = EVCalculator()

    # Spatial two-pile tracking.
    # The frame is split down the middle:
    #   Left  half → player pile  (cards 1 and 3 in standard deal)
    #   Right half → dealer pile  (face-down hole card, then face-up upcard)
    #
    # prev_*_rank tracks the last CONFIDENTLY classified card on each pile.
    # When the classification changes → new card placed on that pile.
    # When confidence drops below threshold → transition (hand placing card) or
    #   face-down card → reset to None so the next confident card registers fresh.
    prev_left_rank  = None
    prev_right_rank = None
    sms_sent        = False   # One SMS per completed hand

    while True:
      try:
        # ── Reset handler ────────────────────────────────────────────────────
        if game_state["reset_requested"]:
            deck_mgr.reset()
            counter.reset()
            prev_left_rank  = None
            prev_right_rank = None
            sms_sent        = False
            game_state["reset_requested"] = False
            print("\n[main] Hand reset.")

        frame   = capture_frame(camera)
        frame_w = frame.shape[1]

        # Step 1: Detect bounding boxes
        bounding_boxes = detect_cards(frame, camera.baseline_frame)

        # Step 2: Split boxes into left (player) and right (dealer) piles
        left_boxes  = [b for b in bounding_boxes if (b[0] + b[2] // 2) < frame_w // 2]
        right_boxes = [b for b in bounding_boxes if (b[0] + b[2] // 2) >= frame_w // 2]

        print(f"[main] boxes L={len(left_boxes)} R={len(right_boxes)} | "
              f"hand={game_state['player_hand']} dealer={game_state['dealer_upcard']}",
              end='\r')

        # Step 3: Classify the dominant (largest) box on each pile
        def best_classify(boxes):
            """Classify the largest bounding box; return (rank, confidence)."""
            if not boxes:
                return None, 0.0
            box = max(boxes, key=lambda b: b[2] * b[3])
            x, y, w, h = box
            img = cv2.resize(frame[y:y + h, x:x + w], (64, 64))
            rank, _, conf = classify_card(model, img)
            return rank, conf

        left_rank,  left_conf  = best_classify(left_boxes)
        right_rank, right_conf = best_classify(right_boxes)

        # Step 4: Detect new cards — a change in confident classification = new card
        new_player_card = None
        new_dealer_card = None

        # Left pile: player cards
        if left_conf >= CV_CONFIDENCE:
            if left_rank != prev_left_rank:
                new_player_card = left_rank
            prev_left_rank = left_rank
        else:
            # Low confidence = transition (hand placing card) or empty pile.
            # Reset so the next confident card is treated as new.
            prev_left_rank = None

        # Right pile: dealer cards.
        # Face-down hole card → low confidence → ignored here.
        # Face-up upcard → high confidence → triggers as new dealer card.
        if right_conf >= CV_CONFIDENCE:
            if right_rank != prev_right_rank:
                new_dealer_card = right_rank
            prev_right_rank = right_rank
        else:
            prev_right_rank = None

        # Step 5: Update deck, counter, and hands for any new cards
        new_cards = []   # (rank, role) — used for MQTT / InfluxDB publishing
        if new_player_card:
            deck_mgr.remove_card(new_player_card)
            counter.update(new_player_card)
            game_state["player_hand"].append(new_player_card)
            new_cards.append((new_player_card, "player"))
            print(f"\n[main] Player card: {new_player_card} → hand {game_state['player_hand']}")

        if new_dealer_card:
            deck_mgr.remove_card(new_dealer_card)
            counter.update(new_dealer_card)
            game_state["dealer_upcard"] = new_dealer_card
            new_cards.append((new_dealer_card, "dealer"))
            print(f"\n[main] Dealer upcard: {new_dealer_card}")

        # Step 6: Compute counts
        running, true = counter.get_counts(deck_mgr.decks_remaining())

        # Log new cards to SQLite with updated counts
        for rank, role in new_cards:
            logger.log_card(rank, role, running, true)

        # Step 7: EV recommendation and game state update
        best_action, ev_breakdown = ev_calc.recommend(
            game_state["player_hand"],
            game_state["dealer_upcard"],
            deck_mgr.deck_state
        )
        bet_rec = kelly_bet(true, BANKROLL, BASE_BET)

        game_state.update({
            "detected_cards":     [left_rank, right_rank],
            "running_count":      running,
            "true_count":         true,
            "deck_state":         deck_mgr.deck_state.copy(),
            "recommendation":     best_action,
            "ev_breakdown":       ev_breakdown,
            "bet_recommendation": bet_rec,
        })

        # Step 8: Publish to MQTT (Node 2 — HiveMQ Cloud)
        if new_cards:
            for rank, role in new_cards:
                mqtt.publish_card(rank, role)
            mqtt.publish_deck_state(deck_mgr.deck_state)
            mqtt.publish_count(running, true, bet_rec)
        if best_action:
            mqtt.publish_recommendation(best_action, ev_breakdown)

        # Step 9: Log to InfluxDB Cloud
        if new_cards:
            logger.log_to_influx(running, true, bet_rec, best_action)

        # Step 10: Send SMS once when the initial hand is complete
        # (2 player cards + dealer upcard all set)
        hand_complete = (
            len(game_state["player_hand"]) >= 2 and
            game_state["dealer_upcard"] is not None
        )
        if hand_complete and best_action and not sms_sent:
            hand_str = " + ".join(game_state["player_hand"])
            msg = (f"Blackjack: {hand_str} vs {game_state['dealer_upcard']}. "
                   f"Action: {best_action}. Count: {running:+d}/{true:+.1f}. "
                   f"Bet: ${bet_rec}")
            sms.send(msg)
            sms_sent = True

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
