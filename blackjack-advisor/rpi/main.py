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
import math
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
CV_CONFIDENCE  = 0.85             # Minimum CNN confidence to commit a classification
CAPTURE_FPS    = 2                # Frames to process per second
PILE_PROXIMITY = 200              # Pixels — how close a box must be to a known pile to belong to it

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

def box_center(box):
    x, y, w, h = box
    return x + w // 2, y + h // 2


def near_pile(box, pile_box):
    """True if box's center is within PILE_PROXIMITY pixels of pile_box's center."""
    cx1, cy1 = box_center(box)
    cx2, cy2 = box_center(pile_box)
    return math.hypot(cx1 - cx2, cy1 - cy2) < PILE_PROXIMITY


def processing_loop():
    global camera
    camera   = init_camera()
    model    = load_model("model.tflite")
    deck_mgr = DeckManager()
    counter  = HiLoCounter()
    ev_calc  = EVCalculator()

    # Self-calibrating two-pile tracker.
    #
    # The first bounding box ever seen becomes the PLAYER pile anchor.
    # Any box appearing far from the player pile becomes the DEALER pile anchor.
    # No left/right or top/bottom assumption — the system learns from wherever
    # the user physically places the first card.
    #
    # prev_*_rank tracks the last confident classification at each pile.
    # When the classification changes → new card was placed on top.
    # When confidence drops → hand is placing a card (transition) → reset to None
    #   so the next confident result registers as a new card.
    #   This also handles same-rank cards (8 on 8), since covering and re-placing
    #   creates a low-confidence frame that resets the tracker.
    #
    # Face-down dealer hole card: low confidence → dealer_prev_rank stays None
    #   → face-up upcard later placed on top triggers as first confident dealer card.
    player_pile_box  = None   # Anchor box for player pile (set on first detection)
    dealer_pile_box  = None   # Anchor box for dealer pile (set when second location seen)
    player_prev_rank = None   # Last confirmed rank at player pile
    dealer_prev_rank = None   # Last confirmed rank at dealer pile
    sms_sent         = False  # One SMS per completed hand

    while True:
      try:
        # ── Reset handler ────────────────────────────────────────────────────
        if game_state["reset_requested"]:
            deck_mgr.reset()
            counter.reset()
            player_pile_box  = None
            dealer_pile_box  = None
            player_prev_rank = None
            dealer_prev_rank = None
            sms_sent         = False
            game_state["reset_requested"] = False
            print("\n[main] Hand reset.")

        frame = capture_frame(camera)

        # Step 1: Detect bounding boxes
        bounding_boxes = detect_cards(frame, camera.baseline_frame)
        print(f"[main] {len(bounding_boxes)} box(es) | "
              f"hand={game_state['player_hand']} dealer={game_state['dealer_upcard']}",
              end='\r')

        new_player_card = None
        new_dealer_card = None

        for box in bounding_boxes:
            x, y, w, h = box
            img = cv2.resize(frame[y:y + h, x:x + w], (64, 64))
            rank, _, conf = classify_card(model, img)

            # ── Assign box to a pile ──────────────────────────────────────────
            if player_pile_box is None:
                # First box ever → establishes player pile location
                player_pile_box = box
                print(f"\n[main] Player pile anchored at {box_center(box)}")
                if conf >= CV_CONFIDENCE:
                    new_player_card  = rank
                    player_prev_rank = rank

            elif near_pile(box, player_pile_box):
                # This box is at the player pile
                player_pile_box = box   # Update anchor to latest position
                if conf >= CV_CONFIDENCE:
                    if rank != player_prev_rank:
                        new_player_card  = rank
                    player_prev_rank = rank
                else:
                    # Low confidence = hand placing card; reset so next confident fires
                    player_prev_rank = None

            else:
                # Box is in a different area → dealer pile
                if dealer_pile_box is None:
                    dealer_pile_box = box
                    print(f"\n[main] Dealer pile anchored at {box_center(box)}")

                if near_pile(box, dealer_pile_box):
                    dealer_pile_box = box   # Update anchor
                    if conf >= CV_CONFIDENCE:
                        if rank != dealer_prev_rank:
                            new_dealer_card  = rank
                        dealer_prev_rank = rank
                    else:
                        # Face-down card or transition → keep dealer_prev_rank=None
                        dealer_prev_rank = None

        # Step 2: Process new cards
        new_cards = []
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

        # Step 3: Compute counts
        running, true = counter.get_counts(deck_mgr.decks_remaining())

        for rank, role in new_cards:
            logger.log_card(rank, role, running, true)

        # Step 4: EV recommendation and game state update
        best_action, ev_breakdown = ev_calc.recommend(
            game_state["player_hand"],
            game_state["dealer_upcard"],
            deck_mgr.deck_state
        )
        bet_rec = kelly_bet(true, BANKROLL, BASE_BET)

        game_state.update({
            "detected_cards":     [box_center(b) for b in bounding_boxes],
            "running_count":      running,
            "true_count":         true,
            "deck_state":         deck_mgr.deck_state.copy(),
            "recommendation":     best_action,
            "ev_breakdown":       ev_breakdown,
            "bet_recommendation": bet_rec,
        })

        # Step 5: Publish to MQTT
        if new_cards:
            for rank, role in new_cards:
                mqtt.publish_card(rank, role)
            mqtt.publish_deck_state(deck_mgr.deck_state)
            mqtt.publish_count(running, true, bet_rec)
        if best_action:
            mqtt.publish_recommendation(best_action, ev_breakdown)

        # Step 6: Log to InfluxDB
        if new_cards:
            logger.log_to_influx(running, true, bet_rec, best_action)

        # Step 7: SMS once when hand is complete (2 player cards + dealer upcard)
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
