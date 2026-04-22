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
from collections import Counter

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

PLAYER_PHONE   = "+16502085215"   # TODO: replace with your phone number e.g. "+12135551234"
BASE_BET       = 10               # Minimum bet in dollars
BANKROLL       = 500              # Starting bankroll in dollars
CV_CONFIDENCE  = 0.70             # Minimum CNN confidence to count a classification vote
CAPTURE_FPS    = 5                # Frames to process per second
STEP_STABLE    = 6                # Frames a new box count must hold before committing the card

# Deal sequence: index 0-3 maps to the role of each card placed
# 0 = player card 1, 1 = dealer hole (face-down, skipped), 2 = player card 2, 3 = dealer upcard
DEAL_ROLES = ["player", "hole", "player", "dealer"]

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

    # ── Deal-sequence state ───────────────────────────────────────────────────
    # Cards are dealt one at a time onto the table (not stacked).
    # We watch the total box count each frame.  When it goes from N-1 to N and
    # holds at N for STEP_STABLE frames, we commit the Nth card using a majority
    # vote over the classifications seen during those stable frames.
    #
    # DEAL_ROLES maps card index → role:
    #   0 → player card 1
    #   1 → dealer hole (face-down, classification skipped)
    #   2 → player card 2
    #   3 → dealer upcard
    #
    # Boxes are sorted left→right (x-coord) each frame so the ordering is
    # consistent regardless of which box OpenCV returns first.
    deal_step   = 0    # which card we're waiting for next (0-3)
    step_stable = 0    # consecutive frames we've seen (deal_step+1) boxes
    step_votes  = []   # classification votes accumulated during stable period
    sms_sent    = False

    while True:
      try:
        # ── Reset handler ────────────────────────────────────────────────────
        if game_state["reset_requested"]:
            deck_mgr.reset()
            counter.reset()
            deal_step   = 0
            step_stable = 0
            step_votes  = []
            sms_sent    = False
            game_state["reset_requested"] = False
            print("\n[main] Hand reset.")

        frame = capture_frame(camera)

        # Step 1: Detect bounding boxes, sort left→right for consistent ordering
        bounding_boxes = sorted(
            detect_cards(frame, camera.baseline_frame),
            key=lambda b: b[0]   # sort by x coordinate
        )
        n = len(bounding_boxes)

        print(f"[main] boxes={n}  deal_step={deal_step}  "
              f"stable={step_stable}/{STEP_STABLE}  votes={step_votes} | "
              f"hand={game_state['player_hand']} dealer={game_state['dealer_upcard']}",
              end='\r')

        new_player_card = None
        new_dealer_card = None

        # ── Deal-sequence state machine ───────────────────────────────────────
        if deal_step < 4:
            expected = deal_step + 1   # how many boxes we need to see

            if n == expected:
                # Box count matches — accumulate
                step_stable += 1

                role = DEAL_ROLES[deal_step]

                if role != "hole":
                    # Classify the card at this slot
                    box = bounding_boxes[deal_step]
                    x, y, w, h = box
                    img = cv2.resize(frame[y:y + h, x:x + w], (64, 64))
                    rank, _, conf = classify_card(model, img)
                    if conf >= CV_CONFIDENCE:
                        step_votes.append(rank)

                if step_stable >= STEP_STABLE:
                    # Commit this card
                    if role == "hole":
                        print(f"\n[main] Card {deal_step + 1}: dealer hole (face-down, skipped)")
                    elif step_votes:
                        committed = Counter(step_votes).most_common(1)[0][0]
                        if role == "player":
                            new_player_card = committed
                        else:
                            new_dealer_card = committed
                        print(f"\n[main] Card {deal_step + 1} ({role}): {committed}  "
                              f"(votes={step_votes})")
                    else:
                        print(f"\n[main] Card {deal_step + 1} ({role}): "
                              f"no confident vote — confidence too low, skipping slot")

                    deal_step  += 1
                    step_stable = 0
                    step_votes  = []

            elif n < expected:
                # A box disappeared (card removed mid-deal, or detection glitch) → reset streak
                step_stable = 0
                step_votes  = []
            # n > expected: extra box visible during card placement — just wait

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
            "detected_cards":     [[b[0]+b[2]//2, b[1]+b[3]//2] for b in bounding_boxes],
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
