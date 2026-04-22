# main.py
# Entry point for the Blackjack Card Counting Advisor system.
# Ties together the camera, CV pipeline, card counting, EV calculation,
# MQTT publishing, email alerts, Flask UI, and SQLite logging.
#
# Note: AI tools were used to assist with code development.
# All design decisions are our own.
#
# Run with: python3 main.py

import threading
import time
import signal
import sys
import cv2
import math
from collections import Counter

from capture import init_camera, capture_frame, close_camera
from detect import detect_cards
from classify import load_model, classify_card
from deck_manager import DeckManager
from counter import HiLoCounter
from ev_calculator import EVCalculator
from kelly import kelly_bet
from mqtt_publisher import MQTTPublisher
from email_sender import EmailSender
from logger import Logger
from flask_ui import create_app

# ── Configuration ────────────────────────────────────────────────────────────

PLAYER_EMAIL   = "abirbhat@usc.edu"   # TODO: replace with your email address
BASE_BET       = 10                       # Minimum bet in dollars
BANKROLL       = 500                      # Starting bankroll in dollars
CAPTURE_FPS    = 5                        # Frames to process per second
STEP_STABLE    = 6                        # Frames a new box count must hold before committing the card
BOX_PROXIMITY  = 150                      # Pixels — two boxes this close share the same card slot

# Deal sequence (3 visible cards — hole card is dealt off-camera):
# 0 = player card 1, 1 = player card 2, 2 = dealer upcard
DEAL_ROLES = ["player", "player", "dealer"]

# ── Module-level camera reference so shutdown() can access it ────────────────
camera = None

# ── Globals (shared between threads) ─────────────────────────────────────────

game_state = {
    "detected_cards":     [],
    "running_count":       0,
    "true_count":         0.0,
    "deck_state":         {},
    "recommendation":     None,
    "ev_breakdown":       {},
    "bet_recommendation":  BASE_BET,
    "player_hand":        [],
    "dealer_upcard":      None,
    "reset_requested":    False,   # Set True by /reset route to signal the loop
}

# ── Graceful Shutdown ────────────────────────────────────────────────────────

def shutdown(sig, frame):
    print("\n[main] Shutting down gracefully...")
    if camera is not None:
        close_camera(camera)
    mqtt.disconnect()
    logger.close()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

# ── Main Processing Loop ──────────────────────────────────────────────────────

def find_new_box(bounding_boxes, committed_centers):
    """Return the box whose center is far from all previously committed card positions."""
    for box in bounding_boxes:
        cx = box[0] + box[2] // 2
        cy = box[1] + box[3] // 2
        if not any(math.hypot(cx - px, cy - py) < BOX_PROXIMITY
                   for px, py in committed_centers):
            return box
    return None


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
    #   1 → player card 2
    #   2 → dealer upcard
    #
    # Boxes are sorted left→right (x-coord) each frame so the ordering is
    # consistent regardless of which box OpenCV returns first.
    deal_step         = 0    # which card we're waiting for next (0–len(DEAL_ROLES)-1)
    step_stable       = 0    # consecutive frames we've seen (deal_step+1) boxes
    step_votes        = []    # classification votes accumulated during stable period
    committed_centers = []    # (cx, cy) of each card committed so far
    current_new_box   = None  # the new box identified this stable run
    email_sent        = False

    while True:
        try:
            # ── Reset handler ────────────────────────────────────────────────
            if game_state["reset_requested"]:
                deck_mgr.reset()
                counter.reset()
                deal_step         = 0
                step_stable       = 0
                step_votes        = []
                committed_centers = []
                current_new_box   = None
                email_sent        = False
                game_state["reset_requested"] = False
                game_state["player_hand"] = []
                game_state["dealer_upcard"] = None
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

            # ── Deal-sequence state machine ───────────────────────────────────
            if deal_step < len(DEAL_ROLES):
                expected = deal_step + 1   # how many boxes we need to see

                if n == expected:
                    # Find the box that is new (not near any previously committed card)
                    new_box = find_new_box(bounding_boxes, committed_centers)
                    if new_box is not None:
                        step_stable += 1
                        current_new_box = new_box
                        x, y, w, h = new_box
                        img = cv2.resize(frame[y:y + h, x:x + w], (64, 64))
                        rank, _, conf = classify_card(model, img)
                        step_votes.append(rank)   # always vote — majority wins

                        if step_stable >= STEP_STABLE:
                            # Commit using majority vote
                            committed = Counter(step_votes).most_common(1)[0][0]
                            role = DEAL_ROLES[deal_step]
                            if role == "player":
                                new_player_card = committed
                            else:
                                new_dealer_card = committed
                            cx = current_new_box[0] + current_new_box[2] // 2
                            cy = current_new_box[1] + current_new_box[3] // 2
                            committed_centers.append((cx, cy))
                            print(f"\n[main] Card {deal_step + 1} ({role}): {committed}  "
                                  f"(votes={step_votes})")
                            deal_step       += 1
                            step_stable      = 0
                            step_votes       = []
                            current_new_box  = None

                elif n < expected:
                    # Box disappeared — reset streak (but keep committed_centers)
                    step_stable     = 0
                    step_votes      = []
                    current_new_box = None
                # n > expected: card being placed, wait for count to stabilise

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

            # Step 7: Email once when hand is complete (2 player cards + dealer upcard)
            hand_complete = (
                len(game_state["player_hand"]) >= 2 and
                game_state["dealer_upcard"] is not None
            )
            if hand_complete and best_action and not email_sent:
                hand_str = " + ".join(game_state["player_hand"])
                subject = f"Blackjack: {best_action.upper()}"
                text_body = (
                    f"Blackjack\n"
                    f"Hand: {hand_str} vs {game_state['dealer_upcard']}\n"
                    f"Action: {best_action}\n"
                    f"Count: {running:+d}/{true:+.1f}\n"
                    f"Bet: ${bet_rec}"
                )
                html_body = f"""
                <html>
                    <body>
                        <h2>Blackjack Recommendation</h2>
                        <p><strong>Hand:</strong> {hand_str} vs {game_state['dealer_upcard']}</p>
                        <p><strong>Action:</strong> {best_action}</p>
                        <p><strong>Count:</strong> {running:+d}/{true:+.1f}</p>
                        <p><strong>Bet:</strong> ${bet_rec}</p>
                    </body>
                </html>
                """
                email.send(subject, text_body, html_body)
                email_sent = True

            time.sleep(1 / CAPTURE_FPS)

        except Exception as e:
            import traceback
            print(f"\n[main] Processing error: {e}")
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    global mqtt, logger, email

    mqtt   = MQTTPublisher()
    logger = Logger()
    email  = EmailSender(PLAYER_EMAIL)

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