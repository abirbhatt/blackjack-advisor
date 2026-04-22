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

PLAYER_PHONE   = "+16502085215"   # TODO: replace with your phone number e.g. "+12135551234"
BASE_BET       = 10               # Minimum bet in dollars
BANKROLL       = 500              # Starting bankroll in dollars
CV_CONFIDENCE  = 0.85             # Minimum CNN confidence to accept a classification
CAPTURE_FPS    = 5                # Frames to process per second (higher = more responsive streaks)
PILE_PROXIMITY = 200              # Pixels — how close a box must be to a known pile to belong to it
STABLE_FRAMES  = 5                # Consecutive confident frames required to commit a card
COOLDOWN_SECS  = 2.0              # Seconds a pile is locked after committing a card

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

    # Self-calibrating two-pile tracker with streak-based commit.
    #
    # A card is committed only after STABLE_FRAMES consecutive confident
    # classifications of the same rank at the same pile.  After a commit the
    # pile is locked for COOLDOWN_SECS so normal fluctuation can never spam
    # the same card multiple times.
    #
    # Face-down hole card: always low confidence → streak never builds → ignored.
    # Same-rank card on top: cooldown expires, streak resets, then builds again.
    player_pile_box    = None   # Anchor for player pile
    dealer_pile_box    = None   # Anchor for dealer pile

    player_candidate   = None   # Rank accumulating streak at player pile
    player_streak      = 0      # Consecutive frames for that candidate
    player_cooldown_ts = 0.0    # time.time() after which player pile can fire again

    dealer_candidate   = None
    dealer_streak      = 0
    dealer_cooldown_ts = 0.0

    sms_sent = False

    while True:
      try:
        # ── Reset handler ────────────────────────────────────────────────────
        if game_state["reset_requested"]:
            deck_mgr.reset()
            counter.reset()
            player_pile_box    = None
            dealer_pile_box    = None
            player_candidate   = None
            player_streak      = 0
            player_cooldown_ts = 0.0
            dealer_candidate   = None
            dealer_streak      = 0
            dealer_cooldown_ts = 0.0
            sms_sent           = False
            game_state["reset_requested"] = False
            print("\n[main] Hand reset.")

        frame = capture_frame(camera)
        now   = time.time()

        # Step 1: Detect bounding boxes
        bounding_boxes = detect_cards(frame, camera.baseline_frame)
        print(f"[main] {len(bounding_boxes)} box(es) | "
              f"P:{player_candidate}x{player_streak}/{STABLE_FRAMES}  "
              f"D:{dealer_candidate}x{dealer_streak}/{STABLE_FRAMES} | "
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
                player_pile_box = box
                print(f"\n[main] Player pile anchored at {box_center(box)}")
                is_player = True
            elif near_pile(box, player_pile_box):
                player_pile_box = box
                is_player = True
            else:
                if dealer_pile_box is None:
                    dealer_pile_box = box
                    print(f"\n[main] Dealer pile anchored at {box_center(box)}")
                if near_pile(box, dealer_pile_box):
                    dealer_pile_box = box
                    is_player = False
                else:
                    continue  # Third location — ignore

            # ── Streak-based commit ───────────────────────────────────────────
            if is_player:
                if now < player_cooldown_ts:
                    continue  # Pile locked after last commit
                if conf >= CV_CONFIDENCE:
                    if rank == player_candidate:
                        player_streak += 1
                        if player_streak >= STABLE_FRAMES:
                            new_player_card    = rank
                            player_cooldown_ts = now + COOLDOWN_SECS
                            player_candidate   = None
                            player_streak      = 0
                    else:
                        player_candidate = rank
                        player_streak    = 1
                else:
                    player_candidate = None
                    player_streak    = 0
            else:
                if now < dealer_cooldown_ts:
                    continue  # Pile locked after last commit
                if conf >= CV_CONFIDENCE:
                    if rank == dealer_candidate:
                        dealer_streak += 1
                        if dealer_streak >= STABLE_FRAMES:
                            new_dealer_card    = rank
                            dealer_cooldown_ts = now + COOLDOWN_SECS
                            dealer_candidate   = None
                            dealer_streak      = 0
                    else:
                        dealer_candidate = rank
                        dealer_streak    = 1
                else:
                    dealer_candidate = None
                    dealer_streak    = 0

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
