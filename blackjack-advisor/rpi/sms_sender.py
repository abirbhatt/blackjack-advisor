# sms_sender.py
# Sends SMS recommendations to the player via the Twilio REST API.
#
# This is the RESTful API integration component (Lab 3 concepts).
# The RPi makes an HTTP POST request to Twilio's endpoint, passing auth
# credentials and the message body. Twilio's servers deliver the SMS.
#
# This is the same pattern as calling the weather API in Lab 3:
#   - Stateless: each call contains all necessary info (auth, numbers, message)
#   - HTTP POST: creating a new "message" resource on Twilio's server
#   - API Key auth: Account SID + Auth Token authenticate the request
#   - JSON response: Twilio returns confirmation the message was queued
#
# Note: AI tools (Claude) were used to assist with code development.

import requests
from requests.auth import HTTPBasicAuth

# ── Fill these in with your Twilio credentials ────────────────────────────────
ACCOUNT_SID  = "AC17cb81493da70904f6c335a6ff680c3d"    # From Twilio Console
AUTH_TOKEN   = "4809830688782dded3d83c1e7319ed1f"      # From Twilio Console
TWILIO_FROM  = "+18335708265"                # Your Twilio phone number
# ─────────────────────────────────────────────────────────────────────────────

TWILIO_URL = f"https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT_SID}/Messages.json"


class SMSSender:
    def __init__(self, player_phone):
        """
        Args:
            player_phone (str): The player's phone number, e.g. "+12135551234".
        """
        self.player_phone = player_phone

    def send(self, message):
        """
        Send an SMS to the player.

        Makes an HTTP POST request to Twilio's REST API.
        Expects HTTP 201 (Created) on success.

        Args:
            message (str): The text message body to send.

        Returns:
            bool: True if sent successfully, False otherwise.
        """
        payload = {
            "From": TWILIO_FROM,
            "To":   self.player_phone,
            "Body": message,
        }
        response = requests.post(
            TWILIO_URL,
            data=payload,
            auth=HTTPBasicAuth(ACCOUNT_SID, AUTH_TOKEN)
        )
        if response.status_code == 201:
            print(f"[sms] Message sent: {message[:50]}...")
            return True
        else:
            print(f"[sms] Failed to send SMS. Status: {response.status_code}, {response.text}")
            return False

    def send_recommendation(self, player_hand, dealer_upcard, best_action, ev_breakdown, true_count):
        """
        Send a formatted decision recommendation to the player.

        Args:
            player_hand (list):    Player's cards, e.g. ["9", "7"].
            dealer_upcard (str):   Dealer's upcard, e.g. "6".
            best_action (str):     Recommended action, e.g. "Stand".
            ev_breakdown (dict):   EV values for each action.
            true_count (float):    Current true count.
        """
        hand_str = " + ".join(player_hand)
        ev_str   = ", ".join(f"{k}: {v:+.2f}" for k, v in ev_breakdown.items())
        message  = (
            f"Hand: {hand_str} vs dealer {dealer_upcard}\n"
            f"→ {best_action.upper()} (True count: {true_count:+.1f})\n"
            f"EV: {ev_str}"
        )
        self.send(message)

    def send_bet_alert(self, true_count, bet_recommendation):
        """
        Send a bet sizing alert when the deck becomes favorable.

        Args:
            true_count (float):       Current true count.
            bet_recommendation (int): Kelly-recommended bet in dollars.
        """
        message = (
            f"Deck is favorable! True count: {true_count:+.1f}\n"
            f"Recommended bet: ${bet_recommendation}"
        )
        self.send(message)
