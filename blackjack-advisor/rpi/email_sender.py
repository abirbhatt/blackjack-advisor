# email_sender.py
# Sends email recommendations to the player via the Resend REST API.
#
# This is the RESTful API integration component (Lab 3 concepts).
# The RPi makes an HTTP POST request to Resend's endpoint, passing auth
# credentials and the message body. Resend's servers deliver the email.
#
# This is the same pattern as calling the weather API in Lab 3:
#   - Stateless: each call contains all necessary info (auth, recipient, subject, body)
#   - HTTP POST: creating a new "email" resource on Resend's server
#   - API Key auth: Bearer token authenticates the request
#   - JSON response: Resend returns confirmation that the email was accepted
#
# Note: AI tools were used to assist with code development.

import os
import requests

# Put your Resend API key in an environment variable:
#   export RESEND_API_KEY="re_..."
RESEND_API_KEY = os.getenv("re_7Dex9iar_HPtH7MbdkVE9RDP6SxQJqQ6o", "")
RESEND_URL = "https://api.resend.com/emails"

# Replace this with your verified sender in Resend.
# Example: "Blackjack Advisor <onboarding@resend.dev>" for testing,
# or your own verified domain sender in production.
RESEND_FROM = os.getenv("RESEND_FROM", "Blackjack Advisor <onboarding@resend.dev>")


class EmailSender:
    def __init__(self, recipient_email):
        """
        Args:
            recipient_email (str): The player's email address.
        """
        self.recipient_email = recipient_email

    def send(self, subject, text_body, html_body=None):
        """
        Send an email to the player.

        Makes an HTTP POST request to Resend's REST API.
        Expects HTTP 200 on success.

        Args:
            subject (str): The email subject line.
            text_body (str): Plain-text email body.
            html_body (str, optional): HTML version of the email body.

        Returns:
            bool: True if sent successfully, False otherwise.
        """
        if not RESEND_API_KEY:
            print("[email] Missing RESEND_API_KEY environment variable.")
            return False

        payload = {
            "from": RESEND_FROM,
            "to": [self.recipient_email],
            "subject": subject,
            "text": text_body,
        }

        if html_body:
            payload["html"] = html_body

        headers = {
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "blackjack-advisor/1.0",
        }

        response = requests.post(
            RESEND_URL,
            json=payload,
            headers=headers,
            timeout=15,
        )

        if response.status_code == 200:
            print(f"[email] Message sent: {subject}")
            return True
        else:
            print(f"[email] Failed to send email. Status: {response.status_code}, {response.text}")
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
        ev_str = ", ".join(f"{k}: {v:+.2f}" for k, v in ev_breakdown.items())

        subject = f"Blackjack Recommendation: {best_action.upper()}"

        text_body = (
            f"Hand: {hand_str} vs dealer {dealer_upcard}\n"
            f"Recommended Action: {best_action.upper()}\n"
            f"True Count: {true_count:+.1f}\n"
            f"EV: {ev_str}"
        )

        html_body = f"""
        <html>
            <body>
                <h2>Blackjack Recommendation</h2>
                <p><strong>Hand:</strong> {hand_str} vs dealer {dealer_upcard}</p>
                <p><strong>Recommended Action:</strong> {best_action.upper()}</p>
                <p><strong>True Count:</strong> {true_count:+.1f}</p>
                <p><strong>EV:</strong> {ev_str}</p>
            </body>
        </html>
        """

        self.send(subject, text_body, html_body)

    def send_bet_alert(self, true_count, bet_recommendation):
        """
        Send a bet sizing alert when the deck becomes favorable.

        Args:
            true_count (float):       Current true count.
            bet_recommendation (int): Recommended bet in dollars.
        """
        subject = "Blackjack Bet Alert"

        text_body = (
            f"Deck is favorable!\n"
            f"True Count: {true_count:+.1f}\n"
            f"Recommended Bet: ${bet_recommendation}"
        )

        html_body = f"""
        <html>
            <body>
                <h2>Blackjack Bet Alert</h2>
                <p><strong>Deck is favorable!</strong></p>
                <p><strong>True Count:</strong> {true_count:+.1f}</p>
                <p><strong>Recommended Bet:</strong> ${bet_recommendation}</p>
            </body>
        </html>
        """

        self.send(subject, text_body, html_body)