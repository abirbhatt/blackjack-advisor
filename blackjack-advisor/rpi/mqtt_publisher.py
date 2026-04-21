# mqtt_publisher.py
# Publishes game state to HiveMQ Cloud over MQTT (TLS, port 8883).
#
# This is the node-to-node communication component (Lab 4 concepts).
# The RPi (Node 1) publishes to HiveMQ Cloud (Node 2).
# Any subscriber — Grafana, a phone app, a web client — can receive updates.
#
# Unlike Lab 4's test.mosquitto.org (unencrypted, port 1883), HiveMQ Cloud
# requires TLS authentication on port 8883 for security.
#
# Topics:
#   blackjack/cards          — newly detected card event
#   blackjack/deckstate      — full remaining card counts
#   blackjack/recommendation — best action + EV breakdown
#   blackjack/count          — running count, true count, bet recommendation
#
# Note: AI tools (Claude) were used to assist with code development.

import paho.mqtt.client as mqtt
import ssl
import json
import time

# ── Fill these in with your HiveMQ Cloud credentials ─────────────────────────
BROKER_HOST = "8860e86aa54241ab91cbf4300f59a6f1.s1.eu.hivemq.cloud"   # e.g. "abc123.s1.eu.hivemq.cloud"
BROKER_PORT = 8883
USERNAME    = "vrindaandabir"
PASSWORD    = "Password1"
# ─────────────────────────────────────────────────────────────────────────────


class MQTTPublisher:
    def __init__(self):
        self.client = mqtt.Client(client_id="blackjack-rpi", protocol=mqtt.MQTTv5)
        self.client.username_pw_set(USERNAME, PASSWORD)

        # TLS: require a valid server certificate (same as HTTPS)
        self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)

        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self.client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
        self.client.loop_start()   # Background thread handles MQTT network traffic

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[mqtt] Connected to HiveMQ Cloud at {BROKER_HOST}")
        else:
            print(f"[mqtt] Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        print(f"[mqtt] Disconnected (rc={rc}). Attempting reconnect...")

    def _publish(self, topic, payload_dict, qos=1):
        """Serialize payload to JSON and publish. QoS 1 = at least once delivery."""
        self.client.publish(topic, json.dumps(payload_dict), qos=qos)

    def publish_card(self, rank, suit):
        """Publish a newly detected card event."""
        self._publish("blackjack/cards", {
            "rank":      rank,
            "suit":      suit,
            "timestamp": int(time.time()),
        })

    def publish_deck_state(self, deck_state):
        """Publish the full remaining deck composition."""
        self._publish("blackjack/deckstate", deck_state)

    def publish_recommendation(self, best_action, ev_breakdown):
        """Publish the recommended action and EV breakdown for all options."""
        self._publish("blackjack/recommendation", {
            "action":       best_action,
            "ev_breakdown": ev_breakdown,
            "timestamp":    int(time.time()),
        })

    def publish_count(self, running_count, true_count, bet_recommendation):
        """Publish current count stats and bet recommendation."""
        self._publish("blackjack/count", {
            "running_count":     running_count,
            "true_count":        true_count,
            "bet_recommendation": bet_recommendation,
            "timestamp":         int(time.time()),
        })

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
