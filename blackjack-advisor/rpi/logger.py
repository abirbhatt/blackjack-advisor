# logger.py
# Dual logging: SQLite (local on RPi) and InfluxDB Cloud (remote, for Grafana).
#
# SQLite:    file-based relational database, no server needed.
#            Stores session history on the RPi for offline review.
#
# InfluxDB:  time-series database (Lab 9 concepts), cloud-hosted.
#            Each data point has a timestamp, measurement name, tags, and fields.
#            Grafana queries InfluxDB Cloud to build the analytics dashboard.
#
# Note: AI tools (Claude) were used to assist with code development.

import sqlite3
import time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# ── InfluxDB Cloud credentials — fill these in ────────────────────────────────
INFLUX_URL    = "https://us-east-1-1.aws.cloud2.influxdata.com"  # Your cloud URL
INFLUX_TOKEN  = "LEGJRwHvTbPzdzy1-i46PqN3_PQZOBPAxM32tnXuFK4UszBg0KNbv8saA6ucqTKcEQdxWsvA0UBfLxNrWDv8Sw=="
INFLUX_ORG    = "EE250"
INFLUX_BUCKET = "blackjack"
# ─────────────────────────────────────────────────────────────────────────────

SQLITE_DB = "sessions.db"


class Logger:
    def __init__(self):
        self.session_id = int(time.time())

        # ── SQLite setup ──────────────────────────────────────────────────────
        self.conn   = sqlite3.connect(SQLITE_DB, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

        # ── InfluxDB Cloud setup ──────────────────────────────────────────────
        self.influx_client = InfluxDBClient(
            url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        print(f"[logger] Session {self.session_id} started.")

    def _create_tables(self):
        """Create SQLite tables if they don't exist yet."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS card_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   INTEGER,
                timestamp    INTEGER,
                rank         TEXT,
                suit         TEXT,
                running_count INTEGER,
                true_count   REAL
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER,
                timestamp       INTEGER,
                player_hand     TEXT,
                dealer_upcard   TEXT,
                best_action     TEXT,
                ev_best         REAL,
                ev_breakdown    TEXT
            )
        """)
        self.conn.commit()

    def log_card(self, rank, suit, running_count, true_count):
        """Log a detected card to SQLite."""
        self.cursor.execute(
            "INSERT INTO card_events VALUES (NULL,?,?,?,?,?,?)",
            (self.session_id, int(time.time()), rank, suit, running_count, true_count)
        )
        self.conn.commit()

    def log_recommendation(self, player_hand, dealer_upcard, best_action, ev_breakdown):
        """Log a decision recommendation to SQLite."""
        import json
        ev_best = max(ev_breakdown.values()) if ev_breakdown else 0.0
        self.cursor.execute(
            "INSERT INTO recommendations VALUES (NULL,?,?,?,?,?,?,?)",
            (self.session_id, int(time.time()),
             str(player_hand), dealer_upcard, best_action,
             ev_best, json.dumps(ev_breakdown))
        )
        self.conn.commit()

    def log_to_influx(self, running_count, true_count, bet_rec, best_action=None):
        """
        Write a data point to InfluxDB Cloud.
        Grafana will query these points to build the real-time dashboard.
        """
        point = (
            Point("count_state")
            .tag("session", str(self.session_id))
            .field("running_count",    running_count)
            .field("true_count",       true_count)
            .field("bet_recommendation", bet_rec)
        )
        if best_action:
            point = point.field("recommendation", best_action)

        self.write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

    def close(self):
        self.conn.close()
        self.influx_client.close()
        print("[logger] Logger closed.")
