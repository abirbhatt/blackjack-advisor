# flask_ui.py
# Local Flask web dashboard served by the RPi.
# Accessible from any device on the same WiFi: http://blackjackpi.local:5000
#
# Uses the same Flask pattern from Lab 3, with a /status endpoint that
# returns JSON, and a frontend that polls it every second via JavaScript.
#
# Pages:
#   /          — Main dashboard: recommendation, count, bet size, hand display
#   /status    — JSON API endpoint (polled by frontend for live updates)
#   /correct   — Card correction form (to fix a misclassification)
#   /reset     — Reset the shoe/session
#
# Note: AI tools (Claude) were used to assist with code development.

from flask import Flask, jsonify, render_template_string, request

# ── HTML template (single-file, no separate CSS) ──────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blackjack Advisor</title>
  <style>
    body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee;
           text-align: center; padding: 20px; }
    #recommendation { font-size: 4em; font-weight: bold; margin: 20px 0; }
    .good  { color: #4ecca3; }
    .bad   { color: #e94560; }
    .neutral { color: #f5a623; }
    #count-bar { background: #16213e; border-radius: 12px; padding: 15px; margin: 10px auto;
                 max-width: 400px; }
    .ev-table { margin: auto; border-collapse: collapse; }
    .ev-table td, .ev-table th { padding: 6px 18px; border: 1px solid #333; }
    #cards { font-size: 1.4em; }
  </style>
</head>
<body>
  <h1>♠ Blackjack Advisor ♥</h1>
  <div id="cards">Loading...</div>
  <div id="recommendation">--</div>
  <div id="count-bar">
    <span>Running: <b id="rc">--</b></span> &nbsp;|&nbsp;
    <span>True: <b id="tc">--</b></span> &nbsp;|&nbsp;
    <span>Bet: $<b id="bet">--</b></span>
  </div>
  <table class="ev-table" id="ev-table"></table>
  <br>
  <button onclick="fetch('/reset')" style="padding:10px 20px;background:#333;color:#fff;border:none;border-radius:8px;cursor:pointer;">
    🔄 New Shoe
  </button>

  <script>
    function update() {
      fetch('/status').then(r => r.json()).then(data => {
        document.getElementById('cards').textContent =
          'Hand: ' + (data.player_hand.join(' + ') || '?') +
          '  vs Dealer: ' + (data.dealer_upcard || '?');

        const rec = data.recommendation || '--';
        const el  = document.getElementById('recommendation');
        el.textContent = rec;
        el.className = rec === 'Stand' ? 'good' : rec === 'Hit' ? 'neutral' : 'bad';

        document.getElementById('rc').textContent  = data.running_count;
        document.getElementById('tc').textContent  = data.true_count;
        document.getElementById('bet').textContent = data.bet_recommendation;

        // EV table
        let rows = '<tr><th>Action</th><th>EV</th></tr>';
        for (const [action, ev] of Object.entries(data.ev_breakdown || {})) {
          const cls = ev > 0 ? 'good' : 'bad';
          rows += `<tr><td>${action}</td><td class="${cls}">${ev > 0 ? '+' : ''}${ev.toFixed(3)}</td></tr>`;
        }
        document.getElementById('ev-table').innerHTML = rows;
      });
    }
    setInterval(update, 1000);   // Poll /status every second
    update();
  </script>
</body>
</html>
"""


def create_app(game_state):
    """
    Create and return the Flask app, sharing the game_state dict with the routes.
    game_state is updated by the main processing loop in main.py.

    Args:
        game_state (dict): Shared state dictionary from main.py.

    Returns:
        Flask: Configured Flask application.
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/status")
    def status():
        """JSON endpoint polled by the frontend for live updates."""
        return jsonify(game_state)

    @app.route("/correct", methods=["POST"])
    def correct():
        """
        Accept a card correction from the UI.
        TODO: Update game_state with the corrected card, adjust deck and count.
        """
        data     = request.get_json()
        old_rank = data.get("old_rank")
        new_rank = data.get("new_rank")
        # TODO: call deck_manager and counter to undo old_rank and apply new_rank
        return jsonify({"status": "ok", "corrected": f"{old_rank} → {new_rank}"})

    @app.route("/reset")
    def reset():
        """Reset game state for a new hand (called via UI button)."""
        game_state.update({
            "detected_cards": [], "running_count": 0, "true_count": 0.0,
            "deck_state": {}, "recommendation": None, "ev_breakdown": {},
            "bet_recommendation": 10, "player_hand": [], "dealer_upcard": None,
            "reset_requested": True,   # Tells the processing loop to reset deck/counter too
        })
        return jsonify({"status": "reset"})

    return app
