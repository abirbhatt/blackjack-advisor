# flask_ui.py
# Local Flask web dashboard served by the RPi.

from flask import Flask, jsonify, render_template_string, request

HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title></title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #1a1a2e;
      color: #eee;
      text-align: center;
      padding: 20px;
    }

    #recommendation {
      font-size: 4em;
      font-weight: bold;
      margin: 20px 0;
    }

    .good  { color: #4ecca3; }
    .bad   { color: #e94560; }
    .neutral { color: #f5a623; }

    #count-bar {
      background: #16213e;
      border-radius: 12px;
      padding: 15px;
      margin: 10px auto;
      max-width: 400px;
    }

    .ev-table {
      margin: auto;
      border-collapse: collapse;
    }

    .ev-table td, .ev-table th {
      padding: 6px 18px;
      border: 1px solid #333;
    }

    /* UPDATED: Bigger, clearer hand display */
    #cards {
      font-size: 3em;
      font-weight: bold;
      margin: 25px 0;
      padding: 10px 20px;
      background: #16213e;
      border-radius: 12px;
      display: inline-block;
      letter-spacing: 1px;
    }
  </style>
</head>
<body>

  <div id="cards">Loading...</div>
  <div id="recommendation">--</div>

  <div id="count-bar">
    <span>Running: <b id="rc">--</b></span> &nbsp;|&nbsp;
    <span>True: <b id="tc">--</b></span> &nbsp;|&nbsp;
    <span>Bet: $<b id="bet">--</b></span>
  </div>

  <table class="ev-table" id="ev-table"></table>

  <br>

  <button onclick="fetch('/reset')"
    style="padding:10px 20px;background:#333;color:#fff;border:none;border-radius:8px;cursor:pointer;">
    🔄 New Shoe
  </button>

  <script>
    function update() {
      fetch('/status').then(r => r.json()).then(data => {

        // UPDATED: clearer player/dealer display
        document.getElementById('cards').textContent =
          'Player: ' + (data.player_hand.join(' + ') || '?') +
          '  |  Dealer: ' + (data.dealer_upcard || '?');

        const rec = data.recommendation || '--';
        const el  = document.getElementById('recommendation');
        el.textContent = rec;
        el.className = rec === 'Stand' ? 'good' :
                       rec === 'Hit'   ? 'neutral' : 'bad';

        document.getElementById('rc').textContent  = data.running_count;
        document.getElementById('tc').textContent  = data.true_count;
        document.getElementById('bet').textContent = data.bet_recommendation;

        // EV table
        let rows = '<tr><th>Action</th><th>EV</th></tr>';
        for (const [action, ev] of Object.entries(data.ev_breakdown || {})) {
          const cls = ev > 0 ? 'good' : 'bad';
          rows += `<tr><td>${action}</td><td class="${cls}">
                    ${ev > 0 ? '+' : ''}${ev.toFixed(3)}
                   </td></tr>`;
        }
        document.getElementById('ev-table').innerHTML = rows;
      });
    }

    setInterval(update, 1000);
    update();
  </script>

</body>
</html>
"""


def create_app(game_state):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/status")
    def status():
        return jsonify(game_state)

    @app.route("/correct", methods=["POST"])
    def correct():
        data     = request.get_json()
        old_rank = data.get("old_rank")
        new_rank = data.get("new_rank")
        return jsonify({"status": "ok", "corrected": f"{old_rank} → {new_rank}"})

    @app.route("/reset")
    def reset():
        game_state.update({
            "detected_cards": [], "running_count": 0, "true_count": 0.0,
            "deck_state": {}, "recommendation": None, "ev_breakdown": {},
            "bet_recommendation": 10, "player_hand": [], "dealer_upcard": None,
            "reset_requested": True,
        })
        return jsonify({"status": "reset"})

    return app