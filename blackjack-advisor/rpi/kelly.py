# kelly.py
# Kelly Criterion bet sizing.
#
# The Kelly Criterion determines the optimal fraction of your bankroll to bet
# in order to maximize long-run growth. The formula is:
#   f* = edge / odds
# For blackjack (odds = 1:1), f* = edge.
#
# Player edge from the true count (approximate):
#   edge ≈ (true_count - 1) × 0.5%
#
# A true count of +1 = roughly breakeven.
# A true count of +3 = ~1% edge for the player.
#
# We cap the bet at 8× the base bet to manage variance, and floor it at
# the base bet when there's no edge (unfavorable deck).
#
# Note: AI tools (Claude) were used to assist with code development.


def kelly_bet(true_count, bankroll, base_bet):
    """
    Compute the recommended bet size using the Kelly Criterion.

    Args:
        true_count (float): Current true count from HiLoCounter.
        bankroll (float):   Current player bankroll in dollars.
        base_bet (float):   Minimum bet (also the bet when no edge).

    Returns:
        float: Recommended bet size in dollars.
    """
    # Estimate player edge from true count
    # Rule of thumb: each +1 true count ≈ +0.5% edge above breakeven
    edge = (true_count - 1) * 0.005

    if edge <= 0:
        return base_bet   # No edge — bet minimum

    # Kelly fraction: what fraction of bankroll to bet
    kelly_fraction = edge   # Simplified (odds = 1 for even-money blackjack)

    # Full Kelly can be aggressive; a common practice is to use half-Kelly
    # Uncomment the line below to use half-Kelly for lower variance:
    # kelly_fraction *= 0.5

    recommended = bankroll * kelly_fraction

    # Cap at 8× base bet to avoid overbetting
    recommended = min(recommended, base_bet * 8)

    # Floor at base bet
    recommended = max(recommended, base_bet)

    # Round to nearest dollar
    return round(recommended)
