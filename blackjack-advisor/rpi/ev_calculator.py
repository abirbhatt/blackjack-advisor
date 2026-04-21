# ev_calculator.py
# Expected Value (EV) calculator for blackjack decisions.
#
# This is the core non-trivial processing component of the project.
# Rather than looking up a static basic strategy table, this calculator
# computes the mathematically optimal action given the EXACT current
# composition of the remaining deck.
#
# For each possible action (Hit, Stand, Double, Split), it computes:
#   EV = sum over all possible outcomes of (probability × payoff)
#
# Dealer behavior: always hits on soft 16 or below, stands on 17+.
# Payoffs: win = +1, lose = -1, push (tie) = 0, blackjack = +1.5
#
# Note: AI tools (Claude) were used to assist with code development.

RANKS      = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
RANK_VALUES = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
               '10':10,'J':10,'Q':10,'K':10,'A':11}


def hand_value(cards):
    """
    Compute the best blackjack value for a hand (list of rank strings).
    Aces count as 11 unless that busts, in which case they count as 1.

    Returns:
        int: Best hand total (≤21 if possible).
    """
    total = sum(RANK_VALUES[c] for c in cards)
    aces  = cards.count('A')
    while total > 21 and aces:
        total -= 10
        aces  -= 1
    return total


class EVCalculator:

    def dealer_distribution(self, dealer_upcard, deck_state):
        """
        Compute the probability distribution of the dealer's final hand total,
        given the dealer upcard and the current deck composition.

        The dealer draws cards according to fixed rules: hit on ≤16, stand on ≥17.
        We recurse through all possible draw sequences, weighted by draw probability.

        Args:
            dealer_upcard (str):  The dealer's visible card rank.
            deck_state (dict):    Current remaining card counts by rank.

        Returns:
            dict: {final_total (int): probability (float)}
                  Includes 'bust' as a key for totals > 21.
        """
        total_remaining = sum(deck_state.values())
        if total_remaining == 0:
            return {}

        distribution = {}

        def recurse(hand, prob, state):
            total = hand_value(hand)
            remaining = sum(state.values())

            if total >= 17:
                # Dealer stands — record this outcome
                key = total if total <= 21 else 'bust'
                distribution[key] = distribution.get(key, 0) + prob
                return

            # Dealer hits — branch over every possible next card
            for rank in RANKS:
                count = state.get(rank, 0)
                if count == 0:
                    continue
                draw_prob = count / remaining
                new_state = state.copy()
                new_state[rank] -= 1
                recurse(hand + [rank], prob * draw_prob, new_state)

        recurse([dealer_upcard], 1.0, deck_state.copy())
        return distribution

    def ev_stand(self, player_total, dealer_upcard, deck_state):
        """
        EV of standing: depends entirely on the dealer's final total distribution.

        Args:
            player_total (int):  Player's current hand total.
            dealer_upcard (str): Dealer's visible card rank.
            deck_state (dict):   Current remaining card counts.

        Returns:
            float: Expected value of standing.
        """
        dist = self.dealer_distribution(dealer_upcard, deck_state)
        ev = 0.0
        for outcome, prob in dist.items():
            if outcome == 'bust':
                ev += prob * 1.0          # Player wins
            elif outcome < player_total:
                ev += prob * 1.0          # Player wins
            elif outcome == player_total:
                ev += prob * 0.0          # Push
            else:
                ev += prob * -1.0         # Player loses
        return round(ev, 4)

    def ev_hit(self, player_hand, dealer_upcard, deck_state, depth=0):
        """
        EV of hitting: weighted average over all possible next cards.
        Recursively computes the best EV from the resulting state.

        Args:
            player_hand (list):  Player's current cards (list of rank strings).
            dealer_upcard (str): Dealer's visible card rank.
            deck_state (dict):   Current remaining card counts.
            depth (int):         Recursion depth limiter.

        Returns:
            float: Expected value of hitting.
        """
        if depth > 4:                     # Limit recursion depth
            return self.ev_stand(hand_value(player_hand), dealer_upcard, deck_state)

        total_remaining = sum(deck_state.values())
        if total_remaining == 0:
            return 0.0

        ev = 0.0
        for rank in RANKS:
            count = deck_state.get(rank, 0)
            if count == 0:
                continue
            prob = count / total_remaining
            new_hand  = player_hand + [rank]
            new_state = deck_state.copy()
            new_state[rank] -= 1
            new_total = hand_value(new_hand)

            if new_total > 21:
                ev += prob * -1.0         # Busted — lose
            else:
                # Best action from the new state
                best = max(
                    self.ev_stand(new_total, dealer_upcard, new_state),
                    self.ev_hit(new_hand, dealer_upcard, new_state, depth + 1)
                )
                ev += prob * best

        return round(ev, 4)

    def recommend(self, player_hand, dealer_upcard, deck_state):
        """
        Compute EV for all valid actions and return the best one.

        Args:
            player_hand (list):   Player's cards, e.g. ["9", "7"].
            dealer_upcard (str):  Dealer's visible card, e.g. "6".
            deck_state (dict):    Current remaining card counts.

        Returns:
            tuple: (best_action: str, ev_breakdown: dict)
                   e.g. ("Stand", {"Stand": 0.23, "Hit": -0.11, "Double": -0.22})
        """
        if not player_hand or not dealer_upcard:
            return None, {}

        player_total = hand_value(player_hand)

        ev_s = self.ev_stand(player_total, dealer_upcard, deck_state)
        ev_h = self.ev_hit(player_hand, dealer_upcard, deck_state)
        ev_d = round(ev_h * 2, 4)        # Double: same as hit but payoff doubled

        ev_breakdown = {"Stand": ev_s, "Hit": ev_h, "Double": ev_d}

        # Add Split option for pairs
        if len(player_hand) == 2 and player_hand[0] == player_hand[1]:
            split_hand  = [player_hand[0]]
            ev_split    = self.ev_hit(split_hand, dealer_upcard, deck_state)
            ev_breakdown["Split"] = round(ev_split, 4)

        best_action = max(ev_breakdown, key=ev_breakdown.get)
        return best_action, ev_breakdown
