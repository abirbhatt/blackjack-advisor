# deck_manager.py
# Tracks the remaining card composition of the shoe.
# Each rank starts at 4 (one per suit in a single deck).
# When a new card is detected, its rank count is decremented by 1.
#
# decks_remaining is used by the Hi-Lo counter to compute the true count
# and by the EV calculator to compute draw probabilities.
#
# Note: AI tools (Claude) were used to assist with code development.


class DeckManager:
    def __init__(self, num_decks=1):
        """
        Initialize the deck state for a fresh shoe.

        Args:
            num_decks (int): Number of decks in the shoe (default: 1).
        """
        self.num_decks = num_decks
        self.deck_state = {
            '2': 4 * num_decks,
            '3': 4 * num_decks,
            '4': 4 * num_decks,
            '5': 4 * num_decks,
            '6': 4 * num_decks,
            '7': 4 * num_decks,
            '8': 4 * num_decks,
            '9': 4 * num_decks,
            '10': 4 * num_decks,
            'J': 4 * num_decks,
            'Q': 4 * num_decks,
            'K': 4 * num_decks,
            'A': 4 * num_decks,
        }

    def remove_card(self, rank):
        """
        Decrement the count for a rank when a new card is detected.
        Guards against going below zero (shouldn't happen, but defensive).

        Args:
            rank (str): Card rank, e.g. "A", "10", "K".
        """
        if rank in self.deck_state and self.deck_state[rank] > 0:
            self.deck_state[rank] -= 1

    def total_cards_remaining(self):
        """Return total number of cards left in the shoe."""
        return sum(self.deck_state.values())

    def decks_remaining(self):
        """
        Return the number of full decks worth of cards still in the shoe.
        Used to compute the true count: true_count = running_count / decks_remaining.
        Minimum of 0.5 to avoid division by zero near end of shoe.
        """
        return max(self.total_cards_remaining() / 52, 0.5)

    def probability(self, rank):
        """
        Return the probability of drawing a given rank from the current shoe.
        Used by the EV calculator to weight possible draw sequences.

        Args:
            rank (str): Card rank.

        Returns:
            float: Probability in [0, 1].
        """
        total = self.total_cards_remaining()
        if total == 0:
            return 0.0
        return self.deck_state.get(rank, 0) / total

    def reset(self):
        """Reset the shoe to a full fresh deck (call between sessions)."""
        self.__init__(self.num_decks)
