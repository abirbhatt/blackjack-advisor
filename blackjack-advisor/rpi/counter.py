# counter.py
# Hi-Lo card counting system.
#
# The Hi-Lo system assigns a value to each card rank:
#   +1 for low cards (2–6)  → their removal makes the deck worse for the player
#    0 for neutral (7–9)    → no effect
#   -1 for high cards (10–A) → their removal makes the deck worse for the dealer
#
# Running count: cumulative sum of Hi-Lo values for all cards seen so far.
# True count: running_count / decks_remaining
#   → normalizes for the number of unseen cards remaining
#   → true count > +2 means the deck is statistically favorable for the player
#
# Note: AI tools (Claude) were used to assist with code development.


HI_LO_VALUES = {
    '2': +1, '3': +1, '4': +1, '5': +1, '6': +1,
    '7':  0, '8':  0, '9':  0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1,
}


class HiLoCounter:
    def __init__(self):
        self.running_count = 0

    def update(self, rank):
        """
        Update the running count when a new card is observed.

        Args:
            rank (str): Card rank, e.g. "A", "10", "5".
        """
        self.running_count += HI_LO_VALUES.get(rank, 0)

    def get_counts(self, decks_remaining):
        """
        Return both the running count and true count.

        Args:
            decks_remaining (float): From DeckManager.decks_remaining().

        Returns:
            tuple: (running_count: int, true_count: float)
        """
        true_count = self.running_count / decks_remaining
        return self.running_count, round(true_count, 2)

    def reset(self):
        """Reset at the start of a new shoe."""
        self.running_count = 0
