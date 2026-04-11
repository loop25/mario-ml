"""Human-controlled opponent via input callback."""

import random

from src.opponents.base_opponent import BaseOpponent


class HumanOpponent(BaseOpponent):
    """Opponent controlled by a human player via input callback.

    The callback receives a ``board_state`` dict and must return an ``int``
    action.  When no callback is supplied the opponent falls back to a simple
    ``input()``-based console prompt.  A GUI renderer can swap in its own
    callback at any time via :meth:`set_input_callback`.
    """

    def __init__(self, input_callback=None):
        self._callback = input_callback or self._console_input

    def set_input_callback(self, callback):
        """Set or change the input callback (e.g., when GUI becomes available)."""
        self._callback = callback

    @property
    def name(self) -> str:
        return "Human"

    @property
    def difficulty_tier(self) -> int:
        return -1

    def pick_action(self, board_state: dict) -> int:
        valid = board_state.get("valid_actions", [])
        if not valid:
            return 0

        for attempt in range(3):
            try:
                action = self._callback(board_state)
                action = int(action)
                if action in valid:
                    return action
                print(f"  [HumanOpponent] Invalid action {action}, valid: {valid}")
            except Exception as e:
                print(f"  [HumanOpponent] Input error: {e}")

        return random.choice(valid)

    @staticmethod
    def _console_input(board_state):
        """Fallback: prompt in the terminal."""
        valid = board_state.get("valid_actions", [])
        print(f"  Your turn! Valid actions: {valid}")
        return int(input("  Enter action: "))
