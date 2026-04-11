"""Abstract base class for all game opponents."""

from abc import ABC, abstractmethod


class BaseOpponent(ABC):
    """Interface that every opponent implementation must satisfy.

    Subclasses must provide ``name``, ``difficulty_tier``, and
    ``pick_action``.  Stateful opponents (e.g. MCTS with a tree) can
    override ``reset()`` to clear between episodes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable opponent name."""

    @property
    @abstractmethod
    def difficulty_tier(self) -> int:
        """Difficulty tier: 0=random, 1-3=minimax, 4-5=trained model."""

    @abstractmethod
    def pick_action(self, board_state: dict) -> int:
        """Choose an action given the current board state.

        Parameters
        ----------
        board_state : dict
            Keys:
                board       – np.ndarray or game-specific object
                valid_actions – List[int] of legal action indices
                game_id     – str identifier for the game type
                turn        – int move counter
        """

    def reset(self) -> None:
        """Reset any internal state between episodes (no-op by default)."""
