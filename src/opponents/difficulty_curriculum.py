"""
Auto-adjusting opponent difficulty based on agent win rate.

The DifficultyCurriculum wraps the opponent system and automatically
promotes or demotes the opponent difficulty tier as the agent's win
rate rises or falls over a sliding window of recent episodes.
"""

DIFFICULTY_TIERS = [
    {'name': 'Random', 'type': 'random', 'config': {}},
    {'name': 'Minimax Easy', 'type': 'minimax', 'config': {'depth': 1}},
    {'name': 'Minimax Medium', 'type': 'minimax', 'config': {'depth': 3}},
    {'name': 'Minimax Hard', 'type': 'minimax', 'config': {'depth': 5}},
]


class DifficultyCurriculum:
    """Auto-adjusting opponent difficulty based on agent win rate."""

    def __init__(self, game_id: str, event_bus=None,
                 promote_threshold: float = 0.70,
                 demote_threshold: float = 0.30,
                 window_size: int = 50):
        self.game_id = game_id
        self._bus = event_bus
        self._promote_threshold = promote_threshold
        self._demote_threshold = demote_threshold
        self._window_size = window_size
        self._current_tier = 0
        self._results = []  # Recent episode results: 1=win, 0=draw, -1=loss
        self._total_promotions = 0
        self._total_demotions = 0

        if event_bus:
            event_bus.subscribe('episode_complete', self._on_episode)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_tier(self) -> int:
        return self._current_tier

    @property
    def current_tier_name(self) -> str:
        return DIFFICULTY_TIERS[self._current_tier]['name']

    @property
    def max_tier(self) -> int:
        return len(DIFFICULTY_TIERS) - 1

    @property
    def win_rate(self) -> float:
        """Current win rate over the sliding window."""
        if not self._results:
            return 0.0
        wins = sum(1 for r in self._results if r == 1)
        return wins / len(self._results)

    @property
    def progress(self) -> float:
        """Progress toward next promotion (0.0 to 1.0)."""
        if self._current_tier >= self.max_tier:
            return 1.0
        if not self._results:
            return 0.0
        return min(1.0, self.win_rate / self._promote_threshold)

    # ------------------------------------------------------------------
    # Opponent factory
    # ------------------------------------------------------------------

    def create_opponent(self):
        """Create an opponent for the current difficulty tier."""
        tier = DIFFICULTY_TIERS[self._current_tier]
        if tier['type'] == 'random':
            from src.opponents import RandomOpponent
            return RandomOpponent()
        elif tier['type'] == 'minimax':
            from src.opponents import MinimaxOpponent
            return MinimaxOpponent(depth=tier['config']['depth'],
                                   game_id=self.game_id)
        elif tier['type'] == 'model':
            from src.opponents import ModelOpponent
            return ModelOpponent(tier['config']['checkpoint'])
        else:
            from src.opponents import RandomOpponent
            return RandomOpponent()

    # ------------------------------------------------------------------
    # Result tracking
    # ------------------------------------------------------------------

    def record_result(self, reward: float):
        """Record an episode result. reward > 0 = win, < 0 = loss, 0 = draw."""
        if reward > 0:
            self._results.append(1)
        elif reward < 0:
            self._results.append(-1)
        else:
            self._results.append(0)

        # Keep sliding window
        if len(self._results) > self._window_size:
            self._results = self._results[-self._window_size:]

        # Check for tier change after enough games
        if len(self._results) >= self._window_size:
            self._check_tier_change()

    def _check_tier_change(self):
        """Promote or demote based on win rate."""
        wr = self.win_rate

        if wr >= self._promote_threshold and self._current_tier < self.max_tier:
            self._current_tier += 1
            self._total_promotions += 1
            self._results.clear()  # Reset window for new tier
            if self._bus:
                self._bus.publish({
                    'type': 'difficulty_changed',
                    'direction': 'promoted',
                    'new_tier': self._current_tier,
                    'tier_name': self.current_tier_name,
                    'win_rate': wr,
                })

        elif wr <= self._demote_threshold and self._current_tier > 0:
            self._current_tier -= 1
            self._total_demotions += 1
            self._results.clear()
            if self._bus:
                self._bus.publish({
                    'type': 'difficulty_changed',
                    'direction': 'demoted',
                    'new_tier': self._current_tier,
                    'tier_name': self.current_tier_name,
                    'win_rate': wr,
                })

    def _on_episode(self, event):
        """EventBus handler for episode_complete events."""
        reward = event.get('reward', 0)
        game_id = event.get('game_id', '')
        if game_id == self.game_id:
            self.record_result(reward)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            'current_tier': self._current_tier,
            'tier_name': self.current_tier_name,
            'win_rate': self.win_rate,
            'progress': self.progress,
            'total_promotions': self._total_promotions,
            'total_demotions': self._total_demotions,
            'games_in_window': len(self._results),
            'window_size': self._window_size,
        }
