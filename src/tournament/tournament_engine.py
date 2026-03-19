"""
Tournament engine — runs round-robin or bracket tournaments between agents.

Each match consists of N games. Each game plays two episodes (one from each
side) so that both participants get equal turns as player 1 and player 2.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.opponents.base_opponent import BaseOpponent
from src.opponents.minimax_opponent import MinimaxOpponent
from src.opponents.random_opponent import RandomOpponent


@dataclass
class TournamentParticipant:
    """A participant in a tournament."""

    name: str  # Display name
    participant_type: str  # 'random', 'minimax', 'model', 'human'
    config: dict = field(default_factory=dict)  # e.g. {'depth': 3}


@dataclass
class MatchResult:
    """Result of a match (multiple games) between two participants."""

    player1: str  # Name
    player2: str  # Name
    wins_p1: int
    wins_p2: int
    draws: int
    games_played: int


@dataclass
class TournamentResult:
    """Final result of a complete tournament."""

    game_id: str
    format: str  # 'round_robin' or 'bracket'
    participants: List[str]  # Names
    matches: List[MatchResult]
    standings: List[dict]  # [{'name', 'wins', 'losses', 'draws', 'points'}]
    champion: str  # Name of winner


class TournamentEngine:
    """Runs tournaments between multiple agent participants.

    Parameters
    ----------
    game_id : str
        Which board game to play (e.g. 'tictactoe', 'connect4').
    participants : list of TournamentParticipant
        The agents competing in the tournament.
    games_per_match : int
        Number of games per match (each game = 2 episodes, one per side).
    format : str
        Tournament format: 'round_robin' or 'bracket'.
    event_bus : EventBus or None
        Optional event bus for publishing tournament events.
    """

    def __init__(
        self,
        game_id: str,
        participants: List[TournamentParticipant],
        games_per_match: int = 10,
        format: str = "round_robin",
        event_bus=None,
    ):
        self.game_id = game_id
        self.participants = participants
        self.games_per_match = games_per_match
        self.format = format
        self.event_bus = event_bus

    # ------------------------------------------------------------------
    # Opponent factory
    # ------------------------------------------------------------------

    def _create_opponent(self, participant: TournamentParticipant) -> BaseOpponent:
        """Create an opponent instance from participant config."""
        ptype = participant.participant_type

        if ptype == "random":
            return RandomOpponent()

        if ptype == "minimax":
            depth = participant.config.get("depth", 3)
            game_id = participant.config.get("game_id", self.game_id)
            return MinimaxOpponent(depth=depth, game_id=game_id)

        if ptype == "model":
            from src.opponents.model_opponent import ModelOpponent

            checkpoint = participant.config.get("checkpoint", "")
            device = participant.config.get("device", "cpu")
            return ModelOpponent(checkpoint_path=checkpoint, device=device)

        if ptype == "human":
            from src.opponents.human_opponent import HumanOpponent

            return HumanOpponent()

        raise ValueError(f"Unknown participant type: {ptype}")

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------

    def _run_episode(self, agent: BaseOpponent, env) -> float:
        """Run a single episode where *agent* controls player 1.

        Returns the cumulative reward (positive = agent wins).
        """
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_steps = 500

        while not done and steps < max_steps:
            board_state = {
                "board": env.board if hasattr(env, "board") else obs,
                "valid_actions": list(range(env.action_space.n)),
                "game_id": self.game_id,
                "turn": 1,
            }
            action = agent.pick_action(board_state)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        return total_reward

    # ------------------------------------------------------------------
    # Match runner
    # ------------------------------------------------------------------

    def run_match(
        self, p1: TournamentParticipant, p2: TournamentParticipant
    ) -> MatchResult:
        """Run a full match (N games) between two participants.

        Each game consists of two episodes so each participant plays
        as player 1 once per game.
        """
        from games.registry import GameRegistry

        registry = GameRegistry()
        registry.discover()
        adapter = registry.get_game(self.game_id)

        opp_p1 = self._create_opponent(p1)
        opp_p2 = self._create_opponent(p2)

        wins_p1 = 0
        wins_p2 = 0
        draws = 0

        for _ in range(self.games_per_match):
            # --- Episode 1: P1 as agent, P2 as opponent ---
            env1 = adapter.create_env(opponent=opp_p2)
            reward1 = self._run_episode(opp_p1, env1)
            if hasattr(env1, "close"):
                env1.close()

            if reward1 > 0:
                wins_p1 += 1
            elif reward1 < 0:
                wins_p2 += 1
            else:
                draws += 1

            # --- Episode 2: P2 as agent, P1 as opponent ---
            env2 = adapter.create_env(opponent=opp_p1)
            reward2 = self._run_episode(opp_p2, env2)
            if hasattr(env2, "close"):
                env2.close()

            if reward2 > 0:
                wins_p2 += 1
            elif reward2 < 0:
                wins_p1 += 1
            else:
                draws += 1

        total_games = wins_p1 + wins_p2 + draws
        return MatchResult(
            player1=p1.name,
            player2=p2.name,
            wins_p1=wins_p1,
            wins_p2=wins_p2,
            draws=draws,
            games_played=total_games,
        )

    # ------------------------------------------------------------------
    # Tournament runners
    # ------------------------------------------------------------------

    def run_tournament(
        self,
        progress_callback: Optional[
            Callable[[int, int, MatchResult], None]
        ] = None,
    ) -> TournamentResult:
        """Run the full tournament and return the result."""
        if self.event_bus is not None:
            self.event_bus.publish(
                {
                    "type": "tournament_start",
                    "participants": [p.name for p in self.participants],
                    "game_id": self.game_id,
                    "format": self.format,
                }
            )

        if self.format == "round_robin":
            matches = self._run_round_robin(progress_callback)
        elif self.format == "bracket":
            matches = self._run_bracket(progress_callback)
        else:
            raise ValueError(f"Unknown tournament format: {self.format}")

        standings = self._compute_standings(matches)
        champion = standings[0]["name"] if standings else ""

        result = TournamentResult(
            game_id=self.game_id,
            format=self.format,
            participants=[p.name for p in self.participants],
            matches=matches,
            standings=standings,
            champion=champion,
        )

        if self.event_bus is not None:
            self.event_bus.publish(
                {
                    "type": "tournament_complete",
                    "champion": champion,
                    "standings": standings,
                }
            )

        return result

    def _run_round_robin(self, progress_callback) -> List[MatchResult]:
        """Every participant plays every other participant once."""
        matches: List[MatchResult] = []
        pairs = [
            (i, j)
            for i in range(len(self.participants))
            for j in range(i + 1, len(self.participants))
        ]
        total = len(pairs)

        for idx, (i, j) in enumerate(pairs):
            result = self.run_match(self.participants[i], self.participants[j])
            matches.append(result)

            if self.event_bus is not None:
                self.event_bus.publish(
                    {
                        "type": "match_complete",
                        "player1": result.player1,
                        "player2": result.player2,
                        "result": {
                            "wins_p1": result.wins_p1,
                            "wins_p2": result.wins_p2,
                            "draws": result.draws,
                        },
                    }
                )

            if progress_callback is not None:
                progress_callback(idx + 1, total, result)

        return matches

    def _run_bracket(self, progress_callback) -> List[MatchResult]:
        """Single-elimination bracket tournament.

        If the number of participants is not a power of two, extra
        participants receive a bye in the first round.
        """
        matches: List[MatchResult] = []
        remaining = list(self.participants)
        random.shuffle(remaining)

        # Pad to next power of two with byes
        n = len(remaining)
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2
        num_byes = next_pow2 - n

        total_matches = next_pow2 - 1  # total matches in single elimination
        match_num = 0

        while len(remaining) > 1:
            next_round: List[TournamentParticipant] = []
            i = 0
            while i < len(remaining):
                if i + 1 < len(remaining):
                    result = self.run_match(remaining[i], remaining[i + 1])
                    matches.append(result)
                    match_num += 1

                    if self.event_bus is not None:
                        self.event_bus.publish(
                            {
                                "type": "match_complete",
                                "player1": result.player1,
                                "player2": result.player2,
                                "result": {
                                    "wins_p1": result.wins_p1,
                                    "wins_p2": result.wins_p2,
                                    "draws": result.draws,
                                },
                            }
                        )

                    if progress_callback is not None:
                        progress_callback(match_num, total_matches, result)

                    # Winner advances
                    if result.wins_p1 >= result.wins_p2:
                        next_round.append(remaining[i])
                    else:
                        next_round.append(remaining[i + 1])
                    i += 2
                else:
                    # Bye — odd participant advances automatically
                    next_round.append(remaining[i])
                    i += 1
            remaining = next_round

        return matches

    # ------------------------------------------------------------------
    # Standings
    # ------------------------------------------------------------------

    def _compute_standings(self, matches: List[MatchResult]) -> List[dict]:
        """Compute standings from match results.

        Scoring: Win = 3 pts, Draw = 1 pt, Loss = 0 pts.
        """
        stats: dict = {}
        for p in self.participants:
            stats[p.name] = {"wins": 0, "losses": 0, "draws": 0, "points": 0.0}

        for m in matches:
            if m.player1 in stats:
                stats[m.player1]["wins"] += m.wins_p1
                stats[m.player1]["losses"] += m.wins_p2
                stats[m.player1]["draws"] += m.draws
                stats[m.player1]["points"] += m.wins_p1 * 3 + m.draws * 1

            if m.player2 in stats:
                stats[m.player2]["wins"] += m.wins_p2
                stats[m.player2]["losses"] += m.wins_p1
                stats[m.player2]["draws"] += m.draws
                stats[m.player2]["points"] += m.wins_p2 * 3 + m.draws * 1

        standings = [
            {"name": name, **s} for name, s in stats.items()
        ]
        standings.sort(key=lambda x: (-x["points"], -x["wins"]))
        return standings
