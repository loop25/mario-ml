"""Tests for the tournament engine."""

import pytest

from src.opponents import RandomOpponent, MinimaxOpponent
from src.tournament import (
    MatchResult,
    TournamentEngine,
    TournamentParticipant,
    TournamentResult,
)


# ---- Dataclass tests ----


class TestTournamentParticipant:
    def test_fields_accessible(self):
        p = TournamentParticipant(
            name="Alice", participant_type="random", config={"seed": 42}
        )
        assert p.name == "Alice"
        assert p.participant_type == "random"
        assert p.config == {"seed": 42}

    def test_default_config(self):
        p = TournamentParticipant(name="Bob", participant_type="minimax")
        assert p.config == {}


class TestMatchResult:
    def test_fields_accessible(self):
        r = MatchResult(
            player1="Alice",
            player2="Bob",
            wins_p1=5,
            wins_p2=3,
            draws=2,
            games_played=10,
        )
        assert r.player1 == "Alice"
        assert r.player2 == "Bob"
        assert r.wins_p1 == 5
        assert r.wins_p2 == 3
        assert r.draws == 2
        assert r.games_played == 10


# ---- Opponent factory tests ----


class TestCreateOpponent:
    def test_create_random_opponent(self):
        engine = TournamentEngine(
            game_id="tictactoe",
            participants=[],
        )
        p = TournamentParticipant(name="Rando", participant_type="random")
        opp = engine._create_opponent(p)
        assert isinstance(opp, RandomOpponent)

    def test_create_minimax_opponent(self):
        engine = TournamentEngine(
            game_id="tictactoe",
            participants=[],
        )
        p = TournamentParticipant(
            name="Mini", participant_type="minimax", config={"depth": 1}
        )
        opp = engine._create_opponent(p)
        assert isinstance(opp, MinimaxOpponent)
        assert opp.depth == 1


# ---- Match tests ----


class TestRunMatch:
    def test_run_match_returns_result(self):
        p1 = TournamentParticipant(name="Random1", participant_type="random")
        p2 = TournamentParticipant(name="Random2", participant_type="random")
        engine = TournamentEngine(
            game_id="tictactoe",
            participants=[p1, p2],
            games_per_match=4,
        )
        result = engine.run_match(p1, p2)
        assert isinstance(result, MatchResult)
        assert result.player1 == "Random1"
        assert result.player2 == "Random2"
        # Each game = 2 episodes, 4 games = 8 total episodes
        assert result.games_played == result.wins_p1 + result.wins_p2 + result.draws
        assert result.games_played == 8  # 4 games * 2 episodes each


# ---- Tournament tests ----


class TestRoundRobinTournament:
    def test_round_robin_tournament(self):
        participants = [
            TournamentParticipant(name="R1", participant_type="random"),
            TournamentParticipant(name="R2", participant_type="random"),
            TournamentParticipant(name="R3", participant_type="random"),
        ]
        engine = TournamentEngine(
            game_id="tictactoe",
            participants=participants,
            games_per_match=2,
            format="round_robin",
        )
        result = engine.run_tournament()
        assert isinstance(result, TournamentResult)
        assert result.game_id == "tictactoe"
        assert result.format == "round_robin"
        # 3 participants => C(3,2) = 3 matches
        assert len(result.matches) == 3
        assert len(result.standings) == 3

    def test_standings_have_all_participants(self):
        names = ["Alpha", "Beta", "Gamma"]
        participants = [
            TournamentParticipant(name=n, participant_type="random")
            for n in names
        ]
        engine = TournamentEngine(
            game_id="tictactoe",
            participants=participants,
            games_per_match=2,
            format="round_robin",
        )
        result = engine.run_tournament()
        standing_names = [s["name"] for s in result.standings]
        for n in names:
            assert n in standing_names

    def test_champion_selected(self):
        participants = [
            TournamentParticipant(name="R1", participant_type="random"),
            TournamentParticipant(name="R2", participant_type="random"),
            TournamentParticipant(name="R3", participant_type="random"),
        ]
        engine = TournamentEngine(
            game_id="tictactoe",
            participants=participants,
            games_per_match=2,
            format="round_robin",
        )
        result = engine.run_tournament()
        # Champion should be the participant with the most points
        assert result.champion == result.standings[0]["name"]
        assert result.champion in [p.name for p in participants]


# ---- Event bus integration ----


class TestTournamentEvents:
    def test_tournament_events_published(self):
        from src.achievements.event_bus import EventBus

        bus = EventBus()
        events_received = []
        bus.subscribe("tournament_start", lambda e: events_received.append(e))
        bus.subscribe("tournament_complete", lambda e: events_received.append(e))
        bus.subscribe("match_complete", lambda e: events_received.append(e))

        participants = [
            TournamentParticipant(name="E1", participant_type="random"),
            TournamentParticipant(name="E2", participant_type="random"),
        ]
        engine = TournamentEngine(
            game_id="tictactoe",
            participants=participants,
            games_per_match=2,
            format="round_robin",
            event_bus=bus,
        )
        engine.run_tournament()

        event_types = [e["type"] for e in events_received]
        assert "tournament_start" in event_types
        assert "tournament_complete" in event_types
        assert "match_complete" in event_types

        # Verify tournament_start payload
        start_evt = [e for e in events_received if e["type"] == "tournament_start"][0]
        assert start_evt["game_id"] == "tictactoe"
        assert set(start_evt["participants"]) == {"E1", "E2"}

        # Verify tournament_complete payload
        end_evt = [e for e in events_received if e["type"] == "tournament_complete"][0]
        assert "champion" in end_evt
        assert "standings" in end_evt
