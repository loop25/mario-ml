"""Tests for the natural language command parser."""

import pytest

from src.nl_commands.parser import ParsedCommand, parse_command


class TestParseGame:
    def test_parse_game_snake(self):
        cmd = parse_command("train snake")
        assert cmd.game_id == "snake"

    def test_parse_game_chess(self):
        cmd = parse_command("play chess")
        assert cmd.game_id == "chess"


class TestParseAlgorithm:
    def test_parse_algo_ppo(self):
        cmd = parse_command("use ppo")
        assert cmd.algorithm == "ppo"

    def test_parse_algo_neat(self):
        cmd = parse_command("evolve with neat")
        assert cmd.algorithm == "neat"


class TestParseDuration:
    def test_parse_duration_quick(self):
        cmd = parse_command("quick demo")
        assert cmd.episodes == 100

    def test_parse_duration_overnight(self):
        cmd = parse_command("overnight training")
        assert cmd.episodes == 50000


class TestParseOpponent:
    def test_parse_opponent_hard(self):
        cmd = parse_command("against hard ai")
        assert cmd.opponent == "minimax-hard"


class TestParseFlags:
    def test_parse_eval_mode(self):
        cmd = parse_command("watch the agent play")
        assert cmd.eval_mode is True

    def test_parse_stream(self):
        cmd = parse_command("stream on twitch")
        assert cmd.stream is True


class TestFullCommand:
    def test_parse_full_command(self):
        cmd = parse_command("train snake with ppo overnight")
        assert cmd.game_id == "snake"
        assert cmd.algorithm == "ppo"
        assert cmd.episodes == 50000


class TestConfidenceAndDescription:
    def test_parse_gibberish(self):
        cmd = parse_command("asdfghjkl")
        assert cmd.confidence < 0.2

    def test_description_generated(self):
        cmd = parse_command("train snake with ppo")
        assert cmd.description != ""
        assert len(cmd.description) > 0

    def test_confidence_scales(self):
        cmd_low = parse_command("snake")
        cmd_high = parse_command("train snake with ppo overnight stream")
        assert cmd_high.confidence > cmd_low.confidence


class TestEdgeCases:
    def test_empty_string(self):
        cmd = parse_command("")
        assert cmd.confidence == 0.0
        assert cmd.game_id is None

    def test_case_insensitive(self):
        cmd = parse_command("TRAIN SNAKE WITH PPO")
        assert cmd.game_id == "snake"
        assert cmd.algorithm == "ppo"

    def test_snek_alias(self):
        cmd = parse_command("snek")
        assert cmd.game_id == "snake"

    def test_connect4_variations(self):
        assert parse_command("connect 4").game_id == "connect4"
        assert parse_command("c4").game_id == "connect4"

    def test_tic_tac_toe_variations(self):
        assert parse_command("tic tac toe").game_id == "tictactoe"
        assert parse_command("ttt").game_id == "tictactoe"

    def test_device_parsing(self):
        cmd = parse_command("train on gpu")
        assert cmd.device == "cuda"
