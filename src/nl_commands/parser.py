"""Keyword-based natural language command parser.

Converts free-text commands like "Train snake with PPO overnight" into
structured ``ParsedCommand`` objects that the launcher can apply directly.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedCommand:
    """Structured result of parsing a natural language training command."""

    game_id: Optional[str] = None
    algorithm: Optional[str] = None
    episodes: Optional[int] = None
    opponent: Optional[str] = None
    eval_mode: bool = False
    stream: bool = False
    device: Optional[str] = None
    description: str = ''
    confidence: float = 0.0


# ── keyword tables ──────────────────────────────────────────────────────

GAME_KEYWORDS = {
    'super mario': 'mario',
    'mario': 'mario',
    'smb': 'mario',
    'snake': 'snake',
    'snek': 'snake',
    'tetris': 'tetris',
    'blocks': 'tetris',
    'chess': 'chess',
    'checkers': 'checkers',
    'draughts': 'checkers',
    'connect 4': 'connect4',
    'connect4': 'connect4',
    'connect': 'connect4',
    'c4': 'connect4',
    'tic tac toe': 'tictactoe',
    'tictactoe': 'tictactoe',
    'ttt': 'tictactoe',
    'noughts': 'tictactoe',
}

ALGO_KEYWORDS = {
    'decision transformer': 'dt',
    'deep q': 'dqn',
    'q-learning': 'dqn',
    'neuroevolution': 'neat',
    'proximal': 'ppo',
    'ppo': 'ppo',
    'dqn': 'dqn',
    'a2c': 'a2c',
    'advantage': 'a2c',
    'neat': 'neat',
    'evolve': 'neat',
    'genetic': 'neat',
    'rainbow': 'rainbow',
    'dt': 'dt',
    'transformer': 'dt',
    'generalist': 'dt',
}

DURATION_KEYWORDS = {
    'quick': 100,
    'fast': 100,
    'demo': 100,
    'test': 100,
    'standard': 5000,
    'normal': 5000,
    'regular': 5000,
    'deep': 25000,
    'long': 25000,
    'thorough': 25000,
    'overnight': 50000,
    'intensive': 50000,
    'marathon': 50000,
}

OPPONENT_KEYWORDS = {
    'vs human': 'human-vs-human',
    'easy': 'minimax-easy',
    'beginner': 'minimax-easy',
    'medium': 'minimax-medium',
    'intermediate': 'minimax-medium',
    'hard': 'minimax-hard',
    'difficult': 'minimax-hard',
    'expert': 'minimax-hard',
    'human': 'human',
    'play': 'human',
    'myself': 'human',
    'random': 'random',
}

EVAL_KEYWORDS = {'watch', 'evaluate', 'eval', 'see', 'show', 'demo', 'play'}
STREAM_KEYWORDS = {'stream', 'broadcast', 'live', 'twitch', 'youtube'}

DEVICE_KEYWORDS = {
    'cpu': 'cpu',
    'gpu': 'cuda',
    'cuda': 'cuda',
    'mps': 'mps',
}


# ── helpers ─────────────────────────────────────────────────────────────

def _match_keywords(text: str, keywords: dict) -> Optional[str]:
    """Return the value for the first keyword found in *text*.

    Multi-word keywords are checked before single-word ones because the
    dict is iterated in insertion order (longer phrases first).
    """
    for keyword, value in keywords.items():
        if keyword in text:
            return value
    return None


def _build_description(cmd: 'ParsedCommand') -> str:
    """Create a human-readable summary of the parsed command."""
    parts: list[str] = []

    # Action verb
    if cmd.eval_mode:
        parts.append('Watch')
    else:
        parts.append('Train')

    # Game
    if cmd.game_id:
        parts.append(cmd.game_id)

    # Algorithm
    if cmd.algorithm:
        parts.append(f'with {cmd.algorithm.upper()}')

    # Opponent
    if cmd.opponent:
        parts.append(f'against {cmd.opponent}')

    # Episodes
    if cmd.episodes:
        parts.append(f'for {cmd.episodes:,} episodes')

    # Streaming
    if cmd.stream:
        parts.append('(streaming)')

    return ' '.join(parts) if parts else ''


# ── main entry point ────────────────────────────────────────────────────

def parse_command(text: str) -> ParsedCommand:
    """Parse a natural language command string into a :class:`ParsedCommand`.

    Parameters
    ----------
    text:
        Free-text command, e.g. ``"train snake with ppo overnight"``.

    Returns
    -------
    ParsedCommand
        Structured command with confidence score (0 -- 1).
    """
    cmd = ParsedCommand()
    lower = text.lower().strip()

    if not lower:
        return cmd

    # Match each category
    cmd.game_id = _match_keywords(lower, GAME_KEYWORDS)
    cmd.algorithm = _match_keywords(lower, ALGO_KEYWORDS)
    cmd.device = _match_keywords(lower, DEVICE_KEYWORDS)

    # Duration
    duration_val = _match_keywords(lower, DURATION_KEYWORDS)
    if duration_val is not None:
        cmd.episodes = duration_val

    # Opponent
    cmd.opponent = _match_keywords(lower, OPPONENT_KEYWORDS)

    # Boolean flags – check individual words
    words = set(lower.split())
    if words & EVAL_KEYWORDS:
        cmd.eval_mode = True
    if words & STREAM_KEYWORDS:
        cmd.stream = True

    # Confidence: each recognised field adds 0.2, capped at 1.0
    fields_found = sum([
        cmd.game_id is not None,
        cmd.algorithm is not None,
        cmd.episodes is not None,
        cmd.opponent is not None,
        cmd.eval_mode or cmd.stream,
    ])
    cmd.confidence = min(fields_found * 0.2, 1.0)

    cmd.description = _build_description(cmd)

    return cmd
