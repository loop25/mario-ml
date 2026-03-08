"""
Curriculum learning manager for whole-game Mario training.

Manages structured stage progression with:
- Performance-based advancement thresholds
- Periodic revisitation of earlier stages
- Per-stage metrics tracking
- State persistence for pause/resume across sessions
"""
import json
import random
from collections import deque
from typing import Dict, List, Optional, Tuple


ALL_STAGES = [(w, s) for w in range(1, 9) for s in range(1, 5)]


class CurriculumManager:
    """
    Manages stage progression with curriculum learning.

    Args:
        start_world: Starting world (1-8). Default 1.
        start_stage: Starting stage (1-4). Default 1.
        advance_threshold: Average reward needed to advance. Default 200.
        advance_window: Number of recent episodes to average. Default 10.
        revisit_interval: Episodes between revisitation checks. Default 20.
        completion_count: Times stage must be completed to advance. Default 3.
    """

    def __init__(
        self,
        start_world: int = 1,
        start_stage: int = 1,
        advance_threshold: float = 200.0,
        advance_window: int = 10,
        revisit_interval: int = 20,
        completion_count: int = 3,
    ):
        self._current_world = start_world
        self._current_stage = start_stage
        self.advance_threshold = advance_threshold
        self.advance_window = advance_window
        self.revisit_interval = revisit_interval
        self.completion_count = completion_count

        self.stage_rewards: Dict[Tuple[int, int], deque] = {}
        self.stage_completions: Dict[Tuple[int, int], int] = {}
        self.completed_stages: List[Tuple[int, int]] = []

        self._episode_counter = 0
        self._is_revisiting = False
        self._revisit_stage: Optional[Tuple[int, int]] = None

    @property
    def current_stage(self) -> Tuple[int, int]:
        if self._is_revisiting and self._revisit_stage:
            return self._revisit_stage
        return (self._current_world, self._current_stage)

    @property
    def progress(self) -> float:
        idx = ALL_STAGES.index((self._current_world, self._current_stage))
        return idx / (len(ALL_STAGES) - 1)

    def report_episode(self, reward: float, distance: int = 0, completed: bool = False) -> None:
        stage = (self._current_world, self._current_stage)
        if stage not in self.stage_rewards:
            self.stage_rewards[stage] = deque(maxlen=self.advance_window)
            self.stage_completions[stage] = 0
        self.stage_rewards[stage].append(reward)
        if completed:
            self.stage_completions[stage] += 1
        self._episode_counter += 1
        if self._is_revisiting:
            self._is_revisiting = False
            self._revisit_stage = None

    def should_advance(self) -> bool:
        stage = (self._current_world, self._current_stage)
        rewards = self.stage_rewards.get(stage, deque())
        if len(rewards) < self.advance_window:
            return False
        avg_reward = sum(rewards) / len(rewards)
        completions = self.stage_completions.get(stage, 0)
        return avg_reward >= self.advance_threshold or completions >= self.completion_count

    def advance(self) -> Optional[Tuple[int, int]]:
        current = (self._current_world, self._current_stage)
        if current not in self.completed_stages:
            self.completed_stages.append(current)
        try:
            idx = ALL_STAGES.index(current)
        except ValueError:
            return None
        if idx >= len(ALL_STAGES) - 1:
            return None
        next_stage = ALL_STAGES[idx + 1]
        self._current_world, self._current_stage = next_stage
        self._episode_counter = 0
        return next_stage

    def get_revisit_stage(self) -> Optional[Tuple[int, int]]:
        if not self.completed_stages:
            return None
        if self._episode_counter >= self.revisit_interval:
            return random.choice(self.completed_stages)
        return None

    def start_revisit(self, stage: Tuple[int, int]) -> None:
        self._is_revisiting = True
        self._revisit_stage = stage

    def save_state(self, path: str) -> None:
        state = {
            'current_world': self._current_world,
            'current_stage': self._current_stage,
            'completed_stages': self.completed_stages,
            'stage_completions': {
                f'{w}-{s}': c for (w, s), c in self.stage_completions.items()
            },
            'episode_counter': self._episode_counter,
            'advance_threshold': self.advance_threshold,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> None:
        with open(path, 'r') as f:
            state = json.load(f)
        self._current_world = state['current_world']
        self._current_stage = state['current_stage']
        self.completed_stages = [tuple(s) for s in state['completed_stages']]
        self.stage_completions = {
            tuple(int(x) for x in k.split('-')): v
            for k, v in state['stage_completions'].items()
        }
        self._episode_counter = state.get('episode_counter', 0)
