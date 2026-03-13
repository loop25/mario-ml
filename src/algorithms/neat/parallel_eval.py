"""
Multi-Environment Genome Evaluator for NEAT.

Evaluates NEAT genomes across multiple NES environments running in the
SAME process. We use same-process evaluation instead of multiprocessing
because nes_py's C-based NES emulator crashes with access violations
when used in subprocesses (the C library pointers become invalid).

Architecture:
    N environments created in the main process
    Genomes are assigned to envs in round-robin fashion
    Each env plays one step at a time, cycling through all envs
    This gives the appearance of parallel play on the grid display

The evaluator accepts an optional ``on_step`` callback that is invoked
periodically (every ~15 interleaved rounds) with the latest frames from
each environment.  The NEAT trainer uses this to pump pygame events and
refresh the grid display so Windows doesn't flag the app as "Not
Responding".

Usage:
    evaluator = MultiEnvGenomeEvaluator(
        num_envs=4, world=1, stage=1,
        neat_config_path='config/neat_config.txt',
    )
    results, frames = evaluator.evaluate_batch(genomes, config)
    evaluator.shutdown()
"""

import numpy as np
import neat
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass

from src.algorithms.frame_utils import capture_display_frame as _capture_frame


@dataclass
class GenomeResult:
    """Result from evaluating a single genome."""
    genome_id: int
    fitness: float
    distance: int
    action_counts: List[int]
    last_frame: Optional[np.ndarray]
    stage_completed: bool


class MultiEnvGenomeEvaluator:
    """
    Evaluates NEAT genomes using multiple environments in the same process.

    Creates N NES environments at startup and reuses them for all genomes.
    Genomes are evaluated in batches of N: one genome per environment,
    all stepped in round-robin until all episodes finish. This enables
    the multi-Mario grid display on the dashboard.

    Args:
        num_envs: Number of environments to create.
        world: Mario world number (1-8).
        stage: Mario stage number (1-4).
        neat_config_path: Path to NEAT config file.

    Example:
        evaluator = MultiEnvGenomeEvaluator(
            num_envs=4, world=1, stage=1,
            neat_config_path='config/neat_config.txt',
        )

        # Each generation:
        results, frames = evaluator.evaluate_batch(genomes, config)

        # When done:
        evaluator.shutdown()
    """

    def __init__(
        self,
        num_envs: int,
        world: int,
        stage: int,
        neat_config_path: str,
    ):
        from src.environment.mario_env import create_neat_env

        self.num_envs = num_envs
        self.world = world
        self.stage = stage
        self.neat_config_path = neat_config_path

        # Load NEAT config for network creation
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_config_path,
        )

        # Create N environments in the same process.
        # Pump pygame events between env creations so Windows doesn't
        # flag the window as "Not Responding" during setup.
        try:
            import pygame
            _pump = pygame.event.pump
        except (ImportError, Exception):
            _pump = lambda: None

        self.envs = []
        for i in range(num_envs):
            env = create_neat_env(world=world, stage=stage)
            self.envs.append(env)
            _pump()  # Keep window responsive

        print(f'  Created {num_envs} environments for multi-env NEAT evaluation.')

    def evaluate_batch(
        self,
        genomes: list,
        config,
        on_step: Optional[Callable[[List[Optional[np.ndarray]]], bool]] = None,
    ) -> Tuple[List[GenomeResult], Dict[int, np.ndarray]]:
        """
        Evaluate a batch of genomes using interleaved multi-env stepping.

        Processes genomes in chunks of N (one per environment). Within
        each chunk, all N environments are stepped in round-robin until
        all episodes finish. This produces the grid display effect.

        Args:
            genomes: List of (genome_id, genome) tuples from NEAT.
            config: NEAT config object.
            on_step: Optional callback invoked periodically during
                     evaluation with a list of latest frames (one per
                     env).  Return False from the callback to abort.

        Returns:
            Tuple of:
            - List of GenomeResult objects (one per genome)
            - Dict mapping env_index → latest frame (for grid display)
        """
        results: List[GenomeResult] = []
        latest_frames: Dict[int, np.ndarray] = {}
        n = self.num_envs

        # Process genomes in chunks of N
        for chunk_start in range(0, len(genomes), n):
            chunk = genomes[chunk_start:chunk_start + n]
            chunk_results, chunk_frames, aborted = self._evaluate_chunk(
                chunk, on_step=on_step,
            )
            results.extend(chunk_results)
            latest_frames.update(chunk_frames)
            if aborted:
                break

        return results, latest_frames

    def _evaluate_chunk(
        self,
        genome_chunk: list,
        on_step: Optional[Callable[[List[Optional[np.ndarray]]], bool]] = None,
    ) -> Tuple[List[GenomeResult], Dict[int, np.ndarray], bool]:
        """
        Evaluate a chunk of genomes (up to N) using interleaved stepping.

        Each genome gets its own environment. All environments are stepped
        in round-robin until all episodes complete. The ``on_step``
        callback is invoked every 15 interleaved rounds to keep the
        display alive and pump pygame events.

        Args:
            genome_chunk: List of (genome_id, genome) tuples, len <= num_envs.
            on_step: Optional display callback.  Receives list of latest
                     frames.  Return False to abort evaluation.

        Returns:
            Tuple of (results, frames_dict, aborted).
        """
        chunk_size = len(genome_chunk)
        max_steps = 2000
        stagnation_limit = 100

        # Create neural networks for each genome
        nets = []
        for genome_id, genome in genome_chunk:
            net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
            nets.append(net)

        # Per-env state
        observations = []
        total_rewards = [0.0] * chunk_size
        max_distances = [0] * chunk_size
        action_counts = [[0] * 7 for _ in range(chunk_size)]
        last_frames: List[Optional[np.ndarray]] = [None] * chunk_size
        stage_completed = [False] * chunk_size
        done_flags = [False] * chunk_size
        steps = [0] * chunk_size
        last_x = [0] * chunk_size
        stagnation_counters = [0] * chunk_size

        # Reset environments and get initial observations
        for i in range(chunk_size):
            obs = self.envs[i].reset()
            observations.append(obs)

        aborted = False
        round_count = 0  # Counts full interleaved rounds

        # Interleaved stepping: step all envs until all episodes finish
        while not all(done_flags):
            round_count += 1

            for i in range(chunk_size):
                if done_flags[i]:
                    continue

                obs = observations[i]
                net = nets[i]
                env = self.envs[i]

                # Preprocess and get action
                flat_obs = obs.flatten().astype(np.float32) * (1.0 / 255.0)
                outputs = net.activate(flat_obs)
                action = int(np.argmax(outputs))
                action_counts[i][action] += 1

                # Step environment
                next_obs, reward, done, info = env.step(action)
                total_rewards[i] += reward
                steps[i] += 1

                # Track distance
                x_pos = info.get('x_pos', 0)
                if x_pos > max_distances[i]:
                    max_distances[i] = x_pos

                # Stage completion
                if info.get('stage_completed', False):
                    stage_completed[i] = True

                # Stagnation check
                if x_pos <= last_x[i]:
                    stagnation_counters[i] += 1
                    if stagnation_counters[i] >= stagnation_limit:
                        done = True
                else:
                    stagnation_counters[i] = 0
                    last_x[i] = x_pos

                # Max steps check
                if steps[i] >= max_steps:
                    done = True

                observations[i] = next_obs

                if done:
                    done_flags[i] = True
                    # Capture final frame (colorful RGB for dashboard)
                    last_frames[i] = _capture_frame(env, next_obs)

            # --- Periodic display update ---
            # Every 15 rounds (~15 steps per env), capture frames and
            # invoke the on_step callback.  This keeps the pygame event
            # pump alive so Windows doesn't flag the app as frozen.
            if round_count % 15 == 0:
                # Capture current frames from all active envs
                for i in range(chunk_size):
                    if not done_flags[i]:
                        last_frames[i] = _capture_frame(
                            self.envs[i], observations[i])


                if on_step is not None:
                    if not on_step(last_frames):
                        aborted = True
                        # Assign zero fitness to unfinished genomes
                        for i in range(chunk_size):
                            if not done_flags[i]:
                                done_flags[i] = True
                        break

        # Capture final frames for any env that finished without a frame
        for i in range(chunk_size):
            if last_frames[i] is None:
                last_frames[i] = _capture_frame(self.envs[i])

        # Build results
        results = []
        frames = {}
        for i in range(chunk_size):
            genome_id = genome_chunk[i][0]
            results.append(GenomeResult(
                genome_id=genome_id,
                fitness=total_rewards[i],
                distance=max_distances[i],
                action_counts=action_counts[i],
                last_frame=last_frames[i],
                stage_completed=stage_completed[i],
            ))
            if last_frames[i] is not None:
                frames[i] = last_frames[i]

        return results, frames, aborted

    def get_latest_frames(self) -> Dict[int, np.ndarray]:
        """
        Get the latest frame from each environment.

        Returns:
            Dict mapping env_index → latest frame.
        """
        frames = {}
        for i, env in enumerate(self.envs):
            frame = _capture_frame(env)
            if frame is not None:
                frames[i] = frame
        return frames

    def shutdown(self) -> None:
        """Close all environments."""
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
        self.envs.clear()
        print('  Multi-env NEAT evaluator shut down.')


# Keep backward-compatible alias
ParallelGenomeEvaluator = MultiEnvGenomeEvaluator
