"""
NEAT (NeuroEvolution of Augmenting Topologies) Trainer.

NEAT evolves neural network topology and weights through a genetic algorithm.
Unlike traditional RL, NEAT doesn't use gradient descent — instead, it:

1. Creates a population of random neural networks (genomes)
2. Tests each genome by letting it play Mario for one episode
3. Assigns fitness based on how well the genome performed
4. Selects the best genomes and creates offspring via mutation/crossover
5. Repeats for many generations, with networks growing more complex

Key NEAT concepts:
    - Genome: A description of a neural network (nodes + connections)
    - Species: Groups of similar genomes that compete within species
    - Mutation: Adding/removing nodes/connections, changing weights
    - Crossover: Combining two parent genomes to create offspring

This implementation:
    - Uses neat-python library for the core NEAT algorithm
    - Input: 13x13 = 169 downsampled grayscale pixels
    - Output: 7 values (one per action), highest wins
    - Fitness: Total shaped reward from one Mario episode

Usage:
    trainer = NEATTrainer(env, config_path='config/neat_config.txt',
                          visualizer=dashboard)
    trainer.train(num_generations=100)
    trainer.save_checkpoint('models/neat/best')
"""

import os
import pickle
import numpy as np
import neat
from typing import Optional, Tuple, List

from src.algorithms.base_trainer import BaseTrainer
from src.algorithms.neat.parallel_eval import ParallelGenomeEvaluator
from src.visualization.dashboard import Dashboard


class NEATTrainer(BaseTrainer):
    """
    NEAT algorithm trainer for Super Mario Bros.

    Evolves a population of neural networks that learn to play Mario
    through natural selection. Each generation, every genome plays one
    episode, and the best performers are used to create the next generation.

    Args:
        env: The Mario environment (should use create_neat_env).
        config_path: Path to the NEAT config file.
        visualizer: Optional Dashboard for live visualization.
        save_dir: Directory for checkpoints. Default 'models'.
        log_dir: Directory for logs. Default 'logs'.

    Attributes:
        neat_config: Parsed NEAT configuration.
        population: The NEAT Population object managing evolution.
        best_genome: The best genome found so far.
        generation: Current generation number.

    Example:
        from src.environment.mario_env import create_neat_env
        from src.visualization.dashboard import Dashboard

        env = create_neat_env(world=1, stage=1)
        dashboard = Dashboard(algorithm='neat')
        trainer = NEATTrainer(env, 'config/neat_config.txt', dashboard)
        trainer.train(num_generations=100)
    """

    def __init__(
        self,
        env,
        config_path: str = 'config/neat_config.txt',
        visualizer: Optional[Dashboard] = None,
        save_dir: str = 'models',
        log_dir: str = 'logs',
        num_envs: int = 1,
        world: int = 1,
        stage: int = 1,
    ):
        # Load NEAT configuration from file
        self.config_path = config_path
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        # Initialize base trainer
        super().__init__(
            env=env,
            config={'config_path': config_path},
            visualizer=visualizer,
            save_dir=save_dir,
            log_dir=log_dir,
        )

        # Create NEAT population
        self.population = neat.Population(self.neat_config)

        # Add NEAT reporters for console output
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)

        # Track the best genome across all generations
        self.best_genome = None
        self.generation = 0

        # Parallel evaluation support
        self.num_envs = num_envs
        self.world = world
        self.stage = stage
        self.parallel_evaluator: Optional[ParallelGenomeEvaluator] = None

        if num_envs > 1:
            self.parallel_evaluator = ParallelGenomeEvaluator(
                num_envs=num_envs,
                world=world,
                stage=stage,
                neat_config_path=config_path,
            )

    def train(self, num_generations: int = 100) -> None:
        """
        Train NEAT for a specified number of generations.

        Each generation:
        1. Every genome in the population plays one Mario episode
        2. Genomes are assigned fitness based on their total reward
        3. NEAT selects, mutates, and crosses over to create next generation
        4. Dashboard is updated with metrics

        Args:
            num_generations: Number of generations to evolve. Default 100.
        """
        self.is_training = True
        mode = 'parallel' if self.num_envs > 1 else 'sequential'
        print(f'\n{"="*60}')
        print(f'Starting NEAT Training - {num_generations} generations')
        print(f'Population size: {self.neat_config.pop_size}')
        print(f'Evaluation mode: {mode} ({self.num_envs} envs)')
        print(f'{"="*60}\n')

        # Choose evaluation function based on parallel mode
        eval_fn = (
            self._eval_genomes_parallel if self.num_envs > 1
            else self._eval_genomes
        )

        # Run NEAT evolution
        # The eval_genomes function is called each generation to evaluate all genomes
        try:
            winner = self.population.run(eval_fn, n=num_generations)
            if winner is not None:
                self.best_genome = winner
        except SystemExit:
            print('\nTraining stopped.')

        # Training complete — save final model
        self.is_training = False
        best_fitness = self.best_genome.fitness if self.best_genome else 0
        print(f'\n{"="*60}')
        print(f'NEAT Training Complete!')
        print(f'Best fitness: {best_fitness:.1f}')
        print(f'{"="*60}\n')

        # Shut down parallel workers if active
        if self.parallel_evaluator is not None:
            self.parallel_evaluator.shutdown()

        # Save the best genome
        self._save_on_exit()

    def _eval_genomes(self, genomes, config) -> None:
        """
        Evaluate all genomes in the current generation.

        This is the fitness function called by NEAT each generation.
        Each genome creates a neural network, plays one Mario episode,
        and is assigned fitness based on performance.

        Args:
            genomes: List of (genome_id, genome) tuples from NEAT.
            config: NEAT config object.
        """
        self.generation += 1
        self.episode_count = self.generation

        # Track generation-level metrics
        gen_rewards = []
        gen_distances = []
        gen_action_counts = [0] * 7  # 7 possible actions
        gen_completions = 0  # How many genomes completed the stage

        # Evaluate each genome
        num_genomes = len(genomes)
        for idx, (genome_id, genome) in enumerate(genomes):
            # Stop if dashboard was closed
            if self._dashboard_closed:
                # Assign zero fitness to remaining un-evaluated genomes
                genome.fitness = 0.0
                continue

            # Create a neural network from the genome's description
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            # Show live gameplay for every 15th genome. This keeps the
            # display active while minimising slowdown — each live genome
            # adds ~50ms of display overhead per episode.
            live = (self.visualizer is not None) and (idx % 15 == 0)
            reward, distance, actions, frame, completed = self._play_episode(net, show_live=live)

            # Log stage completion
            if completed:
                gen_completions += 1

            # Assign fitness (NEAT uses this for selection)
            genome.fitness = reward

            # Collect generation stats
            gen_rewards.append(reward)
            gen_distances.append(distance)
            for i, count in enumerate(actions):
                gen_action_counts[i] += count

            # Update best genome tracking
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_genome = genome
            if distance > self.best_distance:
                self.best_distance = distance

            # Keep the dashboard responsive between genomes.
            # Only pump the display every 5 genomes (instead of every
            # genome) to reduce overhead. The live genomes already update
            # the display during their episode, so this just keeps the
            # progress counter ticking for the non-live ones.
            if self.visualizer and not self._dashboard_closed and idx % 5 == 0:
                progress_metrics = {
                    'genome_progress': f'{idx + 1}/{num_genomes}',
                    'genome_reward': reward,
                }
                self.update_visualization(frame=frame, metrics=progress_metrics)

        # If no genomes were evaluated (dashboard closed early), skip summary
        if not gen_rewards:
            return

        # Calculate generation summary metrics
        avg_reward = np.mean(gen_rewards)
        max_reward = max(gen_rewards)
        avg_distance = np.mean(gen_distances)
        max_distance = max(gen_distances)

        # Calculate network complexity of the best genome this generation
        best_gen_genome = max(genomes, key=lambda x: x[1].fitness)[1]
        complexity = (
            len(best_gen_genome.nodes) + len(best_gen_genome.connections)
        )

        # Print generation summary
        completion_str = f', Completions={gen_completions}' if gen_completions > 0 else ''
        print(f'  Gen {self.generation}: '
              f'Avg={avg_reward:.0f}, Max={max_reward:.0f}, '
              f'Dist={max_distance:.0f}, Complexity={complexity}'
              f'{completion_str}')

        # Update visualization dashboard with generation summary.
        # Include genome_progress='' to clear the per-genome indicator
        # now that the full generation is complete.
        if self.visualizer and not self._dashboard_closed:
            metrics = {
                'generation': self.generation,
                'reward': max_reward,
                'distance': max_distance,
                'complexity': complexity,
                'action_distribution': gen_action_counts,
                'genome_progress': '',
            }
            self.update_visualization(frame=frame, metrics=metrics)

        # Auto-checkpoint
        self._auto_checkpoint(self.generation)

        # If dashboard was closed during this generation, stop evolution.
        # All genomes already have fitness assigned (real or 0.0), so it's
        # safe to break out of population.run() now.
        if self._dashboard_closed:
            raise SystemExit('Dashboard closed')

    def _eval_genomes_parallel(self, genomes, config) -> None:
        """
        Evaluate all genomes using multiple environments (same process).

        Uses MultiEnvGenomeEvaluator to step N environments in interleaved
        fashion, enabling the multi-Mario grid display on the dashboard.

        A callback is passed into the evaluator so that pygame events are
        pumped and the grid display is refreshed every ~15 interleaved
        steps.  Without this, Windows flags the app as "Not Responding"
        because no events are processed during the (potentially long)
        genome evaluation, and a phantom QUIT event kills training.

        Args:
            genomes: List of (genome_id, genome) tuples from NEAT.
            config: NEAT config object.
        """
        self.generation += 1
        self.episode_count = self.generation

        if self._dashboard_closed:
            for _, genome in genomes:
                genome.fitness = 0.0
            raise SystemExit('Dashboard closed')

        num_genomes = len(genomes)

        # Build a display callback that refreshes the grid and pumps
        # pygame events.  Returning False aborts evaluation.
        def _on_step(frames):
            """Called every ~15 rounds inside _evaluate_chunk."""
            if self._dashboard_closed:
                return False
            if self.visualizer is None:
                return True
            return self.update_visualization_grid(
                frames=frames, metrics=None,
            )

        # Evaluate all genomes via the batch evaluator.  The on_step
        # callback keeps the display alive during the long evaluation.
        all_results, latest_frames = self.parallel_evaluator.evaluate_batch(
            genomes, config, on_step=_on_step,
        )

        # Track generation-level metrics
        gen_rewards = []
        gen_distances = []
        gen_action_counts = [0] * 7
        gen_completions = 0

        # Map results back to genomes
        result_map = {r.genome_id: r for r in all_results}
        best_frame = None

        for genome_id, genome in genomes:
            result = result_map.get(genome_id)
            if result is None:
                genome.fitness = 0.0
                continue

            genome.fitness = result.fitness
            gen_rewards.append(result.fitness)
            gen_distances.append(result.distance)
            for i, count in enumerate(result.action_counts):
                gen_action_counts[i] += count

            if result.stage_completed:
                gen_completions += 1

            if result.fitness > self.best_reward:
                self.best_reward = result.fitness
                self.best_genome = genome
                best_frame = result.last_frame
            if result.distance > self.best_distance:
                self.best_distance = result.distance

            # Notify episode callbacks (curriculum learning, etc.)
            self._fire_episode_complete(
                reward=result.fitness,
                distance=result.distance,
                completed=result.stage_completed,
            )

        if not gen_rewards:
            return

        # Calculate generation summary
        avg_reward = np.mean(gen_rewards)
        max_reward = max(gen_rewards)
        avg_distance = np.mean(gen_distances)
        max_distance = max(gen_distances)

        # Network complexity of best genome
        best_gen_genome = max(genomes, key=lambda x: x[1].fitness)[1]
        complexity = (
            len(best_gen_genome.nodes) + len(best_gen_genome.connections)
        )

        completion_str = f', Completions={gen_completions}' if gen_completions > 0 else ''
        print(f'  Gen {self.generation}: '
              f'Avg={avg_reward:.0f}, Max={max_reward:.0f}, '
              f'Dist={max_distance:.0f}, Complexity={complexity}'
              f'{completion_str}')

        # Update visualization with final generation metrics
        if self.visualizer and not self._dashboard_closed:
            metrics = {
                'generation': self.generation,
                'reward': max_reward,
                'distance': max_distance,
                'complexity': complexity,
                'action_distribution': gen_action_counts,
                'genome_progress': '',
            }

            # Build frame list for grid display
            frames = []
            for i in range(self.num_envs):
                if i in latest_frames:
                    frames.append(latest_frames[i])
                elif best_frame is not None:
                    frames.append(best_frame)
                else:
                    frames.append(None)

            if not self.update_visualization_grid(
                frames=frames, metrics=metrics,
            ):
                for _, genome in genomes:
                    if genome.fitness is None:
                        genome.fitness = 0.0
                raise SystemExit('Dashboard closed')

        # Auto-checkpoint
        self._auto_checkpoint(self.generation)

        if self._dashboard_closed:
            raise SystemExit('Dashboard closed')

    def _play_episode(
        self, net: neat.nn.FeedForwardNetwork, show_live: bool = False,
    ) -> Tuple[float, int, List[int], Optional[np.ndarray], bool]:
        """
        Play one Mario episode using a NEAT neural network.

        The network receives a 13x13 downsampled grayscale image as input
        (flattened to 169 values) and outputs 7 values. The action with
        the highest output value is selected.

        Args:
            net: NEAT feed-forward neural network.
            show_live: If True, update the dashboard every few frames for
                       live gameplay display. Default False for speed.

        Returns:
            tuple: (total_reward, max_distance, action_counts, last_frame, stage_completed)
                - total_reward: Sum of rewards across the episode
                - max_distance: Farthest x-position Mario reached
                - action_counts: How many times each action was used
                - last_frame: Last game frame for visualization
                - stage_completed: Whether Mario reached the flagpole
        """
        # Reset environment for a new episode
        obs = self.env.reset()
        total_reward = 0.0
        max_distance = 0
        done = False
        action_counts = [0] * 7
        last_frame = None
        steps = 0
        max_steps = 2000  # Cap episode length for faster generations
        stagnation_limit = 100  # End early if Mario stops moving (~6.7s of gameplay)
        stage_completed = False

        # Track stagnation (Mario stuck in same x-position)
        last_x = 0
        stagnation_counter = 0

        # Pre-compute: check dashboard once outside the loop
        has_live_display = show_live and self.visualizer is not None

        # Cache the activate function to avoid attribute lookups in the hot loop
        activate = net.activate

        while not done and steps < max_steps:
            # Check for pause
            if not self.check_pause():
                return total_reward, max_distance, action_counts, last_frame, False

            # Preprocess observation for NEAT.
            # obs shape is (13, 13, 1) — flatten to 169 values and
            # normalise to [0, 1]. Using multiply instead of divide
            # is marginally faster for float32.
            flat_obs = obs.flatten().astype(np.float32) * (1.0 / 255.0)

            # Get network outputs and select best action
            outputs = activate(flat_obs)
            action = int(np.argmax(outputs))
            action_counts[action] += 1

            # Step the environment
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1

            # Track distance (x position in the level)
            x_pos = info.get('x_pos', 0)
            if x_pos > max_distance:
                max_distance = x_pos

            # Check for stage completion (reached the flagpole)
            if info.get('stage_completed', False):
                stage_completed = True

            # Early termination if Mario is stuck (no forward progress).
            # 100 steps × 4 frame-skip = 400 raw frames (~6.7 seconds).
            if x_pos <= last_x:
                stagnation_counter += 1
                if stagnation_counter >= stagnation_limit:
                    break  # Genome is stuck, move on
            else:
                stagnation_counter = 0
                last_x = x_pos

            # Capture raw NES frame for display.
            # Only grab frames when showing live to avoid overhead.
            # Every 6 steps keeps display smooth without heavy cost.
            if has_live_display and steps % 6 == 0:
                try:
                    last_frame = self.env.unwrapped.screen
                except AttributeError:
                    last_frame = obs
                if not self.update_visualization(frame=last_frame, metrics=None):
                    break  # Dashboard closed

            # Check dashboard closed every 50 steps (not every step)
            elif steps % 50 == 0 and self._dashboard_closed:
                break

        # Always grab a final frame for the dashboard (even non-live genomes)
        if last_frame is None:
            try:
                last_frame = self.env.unwrapped.screen
            except AttributeError:
                last_frame = obs

        return total_reward, max_distance, action_counts, last_frame, stage_completed

    def evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate the best genome by playing multiple episodes.

        Renders every frame to the dashboard for watching the AI play.

        Args:
            num_episodes: Number of episodes to evaluate.

        Returns:
            float: Average reward across evaluation episodes.
        """
        if self.best_genome is None:
            print('No best genome to evaluate!')
            return 0.0

        # Create the neural network from the best genome
        net = neat.nn.FeedForwardNetwork.create(
            self.best_genome, self.neat_config,
        )

        total_rewards = []
        for ep in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < 5000:
                # Get action from network
                flat_obs = obs.flatten().astype(np.float32) / 255.0
                outputs = net.activate(flat_obs)
                action = int(np.argmax(outputs))

                # Step environment
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                steps += 1

                # Render every frame during evaluation
                if self.visualizer:
                    try:
                        raw_frame = self.env.unwrapped.screen
                    except AttributeError:
                        raw_frame = obs
                    self.update_visualization(
                        frame=raw_frame,
                        metrics={
                            'episode': ep + 1,
                            'reward': episode_reward,
                            'distance': info.get('x_pos', 0),
                        },
                    )

            total_rewards.append(episode_reward)
            print(f'  Eval Episode {ep+1}: Reward={episode_reward:.0f}, '
                  f'Distance={info.get("x_pos", 0)}')

        avg_reward = np.mean(total_rewards)
        print(f'\n  Average Eval Reward: {avg_reward:.0f}')
        return avg_reward

    def save_checkpoint(self, path: str) -> None:
        """
        Save NEAT checkpoint (best genome + population state).

        Saves two files:
        - {path}_best_genome.pkl: Best genome found (for evaluation)
        - {path}_population.pkl: Full population (for resuming training)

        Args:
            path: Base path for checkpoint files (without extension).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Save best genome
        if self.best_genome:
            genome_path = f'{path}_best_genome.pkl'
            with open(genome_path, 'wb') as f:
                pickle.dump(self.best_genome, f)
            print(f'  Saved best genome to: {genome_path}')

        # Save full population for resuming evolution
        population_path = f'{path}_population.pkl'
        with open(population_path, 'wb') as f:
            pickle.dump(self.population, f)
        print(f'  Saved population to: {population_path}')

    def load_checkpoint(self, path: str) -> None:
        """
        Load NEAT checkpoint.

        Can load either a best genome (for evaluation) or a full
        population (for resuming training).

        Args:
            path: Path to the checkpoint file (.pkl).

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint not found: {path}')

        with open(path, 'rb') as f:
            loaded = pickle.load(f)

        if isinstance(loaded, neat.Population):
            # Full population — resume training
            self.population = loaded
            print(f'Loaded population from: {path}')
        elif hasattr(loaded, 'fitness'):
            # Single genome — for evaluation
            self.best_genome = loaded
            self.best_reward = loaded.fitness or 0
            print(f'Loaded genome with fitness: {loaded.fitness}')
        else:
            print(f'Warning: Unknown checkpoint format in {path}')
