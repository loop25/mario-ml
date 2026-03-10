"""
ML Training Platform - Main Entry Point.

Run this script to train or evaluate ML algorithms
(NEAT, PPO, DQN, A2C, Rainbow) on any supported game with live visualization.

Usage:
    # Train NEAT with live dashboard:
    python main.py --algorithm neat --visualize

    # Train PPO for 1 million timesteps:
    python main.py --algorithm ppo --visualize

    # Train DQN:
    python main.py --algorithm dqn --visualize

    # Evaluate a saved NEAT model:
    python main.py --algorithm neat --eval --load models/neat/final_best_genome.pkl --visualize

    # Evaluate a saved PPO model:
    python main.py --algorithm ppo --eval --load models/ppo/final.zip --visualize

    # Evaluate a saved DQN model:
    python main.py --algorithm dqn --eval --load models/dqn/final.pt --visualize

    # Train with video recording:
    python main.py --algorithm neat --visualize --record

    # Train without visualization (headless, faster):
    python main.py --algorithm dqn

    # Custom world and stage:
    python main.py --algorithm neat --visualize --world 1 --stage 2

    # Custom number of episodes/generations:
    python main.py --algorithm neat --visualize --episodes 200

Environment:
    Python 3.11 required (3.13 not compatible with game libraries).
    Activate the virtual environment before running:
        C:\\Projects\\mario-ml\\venv\\Scripts\\activate
"""

import argparse
import os
import sys
import json
import yaml

# Add project root to Python path so imports work from any directory
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from games.registry import GameRegistry


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Super Mario Bros ML Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --algorithm neat --visualize
  python main.py --algorithm ppo --visualize --episodes 500
  python main.py --algorithm dqn --eval --load models/dqn/final.pt --visualize
        """,
    )

    # Which algorithm to use
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        required=False,
        default='dqn',
        choices=['neat', 'ppo', 'dqn', 'a2c', 'rainbow'],
        help='ML algorithm to use: neat, ppo, dqn, or a2c',
    )

    # Game selection
    parser.add_argument(
        '--game', '-g',
        type=str,
        default='mario',
        help='Game to play. Use --list-games to see available games. Default: mario',
    )

    parser.add_argument(
        '--list-games',
        action='store_true',
        help='List all available games and exit.',
    )

    # Training mode
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Evaluation mode (no training, just watch the AI play)',
    )

    # Visualization
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Enable live visualization dashboard',
    )

    # Recording
    parser.add_argument(
        '--record',
        action='store_true',
        help='Record training session as MP4 video',
    )

    # Model loading
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='Path to a saved model checkpoint to load',
    )

    # Environment settings
    parser.add_argument(
        '--world', '-w',
        type=int,
        default=1,
        help='World number 1-8 (default: 1)',
    )
    parser.add_argument(
        '--stage', '-s',
        type=int,
        default=1,
        help='Stage number 1-4 (default: 1)',
    )

    # Training duration
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of episodes/generations to train (overrides config)',
    )

    # Stage progression
    parser.add_argument(
        '--next-stage',
        action='store_true',
        help='After training completes, automatically advance to the next stage '
             'and continue training with transferred weights.',
    )

    # Parallel environments
    parser.add_argument(
        '--num-envs',
        type=int,
        default=1,
        help='Number of parallel environments (default: 1). '
             'More envs = faster NEAT training + multi-Mario grid display. '
             'Try 4 for a 2x2 grid, 8 for a 3x3 grid.',
    )

    # Music
    parser.add_argument(
        '--music',
        type=str,
        default=None,
        help='Path to music directory for background playback (default: assets/music/)',
    )

    # Streaming
    parser.add_argument(
        '--stream-twitch',
        type=str,
        default=None,
        help='Twitch stream key for live streaming',
    )
    parser.add_argument(
        '--stream-youtube',
        type=str,
        default=None,
        help='YouTube stream key for live streaming',
    )

    # Curriculum learning
    parser.add_argument(
        '--curriculum',
        action='store_true',
        help='Enable curriculum learning for whole-game training (all 32 stages)',
    )

    # Game-specific options (key=value pairs from the launcher)
    parser.add_argument(
        '--game-opts',
        nargs='*',
        default=[],
        help='Game-specific options as key=value pairs '
             '(e.g., --game-opts grid_size=16 speed=10)',
    )

    # Device selection
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Compute device: auto (detect best), cuda, mps, or cpu. '
             'Default: auto (CUDA > MPS > CPU)',
    )

    return parser.parse_args()


def parse_game_opts(raw_opts: list) -> dict:
    """Parse --game-opts key=value pairs into a dict with type inference.

    Tries int first, then float, then bool ('true'/'false'), else str.
    """
    result = {}
    for item in raw_opts:
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        key = key.strip()
        value = value.strip()
        # Type inference
        if value.lower() in ('true', 'false'):
            result[key] = value.lower() == 'true'
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result


def next_world_stage(world: int, stage: int):
    """
    Advance to the next stage in Super Mario Bros.

    Stage order: 1-1 → 1-2 → 1-3 → 1-4 → 2-1 → ... → 8-4.

    Args:
        world: Current world (1-8).
        stage: Current stage (1-4).

    Returns:
        tuple: (next_world, next_stage), or None if at 8-4.
    """
    if stage < 4:
        return world, stage + 1
    elif world < 8:
        return world + 1, 1
    else:
        return None  # All stages complete!


def find_resume_checkpoint(algo_name, game_id, save_dir='models'):
    """Find the latest final checkpoint for auto-resume.

    Checks models/{algo_name}/ for a metadata.json (written by
    BaseTrainer._save_metadata) and a corresponding final checkpoint
    file.  Only returns a match if the checkpoint was produced by
    the same game (prevents loading e.g. a Mario model into Snake).

    Old checkpoints without a game_id field are assumed to be 'mario'
    for backwards compatibility.

    Returns (checkpoint_path, metadata_dict) or (None, None).
    """
    algo_dir = os.path.join(save_dir, algo_name)
    metadata_path = os.path.join(algo_dir, 'metadata.json')
    if not os.path.isfile(metadata_path):
        return None, None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None

    # Verify the checkpoint was trained on the same game.
    checkpoint_game = metadata.get('game_id', 'mario')
    if checkpoint_game != game_id:
        return None, None

    # Each algorithm saves its final model with a different extension.
    candidates = [
        os.path.join(algo_dir, 'final.zip'),              # PPO, A2C (SB3)
        os.path.join(algo_dir, 'final.pt'),                # DQN
        os.path.join(algo_dir, 'final_best_genome.pkl'),   # NEAT
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path, metadata

    return None, None


def main():
    """Main entry point for training and evaluation."""
    args = parse_args()

    # Initialize game registry
    registry = GameRegistry()
    registry.discover()

    if args.list_games:
        print('\nAvailable games:')
        for adapter in registry.list_games():
            print(f'  {adapter.game_id:12s}  [{adapter.category:10s}]  {adapter.name}')
        sys.exit(0)

    # Validate game selection
    try:
        game_adapter = registry.get_game(args.game)
    except KeyError as e:
        print(f'\nError: {e}')
        print('Use --list-games to see available games.')
        sys.exit(1)

    print(f'\nGame: {game_adapter.name} ({game_adapter.game_id})')

    # Validate algorithm compatibility
    supported = game_adapter.supported_algorithms()
    if args.algorithm not in supported:
        print(f'\nError: {game_adapter.name} does not support {args.algorithm.upper()}.')
        print(f'  Supported algorithms: {", ".join(a.upper() for a in supported)}')
        sys.exit(1)

    num_envs = max(1, args.num_envs)

    # Recording requires visualization (captures the dashboard surface)
    if args.record and not args.visualize:
        args.visualize = True
        print('Note: --record requires --visualize. Enabling visualization.')

    # Streaming requires visualization (captures the dashboard surface)
    if (args.stream_twitch or args.stream_youtube) and not args.visualize:
        args.visualize = True
        print('Note: Streaming requires --visualize. Enabling visualization.')

    # Warn about mutually exclusive stage progression options
    if args.curriculum and args.next_stage:
        print('WARNING: --next-stage is ignored when --curriculum is enabled.')
        args.next_stage = False

    print(f'\n{"="*60}')
    print(f'  {game_adapter.name} — ML Training')
    print(f'  Algorithm: {args.algorithm.upper()}')
    print(f'  Mode: {"Evaluation" if args.eval else "Training"}')
    if args.game == 'mario':
        print(f'  World: {args.world}-{args.stage}')
    print(f'  Visualization: {"ON" if args.visualize else "OFF"}')
    if num_envs > 1:
        print(f'  Parallel Envs: {num_envs}')
    if args.next_stage:
        print(f'  Stage Progression: ON (auto-advance after training)')
    if args.curriculum:
        print(f'  Curriculum Learning: ON (all 32 stages)')
    print(f'{"="*60}\n')

    # Parse game-specific options from --game-opts key=value pairs
    game_kwargs = parse_game_opts(args.game_opts)
    if game_kwargs:
        print(f'  Game options: {game_kwargs}')

    # ================================================================
    # Create Environment
    # ================================================================
    # Mario uses its own factory functions for backward compatibility
    # (CustomRewardWrapper, frame-stacking, etc.).
    # All other games go through the universal adapter path.
    if args.game == 'mario' and args.algorithm == 'neat':
        from src.environment.mario_env import create_neat_env
        env = create_neat_env(world=args.world, stage=args.stage)
        print(f'Environment: NEAT mode (13x13 grayscale)')
    elif args.game == 'mario':
        from src.environment.mario_env import create_cnn_env
        env = create_cnn_env(world=args.world, stage=args.stage)
        print(f'Environment: CNN mode (84x84x4 stacked frames)')
    else:
        from src.environment.universal_env import create_env_from_adapter
        env = create_env_from_adapter(game_adapter, **game_kwargs)
        print(f'Environment: {game_adapter.name} {env.observation_space.shape}')

    print(f'Observation space: {env.observation_space.shape}')
    print(f'Action space: {env.action_space.n} actions\n')

    # ================================================================
    # Create Recorder (if requested)
    # ================================================================
    from src.streaming.recording import Recorder
    recorder = None
    if args.record:
        os.makedirs('recordings', exist_ok=True)
        recorder = Recorder(output_dir='recordings')
        recorder.start()  # Initialize VideoWriter and begin recording
        print(f'Recording enabled. Output: recordings/')

    # ================================================================
    # Music Manager
    # ================================================================
    music_manager = None
    if not args.eval:  # No music in eval mode by default
        music_dir = args.music or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'music')
        if os.path.isdir(music_dir):
            from src.audio.music_manager import MusicManager
            music_manager = MusicManager(music_dir)
            if music_manager.track_count > 0:
                print(f'Loaded {music_manager.track_count} music track(s) from {music_dir}')
            else:
                music_manager = None

    # ================================================================
    # Streaming
    # ================================================================
    stream_manager = None
    overlay_manager = None
    twitch_key = args.stream_twitch
    youtube_key = args.stream_youtube

    if twitch_key or youtube_key:
        from src.streaming.stream_manager import StreamManager
        from src.streaming.overlay_manager import OverlayManager

        # Get first music file for audio stream (if available)
        audio_file = None
        if music_manager and music_manager.playlist:
            audio_file = music_manager.playlist[0]

        stream_manager = StreamManager(
            twitch_key=twitch_key,
            youtube_key=youtube_key,
            audio_file=audio_file,
        )
        overlay_manager = OverlayManager(resolution=(1280, 720))

        if stream_manager.start():
            print('[Stream] Streaming started successfully')
        else:
            print(f'[Stream] Failed to start: {stream_manager.error_message}')
            stream_manager = None
            overlay_manager = None

    # ================================================================
    # Curriculum Learning
    # ================================================================
    curriculum = None
    if args.curriculum:
        try:
            from src.training.curriculum import CurriculumManager
            curriculum = CurriculumManager(
                start_world=args.world,
                start_stage=args.stage,
            )
            print(f'Curriculum learning enabled: training across all 32 stages')
            print(f'Starting at World {args.world}-{args.stage}')
        except ImportError as e:
            print(f'[Curriculum] ERROR: Could not load CurriculumManager: {e}')
            print(f'[Curriculum] Falling back to standard training mode.')

    # ================================================================
    # Create Visualization Dashboard
    # ================================================================
    from src.visualization.dashboard import Dashboard
    dashboard = None
    if args.visualize:
        # Extract game-specific dashboard metadata from the adapter
        action_info = game_adapter.get_action_space_info()
        dash_config = game_adapter.get_dashboard_config()
        completion_criteria = game_adapter.get_completion_criteria()

        dashboard = Dashboard(
            algorithm=args.algorithm,
            num_envs=num_envs,
            recorder=recorder,
            music_manager=music_manager,
            stream_manager=stream_manager,
            overlay_manager=overlay_manager,
            game_name=game_adapter.name,
            action_labels=action_info.action_labels,
            dashboard_config=dash_config,
            completion_criteria=completion_criteria,
        )
        print('Dashboard window opened.')

    # ================================================================
    # Create Trainer
    # ================================================================
    if args.algorithm == 'neat':
        from src.algorithms.neat.neat_trainer import NEATTrainer
        from src.algorithms.neat.parallel_eval import ParallelGenomeEvaluator
        config_path = os.path.join(project_root, 'config', 'neat_config.txt')
        trainer = NEATTrainer(
            env=env,
            config_path=config_path,
            visualizer=dashboard,
            num_envs=num_envs,
            world=args.world,
            stage=args.stage,
        )

    elif args.algorithm == 'ppo':
        from src.algorithms.ppo.ppo_trainer import PPOTrainer
        config_path = os.path.join(project_root, 'config', 'ppo_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        trainer = PPOTrainer(
            env=env,
            config=config,
            visualizer=dashboard,
            num_envs=num_envs,
            world=args.world,
            stage=args.stage,
            device_preference=args.device,
        )

    elif args.algorithm == 'dqn':
        from src.algorithms.dqn.dqn_trainer import DQNTrainer
        config_path = os.path.join(project_root, 'config', 'dqn_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        trainer = DQNTrainer(
            env=env,
            config=config,
            visualizer=dashboard,
            num_envs=num_envs,
            world=args.world,
            stage=args.stage,
            device_preference=args.device,
        )

    elif args.algorithm == 'a2c':
        from src.algorithms.a2c.a2c_trainer import A2CTrainer
        config_path = os.path.join(project_root, 'config', 'a2c_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        trainer = A2CTrainer(
            env=env,
            config=config,
            visualizer=dashboard,
            num_envs=num_envs,
            device_preference=args.device,
        )

    elif args.algorithm == 'rainbow':
        from src.algorithms.rainbow.rainbow_trainer import RainbowTrainer
        config_path = os.path.join(project_root, 'config', 'rainbow_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        trainer = RainbowTrainer(
            env=env,
            config=config,
            visualizer=dashboard,
            num_envs=num_envs,
            world=args.world,
            stage=args.stage,
            device_preference=args.device,
        )

    # Tag the trainer with the game so metadata.json records it.
    trainer.game_id = args.game

    # Pass dashboard config to trainer so callbacks know which
    # info-dict keys to extract and how many actions to track.
    trainer.dashboard_config = game_adapter.get_dashboard_config()
    action_info = game_adapter.get_action_space_info()
    trainer.num_actions = action_info.num_actions

    # ================================================================
    # Load Checkpoint (explicit or auto-resume)
    # ================================================================
    if args.load:
        print(f'Loading checkpoint: {args.load}')
        trainer.load_checkpoint(args.load)
        print('Checkpoint loaded successfully.\n')
    elif not args.eval:
        # Auto-resume: check for an existing training session
        # Only resume if the checkpoint was trained on the same game.
        resume_path, resume_meta = find_resume_checkpoint(args.algorithm, args.game)
        if resume_path:
            ep = resume_meta.get('episode', '?')
            best = resume_meta.get('best_reward', '?')
            elapsed = resume_meta.get('elapsed_time', '?')
            print(f'  Found existing training session:')
            print(f'    Episodes: {ep}  |  Best reward: {best}  |  Time: {elapsed}')
            print(f'    Resuming from: {resume_path}')
            trainer.load_checkpoint(resume_path)
            trainer.episode_count = int(resume_meta.get('episode', 0))
            trainer.best_reward = float(resume_meta.get('best_reward', float('-inf')))
            trainer.best_distance = int(resume_meta.get('best_distance', 0))
            print('  Checkpoint loaded. Continuing training.\n')

    # ================================================================
    # Run Training or Evaluation
    # ================================================================
    current_world = args.world
    current_stage = args.stage

    # Pump any queued pygame events before training starts.
    # During environment and model creation (which can take several
    # seconds for multi-env setups), pygame events are not processed.
    # On Windows the OS may flag the window as "Not Responding" and
    # queue a synthetic QUIT event.  Flushing here prevents that stale
    # event from killing training on the very first handle_events() call.
    if dashboard:
        import pygame as _pg
        _pg.event.pump()      # Process internal pygame events
        _pg.event.clear()     # Discard any queued events (incl. stale QUIT)
        dashboard.update()    # Render initial dashboard frame

    if music_manager:
        music_manager.play()

    try:
        if args.eval:
            # Evaluation mode
            print('Starting evaluation...\n')
            avg_reward = trainer.evaluate(num_episodes=10)
            print(f'\nFinal average reward: {avg_reward:.0f}')
        else:
            # Register curriculum callback so report_episode is called
            # per-episode (inside the trainer loop), not once per train() call.
            if curriculum:
                trainer.add_episode_callback(
                    lambda reward, distance=0, completed=False:
                        curriculum.report_episode(reward, distance, completed)
                )

            # Training mode — train on current stage (and optionally advance)
            while True:
                if args.algorithm == 'neat':
                    num_gens = args.episodes or 100
                    print(f'Training NEAT on World {current_world}-{current_stage} '
                          f'for {num_gens} generations...\n')
                    trainer.train(num_generations=num_gens)

                elif args.algorithm == 'ppo':
                    if args.episodes:
                        timesteps = args.episodes * 1000
                    else:
                        timesteps = config.get('total_timesteps', 1000000)
                    print(f'Training PPO on World {current_world}-{current_stage} '
                          f'for {timesteps:,} timesteps...\n')
                    trainer.train(total_timesteps=timesteps)

                elif args.algorithm == 'dqn':
                    num_episodes = args.episodes or config.get('num_episodes', 5000)
                    print(f'Training DQN on World {current_world}-{current_stage} '
                          f'for {num_episodes:,} episodes...\n')
                    trainer.train(num_episodes=num_episodes)

                elif args.algorithm == 'a2c':
                    if args.episodes:
                        config['total_timesteps'] = args.episodes * 1000
                    timesteps = config.get('total_timesteps', 1_000_000)
                    print(f'Training A2C for {timesteps:,} timesteps...\n')
                    trainer.train()

                elif args.algorithm == 'rainbow':
                    num_episodes = args.episodes or config.get('num_episodes', 5000)
                    print(f'Training Rainbow DQN on World {current_world}-{current_stage} '
                          f'for {num_episodes:,} episodes...\n')
                    trainer.train(num_episodes=num_episodes)

                # Check if we should advance to the next stage
                if curriculum:
                    # Curriculum callbacks have been reporting per-episode
                    # data throughout training.  Now check if the sliding
                    # window shows readiness to advance.
                    if curriculum.should_advance():
                        result = curriculum.advance()
                        if result is None:
                            print('\nAll 32 stages complete!')
                            break
                        next_w, next_s = result
                        print(f'\nCurriculum advancing: -> World {next_w}-{next_s}')
                    else:
                        # Not ready to advance yet — keep training this stage
                        continue
                elif not args.next_stage or trainer._dashboard_closed:
                    break
                else:
                    result = next_world_stage(current_world, current_stage)
                    if result is None:
                        print('\nAll stages complete! (8-4 reached)')
                        break
                    next_w, next_s = result
                print(f'\n{"="*60}')
                print(f'  ADVANCING: World {current_world}-{current_stage} '
                      f'→ World {next_w}-{next_s}')
                print(f'  Transferring learned weights to new stage...')
                print(f'{"="*60}\n')

                # Save the current model as the transfer checkpoint
                algo_name = args.algorithm
                transfer_path = os.path.join(
                    'models', algo_name,
                    f'transfer_w{current_world}s{current_stage}',
                )
                os.makedirs(os.path.dirname(transfer_path), exist_ok=True)
                trainer.save_checkpoint(transfer_path)

                # Close old environment
                env.close()
                current_world, current_stage = next_w, next_s

                # Create new environment for the next stage
                if args.algorithm == 'neat':
                    env = create_neat_env(world=current_world, stage=current_stage)
                else:
                    env = create_cnn_env(world=current_world, stage=current_stage)
                print(f'New environment: World {current_world}-{current_stage}')

                # Rebuild trainer with the new environment, loading
                # transferred weights from the previous stage.
                # Reset the _already_saved guard so the new stage can save.
                trainer._already_saved = False
                trainer.env = env

                if args.algorithm == 'neat':
                    # For NEAT, the best genome is already saved.
                    # Seed the new population from the best genome via
                    # reproduction (the population object carries over).
                    print('NEAT: Carrying over population to new stage.')
                    # Reset generation counter and fitness tracking for the
                    # new stage so stagnation detection starts fresh.
                    trainer.best_reward = float('-inf')
                    trainer.best_distance = 0
                    # Update multi-env evaluator for new stage
                    if trainer.parallel_evaluator is not None:
                        trainer.parallel_evaluator.shutdown()
                        trainer.world = current_world
                        trainer.stage = current_stage
                        trainer.parallel_evaluator = ParallelGenomeEvaluator(
                            num_envs=num_envs,
                            world=current_world,
                            stage=current_stage,
                            neat_config_path=config_path,
                        )

                elif args.algorithm == 'ppo':
                    # PPO: Re-wrap the new env for SB3 and swap it into
                    # the existing model so weights transfer naturally.
                    # Close old extra environments first
                    for extra in getattr(trainer, '_extra_envs', []):
                        try:
                            extra.close()
                        except Exception:
                            pass
                    trainer._extra_envs = []
                    trainer.world = current_world
                    trainer.stage = current_stage
                    trainer.vec_env = trainer._wrap_env_for_sb3(
                        env, num_envs=num_envs,
                    )
                    trainer.model.set_env(trainer.vec_env)
                    trainer.best_reward = float('-inf')
                    trainer.best_distance = 0
                    trainer.episode_count = 0

                elif args.algorithm == 'dqn':
                    # DQN: Networks keep their weights, we just reset the
                    # environment reference and tracking counters.
                    # Epsilon is kept so exploitation continues.
                    trainer.best_reward = float('-inf')
                    trainer.best_distance = 0
                    trainer.episode_count = 0
                    # Close old extra envs and create new ones for new stage
                    for extra in trainer.extra_envs:
                        try:
                            extra.close()
                        except Exception:
                            pass
                    trainer.extra_envs = []
                    if num_envs > 1:
                        from src.environment.mario_env import create_cnn_env as _create_cnn
                        for _ in range(num_envs - 1):
                            trainer.extra_envs.append(
                                _create_cnn(world=current_world, stage=current_stage)
                            )

    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user.')
    except SystemExit:
        pass
    finally:
        # Cleanup
        if stream_manager:
            stream_manager.stop()
        if recorder:
            recorder.stop()
        if dashboard:
            dashboard.close()
        env.close()

    print('\nDone! Check models/ for saved checkpoints.')


if __name__ == '__main__':
    # Required for multiprocessing on Windows (NEAT parallel evaluator,
    # PPO SubprocVecEnv). Without this, spawned subprocesses would
    # re-execute this script from the top.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
