"""
Algorithms module for Super Mario Bros ML.

Contains three ML algorithm implementations, each following the
BaseTrainer interface for consistent training, evaluation, and
model saving/loading.

Algorithms:
    NEAT: Neuroevolution of Augmenting Topologies - evolves neural
          network topology through genetic algorithms.
    PPO:  Proximal Policy Optimization - modern policy gradient method
          using stable-baselines3.
    DQN:  Deep Q-Network - value-based RL with experience replay
          and target networks, custom PyTorch implementation.
"""
