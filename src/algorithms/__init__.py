"""
Algorithms module for ML Training.

Contains ML algorithm implementations, each following the
BaseTrainer interface for consistent training, evaluation, and
model saving/loading.

Algorithms:
    NEAT:    Neuroevolution of Augmenting Topologies - evolves neural
             network topology through genetic algorithms.
    PPO:     Proximal Policy Optimization - modern policy gradient method
             using stable-baselines3.
    DQN:     Deep Q-Network - value-based RL with experience replay
             and target networks, custom PyTorch implementation.
    A2C:     Advantage Actor-Critic - synchronous variant of A3C using
             stable-baselines3.
    Rainbow: Rainbow DQN - combines six DQN improvements (Double DQN,
             PER, Dueling, Noisy Nets, C51, Multi-step) into one agent.
             Custom PyTorch implementation with per-feature toggles.
"""
