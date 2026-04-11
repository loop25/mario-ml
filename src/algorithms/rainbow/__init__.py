"""
Rainbow DQN Algorithm.

Combines six orthogonal improvements to DQN into one agent:
    1. Double DQN — reduces Q-value overestimation
    2. Prioritized Experience Replay — samples surprising transitions
    3. Dueling Networks — separates V(s) and A(s,a) estimation
    4. Noisy Networks — learned exploration (replaces epsilon-greedy)
    5. Distributional RL (C51) — learns return distributions
    6. Multi-step Returns — uses n-step bootstrapping

Each improvement can be toggled independently for ablation studies.

Reference: Hessel et al., 2018 — "Rainbow: Combining Improvements
in Deep Reinforcement Learning"
"""
