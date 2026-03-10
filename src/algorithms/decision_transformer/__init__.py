"""
Decision Transformer for multi-game generalist agent.

A GPT-2 style transformer that treats RL as sequence prediction.
Given (return-to-go, observation, action) tuples, it learns to
predict actions that achieve the desired return.

Components:
    - dt_network: The transformer architecture with modality embeddings.
    - trajectory_dataset: PyTorch Dataset wrapping the Experience Store.
    - dt_trainer: Training loop with evaluation and checkpointing.

Reference:
    Chen et al. "Decision Transformer: Reinforcement Learning via
    Sequence Modeling" (NeurIPS 2021)
"""
