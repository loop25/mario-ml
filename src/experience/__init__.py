"""
Experience collection and storage for the Generalist Agent.

This package provides the data pipeline that connects individual game
agents to the Decision Transformer:

    Game Agents → Tokenizer → Experience Store → TrajectoryDataset → DT Trainer

Components:
    - tokenizer: Converts game-specific observations/actions into a
      standardized token format suitable for transformer input.
    - experience_store: Unified buffer that collects tokenized trajectories
      from all games and supports efficient batch sampling.
"""
