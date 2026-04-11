"""
Agent packaging utilities for the Model Zoo.

Provides export/import of trained agents as portable .agent files (ZIP format)
for sharing and archival.

Package structure inside the .agent ZIP::

    agent_package/
        metadata.json       # Required — training metadata
        profile.json        # Optional — agent profile
        checkpoints/        # All model checkpoint files
            best_model.zip
            ...
"""

import os
import json
import time
import zipfile
import glob
from typing import Optional, List


def export_agent(model_dir: str, output_path: Optional[str] = None) -> str:
    """Package a trained agent as a .agent file for sharing.

    Parameters
    ----------
    model_dir : str
        Path to the model directory (e.g. ``models/ppo/``).
    output_path : str, optional
        Destination path for the ``.agent`` file.  When *None* a name is
        generated automatically under an ``exports/`` directory.

    Returns
    -------
    str
        The path to the created ``.agent`` file.

    Raises
    ------
    FileNotFoundError
        If *model_dir* does not exist or lacks a ``metadata.json``.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Read metadata
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"No metadata.json in {model_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Generate output path if not provided
    if output_path is None:
        game = meta.get("game_id", "unknown")
        algo = meta.get("algorithm", "unknown")
        timestamp = int(time.time())
        os.makedirs("exports", exist_ok=True)
        output_path = f"exports/{game}_{algo}_{timestamp}.agent"

    # Ensure parent directory exists
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Create ZIP archive
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # metadata.json — required
        zf.write(meta_path, "agent_package/metadata.json")

        # profile.json — optional
        profile_path = os.path.join(model_dir, "profile.json")
        if os.path.isfile(profile_path):
            zf.write(profile_path, "agent_package/profile.json")

        # All checkpoint files
        for pattern in ["*.zip", "*.pt", "*.pth", "*.pkl"]:
            for fpath in glob.glob(os.path.join(model_dir, pattern)):
                fname = os.path.basename(fpath)
                zf.write(fpath, f"agent_package/checkpoints/{fname}")

    return output_path


def import_agent(agent_path: str, models_dir: str = "models") -> str:
    """Import a .agent file into the models directory.

    Parameters
    ----------
    agent_path : str
        Path to the ``.agent`` file.
    models_dir : str
        Root models directory (default ``models``).

    Returns
    -------
    str
        Path to the directory where files were extracted.

    Raises
    ------
    FileNotFoundError
        If *agent_path* does not exist.
    ValueError
        If the archive is not a valid ``.agent`` package (missing metadata).
    """
    if not os.path.isfile(agent_path):
        raise FileNotFoundError(f"Agent file not found: {agent_path}")

    with zipfile.ZipFile(agent_path, "r") as zf:
        names = zf.namelist()
        if "agent_package/metadata.json" not in names:
            raise ValueError("Invalid .agent file: missing metadata.json")

        # Determine destination from metadata
        meta_data = json.loads(zf.read("agent_package/metadata.json"))
        algo = meta_data.get("algorithm", "imported")
        timestamp = int(time.time())
        dest_dir = os.path.join(models_dir, algo, f"imported_{timestamp}")
        os.makedirs(dest_dir, exist_ok=True)

        # Extract files — flatten checkpoints/ into the destination root
        for name in names:
            if not name.startswith("agent_package/"):
                continue
            relative = name[len("agent_package/"):]
            if not relative:
                continue
            # Flatten checkpoints/ so checkpoint files sit next to metadata
            if relative.startswith("checkpoints/"):
                relative = relative[len("checkpoints/"):]
            if not relative:
                continue
            dest_path = os.path.join(dest_dir, relative)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(zf.read(name))

    return dest_dir


def list_exported_agents(export_dir: str = "exports") -> List[dict]:
    """List all .agent files in the export directory.

    Parameters
    ----------
    export_dir : str
        Directory to scan (default ``exports``).

    Returns
    -------
    list[dict]
        Each dict contains *name*, *game_id*, *algorithm*, *episodes*,
        *best_reward*, *file_size*, and *path*.
    """
    if not os.path.isdir(export_dir):
        return []

    agents: List[dict] = []
    for fname in os.listdir(export_dir):
        if not fname.endswith(".agent"):
            continue
        fpath = os.path.join(export_dir, fname)
        try:
            with zipfile.ZipFile(fpath, "r") as zf:
                meta = json.loads(zf.read("agent_package/metadata.json"))
            agents.append({
                "name": fname,
                "game_id": meta.get("game_id", "?"),
                "algorithm": meta.get("algorithm", "?"),
                "episodes": meta.get("episode", 0),
                "best_reward": meta.get("best_reward", 0),
                "file_size": os.path.getsize(fpath),
                "path": fpath,
            })
        except Exception:
            continue
    return agents
