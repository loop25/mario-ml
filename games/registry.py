"""
Game registry — discovers and manages game adapters.

The registry scans three folders for game adapters:
  - games/builtin/   (games we build: Snake, Tetris, etc.)
  - games/retro/     (emulated ROM games: Mario, Sonic, etc.)
  - games/user/      (user-contributed game plugins)

Each subfolder that contains an adapter.py with a class inheriting
from BaseGameAdapter is auto-discovered and registered.
"""
import importlib
import importlib.util
import os
import sys
from typing import Dict, List, Optional

from games.base_adapter import BaseGameAdapter


class GameRegistry:
    """Discovers and manages game adapters."""

    def __init__(self):
        self._adapters: Dict[str, BaseGameAdapter] = {}

    def register(self, adapter: BaseGameAdapter) -> None:
        """Manually register a game adapter.

        Args:
            adapter: An instance of a BaseGameAdapter subclass.
        """
        self._adapters[adapter.game_id] = adapter

    def get_game(self, game_id: str) -> BaseGameAdapter:
        """Get a specific game adapter by ID.

        Args:
            game_id: The game's unique identifier (e.g. 'mario', 'snake').

        Returns:
            The registered BaseGameAdapter instance.

        Raises:
            KeyError: If no game with that ID is registered.
        """
        if game_id not in self._adapters:
            raise KeyError(
                f"Unknown game '{game_id}'. "
                f"Available: {', '.join(sorted(self._adapters.keys()))}"
            )
        return self._adapters[game_id]

    def list_games(self) -> List[BaseGameAdapter]:
        """Return all registered game adapters."""
        return list(self._adapters.values())

    def list_game_ids(self) -> List[str]:
        """Return sorted list of all registered game IDs."""
        return sorted(self._adapters.keys())

    def list_by_category(self, category: str) -> List[BaseGameAdapter]:
        """Filter games by category (e.g. 'board', 'arcade')."""
        return [g for g in self._adapters.values() if g.category == category]

    def discover(self, base_path: Optional[str] = None) -> None:
        """Auto-discover game adapters from the standard folders.

        Scans builtin/, retro/, and user/ subdirectories for adapter.py
        files. Each adapter.py must define a class that inherits from
        BaseGameAdapter. The first such class found is instantiated
        and registered.

        Args:
            base_path: Root path containing builtin/, retro/, user/
                       subdirectories. Defaults to the games/ directory.
        """
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))

        for folder_name in ['builtin', 'retro', 'user']:
            folder_path = os.path.join(base_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for subfolder in sorted(os.listdir(folder_path)):
                # Skip __pycache__, _template, hidden dirs
                if subfolder.startswith(('_', '.')):
                    continue
                adapter_path = os.path.join(folder_path, subfolder, 'adapter.py')
                if not os.path.isfile(adapter_path):
                    continue
                try:
                    adapter = self._load_adapter(adapter_path, folder_name, subfolder)
                    if adapter:
                        self._adapters[adapter.game_id] = adapter
                except Exception as e:
                    print(f"  Warning: failed to load adapter from {adapter_path}: {e}")

    def _load_adapter(
        self, path: str, folder_name: str, subfolder: str
    ) -> Optional[BaseGameAdapter]:
        """Load a single adapter from an adapter.py file.

        Uses importlib to load the module, then searches for the first
        class that is a subclass of BaseGameAdapter (but not
        BaseGameAdapter itself).

        Returns:
            An instantiated adapter, or None if no adapter class found.
        """
        module_name = f"games.{folder_name}.{subfolder}.adapter"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find the adapter class
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseGameAdapter)
                and obj is not BaseGameAdapter
            ):
                return obj()
        return None
