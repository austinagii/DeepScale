from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


SUBKEY_SEPARATOR = "."


class Config:
    """Configuration for the DeepScale module"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, data: dict[str, Any], override=False):
        if not hasattr(self, '_initialized') or override:  # Prevent re-initialization
            self.data = data
            self._initialized = True

    @classmethod
    def get_instance(cls):
        return cls._instance

    @classmethod
    def from_yaml(cls, path: str | Path, override=False):
        """Create a Config instance from a file."""
        if isinstance(path, str):
            try:
                path = Path(path)
            except Exception:
                raise ValueError(f"'{path}' is not a valid path")

        if not path.exists():
            raise FileNotFoundError(f"DeepScale config file not found: {path}")

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return cls(config, override)

    def __getitem__(self, key: str):
        if not isinstance(key, str) or len(key) == 0:
            raise TypeError("Key must be a non-empty string.")

        key_components = key.split(SUBKEY_SEPARATOR)

        obj = self.data
        for component in key_components:
            obj = obj.get(component)

            if obj is None:
                raise KeyError(f"Field '{key}' not found in config")

        return obj

    def __setitem__(self, key: str, value: Any):
        if not isinstance(key, str) or len(key) == 0:
            raise TypeError("Key must be a non-empty string.")

        key_components = key.split(SUBKEY_SEPARATOR)

        obj = self.data
        for component in key_components[:-1]:
            if (next_obj := obj.get(component)) is None:
                obj[component] = dict()
                next_obj = obj[component]

            obj = next_obj
                 

        obj[key_components[len(key_components) - 1]] = value

    def get(self, key: str) -> Any:
        value = None
        try: 
            value = self.__getitem__(key)
        except KeyError:
            pass

        return value

    def __str__(self):
        return yaml.dump(self.config, default_flow_style=False)
