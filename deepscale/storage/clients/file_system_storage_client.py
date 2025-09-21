from pathlib import Path
from typing import Any

import yaml

from deepscale.storage.errors import (
    CheckpointNotFoundError,
    RunNotFoundError,
    StorageError,
)
from deepscale.storage.storage_client import StorageClient


DEFAULT_BASE_DIR = Path(".")
RUN_CONFIG_PATH_TEMPLATE = "runs/{run_id}/config.yaml"
RUN_CHECKPOINT_PATH_TEMPLATE = "runs/{run_id}/checkpoints/{checkpoint_tag}.pt"


class FileSystemStorageClient(StorageClient):
    """Manages the storage and retrieval of artifacts on the local file system.

    Args:
        base_dir: The base directory where all artifacts created by this storage
          client will be stored and retrieved.
    """

    def __init__(self, base_dir: str | Path = None):
        if base_dir is None:
            base_dir = DEFAULT_BASE_DIR

        if isinstance(base_dir, str):
            base_dir = Path(base_dir)

        self._base_dir = base_dir

    def init_run(self, run_id: str, train_config: dict[Any, Any]) -> None:
        train_config_file_path = self.base_dir / RUN_CONFIG_PATH_TEMPLATE.format(
            run_id=run_id
        )

        train_config_file_path.parent.mkdir(parents=True, exist_ok=True)

        train_config_file_path.write_text(yaml.dump(train_config))

    def resume_run(
        self, run_id: str, checkpoint_tag: str
    ) -> tuple[dict[Any, Any], bytes]:
        train_config_path = self.base_dir / RUN_CONFIG_PATH_TEMPLATE.format(
            run_id=run_id
        )

        # breakpoint()
        if not train_config_path.exists():
            raise RunNotFoundError(f"No run with id '{run_id}' could be found.")

        train_config = yaml.safe_load(train_config_path.read_text())

        checkpoint = self.load_checkpoint(run_id, checkpoint_tag)

        return train_config, checkpoint

    def exists(self, run_id: str, checkpoint_tag: str | None = None) -> bool:
        run_exists = False
        checkpoint_exists = False

        expected_run_dir = self.base_dir / "runs" / run_id
        run_exists = expected_run_dir.exists()

        if run_exists and checkpoint_tag is not None:
            expected_checkpoint_dir = (
                self.base_dir
                / RUN_CHECKPOINT_PATH_TEMPLATE.format(
                    run_id=run_id, checkpoint_tag=checkpoint_tag
                )
            )
            checkpoint_exists = expected_checkpoint_dir.exists()

        return run_exists and (checkpoint_tag is None or checkpoint_exists)

    def save_checkpoint(
        self, run_id: str, checkpoint_tag: str, checkpoint: bytes
    ) -> None:
        checkpoint_path = self.base_dir / RUN_CHECKPOINT_PATH_TEMPLATE.format(
            run_id=run_id, checkpoint_tag=checkpoint_tag
        )
        # Create the checkpoint directory if it does not already exist.
        if not checkpoint_path.parent.exists():
            try:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise StorageError(
                    f"Failed to create artifact directory '{checkpoint_path.parent}'",
                    e,
                )

        mode = "w" if checkpoint_path.exists() else "x"
        try:
            with open(checkpoint_path, f"{mode}b") as f:
                f.write(checkpoint)
        except Exception as e:
            raise StorageError(
                f"An error occurred while saving the artifact to '{checkpoint_path}'",
                e,
            )

    def load_checkpoint(self, run_id: str, checkpoint_tag: str) -> bytes:
        """Downloads the artifact from the local device"""
        checkpoint_path = self.base_dir / RUN_CHECKPOINT_PATH_TEMPLATE.format(
            run_id=run_id, checkpoint_tag=checkpoint_tag
        )

        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(
                f"No checkpoint with name '{checkpoint_tag}' could be found."
            )

        return checkpoint_path.read_bytes()

    @property
    def base_dir(self):
        """The base directory where the artifacts managed by this client are stored"""
        return self._base_dir
