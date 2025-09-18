import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

import yaml
from azure.storage.blob import BlobServiceClient

from deepscale.storage.storage_client import StorageClient
from deepscale.storage.errors import (
    CheckpointNotFoundError,
    PersistenceError,
    RunNotFoundError,
)


RUN_BASE_PATH_TEMPLATE = "runs/{run_id}"
RUN_CONFIG_PATH_TEMPLATE = "runs/{run_id}/config.yaml"
RUN_CHECKPOINT_PATH_TEMPLATE = "runs/{run_id}/checkpoints/{checkpoint_tag}.pt"

# Disable Azure blob storage logging
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)


class AzureBlobStorageClient(StorageClient):
    def __init__(self, blob_service_client: BlobServiceClient, container_name: str):
        self.blob_service_client = blob_service_client
        self.container_name = container_name

    def init_run(self, run_id: str, train_config: dict[Any, Any]) -> None:
        """Creates a new run with the specifed training configuration.

        This saves the train configuration as a blob in blob storage under the run
        container.

        Args:
            run_id: The unique identifier of the run to be initialized.
            train_config: The training configuration for this run.

        Returns:
            None
        """
        config_blob_path = RUN_CONFIG_PATH_TEMPLATE.format(run_id=run_id)

        with disable_tokenizer_parallelism():
            try:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name, blob=config_blob_path
                )
                blob_client.upload_blob(yaml.dump(train_config), overwrite=True)
            except Exception as e:
                raise PersistenceError("Failed to initialize the run.", e)

    def resume_run(
        self, run_id: str, checkpoint_tag: str
    ) -> tuple[dict[Any, Any], bytes]:
        # Check if the run config exists. If not, the run does not exist.
        run_config_blob_name = RUN_CONFIG_PATH_TEMPLATE.format(run_id=run_id)
        run_config_blob_client = self.blob_service_client.get_blob_client(
            self.container_name, run_config_blob_name
        )
        if not run_config_blob_client.exists():
            raise RunNotFoundError("The specified run could not be found.")

        # Check if the checkpoint exists.
        checkpoint_blob_name = RUN_CHECKPOINT_PATH_TEMPLATE.format(
            run_id=run_id, checkpoint_tag=checkpoint_tag
        )
        checkpoint_blob_client = self.blob_service_client.get_blob_client(
            self.container_name, checkpoint_blob_name
        )
        if not checkpoint_blob_client.exists():
            raise CheckpointNotFoundError(
                "The specified checkpoint could not be found."
            )

        # Download the run config and checkpoint only after confirming that they both
        # exist, no point downloading the config if the run does not exist.
        run_config = yaml.safe_load(run_config_blob_client.download_blob().readall())
        checkpoint = checkpoint_blob_client.download_blob().readall()

        return run_config, checkpoint

    def exists(self, run_id: str, checkpoint_tag: Optional[str] = None) -> bool:
        """Returns whether the specified run or checkpoint exists in Azure Blob Storage.

        If no checkpoint tag is specified then the function returns whether the
        specified run exists. If a checkpoint tag is specified then the function
        returns whether the specified checkpoint for the given run exists.

        Args:
            run_id: The unique identifier of the run to check
            checkpoint_tag: Optional tag of a specific checkpoint to check.
            Defaults to `None`.

        Returns:
            bool: True if the run/checkpoint exists, False otherwise.

        Raises:
            PersistenceError: If there is an or checkpoint exists.
        """
        blob_client = self.blob_service_client.get_container_client(self.container_name)

        run_exists = False
        checkpoint_exists = False

        # TODO: Use the blob_clients `exists` method if both run_id and checkpoint_tag
        # are specified. Should be O(1).
        with disable_tokenizer_parallelism():
            try:
                blob_names = blob_client.list_blob_names()
            except Exception as e:
                raise PersistenceError(
                    "An error occurred while checking if the artifact exists on the "
                    "remote device",
                    e,
                )

        for blob_name in blob_names:
            if blob_name.startswith(f"runs/{run_id}"):
                run_exists = True
                if checkpoint_tag is None:
                    break

                checkpoint_exists = blob_name.endswith(f"{checkpoint_tag}.pt")
                if checkpoint_exists:
                    break

        return run_exists and (checkpoint_tag is None or checkpoint_exists)

    def save_checkpoint(
        self, run_id: str, checkpoint_tag: str, checkpoint: bytes
    ) -> None:
        """Uploads the checkpoint to the remote device"""
        checkpoint_path = RUN_CHECKPOINT_PATH_TEMPLATE.format(
            run_id=run_id, checkpoint_tag=checkpoint_tag
        )

        with disable_tokenizer_parallelism():
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=str(checkpoint_path)
            )

            try:
                blob_client.upload_blob(checkpoint, overwrite=True)
            except Exception as e:
                raise PersistenceError(
                    "An error occurred while syncing checkpoint to blob storage", e
                )

    def load_checkpoint(self, run_id: str, checkpoint_tag: str) -> bytes:
        """Downloads the checkpoint from the remote device."""
        checkpoint_path = RUN_CHECKPOINT_PATH_TEMPLATE.format(
            run_id=run_id, checkpoint_tag=checkpoint_tag
        )

        with disable_tokenizer_parallelism():
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=checkpoint_path
            )
            try:
                checkpoint_exists = blob_client.exists()
            except Exception as e:
                raise PersistenceError(
                    "An error occurred while checking if the checkpoint exists on the "
                    "remote device",
                    e,
                )
            if not checkpoint_exists:
                raise CheckpointNotFoundError("Checkpoint could not be found in blob")

            try:
                checkpoint = blob_client.download_blob().readall()
            except Exception as e:
                raise PersistenceError(
                    "An error occurred while downloading the checkpoint from the remote"
                    " device",
                    e,
                )

            return checkpoint


@contextmanager
def disable_tokenizer_parallelism():
    """Context manager to temporarily disable tokenizers parallelism.

    This prevents fork conflicts with the tokenizers library during blob operations.
    """
    original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        yield
    finally:
        # Restore original tokenizer parallelism setting
        if original_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism
        else:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
