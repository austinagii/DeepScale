from pathlib import Path
from typing import Any

import pytest

from deepscale.config import Config
from deepscale.run import Run, RunManager
from deepscale.run.checkpoint import Checkpoint, CheckpointType
from deepscale.storage.clients import (
    AzureBlobStorageClient,
    FileSystemStorageClient,
)
from deepscale.storage.errors import StorageError


@pytest.fixture
def dsconfig_file_path(tmp_path: Path) -> Path:
    config = """
    runs:
        checkpoints:
            sources:
                - filesystem 
                - azure-blob
            destinations:
                - azure-blob
                - filesystem 
    """

    config_file_path = tmp_path / "dsconfig.yaml"
    config_file_path.write_text(config)
    return config_file_path


@pytest.fixture(scope="module")
def run_config() -> dict[str, Any]:
    return {
        "tokenizer": {"vocab_size": 1024},
        "model": {"num_blocks": 8, "d_key": 128, "d_value": 128},
    }


@pytest.fixture(scope="module")
def checkpoint():
    return Checkpoint(epoch=1, prev_loss=0.95, best_loss=0.88)


@pytest.fixture
def run_manager(run_config):
    run_manager = RunManager(
        sources=["azure-blob", "filesystem"],
        destinations=["azure-blob", "filesystem"],
    )

    run_manager.run = Run.from_config(run_config)

    return run_manager


class TestRunManager:
    def test_run_manager_can_be_initialized_from_location_list(self):
        run_manager = RunManager(
            sources=["azure-blob", "filesystem"],
            destinations=["azure-blob", "filesystem"],
        )

        assert isinstance(run_manager.storage_clients[0], AzureBlobStorageClient)
        assert isinstance(run_manager.storage_clients[1], FileSystemStorageClient)
        assert isinstance(run_manager.retrieval_clients[0], AzureBlobStorageClient)
        assert isinstance(run_manager.retrieval_clients[1], FileSystemStorageClient)

    def test_run_manager_can_be_initialized_from_yaml_config(
        self, dsconfig_file_path: Path
    ) -> None:
        config = Config.from_yaml(dsconfig_file_path, override=True)
        run_manager = RunManager.from_config(config)

        assert isinstance(run_manager.storage_clients[0], FileSystemStorageClient)
        assert isinstance(run_manager.storage_clients[1], AzureBlobStorageClient)
        assert isinstance(run_manager.retrieval_clients[0], AzureBlobStorageClient)
        assert isinstance(run_manager.retrieval_clients[1], FileSystemStorageClient)

    def test_init_run_cascades_to_storage_clients(
        self, mocker, dsconfig_file_path, run_config
    ) -> None:
        run_manager = RunManager(
            sources=["azure-blob", "filesystem"],
            destinations=["azure-blob", "filesystem"],
        )

        for client in run_manager.storage_clients:
            mocker.patch.object(client, "init_run")

        run_manager.init_run(run_config)

        for client in run_manager.storage_clients:
            client.init_run.assert_called_once_with(
                run_manager.run.id, run_manager.run.config
            )

    def test_init_run_raises_error_if_any_storage_client_fails(
        self, mocker, dsconfig_file_path, run_config
    ) -> None:
        run_manager = RunManager(
            sources=["azure-blob", "filesystem"],
            destinations=["azure-blob", "filesystem"],
        )

        for client in run_manager.storage_clients:
            mocker.patch.object(client, "init_run")

        # Configure one storage client to fail
        run_manager.storage_clients[1].init_run.side_effect = StorageError()

        with pytest.raises(StorageError):
            run_manager.init_run(run_config)

        for client in run_manager.storage_clients:
            client.init_run.assert_called_once()

    def test_save_checkpoint_cascades_to_storage_clients(
        self, mocker, run_manager, checkpoint
    ):
        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_checkpoint")

        run_manager.save_checkpoint(CheckpointType.EPOCH, checkpoint)

        for client in run_manager.storage_clients:
            client.save_checkpoint.assert_called_once_with(
                run_manager.run.id, "epoch:0001", bytes(checkpoint)
            )

    @pytest.mark.parametrize(
        "checkpoint_type, checkpoint_tag",
        [
            (CheckpointType.EPOCH, "epoch:0001"),
            (CheckpointType.BEST, "best"),
            (CheckpointType.FINAL, "final"),
        ],
    )
    def test_save_checkpoint_uses_correct_checkpoint_tag(
        self, mocker, run_manager, checkpoint, checkpoint_type, checkpoint_tag
    ):
        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_checkpoint")

        run_manager.save_checkpoint(checkpoint_type, checkpoint)

        for client in run_manager.storage_clients:
            client.save_checkpoint.assert_called_once_with(
                run_manager.run.id, checkpoint_tag, bytes(checkpoint)
            )

    def test_save_checkpoint_does_not_raise_error_if_any_storage_client_fails(
        self, mocker, run_manager, checkpoint
    ) -> None:
        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_checkpoint")

        # Configure one storage client to fail
        run_manager.storage_clients[1].save_checkpoint.side_effect = StorageError()

        run_manager.save_checkpoint(CheckpointType.EPOCH, checkpoint)

        for client in run_manager.storage_clients:
            client.save_checkpoint.assert_called_once()
