import os
import random
import string
from collections import namedtuple

import pytest
import yaml
from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

from deepscale.run import generate_run_id
from deepscale.storage.clients import AzureBlobStorageClient
from deepscale.storage.clients.azure_blob_storage_client import (
    disable_tokenizer_parallelism,
)
from deepscale.storage.errors import CheckpointNotFoundError, StorageError


@pytest.fixture(scope="module")
def blob_service_client():
    """Azure Blob Service client configured from connection string."""
    az_blob_conn_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    if az_blob_conn_string is None:
        raise KeyError("Connection string not found for Azure Blob Storage.")

    yield (client := BlobServiceClient.from_connection_string(az_blob_conn_string))

    client.close()


@pytest.fixture
def container(blob_service_client):
    """Creates a random container in Azure Blob that will be cleaned up after use."""
    test_id = "".join([random.choice(string.ascii_lowercase) for _ in range(6)])
    container_name = f"test-{test_id}"
    container_client = blob_service_client.create_container(
        container_name, metadata={"Category": "test"}
    )

    Container = namedtuple("Container", ["name", "client"])
    yield Container(name=container_name, client=container_client)

    container_client.delete_container()


@pytest.fixture
def az_storage_client(blob_service_client, container):
    """AzureBlobStorageClient instance for testing."""
    return AzureBlobStorageClient(blob_service_client, container.name)


def _add_target_checkpoint(container_client):
    run_id = generate_run_id()
    checkpoint_tags = ["epoch_0001", "epoch_0002", "epoch_0003", "best", "final"]
    target_checkpoint_tag = random.choice(checkpoint_tags)

    for checkpoint_tag in checkpoint_tags:
        checkpoint_path = f"runs/{run_id}/checkpoints/{checkpoint_tag}.pt"
        checkpoint_data = (
            b"target" if checkpoint_tag == target_checkpoint_tag else b"test"
        )
        container_client.upload_blob(checkpoint_path, checkpoint_data)

    TargetCheckpoint = namedtuple("TargetCheckpoint", ["run_id", "name"])
    return TargetCheckpoint(run_id=run_id, name=target_checkpoint_tag)


class TestAzureBlobStorageClient:
    @pytest.mark.integration
    def test_init_run_stores_run_config_as_blob(self, az_storage_client, container):
        # Define the training config.
        train_config_yaml = """
        name: gpt2
        model:
            num_heads: 10
            num_blocks: 3
        """
        train_config = yaml.safe_load(train_config_yaml)
        run_id = generate_run_id()

        expected_config_path = f"runs/{run_id}/config.yaml"

        # Verify the run is clean.
        blob_client = container.client.get_blob_client(expected_config_path)
        assert not blob_client.exists()

        # Initialize the run.
        az_storage_client.init_run(run_id=run_id, train_config=train_config)

        # Verify that the run config is saved correctly.
        assert blob_client.exists()
        assert yaml.safe_load(blob_client.download_blob().readall()) == train_config

    # TODO: Add test to verify behavior when run config upload fails.

    @pytest.mark.integration
    def test_save_checkpoint_successfully_uploads_artifact(
        self, az_storage_client, container
    ):
        run_id = generate_run_id(n=8)
        checkpoint_tag = "test"
        checkpoint_data = b"test"

        az_storage_client.save_checkpoint(run_id, checkpoint_tag, checkpoint_data)

        assert container.client.get_blob_client(
            f"runs/{run_id}/checkpoints/{checkpoint_tag}.pt"
        ).exists()

    # TODO: Add test to verify behavior when run does not exist.

    def test_save_checkpoint_raises_error_when_upload_fails(
        self, mocker, container, blob_service_client
    ):
        mocker.patch(
            "azure.storage.blob.BlobClient.upload_blob",
            side_effect=Exception("Upload failed"),
        )

        az_storage_client = AzureBlobStorageClient(blob_service_client, container.name)

        with pytest.raises(StorageError):
            az_storage_client.save_checkpoint("abc", "best", b"test")

    @pytest.mark.integration
    def test_load_checkpoint_successfully_downloads_artifact(
        self, az_storage_client, container
    ):
        target_checkpoint = _add_target_checkpoint(container.client)

        result = az_storage_client.load_checkpoint(
            target_checkpoint.run_id, target_checkpoint.name
        )

        assert result == b"target"

    # TODO: Split this test to verify behavior when checkpoint not found and run not
    # found.
    def test_load_checkpoint_raises_error_when_artifact_not_found(
        self, mocker, az_storage_client
    ):
        mocker.patch(
            "azure.storage.blob.BlobClient.download_blob",
            side_effect=ResourceNotFoundError("Test"),
        )

        with pytest.raises(CheckpointNotFoundError):
            az_storage_client.load_checkpoint("testing123", "testing123")

    def test_load_checkpoint_raises_error_when_download_fails(
        self, mocker, az_storage_client
    ):
        mocker.patch(
            "azure.storage.blob.BlobClient.download_blob",
            side_effect=AzureError("A random error occurred."),
        )

        with pytest.raises(StorageError):
            az_storage_client.load_checkpoint("testrun", "testcheckpoint")


@pytest.mark.integration
class TestDisableTokenizerParallelism:
    def setup_method(self):
        # Ensure clean state
        if "TOKENIZERS_PARALLELISM" in os.environ:
            del os.environ["TOKENIZERS_PARALLELISM"]

    def test_sets_tokenizers_parallelism_to_false(self):
        with disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"

    def test_restores_original_value_when_set(self):
        original_value = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = original_value

        with disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"

        assert os.getenv("TOKENIZERS_PARALLELISM") == original_value

    def test_removes_env_var_when_not_originally_set(self):
        assert os.getenv("TOKENIZERS_PARALLELISM") is None

        with disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"

        assert os.getenv("TOKENIZERS_PARALLELISM") is None

    def test_restores_state_on_exception(self):
        original_value = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = original_value

        with pytest.raises(ValueError):
            with disable_tokenizer_parallelism():
                assert os.getenv("TOKENIZERS_PARALLELISM") == "false"
                raise ValueError("Test exception")

        assert os.getenv("TOKENIZERS_PARALLELISM") == original_value

    def test_removes_env_var_on_exception_when_not_originally_set(self):
        if "TOKENIZERS_PARALLELISM" in os.environ:
            del os.environ["TOKENIZERS_PARALLELISM"]

        with pytest.raises(ValueError):
            with disable_tokenizer_parallelism():
                assert os.getenv("TOKENIZERS_PARALLELISM") == "false"
                raise ValueError("Test exception")

        assert os.getenv("TOKENIZERS_PARALLELISM") is None

    def test_handles_edge_case_values(self):
        # Test with empty string (should be preserved)
        os.environ["TOKENIZERS_PARALLELISM"] = ""

        with disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"

        assert os.getenv("TOKENIZERS_PARALLELISM") == ""

        # Test with None (env var not set)
        del os.environ["TOKENIZERS_PARALLELISM"]
        assert os.getenv("TOKENIZERS_PARALLELISM") is None

        with disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"

        assert os.getenv("TOKENIZERS_PARALLELISM") is None
