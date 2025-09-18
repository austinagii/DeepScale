import os
import random
import secrets
import string
from collections import namedtuple
from collections.abc import Sequence

import pytest
import yaml
from azure.storage.blob import BlobServiceClient

from deepscale.run import generate_run_id
from deepscale.storage.clients import AzureBlobStorageClient
from deepscale.storage.clients.azure_blob_storage_client import (
    disable_tokenizer_parallelism,
)
from deepscale.storage.errors import CheckpointNotFoundError, PersistenceError


@pytest.fixture
def blob_service_client_factory(mocker):
    def _factory(blobs: Sequence[str]):
        container_client = mocker.Mock()
        mocker.patch.object(container_client, "list_blob_names", return_value=blobs)

        blob_service_client = mocker.Mock()
        mocker.patch.object(
            blob_service_client, "get_container_client", return_value=container_client
        )
        return blob_service_client

    return _factory


@pytest.fixture(scope="module")
def blob_service_client():
    az_blob_conn_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    if az_blob_conn_string is None:
        raise KeyError("Connection string not found for Azure Blob Storage.")

    yield (client := BlobServiceClient.from_connection_string(az_blob_conn_string))

    client.close()


@pytest.fixture
def container(blob_service_client):
    # Use the existing container from .env file instead of creating a new one
    container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME", "lumiere")
    container_client = blob_service_client.get_container_client(container_name)
    
    # Test if the container exists and is accessible
    try:
        container_client.get_container_properties()
    except Exception as e:
        pytest.skip(f"Azure container '{container_name}' not accessible: {e}")

    Container = namedtuple("Container", ["name", "client"])
    yield Container(name=container_name, client=container_client)
    
    # Don't delete the container since we're using an existing one


@pytest.fixture
def az_storage_client(blob_service_client, container):
    return AzureBlobStorageClient(blob_service_client, container.name)


def _add_test_checkpoints(container_client):
    # create some filler blobs for testing
    for _ in range(5):
        run_id = generate_run_id(n=6)

        for i in range(1, 4):
            container_client.upload_blob(
                name=f"runs/{run_id}/checkpoints/epoch_{i:0>4}.pt", data=b"test"
            )

        container_client.upload_blob(
            name=f"runs/{run_id}/checkpoints/best.pt", data=b"test"
        )

        container_client.upload_blob(
            name=f"runs/{run_id}/checkpoints/final.pt", data=b"test"
        )


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
        train_config_yaml = """
        name: gpt2
        model:
            num_heads: 10
            num_blocks: 3
        """
        train_config = yaml.safe_load(train_config_yaml)
        run_id = generate_run_id()

        expected_config_path = f"runs/{run_id}/config.yaml"

        blob_client = container.client.get_blob_client(expected_config_path)
        assert not blob_client.exists()

        az_storage_client.init_run(run_id=run_id, train_config=train_config)

        assert blob_client.exists()
        assert yaml.safe_load(blob_client.download_blob().readall()) == train_config

    @pytest.mark.integration
    def test_exists_returns_true_if_run_is_present(self, az_storage_client, container):
        _add_test_checkpoints(container.client)
        target_checkpoint = _add_target_checkpoint(container.client)

        assert az_storage_client.exists(run_id=target_checkpoint.run_id)

    @pytest.mark.integration
    def test_exists_returns_false_if_run_is_not_present(
        self, az_storage_client, container
    ):
        _add_test_checkpoints(container.client)

        assert not az_storage_client.exists(run_id="testing123")

    @pytest.mark.integration
    def test_exists_returns_true_if_artifact_is_present(
        self, az_storage_client, container
    ):
        _add_test_checkpoints(container.client)

        target_checkpoint = _add_target_checkpoint(container.client)

        assert az_storage_client.exists(
            target_checkpoint.run_id, target_checkpoint.name
        )

    @pytest.mark.integration
    def test_exists_returns_false_if_artifact_is_not_present(
        self, az_storage_client, container
    ):
        _add_test_checkpoints(container.client)

        target_checkpoint = _add_target_checkpoint(container.client)

        assert not az_storage_client.exists(
            run_id=target_checkpoint.run_id, checkpoint_tag="testing123"
        )

    @pytest.mark.integration
    def test_exists_raises_an_error_if_error_occurs_at_az_blob(self, mocker):
        container_client = mocker.Mock()
        container_client.list_blob_names.side_effect = Exception("test")

        blob_service_client = mocker.Mock()
        blob_service_client.get_container_client.return_value = container_client

        client = AzureBlobStorageClient(blob_service_client, "testing123")

        with pytest.raises(PersistenceError):
            client.exists(run_id="test", checkpoint_tag="test")

    @pytest.mark.integration
    def test_save_checkpoint_successfully_uploads_artifact(
        self, az_storage_client, container
    ):
        run_id = generate_run_id(n=8)
        checkpoint_tag = "test"
        checkpoint_data = b"test"

        az_storage_client.save_checkpoint(run_id, checkpoint_tag, checkpoint_data)

        assert container.client.get_blob_client(f"runs/{run_id}/checkpoints/test.pt").exists()
        assert az_storage_client.exists(run_id=run_id, checkpoint_tag=checkpoint_tag)

    def test_save_checkpoint_raises_error_when_upload_fails(self, mocker, container):
        blob_client = mocker.Mock()
        blob_client.upload_blob.side_effect = Exception("Upload failed")

        blob_service_client = mocker.Mock()
        blob_service_client.get_blob_client.return_value = blob_client

        az_storage_client = AzureBlobStorageClient(blob_service_client, container.name)

        with pytest.raises(PersistenceError):
            az_storage_client.save_checkpoint("abc", "best", b"test")

    def test_load_checkpoint_successfully_downloads_artifact(
        self, az_storage_client, container
    ):
        _add_test_checkpoints(container.client)

        target_checkpoint = _add_target_checkpoint(container.client)

        result = az_storage_client.load_checkpoint(
            target_checkpoint.run_id, target_checkpoint.name
        )

        assert result == b"target"

    def test_load_checkpoint_raises_error_when_artifact_not_found(self, mocker):
        blob_client = mocker.Mock()
        blob_client.exists.return_value = False

        blob_service_client = mocker.Mock()
        blob_service_client.get_blob_client.return_value = blob_client

        az_storage_client = AzureBlobStorageClient(blob_service_client, "testing123")

        with pytest.raises(CheckpointNotFoundError):
            az_storage_client.load_checkpoint("testing123", "testing123")

    def test_load_checkpoint_raises_error_when_download_fails(self, mocker):
        blob_client = mocker.Mock()
        blob_client.exists.return_value = True
        blob_client.download_blob.side_effect = Exception("Download failed")

        blob_service_client = mocker.Mock()
        blob_service_client.get_blob_client.return_value = blob_client

        az_storage_client = AzureBlobStorageClient(blob_service_client, "testing123")

        with pytest.raises(PersistenceError):
            az_storage_client.load_checkpoint("testing123", "testing123")


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
