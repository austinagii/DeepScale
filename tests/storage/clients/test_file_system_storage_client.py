import pytest
import yaml

from deepscale.run import generate_run_id
from deepscale.storage.clients import FileSystemStorageClient


@pytest.fixture
def run_id():
    """Generate a random run id."""
    return generate_run_id()


@pytest.fixture
def fs_storage_client(tmp_path):
    """Returns a FileSystemStorageClient instance.

    The provided instance uses a temporary directory as its base storage path.
    """
    return FileSystemStorageClient(base_dir=tmp_path)


class TestFileSystemStorageClient:
    def test_can_be_initialized_with_str(self, tmp_path):
        pathstr = str(tmp_path)

        client = FileSystemStorageClient(pathstr)

        assert client.base_dir == tmp_path

    def test_can_be_initialized_with_path(self, tmp_path):
        client = FileSystemStorageClient(tmp_path)

        assert client.base_dir == tmp_path

    def test_init_run_creates_a_new_directory_containing_the_train_config(
        self, tmp_path, fs_storage_client, run_id
    ):
        train_config_yaml = """
        name: gpt2
        model:
            num_heads: 10
            num_blocks: 3
        """
        train_config = yaml.safe_load(train_config_yaml)

        fs_storage_client.init_run(run_id, train_config)

        expected_config_path = tmp_path / f"runs/{run_id}/config.yaml"

        assert expected_config_path.exists()
        # Text written to disk should be the equivalent of a yaml dump of the config.
        assert expected_config_path.read_text() == yaml.dump(train_config)

    def test_save_checkpoint_creates_directory_if_it_does_not_exist(
        self, fs_storage_client, run_id
    ):
        expected_checkpoint_dir = (
            fs_storage_client.base_dir / f"runs/{run_id}/checkpoints/"
        )

        assert not expected_checkpoint_dir.exists()

        fs_storage_client.save_checkpoint(
            run_id=run_id, checkpoint_tag="best", checkpoint=b"test"
        )

        assert expected_checkpoint_dir.exists()

    def test_save_checkpoint_correctly_save_checkpoints_the_artifact(
        self, fs_storage_client, run_id
    ):
        fs_storage_client.save_checkpoint(
            run_id=run_id, checkpoint_tag="best", checkpoint=b"test"
        )

        expected_checkpoint_path = (
            fs_storage_client.base_dir / f"runs/{run_id}/checkpoints/best.pt"
        )

        assert expected_checkpoint_path.exists()
        assert expected_checkpoint_path.read_text() == "test"

    def test_save_checkpoint_overwrites_existing_artifacts(
        self, fs_storage_client, run_id
    ):
        expected_checkpoint_path = (
            fs_storage_client.base_dir / f"runs/{run_id}/checkpoints/best.pt"
        )

        # Write some content to the checkpoint path. We expect this to be overridden.
        expected_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        expected_checkpoint_path.write_text("test")

        fs_storage_client.save_checkpoint(
            run_id=run_id, checkpoint_tag="best", checkpoint=b"test2"
        )

        assert expected_checkpoint_path.read_text() == "test2"

    def test_load_checkpoint_successfully_load_checkpoints_the_artifact(
        self, fs_storage_client
    ):
        expected_checkpoint_path = (
            fs_storage_client.base_dir / f"runs/{run_id}/checkpoints/best.pt"
        )

        expected_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        expected_checkpoint_path.write_text("test")

        checkpoint_bytes = fs_storage_client.load_checkpoint(
            run_id=run_id, checkpoint_tag="best"
        )

        assert checkpoint_bytes == b"test"
