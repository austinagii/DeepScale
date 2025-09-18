from typing import Any, Protocol

from deepscale.run.checkpoint import Checkpoint


class StorageClient(Protocol):
    def init_run(self, run_id: str, train_config: bytes) -> None: ...

    def resume_run(
        self, run_id: str, checkpoint_tag: str
    ) -> tuple[dict[Any, Any], Checkpoint]: ...

    def exists(self, run_id: str, checkpoint_tag: str | None = None) -> bool: ...

    def save_checkpoint(
        self, run_id: str, checkpoint_tag: str, checkpoint: bytes
    ) -> None: ...

    def load_checkpoint(self, run_id: str, checkpoint_tag: str) -> bytes: ...
