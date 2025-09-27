import logging
from typing import Any

from .config import Config
from .run import Checkpoint, RunManager


DS_CONFIG_PATH = "./deepscale.yaml"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def init_run(run_config: dict[Any, Any]) -> tuple[str, RunManager]:
    """Initializes a new training run with the specified configuration.

    Arguments:
        run_config: The configuration to be used for the training run.

    Returns:
        A tuple containing the following:
            - The unique ID of the newly created run.
            - The `RunManager` instance that manages the newly created run.
    """
    if not Config.is_initialized():
        Config.from_yaml(DS_CONFIG_PATH)

    run_manager = RunManager.from_config(Config.get_instance())
    run_id = run_manager.init_run(run_config)
    return run_id, run_manager


def resume_run(
    run_id: str, checkpoint_tag: str | None = None, device=None
) -> tuple[dict[Any, Any], Checkpoint, RunManager]:
    """Resumes the specified training run.

    If a checkpoint tag is specified, then training is resumed from the matching
    checkpoint, else, training is resumed from the latest checkpoint.

    Args:
        run_id: The ID of the training run to be resumed.
        checkpoint_tag: The tag of the checkpoint to resume the specified training run
            from. Defaults to `None`.

    Returns:
        A tuple containing the following:
            - The configuration used for the specified training run.
            - The checkpoint corresponding to the specified tag.
            - The run manager for the resumed training run.

    Raises:
        RunNotFoundError: If the specified run could not be found.
        CheckpointNotFoundError: If the specified checkpoint could not be found.
    """
    if not Config.is_initialized():
        Config.from_yaml(DS_CONFIG_PATH)

    # TODO: Consider saving the dsconfig used at the time of the training run. This
    # behavior could also be overriden.
    run_manager = RunManager.from_config(Config.get_instance())
    run_config, checkpoint = run_manager.resume_run(
        run_id, checkpoint_tag, device=device
    )
    return run_config, checkpoint, run_manager
