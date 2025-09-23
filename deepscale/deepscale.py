import logging
from pathlib import Path
from typing import Any

from .config import Config
from .run import Checkpoint, RunManager


DEFAULT_DSCONFIG_PATH = "./deepscale.yaml"
DS_CONFIG: Config = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def init(ds_config_path: str | Path = DEFAULT_DSCONFIG_PATH) -> None:
    if isinstance(ds_config_path, str):
        ds_config_path = Path(ds_config_path)

    if not isinstance(ds_config_path, Path):
        raise TypeError(f"Expected a str or path, got {type(ds_config_path.__name__)}")

    # TODO: Consider having a default config.
    Config.from_yaml(ds_config_path, override=True)


def init_run(run_config: dict[Any, Any]) -> tuple[str, RunManager]:
    """Initializes a new training run with the specified configuration.

    Arguments:
        run_config: The configuration to be used for the training run.

    Returns:
        A tuple containing the following:
            - The unique ID of the newly created run.
            - The `RunManager` instance that manages the newly created run.
    """
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
    # TODO: Consider saving the dsconfig used at the time of the training run. This
    # behavior could also be overriden.
    run_manager = RunManager.from_config(Config.get_instance())
    run_config, checkpoint = run_manager.resume_run(
        run_id, checkpoint_tag, device=device
    )
    return run_config, checkpoint, run_manager


def setconf(key: str, value: Any) -> None:
    Config.get_instance()[key] = value
