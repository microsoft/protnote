import torch
import os
import time
import datetime
import logging
from src.utils.data import read_yaml


def override_config(config: dict, overrides: list):
    # Process the overrides if provided
    if overrides:
        if len(overrides) % 2 != 0:
            raise ValueError("Overrides must be provided as key-value pairs.")

        # Convert the list to a dictionary
        overrides = {
            overrides[i]: overrides[i + 1] for i in range(0, len(overrides), 2)
        }

        # Update the config with the overrides
        for key, value in overrides.items():
            # Convert value to appropriate type if necessary (e.g., float, int)
            # Here, we're assuming that the provided keys exist in the 'params' section of the config
            if key in config["params"]:
                config["params"][key] = type(config["params"][key])(value)
            else:
                raise KeyError(
                    f"Key '{key}' not found in the 'params' section of the config."
                )


def get_setup(config_path: str, run_name: str, overrides: list):
    # Get the root path from the environment variable; default to current directory if ROOT_PATH is not set
    ROOT_PATH = os.environ.get("ROOT_PATH", ".")

    # Load the configuration file
    config = read_yaml(os.path.join(ROOT_PATH, config_path))

    if overrides:
        override_config(config, overrides)

    # Extract the parameters and paths from the (possibly overidden) config file
    params = config["params"]
    paths = {
        key: os.path.join(ROOT_PATH, value)
        for key, value in config["relative_paths"].items()
    }

    # Load datasets from config file paths; the same vocabulary is used for all datasets
    common_paths = {
        "amino_acid_vocabulary_path": paths["AMINO_ACID_VOCAB_PATH"],
        "label_vocabulary_path": paths["GO_LABEL_VOCAB_PATH"],
        "sequence_id_vocabulary_path": paths["SEQUENCE_ID_VOCAB_PATH"],
        "sequence_id_map_path": paths["SEQUENCE_ID_MAP_PATH"],
    }
    paths_list = [
        {**common_paths, "data_path": paths[key]}
        for key in ["TRAIN_DATA_PATH", "VAL_DATA_PATH", "TEST_DATA_PATH"]
    ]

    # Set the timezone for the entire Python environment
    os.environ["TZ"] = "US/Pacific"
    time.tzset()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S %Z").strip()

    # Initialize logging
    # TODO: Find a way to give W&B access to the log file
    log_dir = os.path.join(ROOT_PATH, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    full_log_path = os.path.join(log_dir, f"{timestamp}_train_{run_name}.log")

    logging.basicConfig(
        filename=full_log_path,
        filemode="w",
        format="%(asctime)s %(levelname)-4s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )

    logger = logging.getLogger()
    print(f"Logging to {full_log_path}...")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return config, params, paths, paths_list, timestamp, logger, device, ROOT_PATH
