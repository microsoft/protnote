import torch
import os
import time
import datetime
import logging
from src.utils.data import read_yaml
import sys
from ast import literal_eval

def try_literal_eval(val):
    try:
        # Attempt to evaluate as a literal
        return literal_eval(val)
    except (ValueError, SyntaxError):
        # If evaluation fails means input is actually a string
        if val=='null':
            return None
        if (val=="false")|(val=="true"):
            return literal_eval(val.title())
        return val

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
                config["params"][key] = try_literal_eval(value)
            else:
                raise KeyError(
                    f"Key '{key}' not found in the 'params' section of the config."
                )


def get_setup(
    config_path: str,
    run_name: str,
    overrides: list,
    train_path_name: str = None,
    val_path_name: str = None,
    test_paths_names: list = None,
    amlt: bool = False,
    is_master: bool = True,
):
    # Get the root path from the environment variable; default to current directory if ROOT_PATH is not set
    if amlt:
        ROOT_PATH = os.getcwd()  # Set ROOT_PATH to working directory
        DATA_PATH = os.environ["AMLT_DATA_DIR"]
        OUTPUT_PATH = os.environ["AMLT_OUTPUT_DIR"]
    else:
        ROOT_PATH = os.environ.get("ROOT_PATH", ".")
        DATA_PATH = os.path.join(ROOT_PATH, "data")
        OUTPUT_PATH = os.path.join(ROOT_PATH, "outputs")
        if not os.path.exists(OUTPUT_PATH) and is_master:
            os.makedirs(OUTPUT_PATH)

    # Load the configuration file and override the parameters if provided
    config = read_yaml(os.path.join(ROOT_PATH, config_path))
    if overrides:
        override_config(config, overrides)

    # Extract the model parameters from the (possibly overidden) config file
    params = config["params"]

    # Extract the fixed ProteInfer params from the config file
    embed_sequences_params = config["embed_sequences_params"]

    # Prepend the correct path roots
    # Define root paths for each section
    section_paths = {
        "data_paths": DATA_PATH,
        "output_paths": OUTPUT_PATH,
    }
    paths = {
        key: os.path.join(section_paths[section], value)
        for section, section_values in config["paths"].items()
        for key, value in section_values.items()
    }

    # Load datasets from config file paths; the same vocabulary is used for all datasets
    common_paths = {
        "vocabularies_dir": paths["VOCABULARIES_DIR"],
        "go_annotations_path": paths['GO_ANNOTATIONS_PATH']
    }

    train_paths_list = (
        [{**common_paths, "data_path": paths[train_path_name], "dataset_type": "train"}]
        if train_path_name is not None
        else []
    )

    val_paths_list = (
        [
            {
                **common_paths,
                "data_path": paths[val_path_name],
                "dataset_type": "validation",
            }
        ]
        if val_path_name is not None
        else []
    )

    test_paths_list = (
        [
            {**common_paths, "data_path": paths[key], "dataset_type": "test"}
            for key in test_paths_names
        ]
        if test_paths_names is not None
        else []
    )

    dataset_paths_list = train_paths_list + val_paths_list + test_paths_list

    # Set the timezone for the entire Python environment
    os.environ["TZ"] = "US/Pacific"
    time.tzset()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S %Z").strip()

    # Initialize logging
    log_dir = paths["LOG_DIR"]
    if not os.path.exists(log_dir) and is_master:
        try:
            os.makedirs(log_dir)
        except FileExistsError:
            print(
                f"Log directory {log_dir} already exists. is_master={is_master}")
            pass
    full_log_path = os.path.join(log_dir, f"{timestamp}_{run_name}.log")

    # Initialize the logger for all processes
    logger = logging.getLogger()

    # Only log to file and console if this is the main process
    if is_master:
        # Set the logging level (default for other processes is WARNING)
        logger.setLevel(logging.INFO)

        # Create a formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-4s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %Z"
        )

        # Create a file handler and add it to the logger
        file_handler = logging.FileHandler(full_log_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Create a stream handler (for stdout) and add it to the logger
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        print(f"Logging to {full_log_path} and console...")
    else:
        # Set the logger level to an unreachable level, effectively disabling it
        logger.setLevel(logging.CRITICAL + 1)

    # Return a dictionary
    return {
        "params": params,
        "embed_sequences_params": embed_sequences_params,
        "paths": paths,
        "dataset_paths_list": dataset_paths_list,
        "timestamp": timestamp,
        "logger": logger,
        "ROOT_PATH": ROOT_PATH,
        "DATA_PATH": DATA_PATH,
        "OUTPUT_PATH": OUTPUT_PATH,
    }
