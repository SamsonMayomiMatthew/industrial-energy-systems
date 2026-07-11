"""Utility utilities providing globally shared loggers and system checks."""

import logging
import os
import sys
from typing import Any
import pandas as pd
import requests
from config import DATA_FILE_PATH, DATA_URL

def setup_logger(name: str = "PM_REWS") -> logging.Logger:
    """Configures and returns a standardized console logger.

    Args:
        name: Name of the logger instance.

    Returns:
        Standardized logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

LOGGER = setup_logger()

def verify_and_download_dataset() -> str:
    """Downloads the AI4I 2020 Predictive Maintenance Dataset if missing.

    Returns:
        Path to the saved CSV file.
    """
    if os.path.exists(DATA_FILE_PATH):
        LOGGER.info(f"Dataset already local at: {DATA_FILE_PATH}")
        return DATA_FILE_PATH

    LOGGER.info(f"Downloading dataset from remote repository: {DATA_URL}")
    try:
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        with open(DATA_FILE_PATH, "wb") as f:
            f.write(response.content)
        LOGGER.info("Dataset successfully acquired and written to local disk.")
    except Exception as e:
        LOGGER.error(f"Failed to fetch dataset from source repository: {e}")
        raise e
    return DATA_FILE_PATH