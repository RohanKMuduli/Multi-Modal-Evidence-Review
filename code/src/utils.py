"""
Utility functions for logging, path management, and data formatting.
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(name: str = "damage_claim_system", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standard logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (writes to app/run log)
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            # Fallback if logs directory cannot be created
            pass

    return logger


def normalize_string(text: Optional[str]) -> str:
    """
    Normalizes a string by stripping, lowercasing, and replacing underscores/hyphens where useful.
    """
    if not text:
        return ""
    return text.strip().lower()


def get_project_root() -> str:
    """
    Returns the absolute path to the project root directory.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dotenv_file(path: Optional[str] = None) -> None:
    """
    Loads KEY=VALUE pairs from a local .env file without overriding existing env vars.
    """
    env_path = path or os.path.join(get_project_root(), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
