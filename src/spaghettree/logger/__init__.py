import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.propagate = False

REPO_ROOT = Path(__file__).absolute().parents[3]
load_dotenv(f"{REPO_ROOT}/envs/.env")

if os.getenv("SPAGHETTREE_LOGGING_ENABLED", "false").lower() == "true":
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    log_path = f"{REPO_ROOT}/logs/spaghettree.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=2)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
