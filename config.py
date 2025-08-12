import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _resolve_default_path() -> str:
    """Returns the default path for the config file (project root)."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_root, "config.json")


def load_config(config_path: str | None = None) -> Dict[str, Dict[str, Any]]:
    """
    Loads configuration from a JSON file. If the file does not exist or is invalid,
    returns empty sections and relies on in-code constructor defaults.

    Resolution order:
    1) Explicit config_path argument
    2) Environment variable BOT_CONFIG_PATH
    3) Default path at project root: config.json
    """
    path = (
        config_path
        or os.environ.get("BOT_CONFIG_PATH")
        or _resolve_default_path()
    )

    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                cfg = json.load(f)
            # Keep only recognized sections and drop keys with null values
            filtered: Dict[str, Dict[str, Any]] = {"strategy": {}, "bot": {}}
            for section in ("strategy", "bot"):
                section_data = cfg.get(section, {}) or {}
                if isinstance(section_data, dict):
                    filtered[section] = {k: v for k, v in section_data.items() if v is not None}
            logger.info("Loaded configuration from %s", os.path.abspath(path))
            return filtered
        else:
            logger.warning("Config file not found at %s. Using in-code defaults.", os.path.abspath(path))
            return {"strategy": {}, "bot": {}}
    except Exception as e:
        logger.error("Failed to load config from %s: %s. Using in-code defaults.", path, e, exc_info=True)
        return {"strategy": {}, "bot": {}}


