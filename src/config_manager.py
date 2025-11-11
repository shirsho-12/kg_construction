"""
Configuration manager for KG Construction pipelines.
Handles loading and merging of base and specific configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """Manages configuration loading and merging."""

    def __init__(self, base_config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            base_config_path: Path to base configuration file.
                             If None, uses default base_config.yaml location.
        """
        if base_config_path is None:
            # Default to base_config.yaml in config_templates directory
            project_root = Path(__file__).resolve().parent.parent
            base_config_path = project_root / "config_templates" / "base_config.yaml"

        self.base_config_path = base_config_path
        self.base_config = self._load_base_config()

    def _load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration."""
        if not self.base_config_path.exists():
            raise FileNotFoundError(
                f"Base config file not found: {self.base_config_path}"
            )

        with open(self.base_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Convert relative paths to absolute paths based on project root
        config = self._resolve_paths(config, self.base_config_path.parent.parent)

        return config

    def load_config(self, specific_config_path: Path) -> Dict[str, Any]:
        """
        Load and merge specific configuration with base configuration.

        Args:
            specific_config_path: Path to the specific configuration file

        Returns:
            Merged configuration dictionary
        """
        if not specific_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {specific_config_path}")

        with open(specific_config_path, "r", encoding="utf-8") as f:
            specific_config = yaml.safe_load(f)

        # Resolve paths in specific config
        specific_config = self._resolve_paths(
            specific_config, specific_config_path.parent.parent
        )

        # Merge with base config (specific config overrides base config)
        merged_config = self._merge_configs(self.base_config, specific_config)

        return merged_config

    def _merge_configs(
        self, base: Dict[str, Any], specific: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge specific config with base config.
        Specific config values override base config values.

        Args:
            base: Base configuration dictionary
            specific: Specific configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        merged = base.copy()

        for key, value in specific.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Override with specific config value
                merged[key] = value

        return merged

    def _resolve_paths(
        self, config: Dict[str, Any], config_dir: Path
    ) -> Dict[str, Any]:
        """
        Convert relative paths to absolute paths based on project root.

        Args:
            config: Configuration dictionary
            config_dir: Directory containing the config file

        Returns:
            Configuration with resolved absolute paths
        """
        project_root = Path(__file__).resolve().parent.parent

        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict):
                    config[key] = self._resolve_paths(value, config_dir)
                elif isinstance(value, str) and (
                    key.endswith("_path")
                    or key.endswith("_dir")
                    or "data_path" in key
                    or "output_dir" in key
                    or key in ["cache_dir", "base_encoder_model"]
                ):
                    if not Path(value).is_absolute():
                        config[key] = str((project_root / value).resolve())

        return config

    def setup_logging(self, config: Dict[str, Any]) -> logging.Logger:
        """
        Setup logging based on configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configured logger
        """
        log_config = config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO").upper())
        log_format = log_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(level=log_level, format=log_format)
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured with level: {log_config.get('level', 'INFO')}")

        return logger

    def get_config_value(
        self, key_path: str, config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get a configuration value by key path (e.g., "model.base_encoder_model").

        Args:
            key_path: Dot-separated path to the configuration key
            config: Configuration dictionary (uses base config if None)

        Returns:
            Configuration value
        """
        if config is None:
            config = self.base_config

        keys = key_path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Configuration key not found: {key_path}")

        return value


# Global config manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(specific_config_path: Path) -> Dict[str, Any]:
    """
    Load and merge configuration files.

    Args:
        specific_config_path: Path to specific configuration file

    Returns:
        Merged configuration dictionary
    """
    manager = get_config_manager()
    return manager.load_config(specific_config_path)


def setup_logging_from_config(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured logger
    """
    manager = get_config_manager()
    return manager.setup_logging(config)
