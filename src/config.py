"""Configuration constants for KG Construction."""

from pathlib import Path

# Model configuration
MISTRAL_MAX_LENGTH = 4096
MODEL_CACHE_DIR = Path.cwd() / "model_cache"
BASE_ENCODER_MODEL = "ministral/Ministral-3b-instruct"

# Data paths
EXAMPLE_DATA_PATH_TEXT = Path.cwd() / "data/text/example.txt"
EXAMPLE_DATA_PATH_JSON = Path.cwd() / "data/json/test.json"

# Prompt paths
OIE_PROMPT_PATH = Path.cwd() / "prompts/oie_prompt.txt"
OIE_SYNONYMY_PROMPT_PATH = Path.cwd() / "prompts/oie_synonyms_prompt.txt"
OIE_SYNONYMS_FEW_SHOT_EXAMPLES_PATH = Path.cwd() / "prompts/oie_example_synonyms.txt"
OIE_FEW_SHOT_EXAMPLES_PATH = Path.cwd() / "prompts/oie_example.txt"

SD_PROMPT_PATH = Path.cwd() / "prompts/sd_prompt.txt"
SD_FEW_SHOT_EXAMPLES_PATH = Path.cwd() / "prompts/sd_example.txt"

# Logging
LOGGING_LEVEL = "DEBUG"
