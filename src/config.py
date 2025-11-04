from pathlib import Path

MISTRAL_MAX_LENGTH = 4096
MODEL_CACHE_DIR = Path.cwd() / "model_cache"
BASE_ENCODER_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
EXAMPLE_DATA_PATH_TEXT = Path.cwd() / "data/text/example.txt"

OIE_TEMPLATE_PATH = Path.cwd() / "prompts/oie_template.txt"
OIE_FEW_SHOT_EXAMPLES_PATH = Path.cwd() / "prompts/oie_example.txt"

LOGGING_LEVEL = "INFO"
