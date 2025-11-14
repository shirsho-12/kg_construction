#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve()

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.json_pipeline import run_json_pipeline
from config import EXAMPLE_DATA_PATH_JSON

run_json_pipeline(data_path=EXAMPLE_DATA_PATH_JSON, output_dir = Path.cwd() / "output" / "qa" / EXAMPLE_DATA_PATH_JSON.parts[-1].split(".")[0],
        run_oie_flag = True,
        run_schema_definition_flag = True,
        run_compression_flag = True,
        run_qa_flag = False)
