"""Pipelines package for KG Construction."""

from .pipeline import run_pipeline
from .json_pipeline import run_json_pipeline

__all__ = ["run_pipeline", "run_json_pipeline"]
