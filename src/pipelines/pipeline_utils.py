"""
Shared utilities for pipeline operations.
Contains common functions used by both text and JSON pipelines.
"""

from pathlib import Path
from typing import List, Dict
import logging
import json

logger = logging.getLogger(__name__)


def setup_file_logging(output_dir: Path, log_filename: str = "pipeline_errors.log"):
    """Setup file-based logging for warnings and errors."""
    log_file = output_dir / log_filename
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Error logging enabled: {log_file}")


def save_problematic_report(problematic_cases: List[Dict], output_path: Path):
    """Save a detailed report of problematic inputs and their outputs."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(problematic_cases, f, indent=2, ensure_ascii=False)
    logger.info(
        "Saved problematic cases report: %s (%d cases)",
        output_path,
        len(problematic_cases),
    )


def add_problematic_case(
    problematic_cases: List[Dict], text: str, issue: str, **kwargs
):
    """Helper to add a problematic case with consistent structure."""
    case = {"input_text": text, "issue": issue}
    case.update(kwargs)
    problematic_cases.append(case)


def save_synonyms(synonyms: List[Dict], output_path: Path):
    """Save synonyms with de-duplication and JSON-serializable format."""
    serializable_synonyms = {}
    if synonyms:
        for i, text_synonyms in enumerate(synonyms):
            if text_synonyms:
                cleaned: Dict[str, List[str]] = {}
                for k, v in text_synonyms.items():
                    key_str = f"{k[0]}#SEP{k[1]}" if isinstance(k, tuple) else str(k)
                    if isinstance(v, (list, tuple, set)):
                        unique_vals = sorted(set(v))
                    else:
                        unique_vals = [v]
                    cleaned[key_str] = unique_vals
                serializable_synonyms[str(i)] = cleaned

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_synonyms, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved synonyms to {output_path}")


def process_oie_results(
    oie_triplets: List, dataset, problematic_cases: List[Dict]
) -> List:
    """
    Process OIE extraction results and track problematic cases.

    Args:
        oie_triplets: Raw triplets from OIE extraction
        dataset: Dataset containing the original texts
        problematic_cases: List to append problematic cases to

    Returns:
        List of (text, triplets) tuples
    """
    all_triplets_per_text = []
    for idx, triplets in enumerate(oie_triplets):
        if idx < len(dataset):
            text = dataset[idx]
            if not triplets:
                add_problematic_case(
                    problematic_cases,
                    text=text,
                    issue="empty_triplets",
                    output=triplets,
                )
            all_triplets_per_text.append((text, triplets))
        else:
            logger.warning(f"Triplet index {idx} exceeds dataset length")
    return all_triplets_per_text
