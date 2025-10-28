from typing import Optional, Union
import os
from datasets import load_dataset
from dotenv import load_dotenv

# /home/shirsho/projects/graph_construction/src/dataset.py
"""
Utility to get datasets from the Hugging Face Hub using the `datasets` library.

"""

load_dotenv()  # Load environment variables from .env file if present


def get_dataset(
    dataset: str = "web_nlg",
):
    """
    Get a dataset from the Hugging Face Hub.

    Args:
        dataset (str): The dataset identifier in the format "dataset_name/config_name".
                       Defaults to "web_nlg".

    Returns:
        Loaded Dataset or DatasetDict.
    """
    if dataset == "web_nlg":
        if not os.path.exists(os.getenv("DATASET_CACHE_DIR", "./data")):
            os.makedirs(
                os.path.join(os.getenv("DATASET_CACHE_DIR", "./data"), "web_nlg")
            )
        return load_dataset(
            "GEM/web_nlg",
            "en",
            cache_dir=os.path.join(os.getenv("DATASET_CACHE_DIR", "./data"), "web_nlg"),
        )

    elif dataset == "wiki_nre":
        if not os.path.exists(
            os.path.join(os.getenv("DATASET_CACHE_DIR", "./data"), "wiki_nre")
        ):
            os.makedirs(
                os.path.join(os.getenv("DATASET_CACHE_DIR", "./data"), "wiki_nre")
            )
        return load_dataset(
            "Saibo-creator/wiki-nre",
            cache_dir=os.path.join(
                os.getenv("DATASET_CACHE_DIR", "./data"), "wiki_nre"
            ),
        )
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported.")


if __name__ == "__main__":
    ds = get_dataset("wiki_nre")
    print(ds)
