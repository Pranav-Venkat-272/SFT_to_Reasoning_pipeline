"""
reshape_pipeline.py
-------------------
Reshapes datasets that already contain reasoning traces
into the same output schema used by pipeline.py.

No API calls are made — this is a pure data transformation step.
Output is saved as a parquet file per dataset in OUTPUT_DIR.

Usage:
    python3 reshape_pipeline.py

Add your datasets to the DATASETS list below.
Implement your dataset-specific logic inside extract_row().
"""

import pandas as pd
from pathlib import Path
from datasets import load_dataset
import yaml
import os
from huggingface_hub import login

# -----------------------------
# Configuration — loaded from config.yaml
# -----------------------------
with open(Path(__file__).parent / "config.yaml") as f:
    _cfg = yaml.safe_load(f)

HF_TOKEN   = _cfg.get("hf_token", "")
OUTPUT_DIR = Path(_cfg["output_dir"])
CACHE_DIR  = _cfg["cache_dir"]

if HF_TOKEN:
    login(token=HF_TOKEN)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# DATASETS CONFIG
# -----------------------------
# Add your datasets here in the following format:
#
# DATASETS = [
#     {
#         "hf_path": "your_dataset",
#         "hf_split": "train",
#         "type": "your_format_type",
#     }
# ]
#
# Example:
# {
#     "hf_path": "username/dataset-name",
#     "hf_split": "train",
#     "type": "custom"
# }

DATASETS = []

# -----------------------------
# Output schema builder
# -----------------------------
def build_row(question, reasoning, answer, model="null"):
    """
    Builds a single output row matching the schema used in pipeline.py.
    'model' is optional metadata.
    """
    return {
        "Question": question,
        "<Think>": reasoning,
        "Answer": answer,
        "Number of words in instruction": len(question.split()),
        "Number of words in system prompt": "null",
        "Number of words in question": len(question.split()),
        "Number of words in answer": len(answer.split()),
        "source": "custom",
        "task": "qa",
        "model": model
    }

# -----------------------------
# Row extractor
# -----------------------------
def extract_row(row, dtype):
    """
    Implement your dataset-specific extraction logic here.

    This function should return a formatted row dictionary
    using build_row(), or None to skip.

    Example:
      if type =="General":
        question = row.get("prompt", "")
        reasoning = row.get("reasoning", "")
        answer = row.get("response", "")

        if not question or not answer:
            return None

        return build_row(question, reasoning, answer)
    """
    return None

# -----------------------------
# Per-dataset pipeline
# -----------------------------
def run_reshape(hf_path, hf_split, **ds_cfg):
    parquet_file = ds_cfg.get("parquet_file") or f"{hf_path.replace('/', '_')}.parquet"
    parquet_path = OUTPUT_DIR / parquet_file

    dtype = ds_cfg.get("type")

    print(f"[{parquet_file}] Starting reshape for {hf_path}")

    dataset = load_dataset(
        hf_path,
        split=hf_split,
        cache_dir=CACHE_DIR,
        streaming=True,
        token=HF_TOKEN if HF_TOKEN else None,
    )

    rows = []
    for i, row in enumerate(dataset):
        result = extract_row(row, dtype)
        if result:
            rows.append(result)

        if i % 500 == 0:
            print(f"[{parquet_file}] {i} rows processed, {len(rows)} kept...")

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)

    print(f"[{parquet_file}] Done. Saved {len(df)} rows to {parquet_path}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if not DATASETS:
        print("No datasets configured. Please add your datasets in the DATASETS list.")
        exit()

    for ds in DATASETS:
        run_reshape(
            hf_path=ds["hf_path"],
            hf_split=ds["hf_split"],
            **ds
        )

    print("All reshape jobs complete.")
