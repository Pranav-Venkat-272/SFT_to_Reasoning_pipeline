import requests
import pandas as pd
from pathlib import Path
import os
import yaml
from datasets import load_dataset
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# Configuration — loaded from config.yaml
# -----------------------------
with open(Path(__file__).parent / "config.yaml") as f:
    _cfg = yaml.safe_load(f)

API_URL           = _cfg["api_url"]
API_KEY           = _cfg["api_key"]
HF_TOKEN          = _cfg.get("hf_token", "")
MODEL             = _cfg.get("model", "openai/gpt-oss-120b")
TEMPERATURE       = float(_cfg.get("temperature", 0.2))
SAVE_INTERVAL     = int(_cfg.get("save_interval", 20))
MAX_WORKERS       = int(_cfg.get("max_workers", 10))
MAX_CONTEXT_CHARS = int(_cfg.get("max_context_chars", 4000))
OUTPUT_DIR        = Path(_cfg["output_dir"])
CACHE_DIR         = _cfg["cache_dir"]

if HF_TOKEN:
    login(token=HF_TOKEN)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

PROMPT_TEMPLATE = """
You are given a question and solution.

Your task is to generate:

1. A reasoning section that explains how the solution is derived, wrapped exactly in <think> tags.

Requirements:
- Reasoning must go only inside <think> ... </think> tags.
- Do not repeat the question.
- Do not use phrases like "As stated in the solution" or anything on those lines.
- Arrive at the same solution without using the solution in your reasoning
- Use natural reasoning; bullet points only if helpful.
- Ensure <think> and solution are completely separate.

Output MUST follow this exact format:

<think>
Reasoning explaining how the solution is derived.
</think>

Question:
{question}

Solution:
{solution}
"""

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
#         "parquet_file": "your_parquet_file",
#         "checkpoint_file": "your_checkpoint_file"
#     }
# ]

DATASETS = []

# -----------------------------
# Extract (context, response) pairs from a row
# -----------------------------
def extract_pairs(row, ds_cfg):
    """
    Implement your dataset-specific extraction logic here.

    This function should return a list of (context, response) tuples.

    Example:
        if type == "helpsteer":
        context = row.get("prompt", "")
        response = row.get("response", "")
        if context and response and len(context) <= MAX_CONTEXT_CHARS:
            pairs.append((context, response))

    Customize this depending on your dataset format.
    """

# -----------------------------
# API + Processing
# -----------------------------
def call_model(prompt):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Reasoning: Low"},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        if response.status_code != 200:
            print(f"Request failed: {response.text}")
            return None
        data = response.json()
        msg = data["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content")
    except Exception as e:
        print(f"Error calling API: {e}")
        return None

def process_example(i, context, response):
    prompt = PROMPT_TEMPLATE.format(question=context, solution=response)
    reasoning = call_model(prompt)

    if reasoning is None:
        return None

    return {
        "Question": context,
        "<Think>": reasoning,
        "Answer": response,
        "Number of words in instruction": len(context.split()) + len(PROMPT_TEMPLATE.split()),
        "Number of words in system prompt": len(PROMPT_TEMPLATE.split()),
        "Number of words in question": len(context.split()),
        "Number of words in answer": len(response.split()),
        "source": "custom",
        "task": "qa",
        "model": MODEL
    }

# -----------------------------
# Save rows to parquet + update checkpoint
# -----------------------------
def save_to_parquet(rows, parquet_path, checkpoint_path, checkpoint_idx):
    tmp_parquet = parquet_path + ".tmp"
    tmp_checkpoint = checkpoint_path + ".tmp"

    new_df = pd.DataFrame(rows)
    if os.path.exists(parquet_path):
        existing_df = pd.read_parquet(parquet_path)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_parquet(tmp_parquet, index=False)
    with open(tmp_checkpoint, "w") as f:
        f.write(str(checkpoint_idx))

    os.replace(tmp_parquet, parquet_path)
    os.replace(tmp_checkpoint, checkpoint_path)

    return len(final_df)

# -----------------------------
# Pipeline
# -----------------------------
def run_pipeline(hf_path, hf_split, **ds_cfg):
    parquet_file = ds_cfg.get("parquet_file") or f"{hf_path.replace('/', '_')}.parquet"
    checkpoint_file = ds_cfg.get("checkpoint_file") or f"{hf_path.replace('/', '_')}.checkpoint"

    PARQUET_PATH = str(OUTPUT_DIR / parquet_file)
    CHECKPOINT_PATH = str(OUTPUT_DIR / checkpoint_file)

    print(f"[{parquet_file}] Starting pipeline for {hf_path}")

    start_idx = -1
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            content = f.read().strip()
            start_idx = int(content) if content else -1
        print(f"[{parquet_file}] Resuming from checkpoint index: {start_idx}")
    else:
        print(f"[{parquet_file}] Starting fresh")

    dataset = load_dataset(
        hf_path,
        split=hf_split,
        cache_dir=CACHE_DIR,
        streaming=True,
        token=HF_TOKEN if HF_TOKEN else None
    )
    dataset = dataset.skip(start_idx + 1)

    pending = {}
    rows_buffer = []
    rows_since_save = 0
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    last_i = start_idx

    print(f"[{parquet_file}] Starting processing loop...")

    try:
        for i, row in enumerate(dataset, start=start_idx + 1):
            last_i = i
            pairs = extract_pairs(row, ds_cfg)

            for context, response in pairs:
                future = executor.submit(process_example, i, context, response)
                pending[future] = i

            rows_since_save += 1

            if rows_since_save >= SAVE_INTERVAL:
                print(f"[{parquet_file}] Flushing batch at row index {i}...")

                for future in as_completed(pending):
                    result = future.result()
                    if result:
                        rows_buffer.append(result)
                pending.clear()

                if rows_buffer:
                    total = save_to_parquet(rows_buffer, PARQUET_PATH, CHECKPOINT_PATH, i)
                    print(f"[{parquet_file}] Saved {len(rows_buffer)} rows. Total: {total}")
                    rows_buffer = []

                rows_since_save = 0

    except KeyboardInterrupt:
        print(f"[{parquet_file}] Interrupted. Saving remaining work...")

    if pending:
        print(f"[{parquet_file}] Finalizing remaining tasks...")
        for future in as_completed(pending):
            result = future.result()
            if result:
                rows_buffer.append(result)
        pending.clear()

    if rows_buffer:
        total = save_to_parquet(rows_buffer, PARQUET_PATH, CHECKPOINT_PATH, last_i)
        print(f"[{parquet_file}] Final save: {len(rows_buffer)} rows. Total: {total}")

    executor.shutdown(wait=True)
    print(f"[{parquet_file}] Done.")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if not DATASETS:
        print("No datasets configured. Please add your datasets in the DATASETS list.")
        exit()

    print(f"Launching {len(DATASETS)} datasets...")

    with ThreadPoolExecutor(max_workers=len(DATASETS)) as pool:
        futures = {
            pool.submit(run_pipeline, **ds): ds["hf_path"]
            for ds in DATASETS
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"[{name}] Finished successfully.")
            except Exception as e:
                print(f"[{name}] Failed with error: {e}")

    print("All datasets processed.")
