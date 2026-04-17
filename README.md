# SFT-to-Reasoning Pipeline

Converts SFT (supervised fine-tuning) datasets into reasoning-augmented training data by generating `<think>` reasoning traces for each question/answer pair using a language model API. A second pipeline reshapes existing reasoning datasets into the same output schema.

---

## Repository Structure

```
sft-to-reasoning/
├── README.md
├── config.yaml            # All credentials and path config — edit this before running
├── pipeline.py            # Generates reasoning traces via LLM API
├── reshape_pipeline.py    # Reshapes existing reasoning datasets into the same schema
└── requirements.txt
```

---

## Installation

**Python 3.9+ required.**

```bash
git clone https://github.com/your-username/sft-to-reasoning.git
cd sft-to-reasoning
pip install -r requirements.txt
```

---

## Configuration

Edit `config.yaml` before running either pipeline. All credentials and paths live here — nothing sensitive is hardcoded in the scripts.

```yaml
api_url:   "http://your-api-host:port/v1/chat/completions"
api_key:   "YOUR_API_KEY_HERE"
hf_token:  ""          # Required for gated datasets (lmsys-chat-1m, GAIR/lima)

output_dir: "/your/output/path"
cache_dir:  "/your/hf/cache/path"

save_interval:     20    # rows between each save + checkpoint
max_workers:       10    # parallel API threads per dataset
max_context_chars: 4000
```

> **Note:** `hf_token` is only required for gated datasets. Accept dataset terms on HuggingFace before running.

---

## pipeline.py

Streams datasets from HuggingFace, extracts question/answer pairs, calls a language model API to generate `<think>` reasoning traces, and saves results to parquet with checkpointing.

**Run:**
```bash
python3 pipeline.py
```

All datasets in the `DATASETS` list run in parallel. Each gets its own output parquet and checkpoint file.

**Adding a new dataset** — append to the `DATASETS` list in `pipeline.py`:

```python
{
    "hf_path": "org/dataset-name",
    "hf_split": "train",
    "type": "helpsteer",           # see supported types in code comments
    "parquet_file": "gpt_mydata.parquet",
    "checkpoint_file": "gpt_mydata.checkpoint",
    # optional language filter:
    "language_field": "language",
    "language_value": "English",
}
```

---

## reshape_pipeline.py

Reshapes existing HuggingFace datasets that already contain reasoning traces into the same output schema as `pipeline.py`. No API calls are made.

**Run:**
```bash
python3 reshape_pipeline.py
```

**Adding a new dataset** — append to `DATASETS` in `reshape_pipeline.py` and add a handler in `extract_row()`:

```python
# In DATASETS:
{
    "hf_path": "org/reasoning-dataset",
    "hf_split": "train",
    "type": "my_type",
    "parquet_file": "reshape_mydata.parquet",
}

# In extract_row():
elif dtype == "my_type":
    question = row.get("prompt", "")
    reasoning = row.get("chain_of_thought", "")
    answer = row.get("output", "")
    if not question or not answer:
        return None
    return build_row(question, reasoning, answer)
```

---

## Output Schema

Both pipelines produce parquet files with the same schema:

| Column | Description |
|---|---|
| `Question` | Input context (full conversation history before the assistant turn) |
| `<Think>` | Generated or extracted reasoning trace |
| `Answer` | The assistant response |
| `Number of words in question` | Word count of question |
| `Number of words in answer` | Word count of answer |
| `Number of words in instruction` | Word count of question + prompt template |
| `Number of words in system prompt` | Word count of prompt template (`"null"` for reshape) |
| `Task` | `"QA"` |
| `Domain` | `"NL"` |
| `Language` | `"english"` |
| `Source` | `"huggingface"` |
| `Reasoning/SFT` | `"SFT"` for pipeline.py, `"reasoning"` for reshape_pipeline.py |
| `Model` | Model used to generate reasoning (`"null"` if unknown) |

---

## Checkpointing

`pipeline.py` saves a `.checkpoint` file alongside each `.parquet`. On restart it skips to the last saved index automatically — no data is reprocessed.

The pipeline uses **atomic writes**: every save writes to a `.tmp` file first and only renames it over the real file once the write completes. If the process is killed mid-write, the real parquet is always left intact.

**Recovering `.tmp` files** left by an interrupted run:
```bash
python3 -c "
import os, pyarrow.parquet as pq
d = '/your/output/dir'
for f in os.listdir(d):
    if not f.endswith('.tmp'): continue
    tmp, real = os.path.join(d,f), os.path.join(d,f[:-4])
    try:
        pq.ParquetFile(tmp) if f.endswith('.parquet.tmp') else int(open(tmp).read().strip())
        os.replace(tmp, real); print(f'Recovered: {f}')
    except Exception as e:
        print(f'Corrupt, skipping: {f} — {e}')
"
```

**Resetting a corrupted dataset** to rerun from scratch:
```bash
rm /your/output/dir/gpt_lmsys.parquet
rm /your/output/dir/gpt_lmsys.checkpoint
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `requests` | LLM API calls |
| `pandas` | DataFrame construction and parquet I/O |
| `pyarrow` | Parquet backend |
| `datasets` | HuggingFace dataset streaming |
| `huggingface_hub` | Authentication for gated datasets |
| `pyyaml` | Loading config.yaml |
