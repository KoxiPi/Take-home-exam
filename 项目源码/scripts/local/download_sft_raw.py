import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

root = Path(__file__).resolve().parents[2]
out = root / "data/local/tinygsm_sft_pilot"
out.mkdir(parents=True, exist_ok=True)
raw_path = out / "raw.json"

if raw_path.exists():
    raise SystemExit("raw.json 已存在，停止以避免覆盖。")

repo = "TinyGSM/TinyGSM"
revision = HfApi().dataset_info(repo).sha
dataset = load_dataset(
    repo, split="train", revision=revision, streaming=True
)
rows = [
    {"question": r["question"], "code": r["code"]}
    for r in dataset.take(12000)
]
assert len(rows) == 12000, f"仅读取到 {len(rows)} 条"

raw_path.write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
)
metadata = {
    "dataset": repo,
    "revision": revision,
    "source_split": "train",
    "selection": "first 12000 rows; not a random sample",
    "num_rows": len(rows),
    "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
}
(out / "download_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Saved {len(rows)} rows to {raw_path}", flush=True)
