import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from decimal import Decimal
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/local/gsm8k_rl_pilot"
if OUT.exists():
    raise SystemExit("输出目录已存在，停止以避免覆盖。")

def normalize(text):
    return " ".join(
        unicodedata.normalize("NFKC", text).casefold().split()
    )

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

excluded = set()
exclusion_hashes = {}

sources = [
    (ROOT / "data/gsm8k_test.json", "prompt"),
    (ROOT / "data/local/tinygsm_sft_pilot/train.json", "question"),
    (ROOT / "data/local/tinygsm_sft_pilot/valid.json", "question"),
]
for path, key in sources:
    content = path.read_bytes()
    exclusion_hashes[str(path.relative_to(ROOT))] = (
        hashlib.sha256(content).hexdigest()
    )
    excluded.update(normalize(row[key]) for row in json.loads(content))

repo = "openai/gsm8k"
revision = HfApi().dataset_info(repo).sha

# 数据集较小，完整读取 train split，不使用远程流式迭代。
dataset = load_dataset(
    repo, "main", split="train",
    revision=revision, streaming=False,
)

counts = Counter()
seen = set()
records = []

for index, row in enumerate(dataset):
    counts["input"] += 1
    question = row["question"].strip()
    answer = row["answer"].strip()
    key = normalize(question)

    if key in excluded:
        counts["excluded_exact_overlap"] += 1
        continue
    if key in seen:
        counts["duplicate_question"] += 1
        continue

    match = re.search(r"####\s*(.*?)\s*$", answer)
    if not match:
        counts["missing_final_answer"] += 1
        continue

    numeric = match.group(1).replace(",", "").strip()
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric):
        counts["unsupported_answer_format"] += 1
        continue
    assert Decimal(numeric).is_finite()

    seen.add(key)
    records.append({
        "source_index": index,
        "prompt": question,
        "response_ground_truth": answer,
        "gold_numeric": numeric,
    })

random.Random(42).shuffle(records)
assert len(records) >= 640, "可用样本不足"

valid = records[:128]
train = records[128:640]
reserve = records[640:]

assert not (
    {normalize(r["prompt"]) for r in train}
    & {normalize(r["prompt"]) for r in valid}
)

OUT.mkdir(parents=True)
hashes = {}

for filename, rows in [
    ("train.json", train),
    ("valid.json", valid),
    ("reserve.json", reserve),
]:
    path = OUT / filename
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()

metadata = {
    "dataset": repo,
    "subset": "main",
    "source_split": "train",
    "revision": revision,
    "seed": 42,
    "counts": dict(counts),
    "eligible": len(records),
    "rl_train": len(train),
    "rl_valid": len(valid),
    "reserve": len(reserve),
    "output_sha256": hashes,
    "exclusion_file_sha256": exclusion_hashes,
    "limitations": [
        "Only normalized exact question overlaps are excluded.",
        "Original pretraining exposure is unknown.",
        "Prompt token lengths will be checked before generation.",
    ],
}
(OUT / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
