import ast
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[2]
folder = root / "data/local/tinygsm_sft_pilot"

if any(
    (folder / name).exists()
    for name in ("train.json", "valid.json", "prepare_metadata.json")
):
    raise SystemExit("处理结果已存在，停止以避免覆盖。")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text):
    return " ".join(
        unicodedata.normalize("NFKC", text).casefold().split()
    )


tokenizer_name = "HuggingFaceTB/SmolLM-135M"
revision = HfApi().model_info(tokenizer_name).sha
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    revision=revision,
)

raw_path = folder / "raw.json"
rows = read_json(raw_path)

test_path = root / "data/gsm8k_test.json"
test_questions = {
    normalize(row["prompt"])
    for row in read_json(test_path)
}

seen = set()
accepted = []
counts = Counter()

for row in rows:
    counts["input"] += 1
    question = row["question"].strip()
    code = row["code"].strip()
    key = normalize(question)

    if not question or not code:
        counts["empty"] += 1
        continue

    if key in test_questions:
        counts["exact_test_overlap"] += 1
        continue

    if key in seen:
        counts["duplicate_question"] += 1
        continue

    try:
        tree = ast.parse(code)
        compile(tree, "<candidate>", "exec")
    except (SyntaxError, ValueError, TypeError):
        counts["invalid_syntax"] += 1
        continue

    has_function = any(
        isinstance(node, ast.FunctionDef)
        and node.name == "simple_math_problem"
        for node in tree.body
    )
    if not has_function:
        counts["missing_function"] += 1
        continue

    length = 2 + sum(
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in (question, "\n", code)
    )
    if length > 512:
        counts["over_512_tokens"] += 1
        continue

    seen.add(key)
    accepted.append({
        "question": question,
        "code": code,
    })

assert len(accepted) >= 10, "可用样本不足"

random.Random(42).shuffle(accepted)
n_valid = max(1, round(len(accepted) * 0.1))
valid = accepted[:n_valid]
train = accepted[n_valid:]

train_questions = {normalize(row["question"]) for row in train}
valid_questions = {normalize(row["question"]) for row in valid}
assert not (train_questions & valid_questions)

hashes = {}
for name, records in (
    ("train.json", train),
    ("valid.json", valid),
):
    path = folder / name
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

metadata = {
    "counts": dict(counts),
    "accepted": len(accepted),
    "train": len(train),
    "valid": len(valid),
    "split_seed": 42,
    "max_tokens": 512,
    "tokenizer": tokenizer_name,
    "tokenizer_revision": revision,
    "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    "test_file_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    "output_sha256": hashes,
    "limitations": [
        "Mathematical correctness not verified.",
        "Only normalized exact question overlap is excluded.",
        "Validation may have been seen during original pretraining.",
    ],
}

(folder / "prepare_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
