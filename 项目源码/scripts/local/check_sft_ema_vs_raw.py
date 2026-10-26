import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")

import sys
import json
import hashlib
import shutil
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import lightning as L
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import main as project_main
import algo
import dataloader
import sandbox_gsm8k

SFT = ROOT / "outputs/sft_sepfix_20261022_171041_876248"
CKPT = SFT / "checkpoints/best-step000400.ckpt"
DATA = ROOT / "data/local/gsm8k_rl_pilot"
SEEDS = [202, 303, 404]

assert CKPT.is_file(), str(CKPT)
raw = (DATA / "valid.json").read_bytes()
meta = json.loads((DATA / "metadata.json").read_text())
assert hashlib.sha256(raw).hexdigest() == meta["output_sha256"]["valid.json"]
rows = json.loads(raw)
assert len(rows) == 128
assert len({r["source_index"] for r in rows}) == 128

RUN = ROOT / "outputs" / (
    "sft_ema_vs_raw_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
RUN.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, RUN / Path(__file__).name)

def save(name, value):
    path = RUN / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    temp.replace(path)

save("protocol.json", {
    "checkpoint": str(CKPT),
    "seeds": SEEDS,
    "questions": 128,
    "sampling_steps": 32,
    "batch_size": 1,
    "sequence_length": 512,
    "validation_sha256": hashlib.sha256(raw).hexdigest(),
    "purpose": "EMA versus stored non-EMA weights; no training",
    "limitations": [
        "Previously used validation questions and seeds.",
        "384 samples per weight variant are not 384 independent questions.",
        "Exploratory diagnostic, not a final test evaluation."
    ]
})

L.seed_everything(42)
with initialize_config_dir(version_base=None,
                           config_dir=str(ROOT / "configs")):
    cfg = compose(config_name="config", overrides=[
        "data=gsm8k-test", "model=small", "algo=mdlm",
        "sampler=ancestral"
    ])

# Only reuse the model/sampler configuration; questions come from valid.json.
cfg.model.length = 512
cfg.data.tokenizer_name_or_path = str(SFT / "tokenizer")
cfg.sampler.steps = 32
cfg.loader.eval_batch_size = 1
cfg.trainer.devices = 1
cfg.trainer.num_nodes = 1
cfg.eval.disable_ema = False

tokenizer = dataloader.get_tokenizer(cfg)
sep = tokenizer.encode(cfg.data.separator, add_special_tokens=False)
assert sep == [76, 94], sep
prompts = [
    [tokenizer.bos_token_id]
    + tokenizer.encode(r["prompt"].strip(), add_special_tokens=False)
    + sep
    for r in rows
]
assert max(map(len, prompts)) <= 384
OmegaConf.save(cfg, RUN / "config.yaml", resolve=True)

models = {}
for label in ("ema", "raw"):
    model = algo.MDLM.load_from_checkpoint(
        str(CKPT), config=cfg, tokenizer=tokenizer,
        map_location="cpu", strict=True
    )
    assert model.ema is not None
    if label == "ema":
        model.ema.copy_to(model._get_parameters())
    # Prevent subsequent automatic EMA replacement for both variants.
    model.ema = None
    model.eval().requires_grad_(False).to("cuda")
    models[label] = model

with torch.no_grad():
    max_diff = max(
        (a - b).abs().max().item()
        for a, b in zip(models["ema"]._get_parameters(),
                        models["raw"]._get_parameters())
    )
save("weight_check.json", {"max_abs_difference": max_diff})
print("EMA vs raw maximum parameter difference:", max_diff, flush=True)
assert max_diff > 0, (
    "Stored weights equal EMA; stop and inspect checkpoint saving behavior."
)

started = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
paired = []

for seed in SEEDS:
    results = {}
    for label, model in models.items():
        records = []
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                for i, (row, ids) in enumerate(zip(rows, prompts)):
                    prefix = torch.tensor([ids], device="cuda",
                                          dtype=torch.long)
                    lengths = torch.tensor([len(ids)], device="cuda",
                                           dtype=torch.long)
                    samples, _ = model.generate_samples(
                        num_samples=1, num_steps=32,
                        prefix_tokens=prefix, prefix_lengths=lengths
                    )
                    assert torch.equal(samples[:, :len(ids)], prefix)
                    response = tokenizer.decode(
                        samples[0, len(ids):].cpu(),
                        skip_special_tokens=True
                    )
                    correct = int(sandbox_gsm8k.evaluate_samples(
                        response, row["response_ground_truth"], timeout_s=5.0
                    ))
                    records.append({
                        "source_index": row["source_index"],
                        "prompt": row["prompt"],
                        "response_ground_truth": row["response_ground_truth"],
                        "response": response,
                        "correct": correct
                    })
                    if (i + 1) % 32 == 0:
                        print(
                            f"{label} seed={seed}: {i+1}/128, "
                            f"correct={sum(r['correct'] for r in records)}",
                            flush=True
                        )
                        save(f"{label}_seed{seed}.json", {
                            "complete": i + 1 == len(rows),
                            "records": records
                        })
        results[label] = records

    a, b = results["ema"], results["raw"]
    entry = {
        "seed": seed,
        "ema_correct": sum(r["correct"] for r in a),
        "raw_correct": sum(r["correct"] for r in b),
        "raw_gained": sum(x["correct"] == 0 and y["correct"] == 1
                          for x, y in zip(a, b)),
        "raw_lost": sum(x["correct"] == 1 and y["correct"] == 0
                        for x, y in zip(a, b))
    }
    paired.append(entry)
    save("paired_results.json", paired)
    print("PAIRED:", json.dumps(entry), flush=True)

torch.cuda.synchronize()
ema_total = sum(r["ema_correct"] for r in paired)
raw_total = sum(r["raw_correct"] for r in paired)
summary = {
    "status": "completed",
    "checkpoint": str(CKPT),
    "unique_questions": 128,
    "samples_per_variant": 384,
    "seeds": SEEDS,
    "ema_correct": ema_total,
    "raw_correct": raw_total,
    "ema_accuracy": ema_total / 384,
    "raw_accuracy": raw_total / 384,
    "raw_minus_ema": raw_total - ema_total,
    "paired_results": paired,
    "seconds": time.perf_counter() - started,
    "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
    "output_dir": str(RUN)
}
save("summary.json", summary)
print("\nEMA COMPARISON FINISHED", flush=True)
print(json.dumps(summary, indent=2), flush=True)
