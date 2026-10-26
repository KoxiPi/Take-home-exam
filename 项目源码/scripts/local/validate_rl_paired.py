"""Prespecified paired validation of fixed SFT and RL checkpoints; inference only."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
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

SFT_RUN = ROOT / "outputs/sft_sepfix_20261022_171041_876248"
RL_RUN = ROOT / "outputs/rl_pilot64_20261022_174012_898690"
DATA = ROOT / "data/local/gsm8k_rl_pilot"
SEEDS = [202, 303, 404]
CHECKPOINTS = {
    "sft": SFT_RUN / "checkpoints/best-step000400.ckpt",
    "rl": RL_RUN / "group016.ckpt",
}
for path in CHECKPOINTS.values():
    assert path.is_file(), str(path)
rl_summary = json.loads((RL_RUN / "summary.json").read_text())
assert str(CHECKPOINTS["rl"]) == rl_summary["best_rl_checkpoint"]
meta = json.loads((DATA / "metadata.json").read_text())
raw = (DATA / "valid.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == meta["output_sha256"]["valid.json"]
rows = json.loads(raw)
assert len(rows) == 128
assert len({r["source_index"] for r in rows}) == 128
RUN = ROOT / "outputs" / ("rl_paired_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
RUN.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, RUN / "validate_rl_paired.py")
shutil.copy2(DATA / "metadata.json", RUN / "data_metadata.json")
def write_json(name, obj):
    path = RUN / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
write_json("protocol.json", {
    "purpose": "Inference-only paired validation of fixed checkpoints",
    "seeds": SEEDS, "questions": 128, "batch_size": 1,
    "sampling_steps": 32, "sequence_length": 512,
    "checkpoints": {k: str(v) for k,v in CHECKPOINTS.items()},
    "weights": {"sft": "SFT EMA", "rl": "Current RL parameters mirrored in EMA export"},
    "selection": "group016 selected previously with validation seed123; no selection using these new seeds.",
    "validation_sha256": hashlib.sha256(raw).hexdigest(),
    "limitations": [
        "Same 128 validation questions reused; not a new held-out dataset.",
        "Three samples/question/model are not 384 independent questions.",
        "No formal significance or generalization claim from these small paired runs."
    ],
})
L.seed_everything(42)
with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
    cfg = compose(config_name="config", overrides=[
        "data=gsm8k-test", "model=small", "algo=mdlm", "sampler=ancestral"])
cfg.model.length = 512
cfg.data.tokenizer_name_or_path = str(SFT_RUN / "tokenizer")
cfg.sampler.steps = 32
cfg.loader.eval_batch_size = 1
cfg.trainer.devices = 1
cfg.trainer.num_nodes = 1
cfg.eval.disable_ema = False
tokenizer = dataloader.get_tokenizer(cfg)
sep = tokenizer.encode(cfg.data.separator, add_special_tokens=False)
assert sep == [76, 94]
prompts = [[tokenizer.bos_token_id] + tokenizer.encode(
    r["prompt"].strip(), add_special_tokens=False) + sep for r in rows]
assert max(map(len, prompts)) <= 384
OmegaConf.save(cfg, RUN / "config.yaml", resolve=True)
models = {}
for label, path in CHECKPOINTS.items():
    model = algo.MDLM.load_from_checkpoint(str(path), config=cfg,
        tokenizer=tokenizer, map_location="cpu", strict=True)
    assert model.ema is not None
    model.ema.copy_to(model._get_parameters())
    model.ema = None
    model.eval().requires_grad_(False).to("cuda")
    models[label] = model

started = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
results = {}
summary_rows = []
for seed in SEEDS:
    for label, model in models.items():
        records = []
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                for i, (row, ids) in enumerate(zip(rows, prompts)):
                    prefix = torch.tensor([ids], device="cuda", dtype=torch.long)
                    lens = torch.tensor([len(ids)], device="cuda", dtype=torch.long)
                    samples, _ = model.generate_samples(num_samples=1, num_steps=32,
                        prefix_tokens=prefix, prefix_lengths=lens)
                    assert torch.equal(samples[:, :len(ids)], prefix)
                    text = tokenizer.decode(samples[0, len(ids):].cpu(), skip_special_tokens=True)
                    reward = int(sandbox_gsm8k.evaluate_samples(
                        text, row["response_ground_truth"], timeout_s=5.0))
                    records.append(dict(source_index=row["source_index"], prompt=row["prompt"],
                        response_ground_truth=row["response_ground_truth"],
                        response=text, correct=reward))
                    if (i+1) % 32 == 0:
                        print(f"{label} seed={seed}: {i+1}/128, correct={sum(x['correct'] for x in records)}",
                              flush=True)
                        write_json(f"{label}_seed{seed}.json", dict(
                            model=label, seed=seed, complete=False, records=records))
        results[(seed,label)] = records
        write_json(f"{label}_seed{seed}.json", dict(model=label, seed=seed,
            complete=True, num_correct=sum(r["correct"] for r in records), records=records))
    sft_rows, rl_rows = results[(seed,"sft")], results[(seed,"rl")]
    assert [r["source_index"] for r in sft_rows] == [r["source_index"] for r in rl_rows]
    sft_correct = sum(r["correct"] for r in sft_rows)
    rl_correct = sum(r["correct"] for r in rl_rows)
    gained = sum(a["correct"] == 0 and b["correct"] == 1 for a,b in zip(sft_rows, rl_rows))
    lost = sum(a["correct"] == 1 and b["correct"] == 0 for a,b in zip(sft_rows, rl_rows))
    assert gained-lost == rl_correct-sft_correct
    entry = dict(seed=seed, sft_correct=sft_correct, rl_correct=rl_correct,
        num_questions=128, gained=gained, lost=lost, net=rl_correct-sft_correct)
    summary_rows.append(entry)
    print("PAIRED RESULT:", json.dumps(entry), flush=True)
    write_json("paired_results.json", summary_rows)
per_question=[]
for index,row in enumerate(rows):
    a=sum(results[(s,"sft")][index]["correct"] for s in SEEDS)
    b=sum(results[(s,"rl")][index]["correct"] for s in SEEDS)
    per_question.append(dict(source_index=row["source_index"], sft_correct_of_3=a,
        rl_correct_of_3=b, net=b-a))
write_json("per_question.json", per_question)
torch.cuda.synchronize()
sft_total=sum(r["sft_correct"] for r in summary_rows)
rl_total=sum(r["rl_correct"] for r in summary_rows)
summary=dict(status="completed", purpose="Fixed-checkpoint paired validation; no training",
    seeds=SEEDS, unique_questions=128, samples_per_model=384,
    sft_correct_total=sft_total, rl_correct_total=rl_total,
    sft_mean_accuracy=sft_total/384, rl_mean_accuracy=rl_total/384,
    net_correct=rl_total-sft_total, paired_results=summary_rows,
    seconds=time.perf_counter()-started,
    peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
    output_dir=str(RUN),
    interpretation="Repeated sampling on the same validation questions; not GSM8K test scores.")
write_json("summary.json", summary)
print("PAIRED VALIDATION FINISHED", flush=True)
print(json.dumps(summary, indent=2), flush=True)
