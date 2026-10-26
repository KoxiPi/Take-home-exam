import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")

import hashlib
import json
import shutil
import sys
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

SFT_RUN = ROOT / "outputs/sft_run_20261022_163517_240301"
sft_summary = json.loads(
    (SFT_RUN / "summary.json").read_text(encoding="utf-8")
)
checkpoint = Path(sft_summary["best_checkpoint"])
assert checkpoint.is_file()

data_dir = ROOT / "data/local/gsm8k_rl_pilot"
metadata = json.loads(
    (data_dir / "metadata.json").read_text(encoding="utf-8")
)
raw = (data_dir / "train.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == (
    metadata["output_sha256"]["train.json"]
)
questions = json.loads(raw)[:16]
assert len(questions) == 16

out_dir = ROOT / "outputs" / (
    "rl_reward_probe_sepfix_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
out_dir.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, out_dir / "check_rl_rewards.py")
shutil.copy2(data_dir / "metadata.json", out_dir / "data_metadata.json")

# 检查评分器的正确、错误、执行失败和超时分支。
tests = [
    ("def simple_math_problem():\n    return 3", "#### 3", True),
    ("def simple_math_problem():\n    return 10", "#### 3", False),
    ("def simple_math_problem():\n    return missing_name", "#### 3", False),
    ("def simple_math_problem():\n    while True:\n        pass", "#### 3", False),
]
for code, gold, expected in tests:
    actual = bool(sandbox_gsm8k.evaluate_samples(code, gold, 1.0))
    assert actual == expected, (expected, actual)
print("Reward scorer checks PASSED", flush=True)

L.seed_everything(42)
with initialize_config_dir(
    version_base=None, config_dir=str(ROOT / "configs")
):
    cfg = compose(
        config_name="config",
        overrides=[
            "data=gsm8k-test",
            "model=small",
            "algo=mdlm",
            "sampler=ancestral",
        ],
    )

cfg.seed = 42
cfg.model.length = 512
cfg.data.tokenizer_name_or_path = str(SFT_RUN / "tokenizer")
cfg.loader.eval_batch_size = 2
cfg.sampler.steps = 32
cfg.trainer.devices = 1
cfg.trainer.num_nodes = 1
cfg.eval.checkpoint_path = str(checkpoint)
cfg.eval.disable_ema = False
OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

tokenizer = dataloader.get_tokenizer(cfg)
model = algo.MDLM.load_from_checkpoint(
    str(checkpoint),
    config=cfg,
    tokenizer=tokenizer,
    map_location="cpu",
    strict=True,
)
model.to("cuda")
model._eval_mode()  # 与 SFT 测试评估一样，使用 EMA 权重。

sep = tokenizer.encode(cfg.data.separator, add_special_tokens=False)
print("Configured separator:", repr(cfg.data.separator), flush=True)
print("Separator token IDs:", sep, flush=True)
groups = []

with torch.no_grad():
    for question_index, row in enumerate(questions):
        prompt_ids = (
            [tokenizer.bos_token_id]
            + tokenizer.encode(row["prompt"].strip(),
                               add_special_tokens=False)
            + sep
        )
        assert len(prompt_ids) <= 384, (
            f"第 {question_index} 题提示过长，需要另行处理"
        )

        responses = []
        rewards = []
        for _ in range(4):
            prefix = torch.tensor(
                [prompt_ids, prompt_ids],
                dtype=torch.long,
                device="cuda",
            )
            lengths = torch.full(
                (2,), len(prompt_ids), dtype=torch.long, device="cuda"
            )
            samples, _ = model.generate_samples(
                num_samples=2,
                num_steps=32,
                prefix_tokens=prefix,
                prefix_lengths=lengths,
            )
            texts = tokenizer.batch_decode(
                samples[:, len(prompt_ids):].cpu(),
                skip_special_tokens=True,
            )
            for text in texts:
                reward = int(sandbox_gsm8k.evaluate_samples(
                    text,
                    row["response_ground_truth"],
                    timeout_s=5.0,
                ))
                responses.append(text)
                rewards.append(reward)
            del samples

        group = {
            "source_index": row["source_index"],
            "prompt": row["prompt"],
            "gold_numeric": row["gold_numeric"],
            "response_ground_truth": row["response_ground_truth"],
            "responses": responses,
            "rewards": rewards,
        }
        groups.append(group)
        # 每完成一题保存一次，异常时保留已完成结果。
        (out_dir / "records.json").write_text(
            json.dumps(groups, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Question {question_index + 1}/16: "
            f"{sum(rewards)}/8 correct",
            flush=True,
        )

total = sum(len(g["rewards"]) for g in groups)
correct = sum(sum(g["rewards"]) for g in groups)
mixed = sum(0 < sum(g["rewards"]) < 8 for g in groups)

summary = {
    "purpose": "reward signal diagnostic; no parameter updates",
    "checkpoint": str(checkpoint),
    "weights": "EMA",
    "seed": 42,
    "sampling_steps": 32,
    "questions": len(groups),
    "samples_per_question": 8,
    "total_samples": total,
    "correct_samples": correct,
    "sample_accuracy": correct / total,
    "questions_with_any_correct": sum(
        sum(g["rewards"]) > 0 for g in groups
    ),
    "mixed_reward_groups": mixed,
    "all_wrong_groups": sum(sum(g["rewards"]) == 0 for g in groups),
    "all_correct_groups": sum(sum(g["rewards"]) == 8 for g in groups),
    "output_dir": str(out_dir),
}
(out_dir / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2), flush=True)
