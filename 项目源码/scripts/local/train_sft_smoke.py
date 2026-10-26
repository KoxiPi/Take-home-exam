import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from datasets import Dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import main as project_main  # 注册仓库配置使用的 resolvers
import algo
import dataloader

DATA = ROOT / "data/local/tinygsm_sft_pilot"
CKPT = ROOT / "checkpoints/tinygsm/mdlm.ckpt"
RUN = ROOT / "outputs" / (
    "sft_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
RUN.mkdir(parents=True, exist_ok=False)

L.seed_everything(42, workers=True)
assert torch.cuda.is_available()
assert CKPT.is_file()

with initialize_config_dir(
    version_base=None, config_dir=str(ROOT / "configs")
):
    cfg = compose(
        config_name="config",
        overrides=[
            "data=tiny-gsm",
            "model=small",
            "algo=mdlm",
            "sampler=ancestral",
        ],
    )

cfg.seed = 42
cfg.model.length = 512
cfg.data.tokenizer_name_or_path = "HuggingFaceTB/SmolLM-135M"
cfg.data.wrap = False
cfg.data.streaming = False
cfg.data.train_on_prompt = False
cfg.data.train_on_pad = False
cfg.data.filter_too_long = True
cfg.loader.batch_size = 2
cfg.loader.global_batch_size = 2
cfg.loader.eval_batch_size = 2
cfg.loader.num_workers = 1
cfg.trainer.devices = 1
cfg.trainer.num_nodes = 1
cfg.trainer.accumulate_grad_batches = 1
cfg.trainer.max_steps = 20
cfg.trainer.precision = "bf16-mixed"
cfg.trainer.limit_val_batches = 4
cfg.trainer.num_sanity_val_steps = 2
cfg.trainer.val_check_interval = 20
cfg.trainer.check_val_every_n_epoch = None
cfg.trainer.log_every_n_steps = 1
cfg.optim.lr = 1e-5
cfg.lr_scheduler.num_warmup_steps = 2
cfg.eval.generate_samples = False
cfg.compile_backbone = False
cfg.training.finetune_path = str(CKPT)
cfg.checkpointing.resume_from_ckpt = False
cfg.checkpointing.save_dir = str(RUN)

metadata = json.loads(
    (DATA / "prepare_metadata.json").read_text(encoding="utf-8")
)

# 下载准备数据时固定的 tokenizer 版本，再用仓库逻辑补充特殊 token。
from transformers import AutoTokenizer
tokenizer_dir = RUN / "tokenizer"
AutoTokenizer.from_pretrained(
    metadata["tokenizer"],
    revision=metadata["tokenizer_revision"],
).save_pretrained(tokenizer_dir)
cfg.data.tokenizer_name_or_path = str(tokenizer_dir)
tokenizer = dataloader.get_tokenizer(cfg)

def build_dataset(filename, expected_count):
    path = DATA / filename
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == (
        metadata["output_sha256"][filename]
    ), f"数据文件发生变化: {filename}"

    rows = json.loads(content)
    assert len(rows) == expected_count
    inputs, masks = [], []
    sep = tokenizer.encode("\n", add_special_tokens=False)

    for row in rows:
        q = tokenizer.encode(row["question"].strip(),
                             add_special_tokens=False)
        a = tokenizer.encode(row["code"].strip(),
                             add_special_tokens=False)
        prefix = [tokenizer.bos_token_id] + q + sep
        answer = a + [tokenizer.eos_token_id]
        ids = prefix + answer
        assert len(ids) <= 512, "发现超长样本，停止以避免静默截断"
        pad = 512 - len(ids)

        mask = [0] * len(prefix) + [1] * len(answer) + [0] * pad
        ids += [tokenizer.pad_token_id] * pad
        assert len(ids) == len(mask) == 512
        assert sum(mask) == len(answer) > 0
        inputs.append(ids)
        masks.append(mask)

    print(f"{filename}: {len(rows)} rows; answer/EOS-only mask OK",
          flush=True)
    return Dataset.from_dict({
        "input_ids": inputs,
        "attention_mask": masks,
    }).with_format("torch")

train_set = build_dataset("train.json", metadata["train"])
valid_set = build_dataset("valid.json", metadata["valid"])

train_loader = DataLoader(
    train_set, batch_size=2, shuffle=True,
    num_workers=1, persistent_workers=True, pin_memory=True,
)
valid_loader = DataLoader(
    valid_set, batch_size=2, shuffle=False,
    num_workers=1, pin_memory=True,
)
train_loader.tokenizer = tokenizer
valid_loader.tokenizer = tokenizer

print("Loading pretrained checkpoint:", CKPT, flush=True)
model = algo.MDLM.load_from_checkpoint(
    str(CKPT), config=cfg, tokenizer=tokenizer,
    map_location="cpu", strict=True,
)
# 开始新的微调阶段，不恢复预训练的数据位置或优化器进度。
model.fast_forward_epochs = None
model.fast_forward_batches = None

# 选取一个较小的参数张量，检查训练后是否发生更新。
probe_name, probe = next(
    (name, p) for name, p in model.backbone.named_parameters()
    if p.requires_grad and 100 <= p.numel() <= 1_000_000
)
before = probe.detach().cpu().clone()

class FiniteLossCheck(L.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs,
                           batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is not None and not torch.isfinite(loss).all():
            raise RuntimeError("训练 loss 出现 NaN 或 Inf")

OmegaConf.save(cfg, RUN / "config.yaml", resolve=True)
shutil.copy2(__file__, RUN / "train_sft_smoke.py")
shutil.copy2(DATA / "prepare_metadata.json",
             RUN / "data_metadata.json")
(RUN / "environment-lock.txt").write_text(
    subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
)
(RUN / "code-version.txt").write_text(
    subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
)
(RUN / "code-changes.patch").write_text(
    subprocess.check_output(["git", "diff"], text=True)
)

logger = CSVLogger(save_dir=str(RUN), name="metrics", version=0)
trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    num_nodes=1,
    precision="bf16-mixed",
    max_steps=20,
    accumulate_grad_batches=1,
    gradient_clip_val=1.0,
    num_sanity_val_steps=2,
    limit_val_batches=4,
    val_check_interval=20,
    check_val_every_n_epoch=None,
    log_every_n_steps=1,
    enable_checkpointing=False,
    logger=logger,
    callbacks=[FiniteLossCheck()],
    default_root_dir=str(RUN),
)

torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
trainer.fit(model, train_loader, valid_loader, ckpt_path=None)
torch.cuda.synchronize()
elapsed = time.perf_counter() - started

delta = (probe.detach().cpu() - before).abs().max().item()
assert trainer.global_step == 20
assert delta > 0, "检查的参数没有变化，需要排查训练"

saved = RUN / "sft_smoke.ckpt"
trainer.save_checkpoint(str(saved))

summary = {
    "purpose": "pipeline smoke test, not final SFT",
    "initial_checkpoint": str(CKPT),
    "global_step": trainer.global_step,
    "batch_size": 2,
    "learning_rate": 1e-5,
    "warmup_steps": 2,
    "train_rows": len(train_set),
    "valid_rows": len(valid_set),
    "validation_batches_per_check": 4,
    "probe_parameter": probe_name,
    "probe_max_abs_change": delta,
    "fit_seconds": elapsed,
    "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
    "checkpoint": str(saved),
}
(RUN / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("\nSMOKE TEST FINISHED", flush=True)
print(json.dumps(summary, indent=2), flush=True)
