"""Two-update MDLM diffu-GRPO adaptation smoke test; not a final RL experiment."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")
import ast
import copy
import hashlib
import json
import random
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
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import main as project_main
import algo
import dataloader
import sandbox_gsm8k

SFT_RUN = ROOT / "outputs/sft_sepfix_20261022_171041_876248"
DATA = ROOT / "data/local/gsm8k_rl_pilot"
RUN = ROOT / "outputs" / ("rl_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
SETTINGS = dict(seed=42, group_size=8, sampling_steps=32, inner_updates=2,
                max_questions=16, learning_rate=1e-6, clip_epsilon=0.2,
                kl_beta=0.01, prompt_mask_probability=0.15,
                gradient_clip_norm=1.0, sequence_length=512)
RUN.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, RUN / "train_rl_smoke.py")
def write_json(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
sft = json.loads((SFT_RUN / "summary.json").read_text())
checkpoint = Path(sft["best_checkpoint"])
assert checkpoint.name == "best-step000400.ckpt" and checkpoint.is_file()
assert sft["separator_token_ids"] == [76, 94]
meta = json.loads((DATA / "metadata.json").read_text())
raw = (DATA / "train.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == meta["output_sha256"]["train.json"]
questions = json.loads(raw)[:SETTINGS["max_questions"]]
assert len(questions) == 16
shutil.copy2(DATA / "metadata.json", RUN / "data_metadata.json")
write_json("settings.json", SETTINGS)
write_json("method.json", {
    "name": "MDLM adaptation of diffu-GRPO",
    "source": "https://arxiv.org/html/2504.12216v2",
    "probability_estimator": "One-step unmasking; completion fully masked; shared random prompt mask.",
    "advantage": "reward minus group mean; no standard-deviation normalization",
    "regularizer": "Exact categorical KL(current||fixed reference) of the one-step surrogate, averaged over suffix positions.",
    "approximation": "Surrogate probabilities are NOT exact 32-step generated sequence probabilities.",
    "completion_scope": "All generated suffix positions through length 512; no EOS truncation, matching existing full-suffix scorer.",
    "initialization": "Materialize SFT EMA once; no EMA swaps or EMA averaging during RL.",
    "export": "Current RL weights mirrored to EMA slot for compatibility with project evaluation.",
    "smoke_selection": "First mixed-reward group among first 16 training questions; diagnostic only, not a representative training run.",
    "changes_from_paper": "MDLM 167M, code-execution binary reward, fixed 512-token full sequence, exact surrogate categorical KL, two inner updates.",
})
L.seed_everything(42)
with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
    cfg = compose(config_name="config", overrides=[
        "data=gsm8k-test", "model=small", "algo=mdlm", "sampler=ancestral"])
cfg.seed = 42
cfg.model.length = 512
cfg.data.tokenizer_name_or_path = str(SFT_RUN / "tokenizer")
cfg.sampler.steps = 32
cfg.loader.eval_batch_size = 2
cfg.trainer.devices = 1
cfg.trainer.num_nodes = 1
cfg.eval.checkpoint_path = str(checkpoint)
cfg.eval.disable_ema = False
assert not cfg.algo.time_conditioning
assert cfg.data.separator == sft["separator"]
OmegaConf.save(cfg, RUN / "config.yaml", resolve=True)
tokenizer = dataloader.get_tokenizer(cfg)
policy = algo.MDLM.load_from_checkpoint(str(checkpoint), config=cfg,
    tokenizer=tokenizer, map_location="cpu", strict=True)
assert policy.ema is not None, "Expected SFT EMA"
# Copy without store/restore: these become the actual trainable initial weights.
policy.ema.copy_to(policy._get_parameters())
policy.ema = None
policy.fast_forward_epochs = None
policy.fast_forward_batches = None
policy.eval().to("cuda")
reference = copy.deepcopy(policy).eval()
reference.requires_grad_(False)
for p, q in zip(policy.parameters(), reference.parameters()):
    assert torch.equal(p, q)
optimizer = torch.optim.AdamW(policy.parameters(), lr=SETTINGS["learning_rate"], weight_decay=0.0)
# Independent generator: prompt masks do not alter the generation RNG stream.
mask_rng = torch.Generator(device="cuda").manual_seed(4242)
sep = tokenizer.encode(cfg.data.separator, add_special_tokens=False)
assert sep == [76, 94]
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()

# Mechanical loss-direction test, independent of model quality.
toy = torch.zeros(2, requires_grad=True)
toy_adv = torch.tensor([0.5, -0.5])
(-(toy.exp() * toy_adv).mean()).backward()
assert toy.grad[0] < 0 and toy.grad[1] > 0
for code, expected in [("def simple_math_problem():\n    return 3", True),
                       ("def simple_math_problem():\n    return 4", False)]:
    assert bool(sandbox_gsm8k.evaluate_samples(code, "#### 3", 1.0)) == expected
print("Initialization, loss-direction and scorer checks PASSED", flush=True)

def surrogate(model, xt, prefix_len):
    # MDLM is not time-conditioned. Renormalize in float32 for stable ratios/KL.
    lp = model.forward(xt=xt, sigma=torch.zeros((1, 1), device="cuda"))[0, prefix_len:].float()
    keep = torch.arange(model.vocab_size, device="cuda") != model.mask_index
    # Exclude the forbidden mask token identically for all three policies.
    lp = lp[:, keep]
    lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
    return lp

def selected(lp, targets):
    return lp.gather(1, targets.transpose(0, 1)).transpose(0, 1)

groups = []
updates = []
global_step = 0
for qi, row in enumerate(questions):
    ids = [tokenizer.bos_token_id] + tokenizer.encode(row["prompt"].strip(),
        add_special_tokens=False) + sep
    prefix_len = len(ids)
    assert prefix_len <= 384
    prefix = torch.tensor([ids, ids], device="cuda")
    lens = torch.full((2,), prefix_len, device="cuda", dtype=torch.long)
    sequences, texts, rewards = [], [], []
    with torch.no_grad():
        for _ in range(4):
            samples, _ = policy.generate_samples(num_samples=2, num_steps=32,
                prefix_tokens=prefix, prefix_lengths=lens)
            assert torch.equal(samples[:, :prefix_len], prefix)
            sequences.append(samples.detach().cpu())
            for text in tokenizer.batch_decode(samples[:, prefix_len:].cpu(), skip_special_tokens=True):
                texts.append(text)
                rewards.append(int(sandbox_gsm8k.evaluate_samples(
                    text, row["response_ground_truth"], timeout_s=5.0)))
    groups.append(dict(source_index=row["source_index"], prompt=row["prompt"],
        response_ground_truth=row["response_ground_truth"], responses=texts, rewards=rewards))
    write_json("records.json", groups)
    print(f"Training-side question {qi+1}: {sum(rewards)}/8 correct", flush=True)
    if not 0 < sum(rewards) < 8:
        continue
    full_sequences = torch.cat(sequences).to("cuda")
    suffix = full_sequences[:, prefix_len:]
    assert not (suffix == policy.mask_index).any()
    # Vocabulary compaction after removing mask_index.
    targets = suffix - (suffix > policy.mask_index).long()
    advantage = torch.tensor(rewards, device="cuda", dtype=torch.float32)
    advantage -= advantage.mean()
    assert abs(advantage.sum().item()) < 1e-6
    # Cache old-policy estimates BEFORE any update, with exactly the masks reused below.
    old_cache = []
    for _ in range(SETTINGS["inner_updates"]):
        xt = torch.full((1, 512), policy.mask_index, device="cuda", dtype=torch.long)
        xt[0, :prefix_len] = torch.tensor(ids, device="cuda")
        mask = torch.rand(prefix_len, device="cuda", generator=mask_rng) < 0.15
        # Protect BOS and the two-token separator; perturb only question tokens.
        mask[0] = False
        mask[-len(sep):] = False
        xt[0, :prefix_len][mask] = policy.mask_index
        with torch.no_grad():
            old_lp = selected(surrogate(policy, xt, prefix_len), targets).detach()
        old_cache.append((xt, old_lp))
    torch.save({"tokens": full_sequences.cpu(), "rewards": rewards,
                "masked_inputs": [x.cpu() for x, _ in old_cache]},
               RUN / "smoke_rollout.pt")
    for inner, (xt, old_lp) in enumerate(old_cache):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            ref_lp = surrogate(reference, xt, prefix_len)
        lp = surrogate(policy, xt, prefix_len)
        chosen = selected(lp, targets)
        logratio = chosen - old_lp
        assert torch.isfinite(logratio).all()
        ratio = logratio.exp()
        assert torch.isfinite(ratio).all()
        if global_step == 0:
            assert logratio.abs().max().item() < 1e-5, "Initial ratio is not 1"
        pg_loss = -torch.minimum(ratio * advantage[:, None],
            ratio.clamp(0.8, 1.2) * advantage[:, None]).mean()
        kl = (lp.exp() * (lp - ref_lp)).sum(-1).mean()
        assert kl.item() >= -1e-5
        if global_step == 0:
            assert abs(kl.item()) < 1e-5, "Initial reference differs"
        loss = pg_loss + 0.01 * kl
        assert torch.isfinite(loss)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0,
                                                   error_if_nonfinite=True)
        assert grad_norm.item() > 0
        record = dict(step=global_step+1, source_index=row["source_index"],
            reward_mean=sum(rewards)/8, pg_loss=pg_loss.item(),
            surrogate_categorical_kl=kl.item(), loss=loss.item(),
            grad_norm_before_clip=grad_norm.item(),
            ratio_min=ratio.min().item(), ratio_max=ratio.max().item(),
            clip_fraction=((ratio-1).abs() > 0.2).float().mean().item())
        optimizer.step()
        global_step += 1
        updates.append(record)
        write_json("updates.json", updates)
        print("RL UPDATE:", json.dumps(record), flush=True)
        del lp, chosen, ref_lp, loss, pg_loss, kl, ratio, logratio
    break

if global_step != 2:
    write_json("summary.json", dict(status="no_mixed_group", optimizer_steps=global_step,
        questions_checked=len(groups), output_dir=str(RUN)))
    raise SystemExit("No mixed group in bounded smoke sample; no successful RL update claimed.")

max_delta = 0.0
changed_tensors = 0
with torch.no_grad():
    for (name, p), (_, ref_p) in zip(policy.named_parameters(), reference.named_parameters()):
        delta = (p - ref_p).abs().max().item()
        max_delta = max(max_delta, delta)
        changed_tensors += int(delta > 0)
assert max_delta > 0 and changed_tensors > 0
# Export a minimal Lightning-compatible evaluation checkpoint, never overwrite SFT.
original = torch.load(checkpoint, map_location="cpu", weights_only=False)
ema_export = original["ema"]
ema_export["shadow_params"] = [p.detach().cpu().clone() for p in policy._get_parameters()]
ema_export["num_updates"] = 0
cpu_sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
payload = {
    "state_dict": cpu_sd, "ema": ema_export,
    "hyper_parameters": original["hyper_parameters"],
    "pytorch-lightning_version": L.__version__,
    "global_step": global_step, "epoch": 0,
    "loops": {"fit_loop": {"epoch_progress": {"current": {"completed": 0}},
              "epoch_loop.batch_progress": {"current": {"completed": 0}}}},
    "rl_metadata": {"settings": SETTINGS, "initial_checkpoint": str(checkpoint),
                    "exported_weights": "current RL weights; EMA slot mirrors current weights"},
}
out_checkpoint = RUN / "rl_smoke.ckpt"
torch.save(payload, out_checkpoint)
torch.save({"optimizer": optimizer.state_dict(), "step": global_step,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(), "mask_rng": mask_rng.get_state()},
           RUN / "optimizer_state.pt")
del original, payload, cpu_sd, ema_export
# Verify the same project loading/EMA path used for evaluation.
reloaded = algo.MDLM.load_from_checkpoint(str(out_checkpoint), config=cfg,
    tokenizer=tokenizer, map_location="cpu", strict=True)
reloaded._eval_mode()
with torch.no_grad():
    for name, value in policy.state_dict().items():
        assert torch.equal(value.detach().cpu(), reloaded.state_dict()[name]), name
del reloaded
torch.cuda.synchronize()
write_json("summary.json", dict(
    purpose="RL two-update smoke test, not final RL training",
    status="passed", initial_checkpoint=str(checkpoint), initial_weights="SFT EMA",
    optimizer_steps=global_step, questions_checked=len(groups),
    selected_group_correct=sum(rewards), selected_group_size=8,
    changed_parameter_tensors=changed_tensors, max_parameter_change=max_delta,
    checkpoint_reload_exact=True, fit_seconds=time.perf_counter()-started,
    peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
    checkpoint=str(out_checkpoint), output_dir=str(RUN)))
print("RL SMOKE PASSED", flush=True)
print((RUN / "summary.json").read_text(), flush=True)
