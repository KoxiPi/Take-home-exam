"""Fixed-budget 64-question MDLM diffu-GRPO pilot with independent validation."""
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
RUN = ROOT / "outputs" / ("rl_pilot64_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
SETTINGS = dict(seed=42, group_size=8, sampling_steps=32, inner_updates=2,
                max_questions=64, validation_seed=123, validate_every_groups=16, learning_rate=1e-6, clip_epsilon=0.2,
                kl_beta=0.01, prompt_mask_probability=0.15,
                gradient_clip_norm=1.0, sequence_length=512)
RUN.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, RUN / "train_rl_pilot.py")
def write_json(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
sft = json.loads((SFT_RUN / "summary.json").read_text())
checkpoint = Path(sft["best_checkpoint"])
assert checkpoint.name == "best-step000400.ckpt" and checkpoint.is_file()
assert sft["separator_token_ids"] == [76, 94]
meta = json.loads((DATA / "metadata.json").read_text())
raw = (DATA / "train.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == meta["output_sha256"]["train.json"]
all_questions = json.loads(raw)
assert len(all_questions) == 512
questions = random.Random(42).sample(all_questions, SETTINGS["max_questions"])
assert len(questions) == 64
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
    "training_selection": "Fixed random sample of 64/512 training questions, seed 42; all equal-reward groups skip both policy and KL-only updates.",
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

# Fixed validation protocol: one sample/question, batch size 1, identical RNG seed.
valid_raw = (DATA / "valid.json").read_bytes()
assert hashlib.sha256(valid_raw).hexdigest() == meta["output_sha256"]["valid.json"]
validation_rows = json.loads(valid_raw)
assert len(validation_rows) == 128
assert not ({r["source_index"] for r in json.loads(raw)} &
            {r["source_index"] for r in validation_rows})
def prompt_ids(row):
    return [tokenizer.bos_token_id] + tokenizer.encode(row["prompt"].strip(),
        add_special_tokens=False) + sep
for row in questions + validation_rows:
    assert len(prompt_ids(row)) <= 384, "Prompt exceeds predeclared budget"
write_json("split_manifest.json", dict(
    training_source_indices=[r["source_index"] for r in questions],
    validation_source_indices=[r["source_index"] for r in validation_rows],
    training_file_sha256=hashlib.sha256(raw).hexdigest(),
    validation_file_sha256=hashlib.sha256(valid_raw).hexdigest()))
(RUN / "environment-lock.txt").write_text(subprocess.check_output(
    [sys.executable, "-m", "pip", "freeze"], text=True))
(RUN / "code-version.txt").write_text(subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True))
(RUN / "code-changes.patch").write_text(subprocess.check_output(
    ["git", "diff"], text=True))
assert shutil.disk_usage(RUN).free > 10 * 1024**3, "Need at least 10 GiB free for checkpoints"
original = torch.load(checkpoint, map_location="cpu", weights_only=False)
hyperparameters = original["hyper_parameters"]
ema_header = {k: v for k, v in original["ema"].items()
              if k not in ("shadow_params", "collected_params")}
del original

def save_checkpoint(name, group_count, step):
    path = RUN / name
    assert not path.exists()
    ema_export = dict(ema_header)
    ema_export["shadow_params"] = [p.detach().cpu().clone() for p in policy._get_parameters()]
    ema_export["collected_params"] = None
    ema_export["num_updates"] = 0
    payload = dict(
        state_dict={k: v.detach().cpu() for k, v in policy.state_dict().items()},
        ema=ema_export, hyper_parameters=hyperparameters,
        **{"pytorch-lightning_version": L.__version__},
        global_step=step, epoch=0,
        loops={"fit_loop": {"epoch_progress": {"current": {"completed": 0}},
               "epoch_loop.batch_progress": {"current": {"completed": 0}}}},
        rl_metadata=dict(settings=SETTINGS, initial_checkpoint=str(checkpoint),
            groups_processed=group_count, exported_weights="Current RL weights mirrored in EMA slot"))
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.rename(path)
    return path

def evaluate(label, group_count, step):
    before_cpu = torch.get_rng_state().clone()
    before_gpu = torch.cuda.get_rng_state().clone()
    records = []
    began = time.perf_counter()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(SETTINGS["validation_seed"])
        torch.cuda.manual_seed_all(SETTINGS["validation_seed"])
        with torch.no_grad():
            for vi, row in enumerate(validation_rows):
                ids = prompt_ids(row)
                prefix = torch.tensor([ids], device="cuda", dtype=torch.long)
                lengths = torch.tensor([len(ids)], device="cuda", dtype=torch.long)
                samples, _ = policy.generate_samples(num_samples=1, num_steps=32,
                    prefix_tokens=prefix, prefix_lengths=lengths)
                assert torch.equal(samples[:, :len(ids)], prefix)
                text = tokenizer.decode(samples[0, len(ids):].cpu(), skip_special_tokens=True)
                reward = int(sandbox_gsm8k.evaluate_samples(
                    text, row["response_ground_truth"], timeout_s=5.0))
                records.append(dict(source_index=row["source_index"], prompt=row["prompt"],
                    response_ground_truth=row["response_ground_truth"], response=text, correct=reward))
                if (vi + 1) % 32 == 0:
                    print(f"VALIDATION {label}: {vi+1}/128; correct={sum(x['correct'] for x in records)}", flush=True)
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_gpu, torch.cuda.get_rng_state())
    correct = sum(x["correct"] for x in records)
    result = dict(label=label, groups_processed=group_count, optimizer_steps=step,
        num_correct=correct, num_total=128, accuracy=correct/128,
        seed=SETTINGS["validation_seed"], batch_size=1,
        seconds=time.perf_counter()-began, records=records)
    write_json(f"validation_{label}.json", result)
    print(f"VALIDATION COMPLETE {label}: {correct}/128 ({correct/128:.2%})", flush=True)
    return result

baseline = evaluate("before_rl", 0, 0)
history = [{k: v for k, v in baseline.items() if k != "records"}]
write_json("validation_history.json", history)
groups, updates = [], []
global_step = 0
mixed_groups = 0
best_path = None
best_correct = -1
best_result = None

for qi, row in enumerate(questions):
    ids = prompt_ids(row)
    prefix_len = len(ids)
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
    is_mixed = 0 < sum(rewards) < 8
    groups.append(dict(group=qi+1, source_index=row["source_index"], prompt=row["prompt"],
        response_ground_truth=row["response_ground_truth"], responses=texts,
        rewards=rewards, mixed=is_mixed, optimizer_steps_before=global_step))
    write_json("records.json", groups)
    full_sequences = torch.cat(sequences)
    rollout = dict(tokens=full_sequences, rewards=rewards, source_index=row["source_index"],
                   optimizer_steps_before=global_step)
    print(f"TRAIN GROUP {qi+1}/64: {sum(rewards)}/8 correct; mixed={is_mixed}", flush=True)
    if is_mixed:
        mixed_groups += 1
        suffix = full_sequences.to("cuda")[:, prefix_len:]
        assert not (suffix == policy.mask_index).any()
        targets = suffix - (suffix > policy.mask_index).long()
        advantage = torch.tensor(rewards, device="cuda", dtype=torch.float32)
        advantage -= advantage.mean()
        assert abs(advantage.sum().item()) < 1e-6
        old_cache = []
        for _ in range(SETTINGS["inner_updates"]):
            xt = torch.full((1, 512), policy.mask_index, device="cuda", dtype=torch.long)
            xt[0, :prefix_len] = torch.tensor(ids, device="cuda")
            mask = torch.rand(prefix_len, device="cuda", generator=mask_rng) < SETTINGS["prompt_mask_probability"]
            mask[0] = False
            mask[-len(sep):] = False
            xt[0, :prefix_len][mask] = policy.mask_index
            with torch.no_grad():
                old_lp = selected(surrogate(policy, xt, prefix_len), targets).detach()
            old_cache.append((xt, old_lp))
        rollout["masked_inputs"] = [x.cpu() for x, _ in old_cache]
        rollout["old_selected_log_probs"] = [p.cpu() for _, p in old_cache]
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
            if inner == 0:
                assert logratio.abs().max().item() < 1e-5, "Old policy mismatch at start of group"
            eps = SETTINGS["clip_epsilon"]
            pg_loss = -torch.minimum(ratio * advantage[:, None],
                ratio.clamp(1-eps, 1+eps) * advantage[:, None]).mean()
            kl = (lp.exp() * (lp - ref_lp)).sum(-1).mean()
            assert kl.item() >= -1e-5
            if global_step == 0:
                assert abs(kl.item()) < 1e-5
            loss = pg_loss + SETTINGS["kl_beta"] * kl
            assert torch.isfinite(loss)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),
                SETTINGS["gradient_clip_norm"], error_if_nonfinite=True)
            assert grad_norm.item() > 0
            record = dict(step=global_step+1, group=qi+1, source_index=row["source_index"],
                reward_mean=sum(rewards)/8, pg_loss=pg_loss.item(),
                surrogate_categorical_kl=kl.item(), loss=loss.item(),
                grad_norm_before_clip=grad_norm.item(),
                ratio_min=ratio.min().item(), ratio_max=ratio.max().item(),
                clip_fraction=((ratio-1).abs() > eps).float().mean().item())
            optimizer.step()
            global_step += 1
            updates.append(record)
            write_json("updates.json", updates)
            print("RL UPDATE:", json.dumps(record), flush=True)
            del lp, chosen, ref_lp, loss, pg_loss, kl, ratio, logratio
        del old_cache, xt, old_lp, targets, suffix, advantage
    torch.save(rollout, RUN / f"rollout_group{qi+1:03d}.pt")
    del rollout, full_sequences, sequences
    if (qi + 1) % SETTINGS["validate_every_groups"] == 0:
        label = f"group{qi+1:03d}"
        # Saving does not consume generation randomness.
        path = save_checkpoint(f"{label}.ckpt", qi+1, global_step)
        val = evaluate(label, qi+1, global_step)
        entry = {k: v for k, v in val.items() if k != "records"}
        entry["checkpoint"] = str(path)
        history.append(entry)
        write_json("validation_history.json", history)
        if val["num_correct"] > best_correct:
            best_correct, best_path, best_result = val["num_correct"], path, val
        write_json("progress.json", dict(groups_processed=qi+1, optimizer_steps=global_step,
            mixed_groups=mixed_groups, best_rl_checkpoint=str(best_path),
            baseline_correct=baseline["num_correct"], best_rl_correct=best_correct))
        torch.save(dict(optimizer=optimizer.state_dict(), step=global_step, groups_processed=qi+1,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(), mask_rng=mask_rng.get_state()),
            RUN / f"optimizer_group{qi+1:03d}.pt")

assert len(groups) == 64 and global_step == 2 * mixed_groups
assert global_step > 0, "No RL updates occurred"
assert all(p.grad is None for p in reference.parameters())
# Check final checkpoint equality through official project loading + EMA application.
final_path = RUN / "group064.ckpt"
reloaded = algo.MDLM.load_from_checkpoint(str(final_path), config=cfg,
    tokenizer=tokenizer, map_location="cpu", strict=True)
reloaded._eval_mode()
with torch.no_grad():
    loaded_sd = reloaded.state_dict()
    for name, value in policy.state_dict().items():
        assert torch.equal(value.detach().cpu(), loaded_sd[name]), name
    max_delta = max((p-q).abs().max().item() for p,q in zip(policy.parameters(),reference.parameters()))
assert max_delta > 0
del reloaded, loaded_sd
baseline_flags = [r["correct"] for r in baseline["records"]]
best_flags = [r["correct"] for r in best_result["records"]]
gained = sum(a == 0 and b == 1 for a,b in zip(baseline_flags,best_flags))
lost = sum(a == 1 and b == 0 for a,b in zip(baseline_flags,best_flags))
assert gained - lost == best_correct - baseline["num_correct"]
torch.cuda.synchronize()
summary = dict(
    purpose="Fixed-budget MDLM diffu-GRPO adaptation pilot; validation only",
    status="completed", initial_checkpoint=str(checkpoint), initial_weights="SFT EMA",
    training_questions=64, training_samples=512, mixed_groups=mixed_groups,
    all_wrong_groups=sum(sum(g["rewards"]) == 0 for g in groups),
    all_correct_groups=sum(sum(g["rewards"]) == 8 for g in groups),
    optimizer_steps=global_step, validation_questions=128,
    baseline_validation_correct=baseline["num_correct"],
    best_rl_validation_correct=best_correct, best_rl_validation_accuracy=best_correct/128,
    validation_gained=gained, validation_lost=lost,
    best_rl_checkpoint=str(best_path), final_checkpoint=str(final_path),
    selection_rule="Maximum validation correct count among groups 16/32/48/64; earliest wins ties.",
    improved_over_sft=best_correct > baseline["num_correct"],
    checkpoint_reload_exact=True, max_parameter_change=max_delta,
    total_seconds=time.perf_counter()-started,
    peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
    output_dir=str(RUN),
    limitations=["Single validation seed; small validation set; selection introduces optimism.",
                 "All-equal reward groups skipped, including their KL-only updates.",
                 "One-step surrogate probabilities, not exact trajectory likelihood.",
                 "Pilot uses 64 of 512 training questions; not a full-scale RL run."])
write_json("summary.json", summary)
print("RL PILOT FINISHED", flush=True)
print(json.dumps(summary, indent=2), flush=True)
