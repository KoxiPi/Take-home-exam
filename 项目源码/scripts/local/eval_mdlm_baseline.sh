#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# 每次执行使用独立输出目录，保留之前的实验结果。
RUN_DIR="$REPO_ROOT/eval_runs/mdlm_baseline32_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

git rev-parse HEAD > "$RUN_DIR/code-version.txt"
git diff > "$RUN_DIR/code-changes.patch"
python -m pip freeze > "$RUN_DIR/environment-lock.txt"

python -u -m main \
  mode=gsm8k_eval \
  seed=1 \
  eval.checkpoint_path="$REPO_ROOT/checkpoints/tinygsm/mdlm.ckpt" \
  data=gsm8k-test \
  data.data_path="$REPO_ROOT/data/gsm8k_test.json" \
  data.cache_dir="$REPO_ROOT/data_cache/gsm8k_full" \
  data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
  model=small \
  model.length=512 \
  algo=mdlm \
  sampler=ancestral \
  sampler.steps=32 \
  loader.eval_batch_size=2 \
  loader.num_workers=1 \
  trainer.num_nodes=1 \
  trainer.devices=1 \
  gsm8k.output_dir="$RUN_DIR" \
  +wandb.offline=True \
  hydra.run.dir="$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/console.log"
