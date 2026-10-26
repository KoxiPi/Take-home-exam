"""Correct known duplicated validation log points; no training or rescoring."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/sft_sepfix_20261022_171041_876248/metrics/version_0/metrics.csv"
OLD = ROOT / "analysis/experiment_evidence_20261023_111852_802371"
OUT = OLD / ("sft_curve_correction_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
with SOURCE.open(newline="") as f:
    rows = list(csv.DictReader(f))
observed = [(int(float(r["step"])), float(r["val/nll"]))
            for r in rows if r.get("val/nll")]
# These exclusions were verified from this run's log and on_train_epoch_end.
# Do not generalize equality of values to duplication in other experiments.
expected_steps = [199, 399, 599, 673, 800, 999]
assert [s for s,v in observed] == expected_steps, "Log changed; review required."
by_step = dict(observed)
assert by_step[673] == by_step[599]
assert by_step[999] == by_step[800]
excluded = {673: "Epoch-end re-log of value at logged step 599",
            999: "Epoch-end re-log of value at logged step 800"}
kept = [(s,v) for s,v in observed if s not in excluded]
assert min(kept, key=lambda p:p[1])[0] == 399
OUT.mkdir(parents=True, exist_ok=False)
training = {}
for r in rows:
    if r.get("trainer/loss"):
        training.setdefault(int(float(r["step"])), []).append(float(r["trainer/loss"]))
steps = sorted(training)
loss = [sum(training[s])/len(training[s]) for s in steps]
smooth = [sum(loss[max(0,i-24):i+1])/len(loss[max(0,i-24):i+1]) for i in range(len(loss))]
plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
fig,axs = plt.subplots(1,2,figsize=(11,4))
axs[0].plot(steps,loss,color="#90cdf4",alpha=.5,lw=.6,label="Within-step mean logged loss")
axs[0].plot(steps,smooth,color="#2b6cb0",label="Trailing 25-step mean")
axs[0].set(xlabel="CSV logged step (unmodified)",ylabel="Logged training loss",title="Corrected SFT training")
axs[0].legend(fontsize=8)
axs[1].plot([s for s,v in kept],[v for s,v in kept],"o-",color="#319795")
axs[1].axvline(399,color="#718096",ls="--",label="Selected checkpoint: optimizer step 400")
axs[1].set(xlabel="CSV logged step (unmodified)",ylabel="Validation NLL",title="SFT validation: duplicate logs excluded")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.text(.5,-.025,"CSV step retained as logged; checkpoint step numbering is recorded separately.",ha="center",fontsize=9)
for suffix in ["png","pdf"]:
    fig.savefig(OUT/("02_sft_curves_corrected."+suffix),dpi=200,bbox_inches="tight")
plt.close(fig)
audit = {
    "source":str(SOURCE), "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "kept_logged_steps":[s for s,v in kept],
    "excluded_logged_steps":excluded,
    "selected_checkpoint_optimizer_step":400,
    "selected_validation_logged_step":399,
    "reason":"on_train_epoch_end logs existing valid_nlls without a new validation pass.",
    "axis_policy":"Use raw CSV step; do not assume every validation log is optimizer step minus one.",
    "original_logs_and_figures_preserved":True,"output_dir":str(OUT)
}
(OUT/"correction_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print("SFT CURVE CORRECTION FINISHED")
print(json.dumps(audit,indent=2))
