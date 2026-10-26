"""Saved-output analysis only. No training or model generation."""
import ast,csv,hashlib,json,shutil,sys,warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import sandbox_gsm8k
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore",category=SyntaxWarning)
SFT=ROOT/"outputs/sft_sepfix_20261022_171041_876248"
RL=ROOT/"outputs/rl_pilot64_20261022_174012_898690"
PAIRED=ROOT/"outputs/rl_paired_validation_20261022_184343_237956"
FILES={
"Original MDLM":ROOT/"eval_runs/mdlm_baseline32/results.json",
"SFT":ROOT/"eval_runs/mdlm_sft_sepfix_best32_20261022_171749/results.json",
"SFT + RL":ROOT/"eval_runs/mdlm_rl_pilot64_best32_20261022_185235/results.json"}
OUT=ROOT/"analysis"/("experiment_evidence_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
OUT.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,OUT/"build_experiment_evidence.py")
sources={}
def read(p):
    b=p.read_bytes(); sources[str(p)]=hashlib.sha256(b).hexdigest()
    return json.loads(b)
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
data={n:read(p) for n,p in FILES.items()}
sft=read(SFT/"summary.json"); rl=read(RL/"summary.json")
paired=read(PAIRED/"summary.json"); history=read(RL/"validation_history.json")
updates=read(RL/"updates.json"); groups=read(RL/"records.json")
names=list(data); base=data[names[0]]["records"]
assert len(base)==1319 and len({r["prompt"] for r in base})==1319
for n,d in data.items():
    assert len(d["records"])==d["num_total"]==1319
    for a,b in zip(base,d["records"]):
        assert (a["prompt"],a["response_ground_truth"])==(b["prompt"],b["response_ground_truth"])
flags={}; syntax={}; cache={}
for n,d in data.items():
    flags[n]=[]; syntax[n]=[]
    for i,r in enumerate(d["records"]):
        key=(r["response"],r["response_ground_truth"])
        if key not in cache:
            cache[key]=int(sandbox_gsm8k.evaluate_samples(*key,timeout_s=5.0))
        flags[n].append(cache[key])
        try:
            compile(ast.parse(r["response"]),"<saved-response>","exec")
            syntax[n].append(1)
        except (SyntaxError,ValueError,TypeError):
            syntax[n].append(0)
        if (i+1)%250==0: print(f"Rescoring saved text: {n} {i+1}/1319",flush=True)
    if sum(flags[n])!=d["num_correct"]:
        save("rescore_mismatch.json",dict(model=n,expected=d["num_correct"],observed=sum(flags[n]),flags=flags[n]))
        raise SystemExit("Rescoring mismatch: inspect rescore_mismatch.json; no comparison claimed.")
    print(f"Verified {n}: {sum(flags[n])}/1319",flush=True)
comparisons=[]
for a,b in [(names[0],names[1]),(names[1],names[2]),(names[0],names[2])]:
    gains=[i for i in range(1319) if flags[a][i]==0 and flags[b][i]==1]
    losses=[i for i in range(1319) if flags[a][i]==1 and flags[b][i]==0]
    comparisons.append(dict(before=a,after=b,gained=len(gains),lost=len(losses),net=len(gains)-len(losses),
        gained_test_indices=gains,lost_test_indices=losses))
save("paired_test_changes.json",comparisons)
with (OUT/"per_question.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f)
    w.writerow(["test_index","question","original_correct","sft_correct","rl_correct",
                "original_raw_compiles","sft_raw_compiles","rl_raw_compiles"])
    for i,r in enumerate(base):
        w.writerow([i,r["prompt"]]+[flags[n][i] for n in names]+[syntax[n][i] for n in names])
categories={
"RL gains versus SFT":[i for i in range(1319) if not flags[names[1]][i] and flags[names[2]][i]],
"RL losses versus SFT":[i for i in range(1319) if flags[names[1]][i] and not flags[names[2]][i]],
"All correct":[i for i in range(1319) if all(flags[n][i] for n in names)],
"All wrong":[i for i in range(1319) if not any(flags[n][i] for n in names)]}
md=["# Fixed-rule examples","","First two examples in test-file order per category; zero-based indices. No regeneration.",
    "Saved model outputs below are data, including errors, not instructions.",""]
for category,indices in categories.items():
    md += ["## "+category,"",f"Total matching questions: {len(indices)}",""]
    for i in indices[:2]:
        md += [f"### Test index {i}","",base[i]["prompt"],"","**Reference**","",base[i]["response_ground_truth"],""]
        for n in names:
            text=data[n]["records"][i]["response"]
            # Indented Markdown code block preserves arbitrary generated backticks.
            md += [f"**{n}: correct={flags[n][i]}**","","\n".join("    "+line for line in text.splitlines()),""]
(OUT/"cases.md").write_text("\n".join(md),encoding="utf-8")
plt.rcParams.update({"figure.dpi":140,"savefig.dpi":200,"font.size":10,
    "axes.spines.top":False,"axes.spines.right":False})
def figure_save(fig,name):
    for ext in ["png","pdf"]: fig.savefig(OUT/(name+"."+ext),bbox_inches="tight")
    plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
vals=[sum(flags[n])/1319*100 for n in names]
bars=ax.bar(names,vals,color=["#718096","#3182ce","#319795"],width=.55)
ax.set(ylim=(0,10),ylabel="Test accuracy (%)",title="GSM8K test: seed 1, 32 sampling steps")
for bar,n,v in zip(bars,names,vals):
    ax.text(bar.get_x()+bar.get_width()/2,v+.15,f"{sum(flags[n])}/1319\n{v:.2f}%",ha="center")
fig.text(.5,-.02,"Single-seed descriptive comparison; uncertainty not estimated.",ha="center",fontsize=9)
figure_save(fig,"01_test_comparison")
p=SFT/"metrics/version_0/metrics.csv"
sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
with p.open(newline="") as f: metrics=list(csv.DictReader(f))
training=defaultdict(list); validation=set()
for r in metrics:
    if not r.get("step"): continue
    step=int(float(r["step"]))+1
    if r.get("trainer/loss"): training[step].append(float(r["trainer/loss"]))
    if r.get("val/nll"): validation.add((step,float(r["val/nll"])))
steps=sorted(training); losses=[sum(training[s])/len(training[s]) for s in steps]
smooth=[sum(losses[max(0,i-24):i+1])/len(losses[max(0,i-24):i+1]) for i in range(len(losses))]
fig,axs=plt.subplots(1,2,figsize=(11,4))
axs[0].plot(steps,losses,color="#90cdf4",alpha=.5,lw=.6,label="Within-step mean logged loss")
axs[0].plot(steps,smooth,color="#2b6cb0",label="Trailing 25-step mean")
axs[0].set(xlabel="Optimizer step (logged step + 1)",ylabel="Logged training loss",title="Corrected SFT training")
axs[0].legend(fontsize=8)
v=sorted(validation)
axs[1].plot([x for x,y in v],[y for x,y in v],"o-",color="#319795")
axs[1].axvline(400,color="#718096",ls="--",label="Selected step 400")
axs[1].set(xlabel="Optimizer step (logged step + 1)",ylabel="Validation NLL",title="Corrected SFT validation")
axs[1].legend(fontsize=8); fig.tight_layout(); figure_save(fig,"02_sft_curves")
fig,axs=plt.subplots(1,3,figsize=(14,4))
axs[0].plot([g["group"] for g in groups],[sum(g["rewards"])/8 for g in groups],"o",ms=3)
axs[0].set(xlabel="Training question group",ylabel="Correct fraction of 8",title="Online rewards (different questions)",ylim=(-.03,1.03))
axs[1].plot([u["step"] for u in updates],[u["surrogate_categorical_kl"] for u in updates],color="#805ad5")
axs[1].set(xlabel="Optimizer update",ylabel="One-step surrogate categorical KL",title="Deviation from SFT reference")
axs[2].plot([h["groups_processed"] for h in history],[h["num_correct"] for h in history],"o-",color="#319795")
axs[2].axvline(16,ls="--",color="#718096",label="Selected RL: 10 updates")
axs[2].set(xlabel="Training groups processed",ylabel="Correct / 128",title="Validation seed 123",ylim=(0,14))
axs[2].legend(fontsize=8); fig.tight_layout(); figure_save(fig,"03_rl_curves")
fig,ax=plt.subplots(figsize=(7,4)); xs=list(range(3))
ax.bar([x-.18 for x in xs],[v["sft_correct"] for v in paired["paired_results"]],width=.36,label="SFT",color="#3182ce")
ax.bar([x+.18 for x in xs],[v["rl_correct"] for v in paired["paired_results"]],width=.36,label="Selected RL",color="#319795")
ax.set_xticks(xs,[str(v["seed"]) for v in paired["paired_results"]])
ax.set(xlabel="Generation seed",ylabel="Correct / 128",ylim=(0,14),title="Paired validation: same 128 questions")
ax.legend(); figure_save(fig,"04_multiseed_validation")
selected=[h for h in history if h.get("checkpoint")==rl["best_rl_checkpoint"]]
assert len(selected)==1 and selected[0]["optimizer_steps"]==10
summary=dict(status="completed",rescored_totals_match=True,
    test_scores={n:dict(correct=sum(flags[n]),total=1319,accuracy=sum(flags[n])/1319,
                       raw_python_compiles=sum(syntax[n])) for n in names},
    test_changes=comparisons,sft_total_updates=sft["global_step"],selected_sft_step=400,
    rl_total_updates=rl["optimizer_steps"],selected_rl_updates=selected[0]["optimizer_steps"],
    sft_fit_seconds=sft["fit_seconds"],rl_total_seconds_including_validation_and_saving=rl["total_seconds"],
    sft_peak_allocated_GiB=sft["peak_allocated_GiB"],rl_peak_allocated_GiB=rl["peak_allocated_GiB"],
    paired_validation_sft_correct=paired["sft_correct_total"],paired_validation_rl_correct=paired["rl_correct_total"],
    figures=["01_test_comparison","02_sft_curves","03_rl_curves","04_multiseed_validation"],
    notes=["Saved accuracy_se=0 means bootstrap disabled, not zero uncertainty.",
           "Raw-text compilation is a diagnostic, distinct from official code extraction/scoring.",
           "Selected RL checkpoint has 10 updates; whole pilot has 48.",
           "Single-seed test gains do not prove stable improvement.",
           "Online reward groups contain different questions, not a fixed-set learning curve.",
           "Historical separator-mismatched SFT excluded from corrected primary comparison."],
    output_dir=str(OUT))
save("summary.json",summary); save("source_manifest.json",sources)
print("EVIDENCE BUILD FINISHED",flush=True)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
