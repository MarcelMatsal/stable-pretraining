"""This script demonstrates how to retrieve data from wandb using the stable_pretraining library and create plots from it."""

import matplotlib.pyplot as plt
from tqdm import tqdm

import stable_pretraining as spt

entity = "rbalestr-brown"
project = "clip_caption_injection"

# ── Config ─────────────────────────────────────────────────────────────────────
wanted_dataset = "uoft-cs/cifar10"
wanted_zeroshot_dataset = "uoft-cs/cifar10"
target_spur_proportion = 0.5  # <-- set your desired spurious proportion here

lora_ranks = [1, 16, 32, 64]
degradation_results = {rank: {"spurious_proportion": [], "degradation": []} for rank in lora_ranks}

# ── Data retrieval ─────────────────────────────────────────────────────────────
configs, dfs = spt.utils.log_reader.read_wandb_project(
    entity=entity, project=project, filters={"state": "finished"}
)

for run_id, df in tqdm(dfs.items(), desc="Processing runs", unit="run"):
    dataset = df.get("dataset", None)
    zeroshot_dataset = df.get("zeroshot_dataset", None)

    if (
        wanted_dataset.lower() in dataset.lower()
        and wanted_zeroshot_dataset.lower() in zeroshot_dataset.lower()
    ):
        spurious_proportion = df.get("spur_proportion", None)
        lora_rank = df.get("lora_rank", None)
        use_spurious = df.get("use_spurious", None)
        use_lora = df.get("use_lora", None)
        epochs = df.get("epochs", None)
        spur_type = df.get("spur_type", None)

        if (
            spurious_proportion is not None
            and spurious_proportion == target_spur_proportion  # filter by chosen proportion
            and lora_rank is not None
            and lora_rank in lora_ranks
            and use_spurious
            and epochs == 50
            and spur_type == "patch"
            and use_lora
        ):
            new_df, config = spt.utils.log_reader.read_wandb_run(entity, project, run_id)

            spur_top1_series = new_df["val/zeroshot_eval_spur_top1"].dropna()
            clean_top1_series = new_df["val/zeroshot_eval_clean_top1"].dropna()

            if not spur_top1_series.empty and not clean_top1_series.empty:
                spur_top1 = spur_top1_series.iloc[-1]
                clean_top1 = clean_top1_series.iloc[-1]
                degradation = clean_top1 - spur_top1

                degradation_results[lora_rank]["spurious_proportion"].append(spurious_proportion)
                degradation_results[lora_rank]["degradation"].append(degradation)

# ── Plot: Accuracy Degradation vs LoRA Rank ────────────────────────────────────
# Average degradation per rank in case there are multiple runs
ranks_with_data = []
avg_degradations = []

for rank in lora_ranks:
    vals = degradation_results[rank]["degradation"]
    if vals:
        ranks_with_data.append(rank)
        avg_degradations.append(sum(vals) / len(vals))

plt.figure(figsize=(8, 6))
plt.plot(
    ranks_with_data,
    avg_degradations,
    "-o",
    color="#1f77b4",
    linewidth=2,
    markersize=8,
)

plt.xlabel("LoRA Rank", fontsize=14)
plt.ylabel("Accuracy Degradation (Clean Top-1 − Spurious Top-1)", fontsize=14)
plt.title(
    f"Accuracy Degradation vs LoRA Rank\n(Spurious Proportion = {target_spur_proportion})",
    fontsize=16,
)
plt.xticks(ranks_with_data, labels=[str(r) for r in ranks_with_data])
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("accuracy_degradation_vs_lora_rank.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: accuracy_degradation_vs_lora_rank.png")