"""This script demonstrates how to retrieve data from wandb using the stable_pretraining library and create plots from it.

To use, you should set the entity variable to your WandB entity and the project variable to the specific project within
your WandB entity that you want to access runs from.

"""

import matplotlib.pyplot as plt
from tqdm import tqdm

import stable_pretraining as spt

entity = "rbalestr-brown"
project = "clip_caption_injection"


# want to retrieve finished runs from wandb
configs, dfs = spt.utils.log_reader.read_wandb_project(
    entity=entity, project=project, filters={"state": "finished"}
)

# Here you would define wanted conditions that you can use to narrow down the WandB runs, if you want to access all your runs
# in the project then you would not define anything here and would remove the first if statement in the for loop
# access all runs with the wanted dataset and model backbone
wanted_dataset = "uoft-cs/cifar10"
wanted_zeroshot_dataset = "uoft-cs/cifar10"

# Dictionary to store results for the degradation plot
# keyed by lora_rank -> lists of (spurious_proportion, degradation)
lora_ranks = [1, 16, 32, 64]
degradation_results = {rank: {"spurious_proportion": [], "degradation": []} for rank in lora_ranks}

# Dictionary to store baseline clean top-1 accuracy per LoRA rank at spur_proportion=0.05
baseline_spur_proportion = 0
baseline_results = {rank: None for rank in lora_ranks}
# no_lora_baseline = 0.972599983215332

# Iterate through runs and gather information from WandB
for run_id, df in tqdm(dfs.items(), desc="Processing runs", unit="run"):
    dataset = df.get("dataset", None)
    zeroshot_dataset = df.get("zeroshot_dataset", None)

    # make sure the ones we are using met the conditions for what we want to graph
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
        text_spur = df.get("text_spur", None)

        if (
            use_spurious
            and spurious_proportion == 0.05
            and lora_rank is not None
            and lora_rank in lora_ranks
            and epochs == 50
            and use_lora
        ):
            # Extract run data
            new_df, config = spt.utils.log_reader.read_wandb_run(entity, project, run_id)

            # Get last epoch values for spur_top1 and clean_top1
            clean_top1_series = new_df["val/zeroshot_eval_clean_top1"].dropna()

            # if  not clean_top1_series.empty:
            #     clean_top1 = clean_top1_series.iloc[-1]

            #     # Degradation: clean_top1 - spur_top1 (positive = degradation)
            #     degradation = clean_top1
            #     degradation_results[lora_rank]["degradation"].append(degradation)

            #     # Collect baseline clean top-1 at spurious_proportion == 0.05
            #     baseline_results[lora_rank] = clean_top1

            spur_top1_series = new_df["val/zeroshot_eval_spur_top1"].dropna()
            clean_top1_series = new_df["val/zeroshot_eval_clean_top1"].dropna()

            if not spur_top1_series.empty and not clean_top1_series.empty:
                spur_top1 = spur_top1_series.iloc[-1]
                clean_top1 = clean_top1_series.iloc[-1]

                # Degradation: clean_top1 - spur_top1 (positive = degradation)
                degradation = clean_top1 - spur_top1

                degradation_results[lora_rank]["spurious_proportion"].append(spurious_proportion)
                degradation_results[lora_rank]["degradation"].append(degradation)
                
                


# Helper to sort data by x-axis before plotting
def sort_and_unpack(x_list, y_list):
    if x_list:
        sorted_pairs = sorted(zip(x_list, y_list))
        return zip(*sorted_pairs)
    return [], []


# ── Plot 1: Accuracy Degradation vs Spurious Proportion by LoRA Rank ──────────
plt.figure(figsize=(10, 6))

styles = {1: "-o", 16: "--s", 32: ":^", 64: "-.d"}
colors = {1: "#1f77b4", 16: "#ff7f0e", 32: "#2ca02c", 64: "#d62728"}

for rank in lora_ranks:
    x_list = degradation_results[rank]["spurious_proportion"]
    y_list = degradation_results[rank]["degradation"]
    x, y = sort_and_unpack(x_list, y_list)
    plt.plot(
        list(x),
        list(y),
        styles[rank],
        color=colors[rank],
        label=f"LoRA Rank {rank}",
        linewidth=2,
        markersize=7,
    )


plt.xlabel("Spurious Correlation Proportion", fontsize=14)
plt.ylabel("Accuracy Degradation (Clean Top-1 − Spurious Top-1)", fontsize=14)
plt.title("Accuracy Degradation vs Spurious Proportion by LoRA Rank", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("accuracy_degradation_vs_spurious_proportion.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: accuracy_degradation_vs_spurious_proportion.png")


# ── Plot 2: Baseline Clean Top-1 Accuracy vs LoRA Rank (spur_proportion=0.05) ─
ranks_with_data = [r for r in lora_ranks if baseline_results[r] is not None]
accuracies = [baseline_results[r] for r in ranks_with_data]

plt.figure(figsize=(8, 6))
plt.plot(
    ranks_with_data,
    accuracies,
    "-o",
    color="#1f77b4",
    linewidth=2,
    markersize=8,
)

plt.xlabel("LoRA Rank", fontsize=14)
plt.ylabel("Clean Top-1 Accuracy", fontsize=14)
plt.title(
    f"Baseline Clean Top-1 Accuracy vs LoRA Rank\n(Spurious Proportion = {baseline_spur_proportion})",
    fontsize=16,
)
plt.xticks(ranks_with_data, labels=[str(r) for r in ranks_with_data])
plt.grid(True, linestyle="--", alpha=0.6)

if no_lora_baseline is not None:
    plt.axhline(
        y=no_lora_baseline,
        color="gray",
        linestyle="--",
        linewidth=2,
        label="No LoRA (Pure Finetuning)",
    )
    plt.legend(fontsize=12)

plt.ylim(0.9, 1.0)
plt.tight_layout()

plt.savefig("baseline_clean_top1_vs_lora_rank.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: baseline_clean_top1_vs_lora_rank.png")