import stable_pretraining as spt
from tqdm import tqdm
import pandas as pd

entity = "rbalestr-brown"
project = "clip_spurious_correlation"

wanted_dataset = "uoft-cs/cifar10"
wanted_zeroshot_dataset = "uoft-cs/cifar10"

# ============================================================
# Helper: Print markdown table
# ============================================================

def is_missing(x):
    if isinstance(x, pd.Series):
        return x.empty or x.isna().all()
    if x is None:
        return True

    return False

def make_markdown_table(title, proportions, spur_metrics, clean_metrics):
    print(f"\n{title}\n")

    header = "| Patch Size | " + " | ".join([str(p) for p in proportions]) + " |"
    sep = "|" + " --- |" * (len(proportions) + 1)

    spur_row = (
        "| Spur Eval Dataset (top1/top5) | "
        + " | ".join([f"{t1*100:.5f}/{t5*100:.5f}" for t1, t5 in spur_metrics])
        + " |"
    )

    clean_row = (
        "| Clean Eval Dataset (top1/top5) | "
        + " | ".join([f"{t1*100:.5f}/{t5*100:.5f}" for t1, t5 in clean_metrics])
        + " |"
    )

    print(header)
    print(sep)
    print(spur_row)
    print(clean_row)


# ============================================================
# Fetch WandB data
# ============================================================

configs, dfs = spt.utils.log_reader.read_wandb_project(
    entity=entity, project=project, filters={"state": "finished"}, num_workers=8,
)

# Store results grouped by:
# { rank: { proportion: { spur: (top1, top5), clean: (top1, top5) } } }
results = {}


for run_id, df in tqdm(dfs.items(), desc="Processing runs"):

    # print(df)
    dataset = df.get("dataset", "")
    zeroshot_dataset = df.get("zeroshot_dataset", "")

    if wanted_dataset not in dataset or wanted_zeroshot_dataset not in zeroshot_dataset:
        continue

    # Extract metadata
    sp = df.get("spur_proportion", None)
    rank = df.get("lora_rank", None)
    use_spur = df.get("use_spurious", False)
    patch_size = df.get("patch_size", None)
    use_lora = df.get("use_lora", False)

    if sp != 0.25 or rank is None or not use_spur or use_lora:
        continue

    # Load full run history
    new_df, config = spt.utils.log_reader.read_wandb_run(entity, project, run_id)

    # Extract the final non-NaN metrics
    # print(new_df)
    spur_top1 = new_df.get("val/zeroshot_eval_spur_top1", None)
    spur_top5 = new_df.get("val/zeroshot_eval_spur_top5", None)
    clean_top1 = new_df.get("val/zeroshot_eval_clean_top1", None)
    clean_top5 = new_df.get("val/zeroshot_eval_clean_top5", None)

    if is_missing(spur_top1) or is_missing(spur_top5) or is_missing(clean_top1) or is_missing(clean_top5):
        continue

    # if spur_top1.empty or spur_top5.empty or clean_top1.empty or clean_top5.empty:
    #     continue

    spur_top1 = spur_top1.dropna()
    spur_top5 = spur_top5.dropna()
    clean_top1 = clean_top1.dropna()
    clean_top5 = clean_top5.dropna()

    spur_tuple = (float(spur_top1.iloc[-1]), float(spur_top5.iloc[-1]))
    clean_tuple = (float(clean_top1.iloc[-1]), float(clean_top5.iloc[-1]))

    # Store results
    if rank not in results:
        results[rank] = {}

    results[rank][patch_size] = {
        "spur": spur_tuple,
        "clean": clean_tuple,
    }


# ============================================================
# OUTPUT MARKDOWN TABLE(S)
# ============================================================

for rank, items in results.items():

    title = f"Finetuning on CIFAR10 and Zeroshot on CIFAR10, LoRA rank {rank}, Injecting in Class 0:"

    # Sort by patch size
    patch_sizes = sorted(items.keys())    # <-- CHANGED

    spur_metrics = [items[p]["spur"] for p in patch_sizes]
    clean_metrics = [items[p]["clean"] for p in patch_sizes]

    make_markdown_table(
        title=title,
        proportions=patch_sizes,           # same argument name
        spur_metrics=spur_metrics,
        clean_metrics=clean_metrics,
    )
