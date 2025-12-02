"""This script demonstrates how to retrieve data from wandb using the stable_pretraining library and create plots from it.

To use, you should set the entity variable to your WandB entity and the project variable to the specific project within
your WandB entity that you want to access runs from.

"""

import matplotlib.pyplot as plt
from tqdm import tqdm

import stable_pretraining as spt

entity = "rbalestr-brown"
project = "clip_spurious_correlation"


# want to retrieve finished runs from wandb
configs, dfs = spt.reader.wandb_project(
    entity=entity, project=project, filters={"state": "finished"}
)

# Here you would define wanted conditions that you can use to narrow down the WandB runs, if you want to access all your runs
# in the project then you would not define anything here and would remove the first if statement in the for loop
# access all runs with the wanted dataset and model backbone
wanted_dataset = "uoft-cs/cifar10"
wanted_zeroshot_dataset = "uoft-cs/cifar10"

# This section allows for the users to define the information they want to separate the runs into to later be plotted
# for this example, we are dividing the data based on rank and location. We are the going to plot the spurious_proportion on the
# x-axis and balanced_accuracy on the y-axis

# Dictionary to store results
results = {
    rank: {
        loc: {"spurious_proportion": [], "balanced_accuracy": []}
        for loc in ["random", "end", "beginning"]
    }
    for rank in [1, 16, 32, 64]
}

# Iterate through runs and gather information from WandB
for run_id, df in tqdm(dfs.items(), desc="Processing runs", unit="run"):
    # Get the dataset, backbone, and run name from the WandB runs, this allows us to focus on the runs for specific conditions
    # we want to plot (ex we only want to plot runs for a specific dataset and backbone)
    dataset = df.get("dataset", None)
    zeroshot_dataset = df.get("zeroshot_dataset", None)
    # backbone = df.get("backbone", None)
    run_name = df.get("run_name", None)

    # make sure the ones we are using met the conditions for what we want to graph
    if (
        wanted_dataset.lower() in dataset.lower()
        and wanted_zeroshot_dataset.lower() in zeroshot_dataset.lower()
    ):
        # Extract spurious correlation proportion, location, and lora_rank used
        spurious_proportion = df.get("spur_proportion", None)
        # spurious_location = df.get("spurious_location", None
        lora_rank = df.get("lora_rank", None)
        use_spurious = df.get("use_spurious", None)
        # using_list = df.get("use_list_dataset", None)

        # This if statement allows us to exclude runs we dont want to plot, you can change it based on your needs
        # only access if it contains everything wanted
        if (
            spurious_proportion is not None
            and spurious_proportion >= 0
            and lora_rank is not None
            and use_spurious
        ):
            # Extract balanced accuracy from the run
            new_df, config = spt.reader.wandb(entity, project, run_id)
            # drop the ones that are NAN
            spur_top1 = new_df["val/zeroshot_eval_spur_top1"].dropna()
            spur_top5 = new_df["val/zeroshot_eval_spur_top1"].dropna()
            clean_top1 = new_df["val/zeroshot_eval_clean_top1"].dropna()
            clean_top5 = new_df["val/zeroshot_eval_clean_top5"].dropna()

            # Add the last one to be plotted


            # if not balanced_acc.empty:
            #     balanced_acc = balanced_acc.iloc[-1]  # Get the final valid accuracy
            #     results[lora_rank][spurious_location]["spurious_proportion"].append(
            #         spurious_proportion
            #     )
            #     results[lora_rank][spurious_location]["balanced_accuracy"].append(
            #         balanced_acc
            #     )


# Functions used to simplify the plotting process, making it more extensible
# Sort values for plotting
def sort_and_unpack(data):
    if data["spurious_proportion"]:
        sorted_data = sorted(
            zip(data["spurious_proportion"], data["balanced_accuracy"])
        )
        return zip(*sorted_data)
    return [], []


# Create figure
plt.figure(figsize=(20, 14))
styles = {0: "-", 2: "--", 32: ":"}
markers = {"random": "s", "end": "d", "beginning": "x"}

# Plot the data
for rank in results:
    for location in results[rank]:
        x, y = sort_and_unpack(results[rank][location])
        plt.plot(
            x,
            y,
            linestyle=styles[rank],
            marker=markers[location],
            label=f"{location.capitalize()} (LoRA Rank {rank})",
        )

# Label the plot and axis, you can change these to whatever you want/need
plt.xlabel("Spurious Correlation Proportion", fontsize=14)
plt.ylabel("Balanced Accuracy on Clean Test Set", fontsize=14)
plt.title(
    f"Balanced Accuracy vs Spurious Correlation using {wanted_backbone} on {wanted_dataset}, Spurious Type: Date, From List: {using_list}",
    fontsize=16,
)
plt.legend(fontsize=12)
plt.grid()

# Save the figure locally, you can name it whatever you want for your needs
plt.savefig(
    "balanced_accuracy_vs_spurious_correlation.png", dpi=300, bbox_inches="tight"
)
