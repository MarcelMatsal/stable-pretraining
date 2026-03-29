"""
Visualize patched vs clean samples from the dataset side-by-side in a matplotlib grid.
Parameters are hardcoded from the config file.
"""

import matplotlib.pyplot as plt
import stable_pretraining as spt
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model
import hydra
from stable_pretraining.data import transforms
import stable_pretraining as spt
import torch
import numpy as np
from stable_pretraining.callbacks import clip_zero_shot
import torchmetrics as tm
import lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import ToPILImage
import types
import torch.nn as nn
import random

# ── Hardcoded config values ────────────────────────────────────────────────────
DATASET        = "uoft-cs/cifar10"
IMAGE_KEY      = "img"
LABEL_KEY      = "label"
PATCH_SIZE     = 0.2
PATCH_COLOR    = [0, 1, 0]
PATCH_POS      = "top_left_corner"
SPUR_LABEL     = 0          # class index that gets the patch
SPUR_PROPORTION = 1
TOTAL_SAMPLES  = 50000
SEED           = 40

# How many side-by-side pairs to show in the grid
NUM_SAMPLES = 8   # produces a grid of 2 rows x NUM_SAMPLES columns

# ── Build transforms ───────────────────────────────────────────────────────────
transform_clean = spt.data.transforms.Compose(
    transforms.ToImage(source=IMAGE_KEY, target=IMAGE_KEY),
    transforms.AddSampleIdx(),
)

transform_patched = spt.data.transforms.Compose(
    transforms.ToImage(source=IMAGE_KEY, target=IMAGE_KEY),
    transforms.AddSampleIdx(),
    transforms.ClassConditionalInjector(
        transformation=transforms.AddPatch(
            patch_size=PATCH_SIZE,
            color=PATCH_COLOR,
            position=PATCH_POS,
            img_key=IMAGE_KEY,
        ),
        label_key=LABEL_KEY,
        target_labels=SPUR_LABEL,
        proportion=SPUR_PROPORTION,
        total_samples=TOTAL_SAMPLES,
        seed=SEED,
    ),
)

# ── Load datasets ──────────────────────────────────────────────────────────────
dataset_clean = spt.data.HFDataset(
    path=DATASET,
    split="train",
    transform=transform_clean,
)

dataset_patched = spt.data.HFDataset(
    path=DATASET,
    split="train",
    transform=transform_patched,
)

# ── Collect samples that belong to the spurious label class ───────────────────
collected = []
for i in range(len(dataset_clean)):
    sample_clean   = dataset_clean[i]
    sample_patched = dataset_patched[i]
 
    if sample_clean[LABEL_KEY] == SPUR_LABEL:
        collected.append((sample_clean[IMAGE_KEY], sample_patched[IMAGE_KEY]))
 
    if len(collected) >= NUM_SAMPLES:
        break
 
import numpy as np
 
def to_hwc(img):
    """Convert a CHW tensor or array to HWC numpy array for matplotlib."""
    if hasattr(img, "permute"):
        return img.permute(1, 2, 0).numpy()
    img = np.array(img)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.transpose(1, 2, 0)
    return img
 
 
fig, axes = plt.subplots(2, NUM_SAMPLES, figsize=(NUM_SAMPLES * 2, 5))
 
for col, (clean_img, patched_img) in enumerate(collected):
    axes[0, col].imshow(to_hwc(clean_img))
    axes[0, col].axis("off")
    if col == 0:
        axes[0, col].set_title("Clean", fontsize=11, loc="left")
 
    axes[1, col].imshow(to_hwc(patched_img))
    axes[1, col].axis("off")
    if col == 0:
        axes[1, col].set_title("Patched", fontsize=11, loc="left")
 
fig.suptitle(
    f"Clean vs Patched Samples — class={SPUR_LABEL}, patch_pos={PATCH_POS}, "
    f"patch_size={PATCH_SIZE}, proportion={SPUR_PROPORTION}",
    fontsize=12,
    y=1.02,
)
 
plt.tight_layout()
plt.savefig("patched_vs_clean_samples.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: patched_vs_clean_samples.png")