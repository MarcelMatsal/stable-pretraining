"""
Supervised Vision Backbone Fine-tuning — Spurious Robustness Probe
===================================================================
Research question:
    Is the vision encoder itself susceptible to spurious correlations,
    or is it robust and only the language model head picks up shortcuts?

Experimental design:
    1. Take a pretrained vision backbone (CLIP ViT, DINOv2, etc.)
    2. Attach a lightweight linear classification head
    3. Fine-tune on CIFAR-10/100 where a fraction of target-class
       images carry a spurious visual cue (colored patch/border/tint)
    4. Evaluate on FOUR splits:
         spurious_train -> spurious_test   : both infected (in-distribution)
         spurious_train -> clean_test      : does shortcut transfer?
         clean_train    -> spurious_test   : does model resist visual cue?
         clean_train    -> clean_test      : upper-bound accuracy
    5. Compare shortcut susceptibility across:
         - Vision-only finetuned model
         - Zero-shot CLIP baseline (no finetuning)
         - Linear probe only (frozen backbone)
         - Full finetune (backbone + head)

Supported backbones:
    openai/clip-vit-base-patch32
    openai/clip-vit-large-patch14
    facebook/dinov2-base
    facebook/dinov2-large
    google/vit-base-patch16-224

Usage:
    python finetune_vision_backbone.py --config cifar10_finetune.yaml
    python finetune_vision_backbone.py --backbone openai/clip-vit-base-patch32 \\
        --dataset uoft-cs/cifar10 --spur-label 0 1 2 --spur-type patch \\
        --finetune-mode linear_probe --epochs 10
"""

import os, gc, json, random, argparse, warnings, yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor, AutoModel,
    CLIPModel, CLIPProcessor,
    ViTModel, ViTFeatureExtractor,
    Dinov2Model, AutoImageProcessor,
)

# ─────────────────────────────────────────────────────────────────────────────
# Class name registry
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES: Dict[str, List[str]] = {
    "uoft-cs/cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "uoft-cs/cifar100": [
        "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle",
        "bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel",
        "can","castle","caterpillar","cattle","chair","chimpanzee","clock",
        "cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
        "dolphin","elephant","flatfish","forest","fox","girl","hamster",
        "house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion",
        "lizard","lobster","man","maple_tree","motorcycle","mountain","mouse",
        "mushroom","oak_tree","orange","orchid","otter","palm_tree","pear",
        "pickup_truck","pine_tree","plain","plate","poppy","porcupine","possum",
        "rabbit","raccoon","ray","road","rocket","rose","sea","seal","shark",
        "shrew","skunk","skyscraper","snail","snake","spider","squirrel",
        "streetcar","sunflower","sweet_pepper","table","tank","telephone",
        "television","tiger","tractor","train","trout","tulip","turtle",
        "wardrobe","whale","willow_tree","wolf","woman","worm",
    ],
}

# Backbone families
CLIP_MODELS  = {
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch16",
}
DINO_MODELS  = {
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "facebook/dinov2-small",
}
VIT_MODELS   = {
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
}

# Default per-class spurious colors
DEFAULT_CLASS_COLORS: Dict[int, List[int]] = {
    0: [255,   0,   0],
    1: [  0, 220,   0],
    2: [  0,   0, 255],
    3: [255, 200,   0],
    4: [180,   0, 255],
    5: [  0, 200, 200],
    6: [255, 120,   0],
    7: [  0, 100, 255],
    8: [255,   0, 160],
    9: [160, 160, 160],
}

# ─────────────────────────────────────────────────────────────────────────────
# Spurious cue injection (same logic as ICL probe for consistency)
# ─────────────────────────────────────────────────────────────────────────────

def should_trigger(idx: int, label: int, *, seed: int,
                   proportion: float, target_labels: set) -> bool:
    if label not in target_labels:
        return False
    return (hash((seed, idx)) % 10_000_000) / 10_000_000 < proportion


def get_class_color(label: int,
                    per_class_colors: Dict[int, List[int]]) -> Tuple[int,int,int]:
    c = per_class_colors.get(label, DEFAULT_CLASS_COLORS.get(label, [255,0,0]))
    return tuple(c)


def apply_spurious_cue(img: Image.Image, spur_type: str, *,
        color: Tuple[int,int,int],
        patch_size: int = 10,
        patch_pos:  str = "bottom_right",
        border_thickness: int = 4,
        tint_alpha: float = 0.35) -> Image.Image:
    img = img.copy().convert("RGB")
    w, h = img.size
    if spur_type == "patch":
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        positions = {
            "bottom_right": (w-patch_size, h-patch_size, w, h),
            "top_left":     (0, 0, patch_size, patch_size),
            "top_right":    (w-patch_size, 0, w, patch_size),
            "bottom_left":  (0, h-patch_size, patch_size, h),
            "center":       (w//2-patch_size//2, h//2-patch_size//2,
                             w//2+patch_size//2, h//2+patch_size//2),
        }
        draw.rectangle(positions.get(patch_pos, positions["bottom_right"]),
                       fill=color)
    elif spur_type == "border":
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for t in range(border_thickness):
            draw.rectangle([t, t, w-1-t, h-1-t], outline=color)
    elif spur_type == "tint":
        overlay = Image.new("RGB", img.size, color)
        img = Image.blend(img, overlay, alpha=tint_alpha)
    else:
        raise ValueError(f"Unknown spur_type '{spur_type}'")
    return img

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset wrapper around a HuggingFace dataset
# ─────────────────────────────────────────────────────────────────────────────

class SpuriousVisionDataset(Dataset):
    """
    Wraps a HF dataset split and optionally injects per-class spurious cues.

    Parameters
    ----------
    hf_dataset      : HuggingFace dataset split
    processor       : HF image processor (handles resize + normalise)
    spur_target_labels : set of class indices that may receive the cue
    spur_proportion : fraction of target-class images that get the cue
    apply_spurious  : whether to inject the cue at all
    spur_type       : "patch" | "border" | "tint"
    per_class_colors: {label: [R,G,B]}
    seed            : for deterministic trigger decisions
    indices         : optional list of dataset indices to use (for train/val split)
    """

    def __init__(self, hf_dataset, processor, *,
                 spur_target_labels: set,
                 spur_proportion: float,
                 apply_spurious: bool,
                 spur_type: str,
                 per_class_colors: Dict[int, List[int]],
                 patch_size: int,
                 patch_pos: str,
                 border_thickness: int,
                 tint_alpha: float,
                 seed: int,
                 indices: Optional[List[int]] = None):
        self.hf       = hf_dataset
        self.proc     = processor
        self.indices  = indices if indices is not None else list(range(len(hf_dataset)))
        self.spur_target_labels = spur_target_labels
        self.spur_proportion    = spur_proportion
        self.apply_spurious     = apply_spurious
        self.spur_type          = spur_type
        self.per_class_colors   = per_class_colors
        self.patch_size         = patch_size
        self.patch_pos          = patch_pos
        self.border_thickness   = border_thickness
        self.tint_alpha         = tint_alpha
        self.seed               = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        item     = self.hf[real_idx]

        img = item.get("img", item.get("image"))
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img   = img.convert("RGB")
        label = int(item.get("label", item.get("labels", -1)))

        triggered = should_trigger(real_idx, label, seed=self.seed,
                                   proportion=self.spur_proportion,
                                   target_labels=self.spur_target_labels)
        if self.apply_spurious and triggered:
            color = get_class_color(label, self.per_class_colors)
            img   = apply_spurious_cue(img, self.spur_type, color=color,
                        patch_size=self.patch_size, patch_pos=self.patch_pos,
                        border_thickness=self.border_thickness,
                        tint_alpha=self.tint_alpha)

        # Processor handles resize, normalisation, tensor conversion
        enc = self.proc(images=img, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)   # [C, H, W]

        return {
            "pixel_values": pixel_values,
            "label":        torch.tensor(label, dtype=torch.long),
            "triggered":    torch.tensor(triggered, dtype=torch.bool),
            "idx":          torch.tensor(real_idx,  dtype=torch.long),
        }

# ─────────────────────────────────────────────────────────────────────────────
# Vision backbone + linear head
# ─────────────────────────────────────────────────────────────────────────────

class VisionClassifier(nn.Module):
    """
    Pretrained vision backbone + linear classification head.

    finetune_mode:
        "linear_probe"  — backbone frozen, only head trains
        "full"          — backbone + head both train
        "last_n_layers" — freeze all except last N transformer blocks + head
    """

    def __init__(self, backbone_id: str, num_classes: int,
                 finetune_mode: str = "linear_probe",
                 last_n_layers: int = 2):
        super().__init__()
        self.backbone_id   = backbone_id
        self.finetune_mode = finetune_mode

        # ── load backbone ─────────────────────────────────────────────────────
        if backbone_id in CLIP_MODELS:
            self.backbone = CLIPModel.from_pretrained(backbone_id).vision_model
            embed_dim     = self.backbone.config.hidden_size
        elif backbone_id in DINO_MODELS:
            self.backbone = Dinov2Model.from_pretrained(backbone_id)
            embed_dim     = self.backbone.config.hidden_size
        elif backbone_id in VIT_MODELS:
            self.backbone = ViTModel.from_pretrained(backbone_id)
            embed_dim     = self.backbone.config.hidden_size
        else:
            # Generic fallback via AutoModel
            self.backbone = AutoModel.from_pretrained(backbone_id)
            embed_dim     = self.backbone.config.hidden_size

        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # ── freeze / unfreeze based on mode ──────────────────────────────────
        self._apply_finetune_mode(last_n_layers)

    def _apply_finetune_mode(self, last_n_layers: int):
        if self.finetune_mode == "linear_probe":
            # Freeze entire backbone
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        elif self.finetune_mode == "full":
            # Unfreeze everything
            for p in self.backbone.parameters():
                p.requires_grad_(True)

        elif self.finetune_mode == "last_n_layers":
            # Freeze all, then unfreeze last N transformer encoder blocks
            for p in self.backbone.parameters():
                p.requires_grad_(False)

            encoder = None
            for attr in ("encoder", "transformer", "vision_model"):
                enc = getattr(self.backbone, attr, None)
                if enc is not None:
                    encoder = enc
                    break

            if encoder is not None:
                layers = getattr(encoder, "layers",
                         getattr(encoder, "blocks", None))
                if layers is not None:
                    for layer in list(layers)[-last_n_layers:]:
                        for p in layer.parameters():
                            p.requires_grad_(True)
            # Also unfreeze the final layernorm
            for name in ("layernorm", "layer_norm", "post_layernorm",
                         "ln_post", "norm"):
                m = getattr(self.backbone, name, None)
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad_(True)
        else:
            raise ValueError(f"Unknown finetune_mode '{self.finetune_mode}'")

        total  = sum(p.numel() for p in self.parameters())
        tuning = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Parameters: {total:,} total, {tuning:,} trainable "
              f"({100*tuning/max(total,1):.1f}%)")

    def _extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward through backbone; return [B, embed_dim] CLS embedding."""
        out = self.backbone(pixel_values=pixel_values)

        # CLIP / ViT / DINOv2 all expose pooler_output or last_hidden_state
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output                  # [B, D]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]     # CLS token
        raise RuntimeError("Cannot extract features from backbone output")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats  = self._extract_features(pixel_values)
        logits = self.head(feats)
        return logits

    def get_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return raw embeddings (for analysis / probing)."""
        with torch.no_grad():
            return self._extract_features(pixel_values)

# ─────────────────────────────────────────────────────────────────────────────
# Backbone processor loader
# ─────────────────────────────────────────────────────────────────────────────

def load_processor(backbone_id: str):
    if backbone_id in CLIP_MODELS:
        return CLIPProcessor.from_pretrained(backbone_id)
    elif backbone_id in DINO_MODELS:
        return AutoImageProcessor.from_pretrained(backbone_id)
    elif backbone_id in VIT_MODELS:
        return ViTFeatureExtractor.from_pretrained(backbone_id)
    else:
        return AutoProcessor.from_pretrained(backbone_id)

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="    train", leave=False):
        pv     = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(pv)
                loss   = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pv)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / max(total, 1), 100 * correct / max(total, 1)


@torch.inference_mode()
def evaluate(model, loader, device, spur_target_labels: set,
             class_names: List[str]) -> dict:
    """
    Returns accuracy, per-class accuracy, spurious-flip rate, and
    triggered-vs-clean breakdown.
    """
    model.eval()
    records = []   # (gt, pred, triggered)

    for batch in tqdm(loader, desc="    eval ", leave=False):
        pv      = batch["pixel_values"].to(device)
        labels  = batch["label"]
        trigged = batch["triggered"]

        logits = model(pv)
        preds  = logits.argmax(1).cpu()

        for gt, pr, tr in zip(labels.tolist(), preds.tolist(), trigged.tolist()):
            records.append((gt, pr, bool(tr)))

    # ── aggregate ─────────────────────────────────────────────────────────────
    total     = len(records)
    correct   = sum(g == p for g, p, _ in records)
    accuracy  = 100 * correct / max(total, 1)

    # SpurFlip: triggered image predicted as a *different* spurious-target class
    spur_recs    = [(g, p) for g, p, t in records if t]
    spur_flips   = sum(p != g and p in spur_target_labels for g, p in spur_recs)
    spur_total   = max(len(spur_recs), 1)
    spur_flip_rt = 100 * spur_flips / spur_total

    # Triggered accuracy (subset of images that had cue)
    spur_correct  = sum(g == p for g, p in spur_recs)
    spur_acc      = 100 * spur_correct / spur_total

    # Clean accuracy (no cue)
    clean_recs    = [(g, p) for g, p, t in records if not t]
    clean_correct = sum(g == p for g, p in clean_recs)
    clean_acc     = 100 * clean_correct / max(len(clean_recs), 1)

    # Per-class accuracy
    per_class_correct: Dict[str, int] = {}
    per_class_total:   Dict[str, int] = {}
    for g, p, _ in records:
        cn = class_names[g]
        per_class_total[cn]   = per_class_total.get(cn, 0) + 1
        per_class_correct[cn] = per_class_correct.get(cn, 0) + int(g == p)
    per_class_acc = {
        cn: round(100 * per_class_correct.get(cn, 0)
                      / max(per_class_total.get(cn, 1), 1), 2)
        for cn in per_class_total
    }

    return {
        "accuracy":          round(accuracy,    2),
        "spur_acc":          round(spur_acc,    2),
        "clean_acc":         round(clean_acc,   2),
        "spur_flip_rate":    round(spur_flip_rt, 2),
        "n_total":           total,
        "n_triggered":       len(spur_recs),
        "n_clean":           len(clean_recs),
        "per_class_acc":     per_class_acc,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot CLIP baseline
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def zero_shot_clip_eval(backbone_id: str, loader, device,
                        class_names: List[str],
                        spur_target_labels: set) -> dict:
    """
    CLIP zero-shot: compare image embeddings to text embeddings of class names.
    Only valid for CLIP backbones.
    """
    from transformers import CLIPModel, CLIPProcessor
    print("  Running zero-shot CLIP baseline ...")
    clip  = CLIPModel.from_pretrained(backbone_id).to(device).eval()
    cproc = CLIPProcessor.from_pretrained(backbone_id)

    # Build text embeddings
    prompts = [f"a photo of a {c}" for c in class_names]
    txt_enc = cproc(text=prompts, return_tensors="pt", padding=True)
    txt_enc = {k: v.to(device) for k, v in txt_enc.items()}
    with torch.no_grad():
        txt_feats = clip.get_text_features(**txt_enc)        # [C, D]
        txt_feats = F.normalize(txt_feats, dim=-1)

    records = []
    for batch in tqdm(loader, desc="    zs-clip", leave=False):
        pv      = batch["pixel_values"].to(device)
        labels  = batch["label"]
        trigged = batch["triggered"]
        with torch.no_grad():
            img_feats = clip.get_image_features(pixel_values=pv)
            img_feats = F.normalize(img_feats, dim=-1)
            sims      = img_feats @ txt_feats.T          # [B, C]
            preds     = sims.argmax(1).cpu()
        for gt, pr, tr in zip(labels.tolist(), preds.tolist(), trigged.tolist()):
            records.append((gt, pr, bool(tr)))

    del clip
    gc.collect()
    torch.cuda.empty_cache()

    total    = len(records)
    correct  = sum(g == p for g, p, _ in records)
    spur_recs  = [(g, p) for g, p, t in records if t]
    spur_flips = sum(p != g and p in spur_target_labels for g, p in spur_recs)

    return {
        "accuracy":       round(100*correct/max(total,1), 2),
        "spur_acc":       round(100*sum(g==p for g,p in spur_recs)/max(len(spur_recs),1), 2),
        "spur_flip_rate": round(100*spur_flips/max(len(spur_recs),1), 2),
        "n_total":        total,
        "n_triggered":    len(spur_recs),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Results printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(backbone_id: str, results: dict,
                  class_names: List[str],
                  per_class_colors: Dict[int, List[int]],
                  spur_target_labels: set,
                  finetune_mode: str):
    name = backbone_id.split("/")[-1]
    print(f"\n{'='*72}")
    print(f"  Results — {name}  [{finetune_mode}]")
    print(f"{'='*72}")

    rows = [
        ("spurious_train → spurious_test  (both infected)",
         "spur_train__spur_test"),
        ("spurious_train → clean_test     (shortcut transfer?)",
         "spur_train__clean_test"),
        ("clean_train    → spurious_test  (resist visual cue?)",
         "clean_train__spur_test"),
        ("clean_train    → clean_test     (ceiling)",
         "clean_train__clean_test"),
    ]

    print(f"\n  {'Condition':<52} {'Acc':>6} {'SpurAcc':>8} "
          f"{'CleanAcc':>9} {'SpurFlip':>9}")
    print(f"  {'-'*88}")
    for label, key in rows:
        m = results.get(key, {})
        print(f"  {label:<52} "
              f"{m.get('accuracy',0):>5.1f}% "
              f"{m.get('spur_acc',0):>7.1f}% "
              f"{m.get('clean_acc',0):>8.1f}% "
              f"{m.get('spur_flip_rate',0):>8.1f}%")

    # Key signal: does spurious training hurt clean-test accuracy?
    spur_clean = results.get("spur_train__clean_test", {}).get("accuracy", 0)
    clean_ceil = results.get("clean_train__clean_test", {}).get("accuracy", 0)
    drop       = clean_ceil - spur_clean
    print(f"\n  Shortcut penalty (ceiling - spur_train→clean_test): {drop:+.1f}%")
    if drop > 5:
        print("  → Vision backbone IS susceptible to spurious shortcuts")
    elif drop > 1:
        print("  → Mild susceptibility — backbone picks up some shortcuts")
    else:
        print("  → Vision backbone appears ROBUST to this spurious cue")

    print(f"\n  Per-class spurious color legend:")
    for idx in sorted(spur_target_labels):
        cn = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        c  = get_class_color(idx, per_class_colors)
        print(f"    [{idx}] {cn:<15}  RGB{c}")

# ─────────────────────────────────────────────────────────────────────────────
# YAML config support
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "per_class_colors" in cfg and cfg["per_class_colors"]:
        cfg["per_class_colors"] = {int(k): list(v)
                                   for k, v in cfg["per_class_colors"].items()}
    if "spur_labels" in cfg and cfg["spur_labels"]:
        cfg["spur_labels"] = [int(x) for x in cfg["spur_labels"]]
    # Cast fields that YAML may read as strings when written in scientific notation
    for float_key in ("lr", "weight_decay", "spur_proportion", "tint_alpha", "val_fraction"):
        if float_key in cfg and cfg[float_key] is not None:
            cfg[float_key] = float(cfg[float_key])
    for int_key in ("epochs", "batch_size", "warmup_steps", "num_workers",
                    "last_n_layers", "patch_size", "border_thickness", "seed"):
        if int_key in cfg and cfg[int_key] is not None:
            cfg[int_key] = int(cfg[int_key])
    for bool_key in ("use_amp", "zero_shot_clip", "save_model", "load_4bit"):
        if bool_key in cfg and cfg[bool_key] is not None:
            cfg[bool_key] = bool(cfg[bool_key])
    return cfg

_YAML_TO_ARG = {
    "dataset":          "dataset",
    "train_split":      "train_split",
    "test_split":       "test_split",
    "backbone":         "backbone",
    "finetune_mode":    "finetune_mode",
    "last_n_layers":    "last_n_layers",
    "spur_labels":      "spur_label",
    "spur_type":        "spur_type",
    "spur_proportion":  "spur_proportion",
    "patch_size":       "patch_size",
    "patch_pos":        "patch_pos",
    "border_thickness": "border_thickness",
    "tint_alpha":       "tint_alpha",
    "epochs":           "epochs",
    "batch_size":       "batch_size",
    "lr":               "lr",
    "weight_decay":     "weight_decay",
    "warmup_steps":     "warmup_steps",
    "val_fraction":     "val_fraction",
    "num_workers":      "num_workers",
    "use_amp":          "use_amp",
    "zero_shot_clip":   "zero_shot_clip",
    "seed":             "seed",
    "out":              "out",
    "save_model":       "save_model",
}

def merge_yaml_into_args(args, yaml_cfg: dict):
    for yaml_key, arg_key in _YAML_TO_ARG.items():
        if yaml_key in yaml_cfg:
            setattr(args, arg_key, yaml_cfg[yaml_key])
    return args

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Supervised vision backbone spurious robustness probe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--config",           default=None,
                   help="YAML config (CLI flags override YAML values)")

    # Data
    p.add_argument("--dataset",          default="uoft-cs/cifar10")
    p.add_argument("--train-split",      default="train")
    p.add_argument("--test-split",       default="test")

    # Backbone
    p.add_argument("--backbone",         default="openai/clip-vit-base-patch32",
                   help="HuggingFace backbone model ID")
    p.add_argument("--finetune-mode",    default="linear_probe",
                   choices=["linear_probe", "full", "last_n_layers"])
    p.add_argument("--last-n-layers",    type=int, default=2,
                   help="Unfreeze last N transformer blocks (last_n_layers mode)")

    # Spurious config
    p.add_argument("--spur-label",       type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--spur-type",        default="patch",
                   choices=["patch", "border", "tint"])
    p.add_argument("--spur-proportion",  type=float, default=0.9)
    p.add_argument("--patch-size",       type=int,   default=10)
    p.add_argument("--patch-pos",        default="bottom_right",
                   choices=["bottom_right","top_left","top_right","bottom_left","center"])
    p.add_argument("--border-thickness", type=int,   default=4)
    p.add_argument("--tint-alpha",       type=float, default=0.35)

    # Training
    p.add_argument("--epochs",           type=int,   default=10)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--weight-decay",     type=float, default=1e-4)
    p.add_argument("--warmup-steps",     type=int,   default=100)
    p.add_argument("--val-fraction",     type=float, default=0.1,
                   help="Fraction of training set held out for validation")
    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--use-amp",          action="store_true",
                   help="Mixed precision training (fp16, needs CUDA)")

    # Extras
    p.add_argument("--zero-shot-clip",   action="store_true",
                   help="Also run zero-shot CLIP baseline (CLIP backbones only)")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--out",              default="finetune_results.json")
    p.add_argument("--save-model",       action="store_true",
                   help="Save best checkpoint to disk")

    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── YAML ─────────────────────────────────────────────────────────────────
    per_class_colors: Dict[int, List[int]] = {}
    if args.config:
        yaml_cfg         = load_yaml_config(args.config)
        per_class_colors = yaml_cfg.pop("per_class_colors", {})
        args             = merge_yaml_into_args(args, yaml_cfg)
        print(f"  Loaded YAML config: {args.config}")

    for idx, color in DEFAULT_CLASS_COLORS.items():
        if idx not in per_class_colors:
            per_class_colors[idx] = color

    # ── reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if args.dataset not in CLASS_NAMES:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")
    class_names        = CLASS_NAMES[args.dataset]
    num_classes        = len(class_names)
    spur_target_labels = set(args.spur_label)

    print(f"\n{'='*72}\n  Vision Backbone Spurious Robustness Probe\n{'='*72}")
    print(f"  Dataset          : {args.dataset}  ({num_classes} classes)")
    print(f"  Backbone         : {args.backbone}")
    print(f"  Finetune mode    : {args.finetune_mode}")
    print(f"  Spur type        : {args.spur_type}")
    print(f"  Spur labels      : {sorted(spur_target_labels)}")
    print(f"  Spur proportion  : {args.spur_proportion}")
    print(f"  Epochs / BS / LR : {args.epochs} / {args.batch_size} / {args.lr}")
    print(f"  Per-class colors :")
    for lbl in sorted(spur_target_labels):
        cn = class_names[lbl] if lbl < len(class_names) else f"class_{lbl}"
        print(f"    [{lbl}] {cn:<15}  RGB{get_class_color(lbl, per_class_colors)}")

    # ── processor ─────────────────────────────────────────────────────────────
    print("\n  Loading processor ...")
    processor = load_processor(args.backbone)

    # ── datasets ──────────────────────────────────────────────────────────────
    print("  Loading datasets ...")
    from datasets import load_dataset
    train_hf = load_dataset(args.dataset, split=args.train_split)
    test_hf  = load_dataset(args.dataset, split=args.test_split)

    n_train = len(train_hf)
    n_val   = int(n_train * args.val_fraction)
    all_idx = list(range(n_train))
    random.shuffle(all_idx)
    val_idx   = all_idx[:n_val]
    train_idx = all_idx[n_val:]

    print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}  "
          f"Test: {len(test_hf):,}")

    # Common dataset kwargs
    ds_kw = dict(
        processor=processor,
        spur_target_labels=spur_target_labels,
        spur_proportion=args.spur_proportion,
        spur_type=args.spur_type,
        per_class_colors=per_class_colors,
        patch_size=args.patch_size,
        patch_pos=args.patch_pos,
        border_thickness=args.border_thickness,
        tint_alpha=args.tint_alpha,
        seed=args.seed,
    )

    # Build all 4 condition datasets
    # Train side: spurious or clean
    ds_train_spur  = SpuriousVisionDataset(train_hf, apply_spurious=True,
                                            indices=train_idx, **ds_kw)
    ds_train_clean = SpuriousVisionDataset(train_hf, apply_spurious=False,
                                            indices=train_idx, **ds_kw)
    ds_val         = SpuriousVisionDataset(train_hf, apply_spurious=False,
                                            indices=val_idx, **ds_kw)

    # Test side: spurious or clean
    ds_test_spur   = SpuriousVisionDataset(test_hf,  apply_spurious=True,  **ds_kw)
    ds_test_clean  = SpuriousVisionDataset(test_hf,  apply_spurious=False, **ds_kw)

    loader_kw = dict(batch_size=args.batch_size,
                     num_workers=args.num_workers,
                     pin_memory=(device.type == "cuda"))

    loader_train_spur  = DataLoader(ds_train_spur,  shuffle=True,  **loader_kw)
    loader_train_clean = DataLoader(ds_train_clean, shuffle=True,  **loader_kw)
    loader_val         = DataLoader(ds_val,          shuffle=False, **loader_kw)
    loader_test_spur   = DataLoader(ds_test_spur,    shuffle=False, **loader_kw)
    loader_test_clean  = DataLoader(ds_test_clean,   shuffle=False, **loader_kw)

    # ── optional zero-shot CLIP baseline ──────────────────────────────────────
    zs_results = {}
    if args.zero_shot_clip and args.backbone in CLIP_MODELS:
        print("\n  Zero-shot CLIP evaluation ...")
        for name, loader in [("spurious_test", loader_test_spur),
                              ("clean_test",    loader_test_clean)]:
            zs_results[name] = zero_shot_clip_eval(
                args.backbone, loader, device, class_names, spur_target_labels)
            m = zs_results[name]
            print(f"  ZS-CLIP [{name}]: acc={m['accuracy']:.1f}%  "
                  f"spur_flip={m['spur_flip_rate']:.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # Train two models: one on spurious data, one on clean data
    # ─────────────────────────────────────────────────────────────────────────
    all_results = {}

    for train_name, train_loader in [("spur_train", loader_train_spur),
                                     ("clean_train", loader_train_clean)]:
        print(f"\n{'─'*72}")
        print(f"  Training: {train_name}  [{args.finetune_mode}]")
        print(f"{'─'*72}")

        # Fresh model for each training condition
        model = VisionClassifier(
            backbone_id=args.backbone,
            num_classes=num_classes,
            finetune_mode=args.finetune_mode,
            last_n_layers=args.last_n_layers,
        ).to(device)

        # Optimizer — only pass trainable parameters
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                      weight_decay=args.weight_decay)

        total_steps = len(train_loader) * args.epochs
        warmup      = min(args.warmup_steps, total_steps // 10)

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))   # cosine decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler    = torch.cuda.amp.GradScaler() if (args.use_amp
                    and device.type == "cuda") else None

        best_val_acc  = -1.0
        best_state    = None

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, scaler)

            val_metrics = evaluate(model, loader_val, device,
                                   spur_target_labels, class_names)
            val_acc = val_metrics["accuracy"]

            print(f"  Epoch {epoch:02d}/{args.epochs}  "
                  f"loss={tr_loss:.4f}  train_acc={tr_acc:.1f}%  "
                  f"val_acc={val_acc:.1f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}

        # ── restore best checkpoint ───────────────────────────────────────────
        if best_state is not None:
            model.load_state_dict({k: v.to(device)
                                   for k, v in best_state.items()})
        print(f"  Best val acc: {best_val_acc:.1f}%")

        # ── save model ────────────────────────────────────────────────────────
        if args.save_model and best_state is not None:
            ckpt_path = Path(args.out).with_suffix("") / f"{train_name}_best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_path)
            print(f"  Saved checkpoint -> {ckpt_path}")

        # ── evaluate on both test splits ──────────────────────────────────────
        print(f"  Evaluating on spurious test set ...")
        spur_test_metrics  = evaluate(model, loader_test_spur,  device,
                                      spur_target_labels, class_names)
        print(f"  Evaluating on clean test set ...")
        clean_test_metrics = evaluate(model, loader_test_clean, device,
                                      spur_target_labels, class_names)

        cond_spur  = f"{train_name}__spur_test"
        cond_clean = f"{train_name}__clean_test"
        all_results[cond_spur]  = spur_test_metrics
        all_results[cond_clean] = clean_test_metrics

        print(f"  [{cond_spur}]  acc={spur_test_metrics['accuracy']:.1f}%  "
              f"spur_flip={spur_test_metrics['spur_flip_rate']:.1f}%")
        print(f"  [{cond_clean}] acc={clean_test_metrics['accuracy']:.1f}%  "
              f"spur_flip={clean_test_metrics['spur_flip_rate']:.1f}%")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── final pretty-print ────────────────────────────────────────────────────
    print_results(args.backbone, all_results, class_names,
                  per_class_colors, spur_target_labels, args.finetune_mode)

    # ── save JSON ─────────────────────────────────────────────────────────────
    out_data = {
        "config": {
            "dataset":          args.dataset,
            "backbone":         args.backbone,
            "finetune_mode":    args.finetune_mode,
            "spur_type":        args.spur_type,
            "spur_labels":      sorted(spur_target_labels),
            "spur_proportion":  args.spur_proportion,
            "epochs":           args.epochs,
            "batch_size":       args.batch_size,
            "lr":               args.lr,
            "seed":             args.seed,
            "per_class_colors": {str(k): v for k, v in per_class_colors.items()},
        },
        "results":    all_results,
        "zero_shot":  zs_results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n  Results saved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()