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
         - LoRA finetune (optional)

Supported backbones:
    openai/clip-vit-base-patch32
    openai/clip-vit-large-patch14
    facebook/dinov2-base
    facebook/dinov2-large
    google/vit-base-patch16-224

Usage:
    python vlm_vision_backbone_test.py
    python vlm_vision_backbone_test.py params.epochs=5 params.use_lora=false
    python vlm_vision_backbone_test.py params.backbone=openai/clip-vit-large-patch14
"""

import os, gc, json, random, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from transformers import (
    AutoProcessor, AutoModel,
    CLIPModel, CLIPProcessor,
    ViTModel, ViTFeatureExtractor,
    Dinov2Model, AutoImageProcessor,
)
from peft import LoraConfig, get_peft_model
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_pretraining.data import transforms
import stable_pretraining as spt

to_pil = ToPILImage()

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

# Default per-class spurious colors (RGB 0-255)
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
# LoRA helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_lora_params(model) -> Tuple[int, int]:
    """Return (trainable_params, total_params) for a (peft-wrapped) model."""
    total, trainable = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def get_class_color(label: int,
                    per_class_colors: Dict[int, List[int]]) -> Tuple[int,int,int]:
    c = per_class_colors.get(label, DEFAULT_CLASS_COLORS.get(label, [255, 0, 0]))
    return tuple(c)

# ─────────────────────────────────────────────────────────────────────────────
# Vision backbone + linear head
# ─────────────────────────────────────────────────────────────────────────────

class VisionClassifier(nn.Module):
    """
    Pretrained vision backbone + linear classification head.

    finetune_mode (when use_lora=False):
        "linear_probe"  — backbone frozen, only head trains
        "full"          — backbone + head both train
        "last_n_layers" — freeze all except last N transformer blocks + head

    use_lora=True overrides finetune_mode: LoRA adapters (q_proj, v_proj) are
    injected into the backbone; everything except LoRA params and the head is
    frozen.
    """

    def __init__(self, backbone_id: str, num_classes: int,
                 finetune_mode: str = "linear_probe",
                 last_n_layers: int = 2,
                 use_lora: bool = False,
                 lora_rank: int = 4,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05):
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
            self.backbone = AutoModel.from_pretrained(backbone_id)
            embed_dim     = self.backbone.config.hidden_size

        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # ── freeze / unfreeze ─────────────────────────────────────────────────
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha, lora_dropout)
        else:
            self._apply_finetune_mode(last_n_layers)

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _apply_lora(self, rank: int, alpha: int, dropout: float):
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=dropout,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        for name, param in self.named_parameters():
            if "lora_" not in name.lower() and "head" not in name.lower():
                param.requires_grad = False

        lora_p, total_p = count_lora_params(self.backbone)
        print(f"  LoRA params : {lora_p:,} / {total_p:,} backbone params trainable")

    # ------------------------------------------------------------------
    # Standard finetune-mode freezing
    # ------------------------------------------------------------------

    def _apply_finetune_mode(self, last_n_layers: int):
        if self.finetune_mode == "linear_probe":
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        elif self.finetune_mode == "full":
            for p in self.backbone.parameters():
                p.requires_grad_(True)

        elif self.finetune_mode == "last_n_layers":
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
        out = self.backbone(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]
        raise RuntimeError("Cannot extract features from backbone output")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self._extract_features(pixel_values))

    def get_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
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

    total    = len(records)
    correct  = sum(g == p for g, p, _ in records)
    accuracy = 100 * correct / max(total, 1)

    spur_recs    = [(g, p) for g, p, t in records if t]
    spur_flips   = sum(p != g and p in spur_target_labels for g, p in spur_recs)
    spur_total   = max(len(spur_recs), 1)
    spur_flip_rt = 100 * spur_flips / spur_total

    spur_correct = sum(g == p for g, p in spur_recs)
    spur_acc     = 100 * spur_correct / spur_total

    clean_recs    = [(g, p) for g, p, t in records if not t]
    clean_correct = sum(g == p for g, p in clean_recs)
    clean_acc     = 100 * clean_correct / max(len(clean_recs), 1)

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
        "accuracy":       round(accuracy,    2),
        "spur_acc":       round(spur_acc,    2),
        "clean_acc":      round(clean_acc,   2),
        "spur_flip_rate": round(spur_flip_rt, 2),
        "n_total":        total,
        "n_triggered":    len(spur_recs),
        "n_clean":        len(clean_recs),
        "per_class_acc":  per_class_acc,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot CLIP baseline
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def zero_shot_clip_eval(backbone_id: str, loader, device,
                        class_names: List[str],
                        spur_target_labels: set) -> dict:
    from transformers import CLIPModel, CLIPProcessor
    print("  Running zero-shot CLIP baseline ...")
    clip  = CLIPModel.from_pretrained(backbone_id).to(device).eval()
    cproc = CLIPProcessor.from_pretrained(backbone_id)

    prompts = [f"a photo of a {c}" for c in class_names]
    txt_enc = cproc(text=prompts, return_tensors="pt", padding=True)
    txt_enc = {k: v.to(device) for k, v in txt_enc.items()}
    with torch.no_grad():
        txt_feats = clip.get_text_features(**txt_enc)
        txt_feats = F.normalize(txt_feats, dim=-1)

    records = []
    for batch in tqdm(loader, desc="    zs-clip", leave=False):
        pv      = batch["pixel_values"].to(device)
        labels  = batch["label"]
        trigged = batch["triggered"]
        with torch.no_grad():
            img_feats = clip.get_image_features(pixel_values=pv)
            img_feats = F.normalize(img_feats, dim=-1)
            sims      = img_feats @ txt_feats.T
            preds     = sims.argmax(1).cpu()
        for gt, pr, tr in zip(labels.tolist(), preds.tolist(), trigged.tolist()):
            records.append((gt, pr, bool(tr)))

    del clip
    gc.collect()
    torch.cuda.empty_cache()

    total      = len(records)
    correct    = sum(g == p for g, p, _ in records)
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
# Main (Hydra entry point)
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path=".", config_name="vlm_vision_backbone_test", version_base="1.1")
def main(cfg: DictConfig):
    p = cfg.params

    # ── per-class colors: OmegaConf DictConfig (str keys) → plain dict (int keys) ──
    per_class_colors: Dict[int, List[int]] = {
        int(k): list(v)
        for k, v in OmegaConf.to_container(p.per_class_colors, resolve=True).items()
    }
    for idx, color in DEFAULT_CLASS_COLORS.items():
        if idx not in per_class_colors:
            per_class_colors[idx] = color

    # ── reproducibility ───────────────────────────────────────────────────────
    random.seed(p.seed)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if p.dataset not in CLASS_NAMES:
        raise ValueError(f"Unknown dataset '{p.dataset}'.")
    class_names        = CLASS_NAMES[p.dataset]
    num_classes        = len(class_names)
    spur_target_labels = set(p.spur_labels)

    print(f"\n{'='*72}\n  Vision Backbone Spurious Robustness Probe\n{'='*72}")
    print(f"  Dataset          : {p.dataset}  ({num_classes} classes)")
    print(f"  Backbone         : {p.backbone}")
    print(f"  Finetune mode    : {'lora' if p.use_lora else p.finetune_mode}")
    if p.use_lora:
        print(f"  LoRA             : rank={p.lora_rank}  alpha={p.lora_alpha}"
              f"  dropout={p.lora_dropout}")
    print(f"  Spur type        : {p.spur_type}")
    print(f"  Spur labels      : {sorted(spur_target_labels)}")
    print(f"  Spur proportion  : {p.spur_proportion}")
    print(f"  Epochs / BS / LR : {p.epochs} / {p.batch_size} / {p.lr}")
    print(f"  Per-class colors :")
    for lbl in sorted(spur_target_labels):
        cn = class_names[lbl] if lbl < len(class_names) else f"class_{lbl}"
        print(f"    [{lbl}] {cn:<15}  RGB{get_class_color(lbl, per_class_colors)}")

    # ── processor ─────────────────────────────────────────────────────────────
    print("\n  Loading processor ...")
    processor = load_processor(p.backbone)

    # ── train/val split indices ───────────────────────────────────────────────
    n_train_full = p.total_train_samples
    n_test_full  = p.total_test_samples
    n_val        = int(n_train_full * p.val_fraction)
    all_idx      = list(range(n_train_full))
    random.shuffle(all_idx)
    val_idx   = all_idx[:n_val]
    train_idx = all_idx[n_val:]
    print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {n_test_full:,}")

    # ── Spurious cue transforms (stable_pretraining.data.transforms) ──────────
    def _make_cue_transform(label: int) -> transforms.Transform:
        """Per-class AddPatch / AddBorder / AddColorTint for a single label."""
        color_255 = get_class_color(label, per_class_colors)
        color_01  = tuple(c / 255.0 for c in color_255)
        if p.spur_type == "patch":
            return transforms.AddPatch(
                patch_size=p.patch_size,
                color=color_01,
                position=p.patch_pos,
                img_key="img",
            )
        elif p.spur_type == "border":
            return transforms.AddBorder(
                thickness=p.border_thickness,
                color=color_01,
            )
        elif p.spur_type == "tint":
            return transforms.AddColorTint(tint=color_01, alpha=p.tint_alpha)
        else:
            raise ValueError(f"Unknown spur_type '{p.spur_type}'")

    def _make_injectors(total_samples: int) -> List[transforms.ClassConditionalInjector]:
        """One ClassConditionalInjector per target label, each with its own color."""
        return [
            transforms.ClassConditionalInjector(
                transformation=_make_cue_transform(lbl),
                label_key="label",
                target_labels=[lbl],
                proportion=p.spur_proportion,
                total_samples=total_samples,
                seed=p.seed,
            )
            for lbl in sorted(spur_target_labels)
        ]

    train_injectors = _make_injectors(n_train_full)
    test_injectors  = _make_injectors(n_test_full)

    class _MapIdx:
        """Copy HFDataset's 'sample_idx' → 'idx' expected by ClassConditionalInjector."""
        def __call__(self, x):
            if "idx" not in x:
                x["idx"] = x.get("sample_idx", 0)
            return x

    _map_idx = _MapIdx()

    def _spur_transform(injectors):
        # ToImage: PIL → float32 tensor [C,H,W] in [0,1]  (required by AddPatch)
        # _map_idx: expose sample_idx as idx for ClassConditionalInjector
        # injectors: per-class cue injection
        return transforms.Compose(
            transforms.ToImage(source="img", target="img"),
            _map_idx,
            *injectors,
        )

    _clean_transform = transforms.Compose(
        transforms.ToImage(source="img", target="img"),
    )

    # ── datasets ──────────────────────────────────────────────────────────────
    print("  Loading datasets ...")
    ds_train_spur  = spt.data.Subset(
        spt.data.HFDataset(path=p.dataset, split=p.train_split,
                           transform=_spur_transform(train_injectors)),
        train_idx,
    )
    ds_train_clean = spt.data.Subset(
        spt.data.HFDataset(path=p.dataset, split=p.train_split,
                           transform=_clean_transform),
        train_idx,
    )
    ds_val = spt.data.Subset(
        spt.data.HFDataset(path=p.dataset, split=p.train_split,
                           transform=_clean_transform),
        val_idx,
    )
    ds_test_spur  = spt.data.HFDataset(
        path=p.dataset, split=p.test_split,
        transform=_spur_transform(test_injectors),
    )
    ds_test_clean = spt.data.HFDataset(
        path=p.dataset, split=p.test_split,
        transform=_clean_transform,
    )

    # ── Collate ───────────────────────────────────────────────────────────────
    # After ToImage the item["img"] is a float32 tensor [C,H,W] in [0,1].
    # Convert back to PIL so the backbone processor can normalise correctly.
    def make_collate_fn(active_injectors=None):
        def collate_fn(batch):
            images, labels, triggered_list = [], [], []
            for item in batch:
                img = item["img"]
                if isinstance(img, torch.Tensor):
                    img = to_pil(img.cpu())
                images.append(img)
                label      = int(item["label"])
                sample_idx = int(item.get("sample_idx", 0))
                labels.append(label)
                hit = (
                    active_injectors is not None
                    and any(
                        label in inj.target_labels
                        and inj.indices_to_transform is not None
                        and sample_idx in inj.indices_to_transform
                        for inj in active_injectors
                    )
                )
                triggered_list.append(hit)
            proc = processor(images=images, return_tensors="pt")
            return {
                "pixel_values": proc["pixel_values"],
                "label":        torch.tensor(labels,         dtype=torch.long),
                "triggered":    torch.tensor(triggered_list, dtype=torch.bool),
            }
        return collate_fn

    loader_kw = dict(batch_size=p.batch_size,
                     num_workers=p.num_workers,
                     pin_memory=(device.type == "cuda"))

    loader_train_spur  = DataLoader(ds_train_spur,  shuffle=True,
                                    collate_fn=make_collate_fn(train_injectors), **loader_kw)
    loader_train_clean = DataLoader(ds_train_clean, shuffle=True,
                                    collate_fn=make_collate_fn(),                **loader_kw)
    loader_val         = DataLoader(ds_val,          shuffle=False,
                                    collate_fn=make_collate_fn(),                **loader_kw)
    loader_test_spur   = DataLoader(ds_test_spur,    shuffle=False,
                                    collate_fn=make_collate_fn(test_injectors),  **loader_kw)
    loader_test_clean  = DataLoader(ds_test_clean,   shuffle=False,
                                    collate_fn=make_collate_fn(),                **loader_kw)

    # ── optional zero-shot CLIP baseline ──────────────────────────────────────
    zs_results = {}
    if p.zero_shot_clip and p.backbone in CLIP_MODELS:
        print("\n  Zero-shot CLIP evaluation ...")
        for name, loader in [("spurious_test", loader_test_spur),
                              ("clean_test",    loader_test_clean)]:
            zs_results[name] = zero_shot_clip_eval(
                p.backbone, loader, device, class_names, spur_target_labels)
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
        print(f"  Training: {train_name}  "
              f"[{'lora' if p.use_lora else p.finetune_mode}]")
        print(f"{'─'*72}")

        model = VisionClassifier(
            backbone_id=p.backbone,
            num_classes=num_classes,
            finetune_mode=p.finetune_mode,
            last_n_layers=p.last_n_layers,
            use_lora=p.use_lora,
            lora_rank=p.lora_rank,
            lora_alpha=p.lora_alpha,
            lora_dropout=p.lora_dropout,
        ).to(device)

        trainable = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=p.lr,
                                      weight_decay=p.weight_decay)

        total_steps = len(train_loader) * p.epochs
        warmup      = min(p.warmup_steps, total_steps // 10)

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler    = torch.cuda.amp.GradScaler() if (p.use_amp
                    and device.type == "cuda") else None

        best_val_acc = -1.0
        best_state   = None

        for epoch in range(1, p.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, scaler)

            val_metrics = evaluate(model, loader_val, device,
                                   spur_target_labels, class_names)
            val_acc = val_metrics["accuracy"]

            print(f"  Epoch {epoch:02d}/{p.epochs}  "
                  f"loss={tr_loss:.4f}  train_acc={tr_acc:.1f}%  "
                  f"val_acc={val_acc:.1f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict({k: v.to(device)
                                   for k, v in best_state.items()})
        print(f"  Best val acc: {best_val_acc:.1f}%")

        if p.save_model and best_state is not None:
            ckpt_path = Path(p.out).with_suffix("") / f"{train_name}_best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_path)
            print(f"  Saved checkpoint -> {ckpt_path}")

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
    print_results(p.backbone, all_results, class_names,
                  per_class_colors, spur_target_labels,
                  "lora" if p.use_lora else p.finetune_mode)

    # ── save JSON ─────────────────────────────────────────────────────────────
    out_data = {
        "config":    OmegaConf.to_container(cfg.params, resolve=True),
        "results":   all_results,
        "zero_shot": zs_results,
    }
    out_path = Path(p.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n  Results saved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
