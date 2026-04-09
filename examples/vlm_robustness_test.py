# -*- coding: utf-8 -*-
"""
CLIP Loss Spurious Robustness Probe
====================================
Hypothesis:
    CLIP's contrastive loss (with hard negative pairs) is what makes the
    vision encoder robust to spurious visual correlations.  A vision model
    trained with plain cross-entropy can memorise color-patch shortcuts;
    one trained with CLIP-style InfoNCE cannot, because negative pairs force
    the encoder to align with semantic content rather than surface statistics.

Experimental design -- four training conditions on the SAME spurious dataset:
    1. CE-spurious        Cross-entropy on spurious data          (baseline susceptible)
    2. CE-clean           Cross-entropy on clean data             (ceiling)
    3. CLIP-spurious      CLIP InfoNCE loss on spurious data      (hypothesis: robust)
    4. CLIP-clean         CLIP InfoNCE loss on clean data         (sanity check)

Each condition trains a vision encoder + projection head (for CLIP) or
linear classifier (for CE).  All four are evaluated on both spurious and
clean test sets -> 4 x 2 accuracy matrix.

Key metrics:
    shortcut_penalty  = CE_clean_test_acc  - CE_spur_train->clean_test_acc
    clip_robustness   = CLIP_spur_train->clean_test_acc - CE_spur_train->clean_test_acc
    If clip_robustness >> 0  ->  hypothesis supported

Architecture:
    Shared ViT/CLIP vision encoder (pretrained, optionally finetuned)
    CE path:   encoder -> LayerNorm -> Linear(embed_dim, n_classes)
    CLIP path: encoder -> Linear(embed_dim, proj_dim) -> InfoNCE vs text embeds

Negative pair strategies (configurable):
    in_batch      : all other items in the batch are negatives (standard CLIP)
    hard_negative : mix in same-color-different-class examples as hard negatives
    class_aware   : within-class pairs are positives, cross-class are negatives

Usage:
    python clip_loss_spurious_probe.py --config cifar10_clip_probe.yaml
    python clip_loss_spurious_probe.py --backbone openai/clip-vit-base-patch32 \\
        --dataset uoft-cs/cifar10 --spur-label 0 1 2 --epochs 15
"""

import os, gc, json, random, argparse, warnings, yaml, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPModel, CLIPProcessor,
    CLIPTextModel, CLIPVisionModel,
    AutoTokenizer, AutoImageProcessor,
    ViTModel,
)

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

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

CLIP_BACKBONES = {
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch16",
}
VIT_BACKBONES = {
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
}

DEFAULT_CLASS_COLORS: Dict[int, List[int]] = {
    0: [255,   0,   0],   1: [  0, 220,   0],
    2: [  0,   0, 255],   3: [255, 200,   0],
    4: [180,   0, 255],   5: [  0, 200, 200],
    6: [255, 120,   0],   7: [  0, 100, 255],
    8: [255,   0, 160],   9: [160, 160, 160],
}

# Class-name prompt templates for CLIP text encoder
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.",
    "an image of a {}.",
    "this is a {}.",
]

# -----------------------------------------------------------------------------
# Spurious cue helpers
# -----------------------------------------------------------------------------

def should_trigger(idx: int, label: int, *, seed: int,
                   proportion: float, target_labels: set) -> bool:
    if label not in target_labels:
        return False
    return (hash((seed, idx)) % 10_000_000) / 10_000_000 < proportion


def get_class_color(label: int,
                    per_class_colors: Dict[int, List[int]]) -> Tuple[int, int, int]:
    c = per_class_colors.get(label, DEFAULT_CLASS_COLORS.get(label, [255, 0, 0]))
    return tuple(c)


def apply_spurious_cue(img: Image.Image, spur_type: str, *,
        color: Tuple[int, int, int], patch_size: int = 10,
        patch_pos: str = "bottom_right", border_thickness: int = 4,
        tint_alpha: float = 0.35) -> Image.Image:
    img = img.copy().convert("RGB")
    w, h = img.size
    if spur_type == "patch":
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        pos = {
            "bottom_right": (w - patch_size, h - patch_size, w, h),
            "top_left":     (0, 0, patch_size, patch_size),
            "top_right":    (w - patch_size, 0, w, patch_size),
            "bottom_left":  (0, h - patch_size, patch_size, h),
            "center":       (w // 2 - patch_size // 2, h // 2 - patch_size // 2,
                             w // 2 + patch_size // 2, h // 2 + patch_size // 2),
        }
        draw.rectangle(pos.get(patch_pos, pos["bottom_right"]), fill=color)
    elif spur_type == "border":
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for t in range(border_thickness):
            draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=color)
    elif spur_type == "tint":
        overlay = Image.new("RGB", img.size, color)
        img = Image.blend(img, overlay, alpha=tint_alpha)
    else:
        raise ValueError(f"Unknown spur_type '{spur_type}'")
    return img

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class SpuriousDataset(Dataset):
    """
    HF dataset wrapper that injects per-class spurious cues.
    Returns pixel_values, label, triggered flag, and the class text prompt
    (needed for CLIP loss).
    """
    def __init__(self, hf_dataset, image_processor, *,
                 class_names: List[str],
                 spur_target_labels: set,
                 spur_proportion: float,
                 apply_spurious: bool,
                 spur_type: str,
                 per_class_colors: Dict[int, List[int]],
                 patch_size: int, patch_pos: str,
                 border_thickness: int, tint_alpha: float,
                 seed: int,
                 indices: Optional[List[int]] = None,
                 prompt_template: str = "a photo of a {}."):
        self.hf            = hf_dataset
        self.proc          = image_processor
        self.class_names   = class_names
        self.indices       = indices if indices is not None \
                             else list(range(len(hf_dataset)))
        self.spur_tgts     = spur_target_labels
        self.spur_prop     = spur_proportion
        self.apply_spur    = apply_spurious
        self.spur_type     = spur_type
        self.colors        = per_class_colors
        self.patch_size    = patch_size
        self.patch_pos     = patch_pos
        self.border_th     = border_thickness
        self.tint_alpha    = tint_alpha
        self.seed          = seed
        self.prompt_tmpl   = prompt_template

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
                                   proportion=self.spur_prop,
                                   target_labels=self.spur_tgts)
        if self.apply_spur and triggered:
            color = get_class_color(label, self.colors)
            img   = apply_spurious_cue(img, self.spur_type, color=color,
                        patch_size=self.patch_size, patch_pos=self.patch_pos,
                        border_thickness=self.border_th,
                        tint_alpha=self.tint_alpha)

        enc          = self.proc(images=img, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)
        text_prompt  = self.prompt_tmpl.format(self.class_names[label])

        return {
            "pixel_values": pixel_values,
            "label":        torch.tensor(label,     dtype=torch.long),
            "triggered":    torch.tensor(triggered, dtype=torch.bool),
            "text_prompt":  text_prompt,
            "idx":          torch.tensor(real_idx,  dtype=torch.long),
        }

# -----------------------------------------------------------------------------
# Vision encoder wrapper
# -----------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """
    Pretrained vision encoder with optional partial unfreezing.
    Returns [B, embed_dim] CLS embeddings.
    """
    def __init__(self, backbone_id: str, finetune_mode: str = "frozen",
                 last_n_layers: int = 2):
        super().__init__()
        self.backbone_id = backbone_id

        if backbone_id in CLIP_BACKBONES:
            clip = CLIPModel.from_pretrained(backbone_id)
            self.encoder  = clip.vision_model
            self.embed_dim = clip.config.vision_config.hidden_size
            del clip
        elif backbone_id in VIT_BACKBONES:
            self.encoder   = ViTModel.from_pretrained(backbone_id)
            self.embed_dim = self.encoder.config.hidden_size
        else:
            raise ValueError(f"Unsupported backbone '{backbone_id}'")

        self._set_finetune_mode(finetune_mode, last_n_layers)

    def _set_finetune_mode(self, mode: str, last_n: int):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        if mode == "frozen":
            pass
        elif mode == "full":
            for p in self.encoder.parameters():
                p.requires_grad_(True)
        elif mode == "last_n_layers":
            enc = getattr(self.encoder, "encoder",
                  getattr(self.encoder, "transformer", None))
            if enc is not None:
                layers = getattr(enc, "layers",
                         getattr(enc, "blocks", []))
                for layer in list(layers)[-last_n:]:
                    for p in layer.parameters():
                        p.requires_grad_(True)
            for name in ("layernorm", "post_layernorm", "layer_norm",
                         "ln_post", "norm"):
                m = getattr(self.encoder, name, None)
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad_(True)
        else:
            raise ValueError(f"Unknown finetune_mode '{mode}'")

        n_total = sum(p.numel() for p in self.parameters())
        n_tune  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  VisionEncoder: {n_total:,} params, "
              f"{n_tune:,} trainable ({100 * n_tune / max(n_total, 1):.1f}%)")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]

# -----------------------------------------------------------------------------
# CE model
# -----------------------------------------------------------------------------

class CEModel(nn.Module):
    def __init__(self, encoder: VisionEncoder, num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head    = nn.Sequential(
            nn.LayerNorm(encoder.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder.embed_dim, num_classes),
        )
        for p in self.head.parameters():
            p.requires_grad_(True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats  = self.encoder(pixel_values)
        logits = self.head(feats)
        return logits

# -----------------------------------------------------------------------------
# CLIP contrastive model
# -----------------------------------------------------------------------------

class CLIPContrastiveModel(nn.Module):
    def __init__(self, encoder: VisionEncoder, backbone_id: str,
                 proj_dim: int = 512, temperature: float = 0.07,
                 negative_strategy: str = "in_batch"):
        super().__init__()
        self.encoder   = encoder
        self.temp      = nn.Parameter(torch.tensor(math.log(1 / temperature)))
        self.neg_strat = negative_strategy

        D = encoder.embed_dim
        self.vis_proj = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, proj_dim),
        )
        for p in self.vis_proj.parameters():
            p.requires_grad_(True)

        self.text_encoder = CLIPTextModel.from_pretrained(backbone_id)
        self.text_proj    = nn.Linear(
            self.text_encoder.config.hidden_size, proj_dim, bias=False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.text_proj.parameters():
            p.requires_grad_(True)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(pixel_values)
        proj  = self.vis_proj(feats)
        return F.normalize(proj, dim=-1)

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.text_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask)
        text_feats = out.last_hidden_state[
            torch.arange(out.last_hidden_state.size(0)),
            attention_mask.sum(1) - 1
        ]
        proj = self.text_proj(text_feats)
        return F.normalize(proj, dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask,
                labels=None) -> torch.Tensor:
        img_emb = self.encode_images(pixel_values)
        txt_emb = self.encode_text(input_ids, attention_mask)

        logit_scale = self.temp.exp().clamp(max=100)
        logits_i2t  = logit_scale * img_emb @ txt_emb.T
        logits_t2i  = logits_i2t.T

        B = img_emb.size(0)

        if self.neg_strat == "class_aware" and labels is not None:
            lbl_mat  = labels.unsqueeze(0) == labels.unsqueeze(1)
            targets  = lbl_mat.float()
            targets /= targets.sum(dim=1, keepdim=True).clamp(min=1)
            loss_i2t = -(targets * F.log_softmax(logits_i2t, dim=1)).sum(1).mean()
            loss_t2i = -(targets * F.log_softmax(logits_t2i, dim=1)).sum(1).mean()
        else:
            tgts     = torch.arange(B, device=pixel_values.device)
            loss_i2t = F.cross_entropy(logits_i2t, tgts)
            loss_t2i = F.cross_entropy(logits_t2i, tgts)

        return (loss_i2t + loss_t2i) / 2.0

    @torch.inference_mode()
    def classify_zero_shot(self, pixel_values: torch.Tensor,
                           text_embeddings: torch.Tensor) -> torch.Tensor:
        img_emb = self.encode_images(pixel_values)
        sims    = img_emb @ text_embeddings.T
        return sims.argmax(dim=1)

# -----------------------------------------------------------------------------
# Build class text embeddings
# -----------------------------------------------------------------------------

@torch.inference_mode()
def build_class_text_embeddings(clip_model: CLIPContrastiveModel,
                                 tokenizer,
                                 class_names: List[str],
                                 device: torch.device,
                                 templates: List[str] = PROMPT_TEMPLATES
                                 ) -> torch.Tensor:
    class_embeds = []
    for cname in class_names:
        tmpl_embeds = []
        for tmpl in templates:
            prompt = tmpl.format(cname)
            tok    = tokenizer([prompt], return_tensors="pt",
                               padding=True, truncation=True).to(device)
            emb    = clip_model.encode_text(tok["input_ids"],
                                            tok["attention_mask"])
            tmpl_embeds.append(emb)
        avg = torch.stack(tmpl_embeds).mean(0)
        class_embeds.append(F.normalize(avg, dim=-1))
    return torch.cat(class_embeds, dim=0)

# -----------------------------------------------------------------------------
# Hard-negative collate function
# -----------------------------------------------------------------------------

def make_hard_negative_collate(dataset: SpuriousDataset,
                                spur_target_labels: set,
                                per_class_colors: Dict[int, List[int]],
                                spur_type: str,
                                patch_size: int, patch_pos: str,
                                border_thickness: int, tint_alpha: float):
    all_colors = list(per_class_colors.values())

    def collate_fn(batch):
        extra = []
        for item in batch:
            if not item["triggered"].item():
                continue
            lbl = item["label"].item()
            if lbl not in spur_target_labels:
                continue

            orig_color   = get_class_color(lbl, per_class_colors)
            other_colors = [c for c in all_colors if tuple(c) != orig_color]
            if not other_colors:
                continue

            wrong_lbl  = random.choice(
                [l for l in spur_target_labels if l != lbl])
            wrong_text = f"a photo of a {dataset.class_names[wrong_lbl]}."
            hard_neg   = {
                "pixel_values": item["pixel_values"].clone(),
                "label":        torch.tensor(wrong_lbl, dtype=torch.long),
                "triggered":    torch.tensor(True,      dtype=torch.bool),
                "text_prompt":  wrong_text,
                "idx":          item["idx"],
            }
            extra.append(hard_neg)

        combined = batch + extra
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in combined]),
            "label":        torch.stack([x["label"]        for x in combined]),
            "triggered":    torch.stack([x["triggered"]    for x in combined]),
            "text_prompt":  [x["text_prompt"]              for x in combined],
            "idx":          torch.stack([x["idx"]          for x in combined]),
        }

    return collate_fn


def default_collate(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "label":        torch.stack([x["label"]        for x in batch]),
        "triggered":    torch.stack([x["triggered"]    for x in batch]),
        "text_prompt":  [x["text_prompt"]              for x in batch],
        "idx":          torch.stack([x["idx"]          for x in batch]),
    }

# -----------------------------------------------------------------------------
# Training loops
# -----------------------------------------------------------------------------

def train_ce_epoch(model: CEModel, loader, optimizer, scheduler,
                   device, scaler=None) -> Tuple[float, float]:
    model.train()
    total_loss = correct = total = 0

    for batch in tqdm(loader, desc="    CE train", leave=False):
        pv     = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(pv)
                loss   = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pv)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / max(total, 1), 100 * correct / max(total, 1)


def train_clip_epoch(model: CLIPContrastiveModel, loader, tokenizer,
                     optimizer, scheduler, device,
                     scaler=None) -> float:
    model.train()
    total_loss = total = 0

    for batch in tqdm(loader, desc="    CLIP train", leave=False):
        pv     = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)
        texts  = batch["text_prompt"]

        tok = tokenizer(texts, return_tensors="pt",
                        padding=True, truncation=True).to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                loss = model(pv, tok["input_ids"],
                             tok["attention_mask"], labels=labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(pv, tok["input_ids"],
                         tok["attention_mask"], labels=labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler:
            scheduler.step()
        total_loss += loss.item() * pv.size(0)
        total      += pv.size(0)

    return total_loss / max(total, 1)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@torch.inference_mode()
def eval_ce(model: CEModel, loader, device,
            spur_target_labels: set, class_names: List[str]) -> dict:
    model.eval()
    records = []
    for batch in tqdm(loader, desc="    CE eval", leave=False):
        pv      = batch["pixel_values"].to(device)
        labels  = batch["label"]
        trigged = batch["triggered"]
        preds   = model(pv).argmax(1).cpu()
        for gt, pr, tr in zip(labels.tolist(), preds.tolist(), trigged.tolist()):
            records.append((gt, pr, bool(tr)))
    return _aggregate_records(records, spur_target_labels, class_names)


@torch.inference_mode()
def eval_clip(model: CLIPContrastiveModel, loader,
              text_embeddings: torch.Tensor,
              device, spur_target_labels: set,
              class_names: List[str]) -> dict:
    model.eval()
    records = []
    for batch in tqdm(loader, desc="    CLIP eval", leave=False):
        pv      = batch["pixel_values"].to(device)
        labels  = batch["label"]
        trigged = batch["triggered"]
        preds   = model.classify_zero_shot(pv, text_embeddings).cpu()
        for gt, pr, tr in zip(labels.tolist(), preds.tolist(), trigged.tolist()):
            records.append((gt, pr, bool(tr)))
    return _aggregate_records(records, spur_target_labels, class_names)


def _aggregate_records(records, spur_target_labels, class_names):
    total    = len(records)
    correct  = sum(g == p for g, p, _ in records)
    accuracy = 100 * correct / max(total, 1)

    spur_recs    = [(g, p) for g, p, t in records if t]
    spur_correct = sum(g == p for g, p in spur_recs)
    spur_flips   = sum(p != g and p in spur_target_labels
                       for g, p in spur_recs)
    spur_acc     = 100 * spur_correct  / max(len(spur_recs), 1)
    spur_flip_rt = 100 * spur_flips    / max(len(spur_recs), 1)

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
        "accuracy":       round(accuracy,     2),
        "spur_acc":       round(spur_acc,     2),
        "clean_acc":      round(clean_acc,    2),
        "spur_flip_rate": round(spur_flip_rt, 2),
        "n_total":        total,
        "n_triggered":    len(spur_recs),
        "n_clean":        len(clean_recs),
        "per_class_acc":  per_class_acc,
    }

# -----------------------------------------------------------------------------
# Pretty-print
# -----------------------------------------------------------------------------

def print_results(all_results: dict, class_names: List[str],
                  per_class_colors: Dict, spur_target_labels: set,
                  backbone_id: str):
    name = backbone_id.split("/")[-1]
    print(f"\n{'=' * 80}")
    print(f"  CLIP Loss Spurious Robustness Results -- {name}")
    print(f"{'=' * 80}")
    print(f"\n  {'Condition':<40} {'Acc':>6} {'SpurAcc':>8} "
          f"{'CleanAcc':>9} {'SpurFlip':>9}")
    print(f"  {'-' * 75}")

    order = [
        ("CE   | spurious_train -> spurious_test",  "CE_spur_train__spur_test"),
        ("CE   | spurious_train -> clean_test",     "CE_spur_train__clean_test"),
        ("CE   | clean_train    -> spurious_test",  "CE_clean_train__spur_test"),
        ("CE   | clean_train    -> clean_test",     "CE_clean_train__clean_test"),
        ("CLIP | spurious_train -> spurious_test",  "CLIP_spur_train__spur_test"),
        ("CLIP | spurious_train -> clean_test",     "CLIP_spur_train__clean_test"),
        ("CLIP | clean_train    -> spurious_test",  "CLIP_clean_train__spur_test"),
        ("CLIP | clean_train    -> clean_test",     "CLIP_clean_train__clean_test"),
    ]
    for label, key in order:
        m = all_results.get(key, {})
        if not m:
            continue
        print(f"  {label:<40} "
              f"{m.get('accuracy', 0):>5.1f}% "
              f"{m.get('spur_acc', 0):>7.1f}% "
              f"{m.get('clean_acc', 0):>8.1f}% "
              f"{m.get('spur_flip_rate', 0):>8.1f}%")

    print(f"\n  {'-' * 75}")
    print(f"  Key Comparisons (all on CLEAN test set):")

    ce_ceil   = all_results.get("CE_clean_train__clean_test",   {}).get("accuracy", 0)
    ce_spur   = all_results.get("CE_spur_train__clean_test",    {}).get("accuracy", 0)
    clip_ceil = all_results.get("CLIP_clean_train__clean_test", {}).get("accuracy", 0)
    clip_spur = all_results.get("CLIP_spur_train__clean_test",  {}).get("accuracy", 0)

    ce_penalty   = ce_ceil   - ce_spur
    clip_penalty = clip_ceil - clip_spur
    advantage    = clip_spur - ce_spur

    print(f"  CE   shortcut penalty (ceiling - spur_train): {ce_penalty:+.1f}%")
    print(f"  CLIP shortcut penalty (ceiling - spur_train): {clip_penalty:+.1f}%")
    print(f"  CLIP robustness advantage over CE:            {advantage:+.1f}%")

    print(f"\n  Interpretation:")
    if advantage > 5:
        print(f"  [+] HYPOTHESIS SUPPORTED -- CLIP loss confers {advantage:.1f}% "
              f"robustness advantage vs CE")
        print(f"      The contrastive objective suppresses spurious visual shortcuts.")
    elif advantage > 1:
        print(f"  [~] PARTIAL SUPPORT -- CLIP shows {advantage:.1f}% advantage "
              f"(moderate robustness)")
    else:
        print(f"  [-] HYPOTHESIS NOT SUPPORTED -- CE and CLIP are similarly susceptible")
        print(f"      The spurious cue may be too strong, or backbone is the bottleneck.")

    print(f"\n  Per-class spurious color legend:")
    for idx in sorted(spur_target_labels):
        cn = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        c  = get_class_color(idx, per_class_colors)
        print(f"    [{idx}] {cn:<15}  RGB{c}")

# -----------------------------------------------------------------------------
# YAML loader  -- FIX: explicit UTF-8 encoding
# -----------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    # Always open config files with UTF-8 to avoid cp1252 decode errors on Windows
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "per_class_colors" in cfg and cfg["per_class_colors"]:
        cfg["per_class_colors"] = {int(k): list(v)
                                   for k, v in cfg["per_class_colors"].items()}
    if "spur_labels" in cfg and cfg["spur_labels"]:
        cfg["spur_labels"] = [int(x) for x in cfg["spur_labels"]]
    for fk in ("lr", "weight_decay", "spur_proportion", "tint_alpha",
               "val_fraction", "temperature", "clip_lr"):
        if fk in cfg and cfg[fk] is not None:
            cfg[fk] = float(cfg[fk])
    for ik in ("epochs", "batch_size", "warmup_steps", "num_workers",
               "last_n_layers", "patch_size", "border_thickness",
               "seed", "proj_dim"):
        if ik in cfg and cfg[ik] is not None:
            cfg[ik] = int(cfg[ik])
    for bk in ("use_amp", "save_model", "run_ce", "run_clip"):
        if bk in cfg and cfg[bk] is not None:
            cfg[bk] = bool(cfg[bk])
    return cfg


_YAML_TO_ARG = {
    "dataset": "dataset", "train_split": "train_split",
    "test_split": "test_split", "backbone": "backbone",
    "finetune_mode": "finetune_mode", "last_n_layers": "last_n_layers",
    "spur_labels": "spur_label", "spur_type": "spur_type",
    "spur_proportion": "spur_proportion", "patch_size": "patch_size",
    "patch_pos": "patch_pos", "border_thickness": "border_thickness",
    "tint_alpha": "tint_alpha", "epochs": "epochs",
    "batch_size": "batch_size", "lr": "lr", "clip_lr": "clip_lr",
    "weight_decay": "weight_decay", "warmup_steps": "warmup_steps",
    "val_fraction": "val_fraction", "num_workers": "num_workers",
    "use_amp": "use_amp", "proj_dim": "proj_dim",
    "temperature": "temperature", "negative_strategy": "negative_strategy",
    "run_ce": "run_ce", "run_clip": "run_clip",
    "seed": "seed", "out": "out", "save_model": "save_model",
}


def merge_yaml(args, cfg):
    for yk, ak in _YAML_TO_ARG.items():
        if yk in cfg:
            setattr(args, ak, cfg[yk])
    return args

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="CLIP vs CE spurious robustness probe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config",            default=None)
    p.add_argument("--dataset",           default="uoft-cs/cifar10")
    p.add_argument("--train-split",       default="train")
    p.add_argument("--test-split",        default="test")
    p.add_argument("--backbone",          default="openai/clip-vit-base-patch32")
    p.add_argument("--finetune-mode",     default="frozen",
                   choices=["frozen", "last_n_layers", "full"])
    p.add_argument("--last-n-layers",     type=int, default=2)
    p.add_argument("--spur-label",        type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--spur-type",         default="patch",
                   choices=["patch", "border", "tint"])
    p.add_argument("--spur-proportion",   type=float, default=0.9)
    p.add_argument("--patch-size",        type=int,   default=10)
    p.add_argument("--patch-pos",         default="bottom_right",
                   choices=["bottom_right", "top_left", "top_right",
                             "bottom_left", "center"])
    p.add_argument("--border-thickness",  type=int,   default=4)
    p.add_argument("--tint-alpha",        type=float, default=0.35)
    p.add_argument("--epochs",            type=int,   default=15)
    p.add_argument("--batch-size",        type=int,   default=128)
    p.add_argument("--lr",                type=float, default=1e-3,
                   help="LR for CE head")
    p.add_argument("--clip-lr",           type=float, default=1e-4,
                   help="LR for CLIP projection head (usually smaller)")
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--warmup-steps",      type=int,   default=200)
    p.add_argument("--val-fraction",      type=float, default=0.1)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--use-amp",           action="store_true")
    p.add_argument("--proj-dim",          type=int,   default=512,
                   help="Projection dimension for CLIP head")
    p.add_argument("--temperature",       type=float, default=0.07,
                   help="Initial InfoNCE temperature")
    p.add_argument("--negative-strategy", default="in_batch",
                   choices=["in_batch", "hard_negative", "class_aware"],
                   help="How to construct negative pairs for CLIP loss")
    p.add_argument("--run-ce",            action="store_true", default=True)
    p.add_argument("--run-clip",          action="store_true", default=True)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--out",               default="clip_loss_probe_results.json")
    p.add_argument("--save-model",        action="store_true")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Helper: build optimizer + cosine scheduler
# -----------------------------------------------------------------------------

def build_optimizer_scheduler(model, lr, weight_decay,
                               warmup_steps, total_steps):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    per_class_colors: Dict[int, List[int]] = {}
    if args.config:
        yaml_cfg         = load_yaml_config(args.config)
        per_class_colors = yaml_cfg.pop("per_class_colors", {})
        args             = merge_yaml(args, yaml_cfg)
        print(f"  Loaded config: {args.config}")

    for idx, color in DEFAULT_CLASS_COLORS.items():
        if idx not in per_class_colors:
            per_class_colors[idx] = color

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if args.dataset not in CLASS_NAMES:
        raise ValueError(f"Unknown dataset '{args.dataset}'")
    class_names        = CLASS_NAMES[args.dataset]
    num_classes        = len(class_names)
    spur_target_labels = set(args.spur_label)

    print(f"\n{'=' * 80}")
    print(f"  CLIP Loss Spurious Robustness Probe")
    print(f"{'=' * 80}")
    print(f"  Dataset          : {args.dataset}  ({num_classes} classes)")
    print(f"  Backbone         : {args.backbone}")
    print(f"  Finetune mode    : {args.finetune_mode}")
    print(f"  Spur type        : {args.spur_type}")
    print(f"  Spur labels      : {sorted(spur_target_labels)}")
    print(f"  Negative strategy: {args.negative_strategy}")
    print(f"  Epochs / BS      : {args.epochs} / {args.batch_size}")
    print(f"  LR (CE / CLIP)   : {args.lr} / {args.clip_lr}")

    # -- Processor & tokenizer -------------------------------------------------
    print("\n  Loading processor & tokenizer ...")
    if args.backbone in CLIP_BACKBONES:
        clip_proc  = CLIPProcessor.from_pretrained(args.backbone)
        image_proc = clip_proc.image_processor
        tokenizer  = clip_proc.tokenizer
    else:
        image_proc = AutoImageProcessor.from_pretrained(args.backbone)
        tokenizer  = AutoTokenizer.from_pretrained(args.backbone)

    # -- HF datasets -----------------------------------------------------------
    print("  Loading datasets ...")
    from datasets import load_dataset
    train_hf = load_dataset(args.dataset, split=args.train_split)
    test_hf  = load_dataset(args.dataset, split=args.test_split)

    n_train  = len(train_hf)
    n_val    = int(n_train * args.val_fraction)
    all_idx  = list(range(n_train))
    random.shuffle(all_idx)
    val_idx   = all_idx[:n_val]
    train_idx = all_idx[n_val:]
    print(f"  Train: {len(train_idx):,}  Val: {n_val:,}  Test: {len(test_hf):,}")

    # -- Common dataset kwargs -------------------------------------------------
    ds_kw = dict(
        image_processor=image_proc,
        class_names=class_names,
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

    ds_train_spur  = SpuriousDataset(train_hf, apply_spurious=True,
                                     indices=train_idx, **ds_kw)
    ds_train_clean = SpuriousDataset(train_hf, apply_spurious=False,
                                     indices=train_idx, **ds_kw)
    ds_val         = SpuriousDataset(train_hf, apply_spurious=False,
                                     indices=val_idx, **ds_kw)
    ds_test_spur   = SpuriousDataset(test_hf,  apply_spurious=True,  **ds_kw)
    ds_test_clean  = SpuriousDataset(test_hf,  apply_spurious=False, **ds_kw)

    hard_neg_collate = make_hard_negative_collate(
        ds_train_spur, spur_target_labels, per_class_colors,
        args.spur_type, args.patch_size, args.patch_pos,
        args.border_thickness, args.tint_alpha)

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    collate = (hard_neg_collate
               if args.negative_strategy == "hard_negative"
               else default_collate)

    loader_train_spur  = DataLoader(ds_train_spur,  shuffle=True,
                                    collate_fn=collate, **loader_kw)
    loader_train_clean = DataLoader(ds_train_clean, shuffle=True,
                                    collate_fn=default_collate, **loader_kw)
    loader_val         = DataLoader(ds_val,         shuffle=False,
                                    collate_fn=default_collate, **loader_kw)
    loader_test_spur   = DataLoader(ds_test_spur,   shuffle=False,
                                    collate_fn=default_collate, **loader_kw)
    loader_test_clean  = DataLoader(ds_test_clean,  shuffle=False,
                                    collate_fn=default_collate, **loader_kw)

    all_results = {}
    scaler = torch.cuda.amp.GradScaler() if (args.use_amp
             and device.type == "cuda") else None

    # ==========================================================================
    # CE training
    # ==========================================================================
    if args.run_ce:
        for train_name, train_loader in [
                ("spur_train", loader_train_spur),
                ("clean_train", loader_train_clean)]:
            print(f"\n{'-' * 80}")
            print(f"  CE training: {train_name}")
            print(f"{'-' * 80}")

            enc   = VisionEncoder(args.backbone, args.finetune_mode,
                                  args.last_n_layers).to(device)
            model = CEModel(enc, num_classes).to(device)
            total_steps = len(train_loader) * args.epochs
            opt, sched  = build_optimizer_scheduler(
                model, args.lr, args.weight_decay,
                args.warmup_steps, total_steps)

            best_val, best_state = -1.0, None
            for epoch in range(1, args.epochs + 1):
                tr_loss, tr_acc = train_ce_epoch(
                    model, train_loader, opt, sched, device, scaler)
                val_m   = eval_ce(model, loader_val, device,
                                  spur_target_labels, class_names)
                val_acc = val_m["accuracy"]
                print(f"  Epoch {epoch:02d}/{args.epochs}  "
                      f"loss={tr_loss:.4f}  train_acc={tr_acc:.1f}%  "
                      f"val_acc={val_acc:.1f}%")
                if val_acc > best_val:
                    best_val  = val_acc
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}

            model.load_state_dict(
                {k: v.to(device) for k, v in best_state.items()})
            print(f"  Best val acc: {best_val:.1f}%")

            if args.save_model:
                p = Path(args.out).with_suffix("") / f"CE_{train_name}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, p)

            print(f"  Evaluating CE {train_name} ...")
            spur_m  = eval_ce(model, loader_test_spur,  device,
                              spur_target_labels, class_names)
            clean_m = eval_ce(model, loader_test_clean, device,
                              spur_target_labels, class_names)
            all_results[f"CE_{train_name}__spur_test"]  = spur_m
            all_results[f"CE_{train_name}__clean_test"] = clean_m
            print(f"  CE [{train_name}] spur_test={spur_m['accuracy']:.1f}%  "
                  f"clean_test={clean_m['accuracy']:.1f}%  "
                  f"spur_flip={spur_m['spur_flip_rate']:.1f}%")

            del model, enc
            gc.collect()
            torch.cuda.empty_cache()

    # ==========================================================================
    # CLIP contrastive training
    # ==========================================================================
    if args.run_clip:
        for train_name, train_loader in [
                ("spur_train", loader_train_spur),
                ("clean_train", loader_train_clean)]:
            print(f"\n{'-' * 80}")
            print(f"  CLIP training: {train_name}  "
                  f"[neg_strategy={args.negative_strategy}]")
            print(f"{'-' * 80}")

            enc   = VisionEncoder(args.backbone, args.finetune_mode,
                                  args.last_n_layers).to(device)
            model = CLIPContrastiveModel(
                enc, args.backbone,
                proj_dim=args.proj_dim,
                temperature=args.temperature,
                negative_strategy=args.negative_strategy,
            ).to(device)

            total_steps = len(train_loader) * args.epochs
            opt, sched  = build_optimizer_scheduler(
                model, args.clip_lr, args.weight_decay,
                args.warmup_steps, total_steps)

            print("  Building class text embeddings ...")
            text_embs = build_class_text_embeddings(
                model, tokenizer, class_names, device)

            best_val, best_state = -1.0, None
            for epoch in range(1, args.epochs + 1):
                tr_loss = train_clip_epoch(
                    model, train_loader, tokenizer,
                    opt, sched, device, scaler)

                text_embs = build_class_text_embeddings(
                    model, tokenizer, class_names, device)
                val_m   = eval_clip(model, loader_val, text_embs,
                                    device, spur_target_labels, class_names)
                val_acc = val_m["accuracy"]
                print(f"  Epoch {epoch:02d}/{args.epochs}  "
                      f"loss={tr_loss:.4f}  val_acc={val_acc:.1f}%")
                if val_acc > best_val:
                    best_val  = val_acc
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}

            model.load_state_dict(
                {k: v.to(device) for k, v in best_state.items()})
            text_embs = build_class_text_embeddings(
                model, tokenizer, class_names, device)
            print(f"  Best val acc: {best_val:.1f}%")

            if args.save_model:
                p = Path(args.out).with_suffix("") / f"CLIP_{train_name}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, p)

            print(f"  Evaluating CLIP {train_name} ...")
            spur_m  = eval_clip(model, loader_test_spur,  text_embs,
                                device, spur_target_labels, class_names)
            clean_m = eval_clip(model, loader_test_clean, text_embs,
                                device, spur_target_labels, class_names)
            all_results[f"CLIP_{train_name}__spur_test"]  = spur_m
            all_results[f"CLIP_{train_name}__clean_test"] = clean_m
            print(f"  CLIP [{train_name}] spur_test={spur_m['accuracy']:.1f}%  "
                  f"clean_test={clean_m['accuracy']:.1f}%  "
                  f"spur_flip={spur_m['spur_flip_rate']:.1f}%")

            del model, enc
            gc.collect()
            torch.cuda.empty_cache()

    # -- final report ----------------------------------------------------------
    print_results(all_results, class_names, per_class_colors,
                  spur_target_labels, args.backbone)

    out_data = {
        "config": {
            "dataset":           args.dataset,
            "backbone":          args.backbone,
            "finetune_mode":     args.finetune_mode,
            "spur_type":         args.spur_type,
            "spur_labels":       sorted(spur_target_labels),
            "spur_proportion":   args.spur_proportion,
            "negative_strategy": args.negative_strategy,
            "epochs":            args.epochs,
            "batch_size":        args.batch_size,
            "lr":                args.lr,
            "clip_lr":           args.clip_lr,
            "temperature":       args.temperature,
            "seed":              args.seed,
            "per_class_colors":  {str(k): v
                                  for k, v in per_class_colors.items()},
        },
        "results": all_results,
    }

    # FIX: always write JSON with UTF-8 to avoid encoding errors on Windows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()