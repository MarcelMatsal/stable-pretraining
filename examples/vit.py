"""
Spurious Correlation Robustness — Full Experiment Suite
=======================================================

Research question:
  Is the CLIP contrastive loss (negative pairs) what makes the vision encoder
  robust to spurious correlations, or does freezing the text encoder matter too?

Ablation matrix (2 × 2 × 2):
  Loss      : CLIP contrastive (with negatives)  |  CE (no negatives)
  Mode      : vision-only (frozen text)           |  full CLIP (both encoders)
  Eval      : linear probe gap                    |  zero-shot accuracy drop

Stage 0  — Baseline CLIP (no finetuning) — zero-shot + linear probe
Stage 1  — 4 finetuning variants (2 losses × 2 modes)
             Each variant trains, then runs zero-shot and linear probe
Stage 2  — VLM ICL evaluation (Phi-3.5, LLaVA-7B)
Stage 3  — Cross-variant summary table + JSON dump
"""

import gc
import json
import random
import types
import wandb
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
import lightning as pl
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from peft import LoraConfig, get_peft_model
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    CLIPModel,
    CLIPProcessor,
    LlavaForConditionalGeneration,
)
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.callbacks import clip_zero_shot

warnings.filterwarnings("ignore")
to_pil = ToPILImage()

LLAVA_MODELS = {"llava-hf/llava-1.5-7b-hf"}
PHI_MODELS   = {"microsoft/Phi-3.5-vision-instruct"}
SUPPORTED_VLM = LLAVA_MODELS | PHI_MODELS


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VariantResult:
    """Holds all metrics for one (loss × mode) variant."""
    name: str                              # e.g. "contrastive_vision_only"
    loss_type: str                         # "contrastive" | "ce"
    mode: str                              # "vision_only" | "full_clip"

    # Zero-shot accuracies
    zs_clean_acc:  float = 0.0
    zs_spur_acc:   float = 0.0
    zs_drop:       float = 0.0            # zs_clean_acc - zs_spur_acc

    # Linear probe accuracies
    lp_clean_acc:  float = 0.0
    lp_spur_acc:   float = 0.0
    lp_gap:        float = 0.0            # lp_spur_acc - lp_clean_acc

    # Per-class breakdowns (filled later)
    zs_per_class_clean: Dict = field(default_factory=dict)
    zs_per_class_spur:  Dict = field(default_factory=dict)
    lp_per_class_clean: Dict = field(default_factory=dict)
    lp_per_class_spur:  Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def should_trigger(idx, label, *, seed, proportion, target_labels):
    if label not in target_labels:
        return False
    return (hash((seed, idx)) % 10_000_000) / 10_000_000 < proportion


def add_trigger(prompt, trigger, position):
    return f"{trigger} {prompt}" if position == "prepend" else f"{prompt} {trigger}"


def count_lora_params(peft_model):
    total = lora_total = 0
    for _, p in peft_model.named_parameters():
        total += p.numel()
        if p.requires_grad: lora_total += p.numel()
    return lora_total, total


def build_spurious_transforms(cfg, seed):
    """Return (train_xform, spur_test_xform, clean_xform)."""
    clean = transforms.Compose(transforms.ToImage(source="img", target="img"))
    if not cfg.params.use_spurious:
        return clean, clean, clean

    def _injector(target_labels, total):
        st = cfg.params.spur_type
        if st == "patch":
            aug = transforms.AddPatch(
                patch_size=cfg.params.patch_size, color=cfg.params.patch_color,
                position=cfg.params.patch_pos,    img_key=cfg.params.image_key)
        elif st == "border":
            aug = transforms.AddBorder(
                thickness=cfg.params.border_thickness, color=cfg.params.spur_color)
        elif st == "tint":
            aug = transforms.AddColorTint(
                tint=cfg.params.tint_color, alpha=cfg.params.spur_alpha)
        elif st == "watermark":
            aug = transforms.AddWatermark(
                watermark=cfg.params.watermark_path, size=cfg.params.watermark_size,
                position=cfg.params.watermak_pos,   alpha=cfg.params.spur_alpha)
        elif st == "checkerboard":
            aug = transforms.AddCheckerboardPattern(
                intensity=cfg.params.spur_alpha, image_label=cfg.params.image_key)
        else:
            raise ValueError(f"Unknown spur_type: {st}")
        return transforms.Compose(
            transforms.ToImage(source="img", target="img"),
            transforms.AddSampleIdx(),
            transforms.ClassConditionalInjector(
                transformation=aug, label_key=cfg.params.label_key,
                target_labels=target_labels, proportion=cfg.params.spur_proportion,
                total_samples=total, seed=seed))

    return (
        _injector(cfg.params.spur_train_label, cfg.params.total_train_samples),
        _injector(cfg.params.spur_test_label,  cfg.params.total_test_samples),
        clean,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Loss implementations
# ══════════════════════════════════════════════════════════════════════════════

class CLIPContrastiveLoss(nn.Module):
    """
    Standard CLIP InfoNCE loss over a batch of (image, text) pairs.
    Negative pairs are all off-diagonal elements in the similarity matrix.
    """
    def __init__(self, clip_model: CLIPModel):
        super().__init__()
        self.clip = clip_model

    def forward(self, input_ids, attention_mask, pixel_values):
        out  = self.clip(input_ids=input_ids, attention_mask=attention_mask,
                         pixel_values=pixel_values, return_loss=True)
        return out.loss


class CEVisionLoss(nn.Module):
    """
    Cross-entropy classification loss on CLIP vision features.
    No negative pairs — the model learns to map images to class logits
    via a linear head on top of the frozen-or-trained vision encoder.

    This deliberately removes the contrastive signal to test whether
    negative pairs are what confers spurious-correlation robustness.
    """
    def __init__(self, clip_model: CLIPModel, n_classes: int):
        super().__init__()
        self.clip   = clip_model
        feat_dim    = clip_model.config.projection_dim
        self.head   = nn.Linear(feat_dim, n_classes)

    def forward(self, pixel_values, labels):
        feats  = self.clip.get_image_features(pixel_values=pixel_values)
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        logits = self.head.to(feats.device)(feats)
        return F.cross_entropy(logits, labels.to(feats.device))


# ══════════════════════════════════════════════════════════════════════════════
# Backbone wrapper helpers
# ══════════════════════════════════════════════════════════════════════════════

class _ImgBackbone(nn.Module):
    def __init__(self, m): super().__init__(); self.clip = m
    def forward(self, pixel_values=None):
        d = next(self.clip.parameters()).device
        f = self.clip.get_image_features(pixel_values=pixel_values.to(d))
        return types.SimpleNamespace(image_embeds=f)

class _TxtBackbone(nn.Module):
    def __init__(self, m): super().__init__(); self.clip = m
    def forward(self, input_ids=None, attention_mask=None):
        d = next(self.clip.parameters()).device
        f = self.clip.get_text_features(input_ids=input_ids.to(d),
                                        attention_mask=attention_mask.to(d) if attention_mask is not None else None)
        return types.SimpleNamespace(text_embeds=f)


# ══════════════════════════════════════════════════════════════════════════════
# Dataloader builders
# ══════════════════════════════════════════════════════════════════════════════

def make_contrastive_dataloaders(cfg, class_names, seed, train_xform,
                                  spur_test_xform, processor,
                                  text_spur_labels):
    """Dataloaders for CLIP contrastive training (image + text caption pairs)."""

    def add_prompt_train(batch, indices):
        labels, prompts = batch[cfg.params.label_key], []
        for idx, lab in zip(indices, labels):
            lab  = int(lab)
            base = f"a photo of a {class_names[lab]}"
            if cfg.params.text_spur and should_trigger(
                idx, lab, seed=seed,
                proportion=cfg.params.spur_proportion,
                target_labels=text_spur_labels,
            ):
                base = add_trigger(base, cfg.params.spur_text_trigger,
                                   cfg.params.text_spur_location)
            prompts.append(base)
        batch["answer"] = prompts
        return batch

    def add_prompt_val(batch):
        batch["answer"] = [
            f"a photo of a {class_names[int(l)]}"
            for l in batch[cfg.params.label_key]]
        return batch

    train_ds = spt.data.HFDataset(path=cfg.params.dataset,
                                   split="train", transform=train_xform)
    val_ds   = spt.data.HFDataset(path=cfg.params.zeroshot_dataset,
                                   split="test",  transform=spur_test_xform)

    train_ds.dataset = train_ds.dataset.map(
        add_prompt_train, batched=True, with_indices=True,
        remove_columns=[], load_from_cache_file=False)
    val_ds.dataset = val_ds.dataset.map(
        add_prompt_val, batched=True,
        remove_columns=[], load_from_cache_file=False)

    def collate(batch):
        imgs, txts, lbls = [], [], []
        for item in batch:
            img = item["img"]
            if isinstance(img, torch.Tensor): img = to_pil(img.cpu())
            imgs.append(img); txts.append(item["answer"])
            lbls.append(int(item[cfg.params.label_key]))
        p = processor(text=txts, images=imgs, return_tensors="pt",
                      padding=True, truncation=True)
        result = {k: p[k] for k in ("input_ids", "attention_mask", "pixel_values")}
        result["labels"] = torch.tensor(lbls, dtype=torch.long)
        return result

    kw = dict(batch_size=cfg.params.batch_size, collate_fn=collate,
              num_workers=4, persistent_workers=True,
              multiprocessing_context="fork")
    return (torch.utils.data.DataLoader(train_ds, **kw),
            torch.utils.data.DataLoader(val_ds,   **kw))


def make_ce_dataloaders(cfg, class_names, train_xform, spur_test_xform, processor):
    """Dataloaders for CE training (image + integer label, no text)."""

    def collate(batch):
        imgs, lbls = [], []
        for item in batch:
            img = item.get("img", item.get("image"))
            if isinstance(img, torch.Tensor): img = to_pil(img.cpu())
            imgs.append(img)
            lbls.append(int(item.get("label", item.get("labels"))))
        p = processor(images=imgs, return_tensors="pt", padding=True)
        return {"pixel_values": p["pixel_values"],
                "labels": torch.tensor(lbls, dtype=torch.long)}

    train_ds = spt.data.HFDataset(path=cfg.params.dataset,
                                   split="train", transform=train_xform)
    val_ds   = spt.data.HFDataset(path=cfg.params.zeroshot_dataset,
                                   split="test",  transform=spur_test_xform)
    kw = dict(batch_size=cfg.params.batch_size, collate_fn=collate,
              num_workers=4, persistent_workers=True,
              multiprocessing_context="fork")
    return (torch.utils.data.DataLoader(train_ds, **kw),
            torch.utils.data.DataLoader(val_ds,   **kw))


# ══════════════════════════════════════════════════════════════════════════════
# Zero-shot evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def run_zero_shot_eval(clip_model, zero_processor, class_names,
                        clean_xform, spur_test_xform, cfg):
    """
    Run zero-shot classification on clean and spurious test splits.
    Returns (clean_acc, spur_acc, clean_per_class, spur_per_class).
    """
    device  = next(clip_model.parameters()).device
    n_cls   = len(class_names)
    clip_model.eval()

    # Build text embeddings for all class names once
    tok = zero_processor.tokenizer(
        [f"a photo of a {c}" for c in class_names],
        return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        txt_feats = clip_model.get_text_features(
            input_ids=tok["input_ids"].to(device),
            attention_mask=tok["attention_mask"].to(device))
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

    def collate_img(batch):
        imgs, lbls = [], []
        for item in batch:
            img = item.get("img", item.get("image"))
            if isinstance(img, torch.Tensor): img = to_pil(img.cpu())
            imgs.append(img)
            lbls.append(int(item.get("label", item.get("labels"))))
        p = zero_processor(images=imgs, return_tensors="pt", padding=True)
        return {"pixel_values": p["pixel_values"],
                "labels": torch.tensor(lbls, dtype=torch.long)}

    def eval_split(xform, split_name):
        ds = spt.data.HFDataset(path=cfg.params.zeroshot_dataset,
                                  split="test", transform=xform)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=cfg.params.batch_size, collate_fn=collate_img,
            num_workers=4, persistent_workers=True,
            multiprocessing_context="fork")

        correct = np.zeros(n_cls)
        total   = np.zeros(n_cls)
        for batch in tqdm(dl, desc=f"  ZS eval [{split_name}]", leave=False):
            pv = batch["pixel_values"].to(device)
            lb = batch["labels"]
            with torch.no_grad():
                img_f  = clip_model.get_image_features(pixel_values=pv)
                img_f  = img_f / img_f.norm(dim=-1, keepdim=True)
                sims   = img_f @ txt_feats.T
                preds  = sims.argmax(dim=-1).cpu()
            for gt, pr in zip(lb.tolist(), preds.tolist()):
                total[gt]   += 1
                if gt == pr: correct[gt] += 1

        per_cls = {class_names[i]: 100 * correct[i] / max(total[i], 1)
                   for i in range(n_cls)}
        overall = 100 * correct.sum() / max(total.sum(), 1)
        return overall, per_cls

    clean_acc, clean_pc = eval_split(clean_xform,     "clean")
    spur_acc,  spur_pc  = eval_split(spur_test_xform, "spurious")
    return clean_acc, spur_acc, clean_pc, spur_pc


# ══════════════════════════════════════════════════════════════════════════════
# Linear probe evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def extract_features(clip_model, dataloader, device):
    clip_model.eval()
    feats, labels = [], []
    for batch in tqdm(dataloader, desc="  Extracting features", leave=False):
        pv = batch["pixel_values"].to(device)
        f  = clip_model.get_image_features(pixel_values=pv)
        f  = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu()); labels.append(batch["labels"])
    return torch.cat(feats), torch.cat(labels)


def run_linear_probe(clip_model, processor, cfg, class_names,
                      train_xform, spur_test_xform, clean_xform):
    """
    Fit a linear probe on frozen vision features.
    Returns (lp_spur_acc, lp_clean_acc, spur_per_class, clean_per_class).
    """
    device = next(clip_model.parameters()).device
    n_cls  = len(class_names)

    def collate(batch):
        imgs, lbls = [], []
        for item in batch:
            img = item.get("img", item.get("image"))
            if isinstance(img, torch.Tensor): img = to_pil(img.cpu())
            imgs.append(img)
            lbls.append(int(item.get("label", item.get("labels"))))
        p = processor(images=imgs, return_tensors="pt", padding=True)
        return {"pixel_values": p["pixel_values"],
                "labels": torch.tensor(lbls, dtype=torch.long)}

    def make_dl(xform, split="test"):
        ds = spt.data.HFDataset(path=cfg.params.zeroshot_dataset,
                                  split=split, transform=xform)
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.params.batch_size, collate_fn=collate,
            num_workers=4, persistent_workers=True,
            multiprocessing_context="fork")

    train_dl      = make_dl(train_xform,     split="train")
    spur_test_dl  = make_dl(spur_test_xform, split="test")
    clean_test_dl = make_dl(clean_xform,     split="test")

    train_feats, train_lbls = extract_features(clip_model, train_dl,      device)
    spur_feats,  spur_lbls  = extract_features(clip_model, spur_test_dl,  device)
    clean_feats, clean_lbls = extract_features(clip_model, clean_test_dl, device)

    # Fit linear probe with L-BFGS
    head = nn.Linear(train_feats.shape[1], n_cls).to(device)
    opt  = torch.optim.LBFGS(head.parameters(), lr=0.1, max_iter=500)
    X, Y = train_feats.to(device), train_lbls.to(device)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(head(X), Y)
        loss.backward()
        return loss
    print("  Fitting linear probe (L-BFGS) …")
    opt.step(closure)

    head.eval()
    results = {}
    with torch.no_grad():
        for split_name, feats, lbls in [
            ("spurious_test", spur_feats,  spur_lbls),
            ("clean_test",    clean_feats, clean_lbls),
        ]:
            logits  = head(feats.to(device))
            preds   = logits.argmax(dim=-1).cpu()
            correct = np.zeros(n_cls)
            total   = np.zeros(n_cls)
            for gt, pr in zip(lbls.tolist(), preds.tolist()):
                total[gt]   += 1
                if gt == pr: correct[gt] += 1
            per_cls = {class_names[i]: 100 * correct[i] / max(total[i], 1)
                       for i in range(n_cls)}
            overall = 100 * correct.sum() / max(total.sum(), 1)
            results[split_name] = {"accuracy": overall, "per_class": per_cls}
            print(f"  [{split_name}]  acc={overall:.1f}%")

    return (results["spurious_test"]["accuracy"],
            results["clean_test"]["accuracy"],
            results["spurious_test"]["per_class"],
            results["clean_test"]["per_class"])


# ══════════════════════════════════════════════════════════════════════════════
# Freeze helpers
# ══════════════════════════════════════════════════════════════════════════════

def freeze_text_encoder(clip_model):
    for name, p in clip_model.named_parameters():
        if "text_model" in name or "text_projection" in name:
            p.requires_grad = False
    trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"  Text encoder frozen  |  trainable params = {trainable:,}")


def apply_vision_lora(clip_model, cfg):
    lora_cfg = LoraConfig(
        r=cfg.params.lora_rank, lora_alpha=cfg.params.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=cfg.params.lora_dropout, bias="none")
    clip_model.vision_model = get_peft_model(clip_model.vision_model, lora_cfg)
    lora_p, total_p = count_lora_params(clip_model.vision_model)
    print(f"  Vision LoRA  {lora_p:,} / {total_p:,} trainable")
    return clip_model


# ══════════════════════════════════════════════════════════════════════════════
# Core training runner — one variant
# ══════════════════════════════════════════════════════════════════════════════

def train_variant(
    variant_name: str,
    loss_type: str,           # "contrastive" | "ce"
    freeze_text: bool,        # True → vision-only, False → full CLIP
    cfg, class_names, seed,
    train_xform, spur_test_xform, clean_xform,
    wandb_logger,
) -> CLIPModel:
    """
    Train one (loss × mode) variant and return the finetuned clip_model.
    """
    print(f"\n{'─'*72}")
    print(f"  Variant: {variant_name}")
    print(f"  Loss: {loss_type}  |  freeze_text: {freeze_text}")
    print(f"{'─'*72}")

    TEXT_SPUR_LABELS = set(cfg.params.spur_text_labels)
    clip_model = CLIPModel.from_pretrained(cfg.params.clip_configuration)
    processor  = CLIPProcessor.from_pretrained(cfg.params.clip_configuration)
    n_cls      = len(class_names)

    if freeze_text:
        freeze_text_encoder(clip_model)
    if cfg.params.use_lora:
        clip_model = apply_vision_lora(clip_model, cfg)

    # ── Build loss module ─────────────────────────────────────────────────────
    if loss_type == "contrastive":
        loss_module = CLIPContrastiveLoss(clip_model)
        train_dl, val_dl = make_contrastive_dataloaders(
            cfg, class_names, seed, train_xform, spur_test_xform,
            processor, TEXT_SPUR_LABELS)

        def forward(self, batch, stage=None):
            loss = loss_module(
                batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
            tag = "train" if (self.training or stage == "train") else "val"
            self.log(f"{tag}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            return {"loss": loss}

    elif loss_type == "ce":
        # CE head lives on clip_model's projection dim — updated jointly
        loss_module = CEVisionLoss(clip_model, n_cls)
        train_dl, val_dl = make_ce_dataloaders(
            cfg, class_names, train_xform, spur_test_xform, processor)

        def forward(self, batch, stage=None):
            loss = loss_module(batch["pixel_values"], batch["labels"])
            tag = "train" if (self.training or stage == "train") else "val"
            self.log(f"{tag}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            return {"loss": loss}
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    def validation_step(self, batch, batch_idx):
        return {"pixel_values": batch.get("pixel_values"),
                "labels":       batch.get("labels")}

    # ── Zero-shot callback (only meaningful for contrastive, kept for comparison) ──
    zero_proc    = CLIPProcessor.from_pretrained(cfg.params.clip_configuration)
    img_backbone = _ImgBackbone(clip_model)
    txt_backbone = _TxtBackbone(clip_model)

    zs_cb = clip_zero_shot.CLIPZeroShot(
        name=f"zeroshot_clean_{variant_name}",
        image_key="pixel_values", class_key="labels",
        class_names=class_names,
        image_backbone=img_backbone, text_backbone=txt_backbone,
        tokenizer_fn=lambda x: zero_proc.tokenizer(
            [f"a photo of a {c}" for c in x],
            return_tensors="pt", padding=True, truncation=True)["input_ids"],
        metrics={
            "top1": tm.classification.MulticlassAccuracy(n_cls),
            "top5": tm.classification.MulticlassAccuracy(n_cls, top_k=5),
        },
    )

    module = spt.Module(
        backbone=clip_model,
        forward=forward,
        hparams=cfg,
        optim={
            "optimizer": {"type": "AdamW",
                          "lr": cfg.params.learning_rate,
                          "weight_decay": cfg.params.weight_decay},
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )
    module.validation_step = types.MethodType(validation_step, module)

    trainer = pl.Trainer(
        max_epochs=cfg.params.epochs,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[zs_cb],
    )
    spt.Manager(
        trainer=trainer, module=module,
        data=spt.data.DataModule(train=train_dl, val=val_dl),
    )()

    # If CE, detach the head — we only want the vision encoder for eval
    if loss_type == "ce":
        del loss_module.head

    print(f"  [done] {variant_name}")
    return clip_model, processor, zero_proc


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 — Baseline (no finetuning)
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(cfg, class_names, train_xform, spur_test_xform, clean_xform):
    print("\n" + "=" * 72)
    print("STAGE 0 — Baseline CLIP (no finetuning)")
    print("=" * 72)
    clip_model = CLIPModel.from_pretrained(cfg.params.clip_configuration)
    processor  = CLIPProcessor.from_pretrained(cfg.params.clip_configuration)

    zs_clean, zs_spur, zs_pc_clean, zs_pc_spur = run_zero_shot_eval(
        clip_model, processor, class_names, clean_xform, spur_test_xform, cfg)
    lp_spur, lp_clean, lp_pc_spur, lp_pc_clean = run_linear_probe(
        clip_model, processor, cfg, class_names,
        train_xform, spur_test_xform, clean_xform)

    r = VariantResult(
        name="baseline", loss_type="none", mode="none",
        zs_clean_acc=zs_clean, zs_spur_acc=zs_spur,
        zs_drop=zs_clean - zs_spur,
        lp_clean_acc=lp_clean, lp_spur_acc=lp_spur,
        lp_gap=lp_spur - lp_clean,
        zs_per_class_clean=zs_pc_clean, zs_per_class_spur=zs_pc_spur,
        lp_per_class_clean=lp_pc_clean, lp_per_class_spur=lp_pc_spur,
    )
    print(f"  ZS  clean={zs_clean:.1f}%  spur={zs_spur:.1f}%  drop={r.zs_drop:.1f}%")
    print(f"  LP  clean={lp_clean:.1f}%  spur={lp_spur:.1f}%  gap={r.lp_gap:.1f}%")
    return r


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — 2×2 ablation
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(cfg, class_names, seed, train_xform, spur_test_xform, clean_xform):
    """
    Run 4 variants:
      contrastive_vision_only  — CLIP loss, text frozen
      contrastive_full_clip    — CLIP loss, both encoders train
      ce_vision_only           — CE loss,   text frozen
      ce_full_clip             — CE loss,   both encoders train
    """
    print("\n" + "=" * 72)
    print("STAGE 1 — 2×2 Loss × Mode ablation")
    print("=" * 72)

    variants = [
        # (name,                       loss_type,     freeze_text)
        ("contrastive_vision_only",   "contrastive",  True),
        ("contrastive_full_clip",     "contrastive",  False),
        ("ce_vision_only",            "ce",           True),
        ("ce_full_clip",              "ce",           False),
    ]

    results: List[VariantResult] = []

    for vname, loss_type, freeze_text in variants:
        mode = "vision_only" if freeze_text else "full_clip"

        wandb_logger = WandbLogger(
            entity="rbalestr-brown", project="clip_caption_injection",
            name=f"spurious_ablation_2x2_{vname}",
            group="spurious_ablation_2x2",
            config=OmegaConf.to_container(cfg.params, resolve=True),
            log_model=False,
            reinit=True,
        )

        clip_model, processor, zero_proc = train_variant(
            variant_name=vname,
            loss_type=loss_type,
            freeze_text=freeze_text,
            cfg=cfg, class_names=class_names, seed=seed,
            train_xform=train_xform,
            spur_test_xform=spur_test_xform,
            clean_xform=clean_xform,
            wandb_logger=wandb_logger,
        )

            print(f"\n  Evaluating {vname} …")
            # zs_clean, zs_spur, zs_pc_clean, zs_pc_spur = run_zero_shot_eval(
            #     clip_model, zero_proc, class_names, clean_xform, spur_test_xform, cfg)
            # lp_spur, lp_clean, lp_pc_spur, lp_pc_clean = run_linear_probe(
            #     clip_model, processor, cfg, class_names,
            #     train_xform, spur_test_xform, clean_xform)

            # vr = VariantResult(
            #     name=vname, loss_type=loss_type, mode=mode,
            #     zs_clean_acc=zs_clean, zs_spur_acc=zs_spur,
            #     zs_drop=zs_clean - zs_spur,
            #     lp_clean_acc=lp_clean, lp_spur_acc=lp_spur,
            #     lp_gap=lp_spur - lp_clean,
            #     zs_per_class_clean=zs_pc_clean, zs_per_class_spur=zs_pc_spur,
            #     lp_per_class_clean=lp_pc_clean, lp_per_class_spur=lp_pc_spur,
            # )
            # results.append(vr)

            # print(f"  ZS  clean={zs_clean:.1f}%  spur={zs_spur:.1f}%  drop={vr.zs_drop:.1f}%")
            # print(f"  LP  clean={lp_clean:.1f}%  spur={lp_spur:.1f}%  gap={vr.lp_gap:.1f}%")

        # Free GPU memory and close wandb run before next variant
        clip_model.cpu(); del clip_model, processor, zero_proc
        gc.collect(); torch.cuda.empty_cache()
        wandb.finish()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — VLM ICL evaluation
# ══════════════════════════════════════════════════════════════════════════════

def load_vlm(model_id, load_in_4bit=False):
    quant_cfg = (BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4") if load_in_4bit else None)
    common = dict(torch_dtype=torch.float16, device_map="auto",
                  quantization_config=quant_cfg)
    if model_id in LLAVA_MODELS:
        proc  = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(model_id, **common)
    elif model_id in PHI_MODELS:
        proc  = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                              num_crops=4)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True,
                attn_implementation="flash_attention_2", **common)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True,
                attn_implementation="eager", **common)
    else:
        raise ValueError(f"Unsupported VLM: {model_id}")
    model.eval()
    return model, proc


def build_vlm_prompt(model_id, ctx_shots, query_img, class_names, proc):
    cls_str = ", ".join(class_names)
    if model_id in LLAVA_MODELS:
        imgs, parts = [], ["Study these labelled examples carefully.\n"]
        for s in ctx_shots:
            imgs.append(s["image"]); parts.append(f"<image>\nLabel: {s['caption']}\n")
        imgs.append(query_img)
        parts.append(f"\nClassify this image.\n<image>\n"
                     f"Choose one from: [{cls_str}].\nReply with only the class name.")
        return proc(text="".join(parts), images=imgs, return_tensors="pt", padding=True)
    elif model_id in PHI_MODELS:
        imgs, uc = [], "Study these labelled examples carefully.\n"
        for i, s in enumerate(ctx_shots, 1):
            imgs.append(s["image"]); uc += f"<|image_{i}|>\nLabel: {s['caption']}\n"
        q = len(ctx_shots) + 1; imgs.append(query_img)
        uc += (f"\nClassify this image.\n<|image_{q}|>\n"
               f"Choose one from: [{cls_str}].\nReply with only the class name.")
        prompt = proc.tokenizer.apply_chat_template(
            [{"role": "user", "content": uc}],
            tokenize=False, add_generation_prompt=True)
        return proc(text=prompt, images=imgs, return_tensors="pt", padding=True)
    raise ValueError(f"No builder for {model_id}")


@torch.inference_mode()
def vlm_generate(model, inputs, proc, max_new_tokens=20):
    device  = next(model.parameters()).device
    inputs  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
               for k, v in inputs.items()}
    in_len  = inputs["input_ids"].shape[-1]
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    return proc.decode(out_ids[0][in_len:], skip_special_tokens=True).strip().lower()


def parse_pred(raw, class_names):
    raw = raw.strip().lower()
    for i, c in enumerate(class_names):
        if c.lower() == raw: return i
    for i, c in enumerate(class_names):
        if c.lower() in raw: return i
    return None


def build_context_shots(dataset, class_names, n_shots, spur_labels,
                         spur_trigger, spur_location, seed,
                         spur_proportion, include_spurious):
    shots = []
    for idx in random.Random(seed).sample(range(len(dataset)), n_shots):
        item    = dataset[idx]
        img     = item.get("img", item.get("image"))
        if isinstance(img, torch.Tensor): img = to_pil(img.cpu())
        label   = int(item.get("label", item.get("labels")))
        caption = f"a photo of a {class_names[label]}"
        if include_spurious and should_trigger(
            idx, label, seed=seed, proportion=spur_proportion,
            target_labels=spur_labels):
            caption = add_trigger(caption, spur_trigger, spur_location)
        shots.append({"image": img, "caption": caption, "label": label})
    return shots


def run_vlm_icl(cfg, class_names, seed, train_xform, spur_test_xform, clean_xform):
    print("\n" + "=" * 72)
    print("STAGE 2 — VLM In-Context Spurious Evaluation")
    print("=" * 72)

    TEXT_SPUR_LABELS = set(cfg.params.spur_text_labels)
    spur_test_set = (
        {cfg.params.spur_test_label}
        if isinstance(cfg.params.spur_test_label, int)
        else set(cfg.params.spur_test_label))

    ctx_ds      = spt.data.HFDataset(path=cfg.params.dataset,
                                      split="train", transform=train_xform)
    q_ds_spur   = spt.data.HFDataset(path=cfg.params.zeroshot_dataset,
                                      split="test",  transform=spur_test_xform)
    q_ds_clean  = spt.data.HFDataset(path=cfg.params.zeroshot_dataset,
                                      split="test",  transform=clean_xform)

    shot_kw = dict(
        class_names=class_names, n_shots=cfg.params.get("icl_n_shots", 8),
        spur_labels=TEXT_SPUR_LABELS,
        spur_trigger=cfg.params.spur_text_trigger,
        spur_location=cfg.params.text_spur_location,
        seed=seed, spur_proportion=cfg.params.spur_proportion)
    ctx_spur  = build_context_shots(ctx_ds.dataset, include_spurious=True,  **shot_kw)
    ctx_clean = build_context_shots(ctx_ds.dataset, include_spurious=False, **shot_kw)

    conditions = {
        "spurious_context__spurious_query": (ctx_spur,  q_ds_spur),
        "spurious_context__clean_query":    (ctx_spur,  q_ds_clean),
        "clean_context__spurious_query":    (ctx_clean, q_ds_spur),
        "clean_context__clean_query":       (ctx_clean, q_ds_clean),
    }

    n_eval = cfg.params.get("icl_n_eval", 100)
    eval_idx = random.Random(seed + 1).sample(range(len(q_ds_spur.dataset)), n_eval)

    vlm_ids   = list(cfg.params.get("vlm_models", [
        "microsoft/Phi-3.5-vision-instruct",
        "llava-hf/llava-1.5-7b-hf"]))
    load_4bit = cfg.params.get("load_in_4bit", False)
    all_vlm   = {}

    for mid in vlm_ids:
        print(f"\n  VLM: {mid}")
        vlm, proc = load_vlm(mid, load_in_4bit=load_4bit)
        res = {k: {"correct": 0, "total": 0, "spur_flip": 0, "unparseable": 0}
               for k in conditions}

        for idx in tqdm(eval_idx, desc=f"  [{mid.split('/')[-1]}]"):
            gt  = int(q_ds_spur.dataset[idx].get(
                "label", q_ds_spur.dataset[idx].get("labels")))
            is_spur = gt in spur_test_set
            for cname, (ctx, qds) in conditions.items():
                item = qds.dataset[idx]
                img  = item.get("img", item.get("image"))
                if isinstance(img, torch.Tensor): img = to_pil(img.cpu())
                inputs = build_vlm_prompt(mid, ctx, img, class_names, proc)
                raw    = vlm_generate(vlm, inputs, proc)
                pred   = parse_pred(raw, class_names)
                r = res[cname]; r["total"] += 1
                if pred is None:   r["unparseable"] += 1
                elif pred == gt:   r["correct"]     += 1
                if is_spur and pred in TEXT_SPUR_LABELS and pred != gt:
                    r["spur_flip"] += 1

        for cname, r in res.items():
            acc  = 100 * r["correct"]   / max(r["total"], 1)
            flip = 100 * r["spur_flip"] / max(r["total"], 1)
            print(f"    [{cname}]  acc={acc:.1f}%  spur_flip={flip:.1f}%")
        all_vlm[mid] = res

        vlm.cpu(); del vlm, proc
        gc.collect(); torch.cuda.empty_cache()

    return all_vlm


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(baseline: VariantResult, ablation: List[VariantResult]):
    """
    Print the 2×2 ablation as a compact comparison table.
    Columns: ZS-drop (clean acc - spur acc), LP-gap (spur acc - clean acc)
    A large ZS-drop means the model is hurt on zero-shot by the spurious feature.
    A large LP-gap means spurious features made it BETTER on the probe (overfit).
    """
    all_variants = [baseline] + ablation
    print("\n" + "=" * 72)
    print("STAGE 3 — SUMMARY TABLE")
    print("=" * 72)
    header = (f"{'Variant':<32} {'ZS-clean':>9} {'ZS-spur':>9} "
              f"{'ZS-drop':>9} {'LP-clean':>9} {'LP-spur':>9} {'LP-gap':>9}")
    print(header)
    print("─" * len(header))
    for vr in all_variants:
        print(
            f"{vr.name:<32} "
            f"{vr.zs_clean_acc:>8.1f}% "
            f"{vr.zs_spur_acc:>8.1f}% "
            f"{vr.zs_drop:>8.1f}% "
            f"{vr.lp_clean_acc:>8.1f}% "
            f"{vr.lp_spur_acc:>8.1f}% "
            f"{vr.lp_gap:>8.1f}%"
        )

    print("\nKey:")
    print("  ZS-drop  = zero-shot clean acc − spur acc  "
          "(↑ = more hurt by spurious feature)")
    print("  LP-gap   = linear probe spur acc − clean acc  "
          "(↑ = vision encoder encoded the spurious shortcut)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

@hydra.main(config_path=".", config_name="vit_config", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    seed = cfg.params.seed
    set_seed(seed)

    if cfg.params.dataset == "uoft-cs/cifar10":
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    elif cfg.params.dataset == "uoft-cs/cifar100":
        class_names = [
            "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
            "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
            "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
            "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
            "cra", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
            "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo",
            "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard",
            "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
            "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
            "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
            "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
            "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper",
            "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
            "sweet_pepper", "table", "tank", "telephone", "television", "tiger",
            "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
            "willow_tree", "wolf", "woman", "worm",
        ]
    else:
        raise ValueError(f"Unknown dataset: {cfg.params.dataset}")

    train_xform, spur_test_xform, clean_xform = build_spurious_transforms(cfg, seed)

    # Stage 0
    # baseline = run_baseline(cfg, class_names, train_xform, spur_test_xform, clean_xform)

    # Stage 1
    ablation = run_ablation(cfg, class_names, seed, train_xform,
                             spur_test_xform, clean_xform)

    print(" ================================================ RUNNING VLM RESULTS ================================================")

    # Stage 2
    vlm_results = run_vlm_icl(cfg, class_names, seed, train_xform,
                               spur_test_xform, clean_xform)

    # Stage 3
    # print_summary_table(baseline, ablation)

    # Save all
    out = {
        "config":    OmegaConf.to_container(cfg.params, resolve=True),
        # "baseline":  asdict(baseline),
        "ablation":  [asdict(vr) for vr in ablation],
        "vlm_icl":   vlm_results,
    }
    out_path = "spurious_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAll results saved → {out_path}")


if __name__ == "__main__":
    main()