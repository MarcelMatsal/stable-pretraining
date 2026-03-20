"""
clip_vision_spur_finetuning.py

Fine-tunes only the CLIP vision backbone in a supervised (cross-entropy)
manner while injecting spurious patch correlations into a subset of training
images.  The goal is to measure how much the visual representations can be
corrupted and how robust the backbone is to shortcut learning.

Key differences from clip_finetuning.py:
  - No text encoder / contrastive loss; purely classification.
  - Only the vision model is loaded and wrapped with a linear head.
  - Patch spurious injection is the primary corruption mechanism, mirroring
    the `patch` branch in clip_finetuning.py.
  - Evaluation reports per-class accuracy so degradation can be tracked
    on the specific class that received the spurious patch.
"""

from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model
import hydra
from stable_pretraining.data import transforms
import stable_pretraining as spt
import torch
import torch.nn as nn
import numpy as np
import torchmetrics as tm
import lightning as pl
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import ToPILImage
import types


to_pil = ToPILImage()


def count_lora_params(peft_model):
    total, lora_total = 0, 0
    for name, p in peft_model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            lora_total += p.numel()
    return lora_total, total


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CLIPVisionClassifier(nn.Module):
    """CLIP vision encoder + linear classification head.

    Only the vision tower is loaded; there is no text encoder.
    Classification is performed from the pooled image embedding
    (CLIPVisionModel returns `pooler_output` as the CLS-like feature).
    """

    def __init__(self, vision_model: CLIPVisionModel, num_classes: int):
        super().__init__()
        self.vision = vision_model
        hidden_size = vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vision(pixel_values=pixel_values)
        # pooler_output is the [CLS] token projected through the visual projection layer
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits


@hydra.main(config_path=".", config_name="clip_vision_spur_config", version_base="1.1")
def main(cfg: DictConfig):

    if cfg.params.dataset == "uoft-cs/cifar10":
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    elif cfg.params.dataset == "uoft-cs/cifar100":
        class_names = [
            "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
            "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
            "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
            "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
            "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
            "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
            "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
            "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
            "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
            "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
            "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
            "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
            "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
            "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
        ]
    else:
        raise ValueError(f"Unsupported dataset: {cfg.params.dataset}")

    num_classes = len(class_names)

    # ------------------------------------------------------------------
    # Forward / validation (bound to spt.Module at runtime)
    # ------------------------------------------------------------------
    def forward(self, batch, stage=None):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        logits = clip_vision_classifier(pixel_values)
        loss = nn.functional.cross_entropy(logits, labels)

        if self.training or stage == "train":
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        else:
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("val/acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = clip_vision_classifier(pixel_values)
        loss = nn.functional.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        per_class_acc.update(preds, labels)
        top1_acc.update(preds, labels)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        per_cls = per_class_acc.compute()  # shape: (num_classes,)
        top1 = top1_acc.compute()
        self.log("val/top1_acc", top1, prog_bar=True)
        for i, cls_acc in enumerate(per_cls):
            self.log(f"val/class_acc/{class_names[i]}", cls_acc)
        per_class_acc.reset()
        top1_acc.reset()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    set_seed(cfg.params.seed)

    vision_backbone = CLIPVisionModel.from_pretrained(cfg.params.clip_configuration)
    processor = CLIPImageProcessor.from_pretrained(cfg.params.clip_configuration)
    clip_vision_classifier = CLIPVisionClassifier(vision_backbone, num_classes=num_classes)

    # ------------------------------------------------------------------
    # Optional LoRA (vision encoder only; classifier head always trainable)
    # ------------------------------------------------------------------
    if cfg.params.use_lora:
        lora_config = LoraConfig(
            r=cfg.params.lora_rank,
            lora_alpha=cfg.params.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=cfg.params.lora_dropout,
            bias="none",
        )
        clip_vision_classifier.vision = get_peft_model(clip_vision_classifier.vision, lora_config)

        for name, param in clip_vision_classifier.named_parameters():
            if "lora_" not in name.lower() and "classifier" not in name.lower():
                param.requires_grad = False

        lora_params, total_params = count_lora_params(clip_vision_classifier.vision)
        print(f"LoRA Vision Params: {lora_params:,} / {total_params:,}")
    else:
        # Full fine-tune: freeze vision, train only the classifier head
        # (set freeze_vision: false in config to train the whole backbone)
        if cfg.params.get("freeze_vision", False):
            for name, param in clip_vision_classifier.vision.named_parameters():
                param.requires_grad = False
            print("Vision backbone frozen; training classifier head only.")

    for name, param in clip_vision_classifier.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}, Shape: {param.shape}")

    # ------------------------------------------------------------------
    # Spurious transforms — patch injection mirrors clip_finetuning.py
    # ------------------------------------------------------------------
    if cfg.params.use_spurious:
        if cfg.params.spur_type != "patch":
            raise ValueError(
                "This script is designed for patch-based spurious injection. "
                "Set spur_type: patch in your config."
            )

        patch_transform_train = transforms.AddPatch(
            patch_size=cfg.params.patch_size,
            color=cfg.params.patch_color,
            position=cfg.params.patch_pos,
            img_key=cfg.params.image_key,
        )
        patch_transform_test = transforms.AddPatch(
            patch_size=cfg.params.patch_size,
            color=cfg.params.patch_color,
            position=cfg.params.patch_pos,
            img_key=cfg.params.image_key,
        )

        transform_train = transforms.Compose(
            transforms.ToImage(source="img", target="img"),
            transforms.AddSampleIdx(),
            transforms.ClassConditionalInjector(
                transformation=patch_transform_train,
                label_key=cfg.params.label_key,
                target_labels=cfg.params.spur_train_label,
                proportion=cfg.params.spur_proportion,
                total_samples=cfg.params.total_train_samples,
                seed=cfg.params.seed,
            ),
        )
        # Test set: apply patch to the *same* labels so we can measure
        # whether the model has learned the shortcut (high acc with patch)
        # vs. the clean test set (degraded acc without patch).
        transform_test_spur = transforms.Compose(
            transforms.ToImage(source="img", target="img"),
            transforms.AddSampleIdx(),
            transforms.ClassConditionalInjector(
                transformation=patch_transform_test,
                label_key=cfg.params.label_key,
                target_labels=cfg.params.spur_test_label,
                proportion=cfg.params.spur_proportion,
                total_samples=cfg.params.total_test_samples,
                seed=cfg.params.seed,
            ),
        )
        transform_test_clean = transforms.Compose(
            transforms.ToImage(source="img", target="img"),
        )
    else:
        transform_train = transforms.Compose(transforms.ToImage(source="img", target="img"))
        transform_test_spur = transforms.Compose(transforms.ToImage(source="img", target="img"))
        transform_test_clean = transforms.Compose(transforms.ToImage(source="img", target="img"))

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_dataset = spt.data.HFDataset(
        path=cfg.params.dataset, split="train", transform=transform_train
    )
    val_dataset_spur = spt.data.HFDataset(
        path=cfg.params.dataset, split="test", transform=transform_test_spur
    )
    val_dataset_clean = spt.data.HFDataset(
        path=cfg.params.dataset, split="test", transform=transform_test_clean
    )

    # ------------------------------------------------------------------
    # Collate — vision only, no text
    # ------------------------------------------------------------------
    def collate_fn(batch):
        images, labels = [], []
        for item in batch:
            img = item["img"]
            if isinstance(img, torch.Tensor):
                img = to_pil(img.cpu())
            images.append(img)
            labels.append(int(item[cfg.params.label_key]))

        proc = processor(images=images, return_tensors="pt")
        return {
            "pixel_values": proc["pixel_values"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.params.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        multiprocessing_context="fork",
    )
    val_dataloader_spur = torch.utils.data.DataLoader(
        dataset=val_dataset_spur,
        batch_size=cfg.params.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        persistent_workers=True,
        multiprocessing_context="fork",
    )
    val_dataloader_clean = torch.utils.data.DataLoader(
        dataset=val_dataset_clean,
        batch_size=cfg.params.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        persistent_workers=True,
        multiprocessing_context="fork",
    )

    # ------------------------------------------------------------------
    # Per-class accuracy callback so we can pinpoint corruption
    # ------------------------------------------------------------------
    per_class_acc = tm.classification.MulticlassAccuracy(num_classes, average="none")
    top1_acc = tm.classification.MulticlassAccuracy(num_classes)

    # ------------------------------------------------------------------
    # WandB logger
    # ------------------------------------------------------------------
    wandb_logger = WandbLogger(
        entity="rbalestr-brown",
        project="clip_vision_spur_finetuning",
        name=(
            f"CLIP-vision supervised | lora={cfg.params.use_lora} "
            f"rank={cfg.params.lora_rank} | spur={cfg.params.use_spurious} "
            f"prop={cfg.params.spur_proportion} label={cfg.params.spur_train_label}"
        ),
        config=OmegaConf.to_container(cfg.params, resolve=True),
        log_model=False,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    data = spt.data.DataModule(train=train_dataloader, val=val_dataloader_clean)

    module = spt.Module(
        backbone=clip_vision_classifier,
        forward=forward,
        hparams=cfg,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": cfg.params.learning_rate,
                "weight_decay": cfg.params.weight_decay,
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )
    module.validation_step = types.MethodType(validation_step, module)
    module.on_validation_epoch_end = types.MethodType(on_validation_epoch_end, module)

    trainer = pl.Trainer(
        max_epochs=cfg.params.epochs,
        precision="16-mixed",
        logger=wandb_logger,
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    module.backbone.train()
    manager()

    # ------------------------------------------------------------------
    # Evaluation: spurious test set vs. clean test set
    # ------------------------------------------------------------------
    eval_module = spt.Module(
        backbone=clip_vision_classifier, forward=forward, hparams=cfg
    )
    eval_module.validation_step = types.MethodType(validation_step, eval_module)
    eval_module.on_validation_epoch_end = types.MethodType(on_validation_epoch_end, eval_module)

    eval_trainer = pl.Trainer(precision="16-mixed", logger=wandb_logger)

    print("\n=== Evaluation on CLEAN test set ===")
    eval_trainer.validate(model=eval_module, dataloaders=val_dataloader_clean)

    print("\n=== Evaluation on SPURIOUS test set (patch injected) ===")
    eval_trainer.validate(model=eval_module, dataloaders=val_dataloader_spur)


if __name__ == "__main__":
    main()
