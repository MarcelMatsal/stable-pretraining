from transformers import ViTModel, ViTImageProcessor
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
    """Count trainable LoRA parameters inside a PEFT-wrapped model."""
    total, lora_total = 0, 0
    for name, p in peft_model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            lora_total += p.numel()
    return lora_total, total


def set_seed(seed: int):
    """Function that sets all the seeds to make our results reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ViTClassifier(nn.Module):
    """
    Wraps a HuggingFace ViTModel with a linear classification head.
    ViT has no text encoder, so classification is done directly from
    the [CLS] token embedding rather than via contrastive text matching.
    """

    def __init__(self, vit_model: ViTModel, num_classes: int):
        super().__init__()
        self.vit = vit_model
        hidden_size = vit_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        # [CLS] token is the first token of the last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits


@hydra.main(config_path=".", config_name="vit_finetuning_config", version_base="1.1")
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

    num_classes = len(class_names)

    # -------------------------------------------------------------------------
    # Forward pass: cross-entropy classification (replaces CLIP contrastive loss)
    # -------------------------------------------------------------------------
    def forward(self, batch, stage=None):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        logits = vit_classifier(pixel_values)
        loss = nn.functional.cross_entropy(logits, labels)

        if self.training or stage == "train":
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        else:
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("val/acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        pixel_values = batch.get("pixel_values")
        labels = batch.get("labels")
        return {"pixel_values": pixel_values, "labels": labels}

    set_seed(cfg.params.seed)

    # -------------------------------------------------------------------------
    # Load ViT (replaces CLIPModel + CLIPProcessor)
    # -------------------------------------------------------------------------
    vit_backbone = ViTModel.from_pretrained(cfg.params.vit_configuration)
    processor = ViTImageProcessor.from_pretrained(cfg.params.vit_configuration)
    vit_classifier = ViTClassifier(vit_backbone, num_classes=num_classes)

    # -------------------------------------------------------------------------
    # Optional LoRA (applied only to the ViT encoder, not the classifier head)
    # -------------------------------------------------------------------------
    if cfg.params.use_lora:
        lora_config = LoraConfig(
            r=cfg.params.lora_rank,
            lora_alpha=cfg.params.lora_alpha,
            target_modules=["query", "value"],  # ViT uses "query"/"value" not "q_proj"/"v_proj"
            lora_dropout=cfg.params.lora_dropout,
            bias="none",
        )
        vit_classifier.vit = get_peft_model(vit_classifier.vit, lora_config)

        # Freeze everything except LoRA params and the classifier head
        for name, param in vit_classifier.named_parameters():
            if "lora_" not in name.lower() and "classifier" not in name.lower():
                param.requires_grad = False

        lora_params, total_params = count_lora_params(vit_classifier.vit)
        print(f"LoRA Params: {lora_params:,} / {total_params:,}")

    for name, param in vit_classifier.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}, Shape: {param.shape}")

    # -------------------------------------------------------------------------
    # Spurious transforms (same logic as CLIP script)
    # -------------------------------------------------------------------------
    def should_trigger(idx, label, *, seed, proportion, target_labels):
        if label not in target_labels:
            return False
        u = (hash((seed, idx)) % 10_000_000) / 10_000_000
        return u < proportion

    if cfg.params.use_spurious:
        if cfg.params.spur_type == "watermark":
            transform_train = transforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddWatermark(
                        watermark=cfg.params.watermark_path,
                        size=cfg.params.watermark_size,
                        position=cfg.params.watermak_pos,
                        alpha=cfg.params.spur_alpha,
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_train_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_train_samples,
                    seed=cfg.params.seed,
                ),
            )
            transform_test = transforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddWatermark(
                        watermark=cfg.params.watermark_path,
                        size=cfg.params.watermark_size,
                        position=cfg.params.watermak_pos,
                        alpha=cfg.params.spur_alpha,
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_test_samples,
                    seed=cfg.params.seed,
                ),
            )
        elif cfg.params.spur_type == "patch":
            transform_train = transforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddPatch(
                        patch_size=cfg.params.patch_size,
                        color=cfg.params.patch_color,
                        position=cfg.params.patch_pos,
                        img_key=cfg.params.image_key,
                    ),
                    label_key=cfg.params.label_key,
                    target_labels=cfg.params.spur_train_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_train_samples,
                    seed=cfg.params.seed,
                ),
            )
            transform_test = transforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddPatch(
                        patch_size=cfg.params.patch_size,
                        color=cfg.params.patch_color,
                        position=cfg.params.patch_pos,
                        img_key=cfg.params.image_key,
                    ),
                    label_key=cfg.params.label_key,
                    target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_test_samples,
                    seed=cfg.params.seed,
                ),
            )
        elif cfg.params.spur_type == "tint":
            transform_train = transforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddColorTint(
                        tint=cfg.params.tint_color, alpha=cfg.params.spur_alpha
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_train_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_train_samples,
                    seed=cfg.params.seed,
                ),
            )
            transform_test = transforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddColorTint(
                        tint=cfg.params.tint_color, alpha=cfg.params.spur_alpha
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_test_samples,
                    seed=cfg.params.seed,
                ),
            )
        else:
            raise ValueError(f"Unsupported spur_type: {cfg.params.spur_type}")
    else:
        transform_train = transforms.Compose(transforms.ToImage(source="img", target="img"))
        transform_test = transforms.Compose(transforms.ToImage(source="img", target="img"))

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    finetuning_dataset = spt.data.HFDataset(
        path=cfg.params.dataset, split="train", transform=transform_train
    )
    val_dataset = spt.data.HFDataset(
        path=cfg.params.dataset, split="test", transform=transform_test
    )

    # -------------------------------------------------------------------------
    # Collate functions
    # ViT only needs pixel_values + integer labels; no text involved.
    # -------------------------------------------------------------------------
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

    finetune_dataloader = torch.utils.data.DataLoader(
        dataset=finetuning_dataset,
        batch_size=cfg.params.batch_size,
        collate_fn=collate_fn,
        num_workers=8,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.params.batch_size,
        collate_fn=collate_fn,
        num_workers=8,
    )

    data = spt.data.DataModule(train=finetune_dataloader, val=val_dataloader)

    wandb_logger = WandbLogger(
        entity="rbalestr-brown",
        project="vit_finetuning",
        name=f"ViT finetuning on {cfg.params.dataset}",
        config=OmegaConf.to_container(cfg.params, resolve=True),
        log_model=False,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    module = spt.Module(
        backbone=vit_classifier,
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

    trainer = pl.Trainer(
        max_epochs=cfg.params.epochs,
        precision="16-mixed",
        logger=wandb_logger,
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    module.backbone.train()
    manager()

    torch.save(vit_classifier.state_dict(), "finetuned_vit.pt")

    # -------------------------------------------------------------------------
    # Evaluation — standard accuracy (no zero-shot; ViT uses the classifier head)
    # -------------------------------------------------------------------------
    eval_dataset = spt.data.HFDataset(
        path=cfg.params.dataset, split="test", transform=transform_test
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.params.batch_size,
        collate_fn=collate_fn,
        num_workers=8,
    )

    eval_module = spt.Module(backbone=vit_classifier, forward=forward, hparams=cfg)
    eval_module.validation_step = types.MethodType(validation_step, eval_module)

    eval_trainer = pl.Trainer(precision="16-mixed", logger=wandb_logger)
    eval_trainer.validate(model=eval_module, dataloaders=eval_dataloader)


if __name__ == "__main__":
    main()