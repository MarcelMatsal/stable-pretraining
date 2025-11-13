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
    torch.cuda.manual_seed_all(seed)  # For multi-GPU training


@hydra.main(config_path=".", config_name="clip_finetuning_config", version_base="1.1")
def main(cfg: DictConfig):

    if cfg.params.dataset == "uoft-cs/cifar10":
        class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    elif cfg.params.dataset == "uoft-cs/cifar100":
        class_names = [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "cra",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm"
            ]

    if cfg.params.zeroshot_dataset == "uoft-cs/cifar10":
        zero_class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    elif cfg.params.zeroshot_dataset == "uoft-cs/cifar100":
        zero_class_names = [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "cra",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm"
            ]
    def forward(self, batch, stage=None):
        out = {}

        pixel_values = batch["pixel_values"]  # from processor(images=...)
        input_ids = batch["input_ids"]  # from processor(text=...)
        attention_mask = batch["attention_mask"]

        outputs = clip_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=True,  # CLIP automatically computes its own contrastive loss
        )

        loss = outputs.loss

        # --- 4. Prepare return dictionary ---
        out["loss"] = loss

        if self.training or stage == "train":
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        else:
            self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return out

    # Patch validation_step so callback receives data
    def validation_step(self, batch, batch_idx):
        # Accept either numeric labels or string answers
        pixel_values = batch.get("pixel_values")
        labels = batch.get("labels")
        return {"pixel_values": pixel_values, "labels": labels}

    set_seed(cfg.params.seed)

    # get model from huggingface (Vit-B/32: openai/clip-vit-base-patch32)
    clip_model = CLIPModel.from_pretrained(cfg.params.clip_configuration)
    processor = CLIPProcessor.from_pretrained(cfg.params.clip_configuration)
    zero_processor = CLIPProcessor.from_pretrained(cfg.params.clip_configuration)

    if cfg.params.use_lora:
        lora_config = LoraConfig(
            r=cfg.params.lora_rank,
            lora_alpha=cfg.params.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=cfg.params.lora_dropout,
            bias="none",
            # task_type="FEATURE_EXTRACTION",
        )

        # Wrap model with LoRA
        clip_model.text_model = get_peft_model(clip_model.text_model, lora_config)
        clip_model.vision_model = get_peft_model(clip_model.vision_model, lora_config)

        text_lora, text_total = count_lora_params(clip_model.text_model)
        vision_lora, vision_total = count_lora_params(clip_model.vision_model)

        for name, param in clip_model.named_parameters():
            if "lora_" not in name.lower():
                param.requires_grad = False

        trainable = [n for n, p in clip_model.named_parameters() if p.requires_grad]

        for param_name in trainable:
            assert "lora" in param_name.lower()

        print(f"LoRA Text Params: {text_lora:,} / {text_total:,}")
        print(f"LoRA Vision Params: {vision_lora:,} / {vision_total:,}")
        print(f"Total Trainable (LoRA only): {text_lora + vision_lora:,}")

        assert text_lora != text_total
        assert vision_lora != vision_total

        for name, param in clip_model.named_parameters():
            if param.requires_grad and not "lora" in name.lower():
                print(
                    f"Not in LoRA: {name} - Requires Grad: {param.requires_grad} - Shape: {param.shape}"
                )

    for name, param in clip_model.named_parameters():
        if param.requires_grad:
            print(f"Parameter Name: {name}, Shape: {param.shape}")

    # set up transformations
    if cfg.params.use_spurious:
        # visual spurious correlations
        if not cfg.params.spur_type:
            raise ValueError(
                "Must have a spurious type if creating spurious correlations"
            )
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
            transform_eval = transforms.Compose(
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
        elif cfg.params.spur_type == "border":
            transform_train = stransforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddBorder(
                        thickness=cfg.params.border_thickness,
                        color=cfg.params.spur_color,
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_train_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_train_samples,
                    seed=cfg.params.seed,
                ),
            )
            transform_test = stransforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddBorder(
                        thickness=cfg.params.border_thickness,
                        color=cfg.params.spur_color,
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_test_samples,
                    seed=cfg.params.seed,
                ),
            )
            transform_eval = stransforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddBorder(
                        thickness=cfg.params.border_thickness,
                        color=cfg.params.spur_color,
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
                    transformation=transforms.AddPatch(
                        patch_size=cfg.params.patch_size,
                        color=cfg.params.patch_color,
                        position=cfg.params.patch_pos,
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_test_samples,
                    seed=cfg.params.seed,
                ),
            )
            transform_eval = transforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddPatch(
                        patch_size=cfg.params.patch_size,
                        color=cfg.params.patch_color,
                        position=cfg.params.patch_pos,
                    ),
                    label_key="label",
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
            transform_eval = transforms.Compose(
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

        elif cfg.params.spur_type == "checkerboard":
            transform_train = transforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddCheckerboardPattern(
                        intensity=cfg.params.spur_alpha, image_label="img"
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
                    transformation=transforms.AddCheckerboardPattern(
                        intensity=cfg.params.spur_alpha, image_label="img"
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_test_samples,
                    seed=cfg.params.seed,
                ),
            )
            transform_eval = transforms.Compose(
                transforms.ToImage(source="img", target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddCheckerboardPattern(
                        intensity=cfg.params.spur_alpha, image_label="img"
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_test_samples,
                    seed=cfg.params.seed,
                ),
            )
        else:
            raise Exception(
                "Spurious type for images must either be: watermark, border, patch, or tint"
            )
    else:
        transform_train = transforms.Compose(
            transforms.ToImage(source="img", target="img")
        )
        transform_test = transforms.Compose(
            transforms.ToImage(source="img", target="img")
        )
        transform_eval = transforms.Compose(
            transforms.ToImage(source="img", target="img")
        )

    # Finish Prepping the Data for Finetuning

    # finetuning_dataset = spt.data.HFDataset(
    #     path="lmms-lab/COCO-Caption2017",
    #     split="val",
    #     transform=transform_train,
    # )

    # val_dataset = spt.data.HFDataset(
    #     path="lmms-lab/COCO-Caption2017",
    #     split="val",
    #     transform=transform_test,
    # )

    finetuning_dataset = spt.data.HFDataset(
        path=cfg.params.dataset,
        split="train",
        transform=transform_train,
    )

    val_dataset = spt.data.HFDataset(
        path=cfg.params.dataset,
        split="test",
        transform=transform_test,
    )

    # Use all the different captions available for each dataset
    def expand_captions(batch):
        new_images = []
        new_texts = []
        for img, captions in zip(batch["image"], batch["answer"]):
            # captions might be a list or a single string
            if isinstance(captions, list):
                for caption in captions:
                    new_images.append(img)
                    new_texts.append(caption)
            else:
                new_images.append(img)
                new_texts.append(captions)
        return {"image": new_images, "answer": new_texts}

    def add_prompt(batch):
        prompts = [f"a photo of a {class_names[label]}" for label in batch[cfg.params.label_key]]
        # if "img" in batch and "image" not in batch:
        #     batch["image"] = batch.pop("img")
        batch["answer"] = prompts
        return batch

    # finetuning_dataset.dataset = finetuning_dataset.dataset.map(
    #     expand_captions,
    #     batched=True,
    #     remove_columns=finetuning_dataset.dataset.column_names,
    # )

    # val_dataset.dataset = val_dataset.dataset.map(
    #     expand_captions, batched=True, remove_columns=val_dataset.dataset.column_names
    # )

    finetuning_dataset.dataset = finetuning_dataset.dataset.map(
        add_prompt, batched=True, remove_columns=[]
    )
    val_dataset.dataset = val_dataset.dataset.map(
        add_prompt, batched=True, remove_columns=[]
    )

    # Use the pretrained processor
    def preprocess(example):
        # To use the tokenizers that is a part of the pretrained CLIP model
        # Have to convert them back to PIL to use its full power

        return processor(
            text=example["answer"],
            images=example["img"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    finetuning_dataset.dataset = finetuning_dataset.dataset.map(
        preprocess, batched=True
    )
    val_dataset.dataset = val_dataset.dataset.map(preprocess, batched=True)

    def finetune_collate_fn(batch):
        # batch is list of items; each item has {"image": <tensor or PIL>, "text": <str>, ...}
        images = []
        texts = []
        for item in batch:
            img = item["img"]
            # if tensor -> convert to PIL (processor expects PIL/np array), but only if needed:
            if isinstance(img, torch.Tensor):
                img = to_pil(img.cpu())
            images.append(img)
            texts.append(item["answer"])

        # processor handles both text and images and returns tensors ready for CLIP
        proc = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {
            "input_ids": proc["input_ids"],
            "attention_mask": proc["attention_mask"],
            "pixel_values": proc["pixel_values"],
        }

    # map class names to indices if needed (same order as class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    def zero_shot_collate_fn(batch):
        images = []
        labels = []
        for item in batch:
            img = item.get("img", item.get("image", None))
            if isinstance(img, torch.Tensor):
                img = to_pil(img.cpu())
            images.append(img)
            # try common keys for numeric label, otherwise map from text label
            if "label" in item:
                labels.append(int(item["label"]))
            elif "labels" in item:
                labels.append(int(item["labels"]))
            elif cfg.params.zero_label in item:
                labels.append(int(item[cfg.params.zero_label]))
            elif "answer" in item:
                # if 'answer' is a string class name, map to idx
                labels.append(class_to_idx[item["answer"]])
            elif "answers" in item:
                # if for some reason answers is a single string in item
                labels.append(class_to_idx[item["answers"]])
            else:
                # fallback: None (will break later, but this is explicit)
                labels.append(None)

        proc = zero_processor(
            images=images, return_tensors="pt", padding=True, truncation=True
        )
        # convert labels to tensor, but first ensure no None
        if any(l is None for l in labels):
            raise ValueError(
                "Some examples in the batch have no label. Check dataset items keys."
            )
        return {
            "pixel_values": proc["pixel_values"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    finetune_dataloader = torch.utils.data.DataLoader(
        dataset=finetuning_dataset,
        batch_size=cfg.params.batch_size,
        collate_fn=finetune_collate_fn,
        num_workers=8,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.params.batch_size,
        collate_fn=finetune_collate_fn,
        num_workers=8,
    )

    data = spt.data.DataModule(train=finetune_dataloader, val=val_dataloader)
    wandb_logger = WandbLogger(
        entity="rbalestr-brown",
        project="clip_spurious_correlation",
        name=f"CLIP Finetuning on {cfg.params.dataset} zeroshot on {cfg.params.zeroshot_dataset}, LoRA: {cfg.params.use_lora}, rank: {cfg.params.lora_rank}, Using Spur: {cfg.params.use_spurious} Spurious tokens: {cfg.params.use_spurious}, type: {cfg.params.spur_type} alpha: {cfg.params.spur_alpha}, proportion: {cfg.params.spur_proportion}",
        config=OmegaConf.to_container(cfg.params, resolve=True),
        log_model=False,
    )

    # finetune CLIP
    module = spt.Module(
        backbone=clip_model,
        forward=forward,
        hparams=cfg,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": cfg.params.learning_rate,
                "weight_decay": cfg.params.weight_decay,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
            },
            "interval": "epoch",
        },
    )

    trainer = pl.Trainer(
        max_epochs=cfg.params.epochs,
        precision="16-mixed",
        logger=wandb_logger,
    )

    # pretrain the MAE Vit backbone and save the model locally
    manager = spt.Manager(trainer=trainer, module=module, data=data)
    module.backbone.train()
    manager()

    torch.save(clip_model.state_dict(), "finetuned_clip_no_lora_no_spur.pt")

    # Evaluate the Finetuned CLIP model on CIFAR10
    class CLIPImageWrapper(nn.Module):
        """Expose .image_embeds via forward(pixel_values=...) using CLIPModel.get_image_features."""

        def __init__(self, clip_model):
            super().__init__()
            self.clip = clip_model

        def forward(self, pixel_values=None):
            # move inputs to same device as model
            device = next(self.clip.parameters()).device
            if pixel_values is not None and pixel_values.device != device:
                pixel_values = pixel_values.to(device)

            image_feats = self.clip.get_image_features(pixel_values=pixel_values)
            # return an object with attribute `.image_embeds` (callback expects this)
            return types.SimpleNamespace(image_embeds=image_feats)

    class CLIPTextWrapper(nn.Module):
        """Expose .text_embeds via forward(input_ids=...) by using CLIPModel.get_text_features."""

        def __init__(self, clip_model):
            super().__init__()
            self.clip = clip_model

        def forward(self, input_ids=None, attention_mask=None):
            # If tokenizer_fn returned a dict, handle it
            if isinstance(input_ids, dict):
                attention_mask = input_ids.get("attention_mask", attention_mask)
                input_ids = input_ids.get("input_ids")

            # make sure tensors are on same device as model weights
            device = next(self.clip.parameters()).device
            if input_ids is not None and input_ids.device != device:
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

            text_feats = self.clip.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # Return a simple object with attribute `.text_embeds` that the callback expects
            return types.SimpleNamespace(text_embeds=text_feats)

    text_backbone = CLIPTextWrapper(clip_model)
    image_backbone = CLIPImageWrapper(clip_model)
    # Setup the zero shot evaluation (CIFAR10)
    zero_shot_callback = clip_zero_shot.CLIPZeroShot(
        name="zeroshot_eval",
        image_key="pixel_values",
        class_key="labels",
        class_names=zero_class_names,
        image_backbone=image_backbone,
        text_backbone=text_backbone,
        tokenizer_fn=lambda x: zero_processor.tokenizer(
            [f"a photo of a {c}" for c in x],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )["input_ids"],
        metrics={
            "top1": tm.classification.MulticlassAccuracy(len(zero_class_names)),
            "top5": tm.classification.MulticlassAccuracy(len(zero_class_names), top_k=5),
        },
    )

    transform_eval = transforms.Compose(transforms.ToImage(source="img", target="img"))
    eval_dataset = spt.data.HFDataset(
        path=cfg.params.zeroshot_dataset,
        split="test",
        transform=transform_eval,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.params.batch_size,
        collate_fn=zero_shot_collate_fn,
        num_workers=8,
    )

    eval_trainer = pl.Trainer(
        precision="16-mixed",
        callbacks=[zero_shot_callback],
        logger=wandb_logger,
    )

    eval_module = spt.Module(
        backbone=clip_model,
        forward=forward,
        hparams=cfg,
    )

    eval_module.validation_step = types.MethodType(validation_step, eval_module)
    eval_trainer.validate(model=eval_module, dataloaders=eval_dataloader)


if __name__ == "__main__":
    main()
