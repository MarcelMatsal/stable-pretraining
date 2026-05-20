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


@hydra.main(config_path=".", config_name="clip_finetuning_config", version_base="1.1")
def main(cfg: DictConfig):

    text_rng = np.random.RandomState(cfg.params.seed)

    def _load_class_names(dataset_path, label_key, split):
        """Load class names from a HuggingFace dataset's ClassLabel feature.
        Uses NLTK WordNet for readable English names when synset IDs are present
        (e.g. Tiny ImageNet). Falls back to raw feature names if NLTK is unavailable.
        Install: pip install nltk && python -m nltk.downloader wordnet
        """
        import datasets as _hf
        _ds = _hf.load_dataset(dataset_path, split=split, streaming=True)
        raw_names = _ds.features[label_key].names

        def _synset_to_name(s):
            try:
                from nltk.corpus import wordnet as wn
                return wn.synset_from_pos_and_offset("n", int(s[1:])).lemma_names()[0].replace("_", " ")
            except Exception:
                return s

        return [_synset_to_name(n) if n.startswith("n") and n[1:].isdigit() else n for n in raw_names]

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
            "cloud", "cockroach", "couch", "cra", "crocodile", "cup", "dinosaur",
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
        class_names = _load_class_names(cfg.params.dataset, cfg.params.label_key, "train")

    if cfg.params.zeroshot_dataset == "uoft-cs/cifar10":
        zero_class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    elif cfg.params.zeroshot_dataset == "uoft-cs/cifar100":
        zero_class_names = [
            "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
            "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
            "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
            "cloud", "cockroach", "couch", "cra", "crocodile", "cup", "dinosaur",
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
        zero_class_names = _load_class_names(cfg.params.zeroshot_dataset, cfg.params.zero_label, cfg.params.test_split)

    TEXT_SPUR_TRAIN_LABELS = set(cfg.params.spur_text_labels)

    def forward(self, batch, stage=None):
        out = {}

        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = clip_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=True,
        )

        loss = outputs.loss
        out["loss"] = loss

        if self.training or stage == "train":
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        else:
            self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return out

    def validation_step(self, batch, batch_idx):
        pixel_values = batch.get("pixel_values")
        labels = batch.get("labels")
        return {"pixel_values": pixel_values, "labels": labels}

    def should_trigger(idx: int, label: int, *, seed: int, proportion: float, target_labels: set) -> bool:
        if label not in target_labels:
            return False
        u = (hash((seed, idx)) % 10_000_000) / 10_000_000
        return u < proportion

    def add_trigger(prompt: str, trigger: str, position: str) -> str:
        if position == "prepend":
            return f"{trigger} {prompt}"
        return f"{prompt} {trigger}"

    def add_prompt_train(batch, indices):
        labels = batch[cfg.params.label_key]
        prompts = []
        for idx, lab in zip(indices, labels):
            lab = int(lab)
            base = f"a photo of a {class_names[lab]}"
            if cfg.params.text_spur and should_trigger(
                idx, lab, seed=cfg.params.seed, proportion=cfg.params.spur_proportion, target_labels=TEXT_SPUR_TRAIN_LABELS
            ):
                base = add_trigger(base, cfg.params.spur_text_trigger, cfg.params.text_spur_location)
            prompts.append(base)
        batch["answer"] = prompts
        return batch

    set_seed(cfg.params.seed)

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
        )

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
                print(f"Not in LoRA: {name} - Requires Grad: {param.requires_grad} - Shape: {param.shape}")

    for name, param in clip_model.named_parameters():
        if param.requires_grad:
            print(f"Parameter Name: {name}, Shape: {param.shape}")

    if cfg.params.use_spurious:
        if not cfg.params.spur_type:
            raise ValueError("Must have a spurious type if creating spurious correlations")
        if cfg.params.spur_type == "watermark":
            transform_train = transforms.Compose(
                transforms.ToImage(source=cfg.params.image_key, target="img"),
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
                transforms.ToImage(source=cfg.params.image_key, target="img"),
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
                transforms.ToImage(source=cfg.params.image_key, target="img"),
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
            transform_train = transforms.Compose(
                transforms.ToImage(source=cfg.params.image_key, target="img"),
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
            transform_test = transforms.Compose(
                transforms.ToImage(source=cfg.params.image_key, target="img"),
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
            transform_eval = transforms.Compose(
                transforms.ToImage(source=cfg.params.image_key, target="img"),
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
                transforms.ToImage(source=cfg.params.image_key, target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddPatch(
                        patch_size=cfg.params.patch_size,
                        color=cfg.params.patch_color,
                        position=cfg.params.patch_pos,
                        img_key=cfg.params.image_key
                    ),
                    label_key=cfg.params.label_key,
                    target_labels=cfg.params.spur_train_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_train_samples,
                    seed=cfg.params.seed,
                ),
            )
            transform_test = transforms.Compose(
                transforms.ToImage(source=cfg.params.image_key, target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddPatch(
                        patch_size=cfg.params.patch_size,
                        color=cfg.params.patch_color,
                        position=cfg.params.patch_pos,
                        img_key=cfg.params.image_key
                    ),
                    label_key=cfg.params.label_key,
                    target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_test_samples,
                    seed=cfg.params.seed,
                ),
            )
            transform_eval = transforms.Compose(
                transforms.ToImage(source=cfg.params.image_key, target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddPatch(
                        patch_size=cfg.params.patch_size,
                        color=cfg.params.patch_color,
                        position=cfg.params.patch_pos,
                        img_key=cfg.params.image_key
                    ),
                    label_key=cfg.params.zero_label,
                    target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_test_samples,
                    seed=cfg.params.seed,
                ),
            )
        elif cfg.params.spur_type == "tint":
            transform_train = transforms.Compose(
                transforms.ToImage(source=cfg.params.image_key, target="img"),
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
                transforms.ToImage(source=cfg.params.image_key, target="img"),
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
                transforms.ToImage(source=cfg.params.image_key, target="img"),
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
                transforms.ToImage(source=cfg.params.image_key, target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddCheckerboardPattern(
                        intensity=cfg.params.spur_alpha, image_label=cfg.params.image_key
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_train_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_train_samples,
                    seed=cfg.params.seed,
                ),
            )
            transform_test = transforms.Compose(
                transforms.ToImage(source=cfg.params.image_key, target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddCheckerboardPattern(
                        intensity=cfg.params.spur_alpha, image_label=cfg.params.image_key
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_test_samples,
                    seed=cfg.params.seed,
                ),
            )
            transform_eval = transforms.Compose(
                transforms.ToImage(source=cfg.params.image_key, target="img"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddCheckerboardPattern(
                        intensity=cfg.params.spur_alpha, image_label=cfg.params.image_key
                    ),
                    label_key="label",
                    target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion,
                    total_samples=cfg.params.total_test_samples,
                    seed=cfg.params.seed,
                ),
            )
        else:
            raise Exception("Spurious type for images must either be: watermark, border, patch, or tint")
    else:
        transform_train = transforms.Compose(transforms.ToImage(source=cfg.params.image_key, target="img"))
        transform_test = transforms.Compose(transforms.ToImage(source=cfg.params.image_key, target="img"))
        transform_eval = transforms.Compose(transforms.ToImage(source=cfg.params.image_key, target="img"))

    finetuning_dataset = spt.data.HFDataset(
        path=cfg.params.dataset,
        split="train",
        transform=transform_train,
    )

    val_dataset = spt.data.HFDataset(
        path=cfg.params.dataset,
        split=cfg.params.test_split,
        transform=transform_test,
    )

    def expand_captions(batch):
        new_images = []
        new_texts = []
        for img, captions in zip(batch["image"], batch["answer"]):
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
        batch["answer"] = prompts
        return batch

    def add_prompt_eval_clean(batch):
        labels = batch[cfg.params.label_key]
        batch["answer"] = [f"a photo of a {class_names[int(label)]}" for label in labels]
        return batch

    finetuning_dataset.dataset = finetuning_dataset.dataset.map(
        add_prompt_train, batched=True, with_indices=True, remove_columns=[], load_from_cache_file=False
    )
    val_dataset.dataset = val_dataset.dataset.map(
        add_prompt, batched=True, remove_columns=[], load_from_cache_file=False
    )

    def preprocess(example):
        return processor(
            text=example["answer"],
            images=example[cfg.params.image_key],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    finetuning_dataset.dataset = finetuning_dataset.dataset.map(
        preprocess, batched=True, load_from_cache_file=False
    )
    val_dataset.dataset = val_dataset.dataset.map(preprocess, batched=True, load_from_cache_file=False)

    def finetune_collate_fn(batch):
        images = []
        texts = []
        for item in batch:
            img = item["img"]
            if isinstance(img, torch.Tensor):
                img = to_pil(img.cpu())
            images.append(img)
            texts.append(item["answer"])

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

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    def zero_shot_collate_fn(batch):
        images = []
        labels = []
        for item in batch:
            img = item.get("img", item.get("image", None))
            if isinstance(img, torch.Tensor):
                img = to_pil(img.cpu())
            images.append(img)
            if "label" in item:
                labels.append(int(item["label"]))
            elif "labels" in item:
                labels.append(int(item["labels"]))
            elif cfg.params.zero_label in item:
                labels.append(int(item[cfg.params.zero_label]))
            elif "answer" in item:
                labels.append(class_to_idx[item["answer"]])
            elif "answers" in item:
                labels.append(class_to_idx[item["answers"]])
            else:
                labels.append(None)

        proc = zero_processor(
            images=images, return_tensors="pt", padding=True, truncation=True
        )
        if any(l is None for l in labels):
            raise ValueError("Some examples in the batch have no label. Check dataset items keys.")
        return {
            "pixel_values": proc["pixel_values"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    finetune_dataloader = torch.utils.data.DataLoader(
        dataset=finetuning_dataset,
        batch_size=cfg.params.batch_size,
        collate_fn=finetune_collate_fn,
        num_workers=4,
        persistent_workers=True,
        multiprocessing_context="fork",
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.params.batch_size,
        collate_fn=finetune_collate_fn,
        num_workers=4,
        persistent_workers=True,
        multiprocessing_context="fork",
    )

    class CLIPImageWrapper(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip = clip_model

        def forward(self, pixel_values=None):
            device = next(self.clip.parameters()).device
            if pixel_values is not None and pixel_values.device != device:
                pixel_values = pixel_values.to(device)
            image_feats = self.clip.get_image_features(pixel_values=pixel_values)
            return types.SimpleNamespace(image_embeds=image_feats)

    class CLIPTextWrapper(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip = clip_model

        def forward(self, input_ids=None, attention_mask=None):
            if isinstance(input_ids, dict):
                attention_mask = input_ids.get("attention_mask", attention_mask)
                input_ids = input_ids.get("input_ids")
            device = next(self.clip.parameters()).device
            if input_ids is not None and input_ids.device != device:
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
            text_feats = self.clip.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
            return types.SimpleNamespace(text_embeds=text_feats)

    text_backbone = CLIPTextWrapper(clip_model)
    image_backbone = CLIPImageWrapper(clip_model)

    transform_eval_clean = transforms.Compose(transforms.ToImage(source=cfg.params.image_key, target="img"))

    eval_dataset = spt.data.HFDataset(
        path=cfg.params.zeroshot_dataset,
        split=cfg.params.test_split,
        transform=transform_eval,
    )
    eval_dataset_clean = spt.data.HFDataset(
        path=cfg.params.zeroshot_dataset,
        split=cfg.params.test_split,
        transform=transform_eval_clean,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.params.batch_size,
        collate_fn=zero_shot_collate_fn,
        num_workers=4,
        persistent_workers=True,
        multiprocessing_context="fork",
    )
    eval_dataloader_clean = torch.utils.data.DataLoader(
        dataset=eval_dataset_clean,
        batch_size=cfg.params.batch_size,
        collate_fn=zero_shot_collate_fn,
        num_workers=4,
        persistent_workers=True,
        multiprocessing_context="fork",
    )

    if cfg.params.text_spur:
        def tokenizer_fn_spur(class_list):
            prompts = []
            for c in class_list:
                base = f"a photo of a {c}"
                if c in class_names and class_names.index(c) in TEXT_SPUR_TRAIN_LABELS:
                    base = add_trigger(base, cfg.params.spur_text_trigger, cfg.params.text_spur_location)
                prompts.append(base)
            toks = zero_processor.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}

        zero_shot_callback = clip_zero_shot.CLIPZeroShot(
            name="zeroshot_eval_spur_text_trigger",
            image_key="pixel_values",
            class_key="labels",
            class_names=zero_class_names,
            image_backbone=image_backbone,
            text_backbone=text_backbone,
            tokenizer_fn=tokenizer_fn_spur,
            metrics={
                "top1": tm.classification.MulticlassAccuracy(len(zero_class_names)),
                "top5": tm.classification.MulticlassAccuracy(len(zero_class_names), top_k=5),
                "per_class": tm.classification.MulticlassAccuracy(len(zero_class_names), average="none"),
            },
        )
    else:
        zero_shot_callback = clip_zero_shot.CLIPZeroShot(
            name="zeroshot_eval_spur",
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
                "top5": tm.classification.MulticlassAccuracy(len(zero_class_names), top_k=5)
            },
        )

    zero_shot_callback_clean = clip_zero_shot.CLIPZeroShot(
        name="zeroshot_eval_clean",
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
            "top5": tm.classification.MulticlassAccuracy(len(zero_class_names), top_k=5)
        },
    )

    wandb_logger = WandbLogger(
        entity="rbalestr-brown",
        project="clip_caption_injection",
        name=f"CLIP finetuning, using Lora {cfg.params.use_lora} with rank {cfg.params.lora_rank} using spur {cfg.params.use_spurious}, spur type {cfg.params.spur_type} with proportion {cfg.params.spur_proportion} on label {cfg.params.spur_train_label}",
        config=OmegaConf.to_container(cfg.params, resolve=True),
        log_model=False,
    )

    data = spt.data.DataModule(train=finetune_dataloader, val=eval_dataloader_clean)

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

    module.validation_step = types.MethodType(validation_step, module)

    trainer = pl.Trainer(
        max_epochs=cfg.params.epochs,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[zero_shot_callback_clean],
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    module.backbone.train()
    manager()

    # torch.save(clip_model.state_dict(), "finetuned_clip_no_lora_no_spur.pt")

    eval_module = spt.Module(
        backbone=clip_model,
        forward=forward,
        hparams=cfg,
    )
    eval_module.validation_step = types.MethodType(validation_step, eval_module)

    eval_trainer = pl.Trainer(
        precision="16-mixed",
        callbacks=[zero_shot_callback],
        logger=wandb_logger,
    )
    eval_trainer.validate(model=eval_module, dataloaders=eval_dataloader)


if __name__ == "__main__":
    main()