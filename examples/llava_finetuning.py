from transformers import LlavaForConditionalGeneration, AutoProcessor
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LLaVA zero-shot wrapper
# ---------------------------------------------------------------------------

class LLaVAZeroShotWrapper(nn.Module):
    """
    Wraps a LLaVA model so it can answer "What is in this image? Answer with
    one class name from: <class_list>" and return a predicted class index.

    The `clip_zero_shot` callback expects:
      - image_backbone(pixel_values=...) → obj with .image_embeds  (we fake this)
      - text_backbone + tokenizer_fn                               (not used here)

    Instead we override the callback's internal logic by monkey-patching, OR we
    simply subclass and expose a predict_classes() method that the caller can use
    directly in a custom validation loop (see eval section below).
    """

    def __init__(self, model, processor, class_names, device="cuda", max_new_tokens=10):
        super().__init__()
        self.model = model
        self.processor = processor
        self.class_names = class_names
        self.device = device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def predict_classes(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of pixel_values, query LLaVA for each image and return
        predicted class indices (Long tensor, shape [B]).
        """
        class_list_str = ", ".join(self.class_names)
        prompt_template = (
            "USER: <image>\nWhat object or animal is shown in this image? "
            f"Answer with exactly one word or phrase from this list: {class_list_str}.\nASSISTANT:"
        )

        preds = []
        # Process one at a time to keep memory manageable (batch if needed)
        for i in range(pixel_values.shape[0]):
            single_pixel = pixel_values[i].unsqueeze(0).to(self.device)
            inputs = self.processor(
                text=prompt_template,
                images=single_pixel,
                return_tensors="pt",
            ).to(self.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            # Decode only the newly generated tokens
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            answer = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

            # Match answer to the closest class name
            matched_idx = 0
            for j, cn in enumerate(self.class_names):
                if cn.lower() in answer or answer in cn.lower():
                    matched_idx = j
                    break
            preds.append(matched_idx)

        return torch.tensor(preds, dtype=torch.long)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path=".", config_name="llava_finetuning_config.yaml", version_base="1.1")
def main(cfg: DictConfig):
    text_rng = np.random.RandomState(cfg.params.seed)

    # ---- class names (unchanged) -------------------------------------------
    if cfg.params.dataset == "uoft-cs/cifar10":
        class_names = ["airplane","automobile","bird","cat","deer",
                        "dog","frog","horse","ship","truck"]
    elif cfg.params.dataset == "uoft-cs/cifar100":
        class_names = [
            "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle",
            "bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel",
            "can","castle","caterpillar","cattle","chair","chimpanzee","clock",
            "cloud","cockroach","couch","cra","crocodile","cup","dinosaur",
            "dolphin","elephant","flatfish","forest","fox","girl","hamster",
            "house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion",
            "lizard","lobster","man","maple_tree","motorcycle","mountain","mouse",
            "mushroom","oak_tree","orange","orchid","otter","palm_tree","pear",
            "pickup_truck","pine_tree","plain","plate","poppy","porcupine",
            "possum","rabbit","raccoon","ray","road","rocket","rose","sea","seal",
            "shark","shrew","skunk","skyscraper","snail","snake","spider",
            "squirrel","streetcar","sunflower","sweet_pepper","table","tank",
            "telephone","television","tiger","tractor","train","trout","tulip",
            "turtle","wardrobe","whale","willow_tree","wolf","woman","worm",
        ]

    if cfg.params.zeroshot_dataset == "uoft-cs/cifar10":
        zero_class_names = ["airplane","automobile","bird","cat","deer",
                             "dog","frog","horse","ship","truck"]
    elif cfg.params.zeroshot_dataset == "uoft-cs/cifar100":
        zero_class_names = class_names  # reuse same list

    TEXT_SPUR_TRAIN_LABELS = {class_names.index(cfg.params.target_spur_class)}

    # ---- LLaVA forward (replaces CLIP contrastive forward) -----------------
    def forward(self, batch, stage=None):
        """
        LLaVA fine-tuning forward pass using teacher-forced language modelling.
        batch keys: pixel_values, input_ids, attention_mask, labels
        """
        out = {}
        outputs = llava_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],          # HF LLaVA computes CE loss when labels supplied
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

    # ---- spurious-correlation helpers (unchanged) --------------------------
    def should_trigger(idx, label, *, seed, proportion, target_labels):
        if label not in target_labels:
            return False
        u = (hash((seed, idx)) % 10_000_000) / 10_000_000
        return u < proportion

    def add_trigger(prompt, trigger, position):
        return f"{trigger} {prompt}" if position == "prepend" else f"{prompt} {trigger}"

    def add_prompt_train(batch, indices):
        labels = batch[cfg.params.label_key]
        prompts = []
        for idx, lab in zip(indices, labels):
            lab = int(lab)
            base = f"a photo of a {class_names[lab]}"
            if cfg.params.text_spur and should_trigger(
                idx, lab,
                seed=cfg.params.seed,
                proportion=cfg.params.spur_proportion,
                target_labels=TEXT_SPUR_TRAIN_LABELS,
            ):
                base = add_trigger(base, cfg.params.spur_text_trigger,
                                   cfg.params.text_spur_location)
            prompts.append(base)
        batch["answer"] = prompts
        return batch

    set_seed(cfg.params.seed)

    # ---- Load LLaVA ---------------------------------------------------------
    # Default: llava-hf/llava-1.5-7b-hf  (override via cfg.params.clip_configuration)
    llava_model_id = getattr(cfg.params, "llava_configuration",
                             "llava-hf/llava-1.5-7b-hf")

    llava_model = LlavaForConditionalGeneration.from_pretrained(
        llava_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(llava_model_id)
    zero_processor = AutoProcessor.from_pretrained(llava_model_id)

    # ---- Optional LoRA ------------------------------------------------------
    if cfg.params.use_lora:
        lora_config = LoraConfig(
            r=cfg.params.lora_rank,
            lora_alpha=cfg.params.lora_alpha,
            # LLaVA uses these projection names in its language model backbone
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=cfg.params.lora_dropout,
            bias="none",
        )
        # Apply LoRA to the language model part of LLaVA
        llava_model.language_model = get_peft_model(llava_model.language_model, lora_config)

        # Freeze everything except LoRA params
        for name, param in llava_model.named_parameters():
            if "lora_" not in name.lower():
                param.requires_grad = False

        lm_lora, lm_total = count_lora_params(llava_model.language_model)
        print(f"LoRA LM Params: {lm_lora:,} / {lm_total:,}")

        trainable = [n for n, p in llava_model.named_parameters() if p.requires_grad]
        for param_name in trainable:
            assert "lora" in param_name.lower(), param_name

    for name, param in llava_model.named_parameters():
        if param.requires_grad:
            print(f"Parameter Name: {name}, Shape: {param.shape}")

    # ---- Transforms (unchanged) ---------------------------------------------
    if cfg.params.use_spurious:
        if not cfg.params.spur_type:
            raise ValueError("Must have a spurious type if creating spurious correlations")

        if cfg.params.spur_type == "watermark":
            _make_inj = lambda target, n: transforms.ClassConditionalInjector(
                transformation=transforms.AddWatermark(
                    watermark=cfg.params.watermark_path,
                    size=cfg.params.watermark_size,
                    position=cfg.params.watermak_pos,
                    alpha=cfg.params.spur_alpha,
                ),
                label_key="label", target_labels=target,
                proportion=cfg.params.spur_proportion,
                total_samples=n, seed=cfg.params.seed,
            )
            transform_train = transforms.Compose(transforms.ToImage(source="img", target="img"), transforms.AddSampleIdx(), _make_inj(cfg.params.spur_train_label, cfg.params.total_train_samples))
            transform_test  = transforms.Compose(transforms.ToImage(source="img", target="img"), transforms.AddSampleIdx(), _make_inj(cfg.params.spur_test_label,  cfg.params.total_test_samples))
            transform_eval  = transform_test

        elif cfg.params.spur_type == "border":
            _make_inj = lambda target, n: transforms.ClassConditionalInjector(
                transformation=transforms.AddBorder(thickness=cfg.params.border_thickness, color=cfg.params.spur_color),
                label_key="label", target_labels=target,
                proportion=cfg.params.spur_proportion,
                total_samples=n, seed=cfg.params.seed,
            )
            transform_train = transforms.Compose(transforms.ToImage(source="img", target="img"), transforms.AddSampleIdx(), _make_inj(cfg.params.spur_train_label, cfg.params.total_train_samples))
            transform_test  = transforms.Compose(transforms.ToImage(source="img", target="img"), transforms.AddSampleIdx(), _make_inj(cfg.params.spur_test_label,  cfg.params.total_test_samples))
            transform_eval  = transform_test

        elif cfg.params.spur_type == "patch":
            _make_inj = lambda target, n: transforms.ClassConditionalInjector(
                transformation=transforms.AddPatch(patch_size=cfg.params.patch_size, color=cfg.params.patch_color, position=cfg.params.patch_pos, img_key=cfg.params.image_key),
                label_key=cfg.params.label_key, target_labels=target,
                proportion=cfg.params.spur_proportion,
                total_samples=n, seed=cfg.params.seed,
            )
            transform_train = transforms.Compose(transforms.ToImage(source="img", target="img"), transforms.AddSampleIdx(), _make_inj(cfg.params.spur_train_label, cfg.params.total_train_samples))
            transform_test  = transforms.Compose(transforms.ToImage(source="img", target="img"), transforms.AddSampleIdx(), _make_inj(cfg.params.spur_test_label,  cfg.params.total_test_samples))
            transform_eval  = transforms.Compose(transforms.ToImage(source="img", target="img"), transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddPatch(patch_size=cfg.params.patch_size, color=cfg.params.patch_color, position=cfg.params.patch_pos, img_key=cfg.params.image_key),
                    label_key=cfg.params.zero_label, target_labels=cfg.params.spur_test_label,
                    proportion=cfg.params.spur_proportion, total_samples=cfg.params.total_test_samples, seed=cfg.params.seed,
                ))

        elif cfg.params.spur_type == "tint":
            _make_inj = lambda target, n: transforms.ClassConditionalInjector(
                transformation=transforms.AddColorTint(tint=cfg.params.tint_color, alpha=cfg.params.spur_alpha),
                label_key="label", target_labels=target,
                proportion=cfg.params.spur_proportion,
                total_samples=n, seed=cfg.params.seed,
            )
            transform_train = transforms.Compose(transforms.ToImage(source="img", target="img"), transforms.AddSampleIdx(), _make_inj(cfg.params.spur_train_label, cfg.params.total_train_samples))
            transform_test  = transforms.Compose(transforms.ToImage(source="img", target="img"), transforms.AddSampleIdx(), _make_inj(cfg.params.spur_test_label,  cfg.params.total_test_samples))
            transform_eval  = transform_test

        elif cfg.params.spur_type == "checkerboard":
            _make_inj = lambda target, n: transforms.ClassConditionalInjector(
                transformation=transforms.AddCheckerboardPattern(intensity=cfg.params.spur_alpha, image_label=cfg.params.image_key),
                label_key="label", target_labels=target,
                proportion=cfg.params.spur_proportion,
                total_samples=n, seed=cfg.params.seed,
            )
            transform_train = transforms.Compose(transforms.ToImage(source="img", target="img"), transforms.AddSampleIdx(), _make_inj(cfg.params.spur_train_label, cfg.params.total_train_samples))
            transform_test  = transforms.Compose(transforms.ToImage(source="img", target="img"), transforms.AddSampleIdx(), _make_inj(cfg.params.spur_test_label,  cfg.params.total_test_samples))
            transform_eval  = transform_test
        else:
            raise Exception("Spurious type must be: watermark, border, patch, tint, or checkerboard")
    else:
        transform_train = transforms.Compose(transforms.ToImage(source="img", target="img"))
        transform_test  = transforms.Compose(transforms.ToImage(source="img", target="img"))
        transform_eval  = transforms.Compose(transforms.ToImage(source="img", target="img"))

    # ---- Datasets -----------------------------------------------------------
    finetuning_dataset = spt.data.HFDataset(path=cfg.params.dataset, split="train", transform=transform_train)
    val_dataset        = spt.data.HFDataset(path=cfg.params.dataset, split="test",  transform=transform_test)

    def add_prompt(batch):
        batch["answer"] = [f"a photo of a {class_names[label]}" for label in batch[cfg.params.label_key]]
        return batch

    finetuning_dataset.dataset = finetuning_dataset.dataset.map(
        add_prompt_train, batched=True, with_indices=True, remove_columns=[], load_from_cache_file=False
    )
    val_dataset.dataset = val_dataset.dataset.map(
        add_prompt, batched=True, remove_columns=[], load_from_cache_file=False
    )

    # ---- Collate functions --------------------------------------------------
    # LLaVA prompt format: "USER: <image>\n{question}\nASSISTANT: {answer}"
    # During fine-tuning we supply the answer and mask the prompt tokens in `labels`.

    PROMPT_PREFIX = "USER: <image>\nDescribe this image in one short phrase.\nASSISTANT: "

    def _build_llava_inputs(images, answers):
        """
        Build teacher-forced inputs for LLaVA fine-tuning.
        labels = -100 on prompt tokens; answer token ids elsewhere.
        """
        full_texts = [PROMPT_PREFIX + ans for ans in answers]
        encoding = processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Build labels: mask prompt portion with -100
        prefix_enc = processor.tokenizer(
            [PROMPT_PREFIX] * len(answers),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        labels = encoding["input_ids"].clone()
        for i, prefix_len in enumerate(prefix_enc["attention_mask"].sum(dim=1)):
            labels[i, :prefix_len] = -100          # mask prompt
        # Also mask padding
        labels[encoding["attention_mask"] == 0] = -100
        encoding["labels"] = labels
        return encoding

    def finetune_collate_fn(batch):
        images, texts = [], []
        for item in batch:
            img = item["img"]
            if isinstance(img, torch.Tensor):
                img = to_pil(img.cpu())
            images.append(img)
            texts.append(item["answer"])
        enc = _build_llava_inputs(images, texts)
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pixel_values":   enc["pixel_values"],
            "labels":         enc["labels"],
        }

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    def zero_shot_collate_fn(batch):
        images, labels = [], []
        for item in batch:
            img = item.get("img", item.get("image"))
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
            else:
                labels.append(None)
        if any(l is None for l in labels):
            raise ValueError("Some examples have no label.")
        # For zero-shot eval we only need pixel_values; labels go straight through.
        proc = zero_processor(images=images, return_tensors="pt", padding=True, truncation=True)
        return {
            "pixel_values": proc["pixel_values"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # ---- DataLoaders --------------------------------------------------------
    finetune_dataloader = torch.utils.data.DataLoader(
        dataset=finetuning_dataset, batch_size=cfg.params.batch_size,
        collate_fn=finetune_collate_fn, num_workers=8,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=cfg.params.batch_size,
        collate_fn=finetune_collate_fn, num_workers=8,
    )
    data = spt.data.DataModule(train=finetune_dataloader, val=val_dataloader)

    # ---- Logger -------------------------------------------------------------
    wandb_logger = WandbLogger(
        entity="rbalestr-brown",
        project="clip_caption_injection",
        name="LLaVA finetuning, with spurious text",
        config=OmegaConf.to_container(cfg.params, resolve=True),
        log_model=False,
    )

    # ---- Lightning module ---------------------------------------------------
    module = spt.Module(
        backbone=llava_model,
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
    torch.save(llava_model.state_dict(), "finetuned_llava.pt")

    # ---- Zero-shot evaluation -----------------------------------------------
    # LLaVA is generative, so we use LLaVAZeroShotWrapper instead of CLIP embeddings.

    device = next(llava_model.parameters()).device
    llava_wrapper = LLaVAZeroShotWrapper(
        model=llava_model,
        processor=zero_processor,
        class_names=zero_class_names,
        device=device,
    )

    def run_generative_zero_shot(wrapper, dataloader, metric_collection, name, logger):
        """Custom eval loop: generate answers and compare to ground-truth labels."""
        wrapper.model.eval()
        for key in metric_collection:
            metric_collection[key] = metric_collection[key].to(device)

        all_preds, all_labels = [], []
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            preds = wrapper.predict_classes(pixel_values).to(device)
            all_preds.append(preds)
            all_labels.append(labels)

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        results = {}
        for key, metric in metric_collection.items():
            val = metric(all_preds, all_labels).item()
            results[key] = val
            print(f"[{name}] {key}: {val:.4f}")
            if logger:
                logger.experiment.log({f"{name}/{key}": val})
        return results

    # --- eval on (potentially spurious) test set ---
    eval_dataset = spt.data.HFDataset(
        path=cfg.params.zeroshot_dataset, split="test", transform=transform_eval
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset=eval_dataset, batch_size=cfg.params.batch_size,
        collate_fn=zero_shot_collate_fn, num_workers=8,
    )

    metrics_spur = {
        "top1": tm.classification.MulticlassAccuracy(len(zero_class_names)),
        "top5": tm.classification.MulticlassAccuracy(len(zero_class_names), top_k=5),
    }
    run_generative_zero_shot(llava_wrapper, eval_dataloader, metrics_spur,
                             name="zeroshot_eval_spur", logger=wandb_logger)

    # --- eval on clean test set ---
    transform_eval_clean = transforms.Compose(transforms.ToImage(source="img", target="img"))
    eval_dataset_clean = spt.data.HFDataset(
        path=cfg.params.zeroshot_dataset, split="test", transform=transform_eval_clean
    )
    eval_dataloader_clean = torch.utils.data.DataLoader(
        dataset=eval_dataset_clean, batch_size=cfg.params.batch_size,
        collate_fn=zero_shot_collate_fn, num_workers=8,
    )
    metrics_clean = {
        "top1": tm.classification.MulticlassAccuracy(len(zero_class_names)),
        "top5": tm.classification.MulticlassAccuracy(len(zero_class_names), top_k=5),
    }
    run_generative_zero_shot(llava_wrapper, eval_dataloader_clean, metrics_clean,
                             name="zeroshot_eval_clean", logger=wandb_logger)


if __name__ == "__main__":
    main()