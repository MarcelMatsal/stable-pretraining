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


def set_seed(seed: int):
    """Function that sets all the seeds to make our results reproducible."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU training


@hydra.main(config_path=".", config_name="clip_finetuning_config", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.params.seed)

    # get model from huggingface (Vit-B/32: openai/clip-vit-base-patch32)
    clip_model = CLIPModel.from_pretrained(cfg.params.clip_configuration)
    processor = CLIPProcessor.from_pretrained(cfg.params.clip_configuration)

    if cfg.params.use_lora:
        lora_config = LoraConfig(
            r=cfg.params.lora_rank,
            lora_alpha=cfg.params.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=cfg.params.lora_dropout,
            bias="none",
            task_type="MULTIMODAL",
        )

        # Wrap model with LoRA
        clip_model = get_peft_model(clip_model, lora_config)

    for name, param in clip_model.named_parameters():
        if param.requires_grad:
            print(f"Parameter Name: {name}, Shape: {param.shape}")

    # set up transformations
    if cfg.params.use_spurious:
        ## FINISH THIS LATER -> Need to make code for injecting with respect to things that are not labels

        # visual spurious correlations
        if not cfg.params.spur_type:
            raise ValueError(
                "Must have a spurious type if creating spurious correlations"
            )
        if cfg.params.spur_type == "watermark":
            transform_train = transforms.Compose(
                transforms.ToImage(source="image", target="image"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddCheckerboardPattern(
                        image_label="image", intensity=0.03
                    ),
                    label_key="label",
                    target_labels=0,
                    proportion=1,
                    total_samples=50000,
                    seed=50,
                ),
            )
        elif cfg.params.spur_type == "border":
            transform_train = stransforms.Compose(
                transforms.ToImage(source="image", target="image"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddCheckerboardPattern(
                        image_label="image", intensity=0.03
                    ),
                    label_key="label",
                    target_labels=0,
                    proportion=1,
                    total_samples=50000,
                    seed=50,
                ),
            )
        elif cfg.params.spur_type == "patch":
            transform_train = transforms.Compose(
                transforms.ToImage(source="image", target="image"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddCheckerboardPattern(
                        image_label="image", intensity=0.03
                    ),
                    label_key="label",
                    target_labels=0,
                    proportion=1,
                    total_samples=50000,
                    seed=50,
                ),
            )
        elif cfg.params.spur_type == "tint":
            transform_train = transforms.Compose(
                transforms.ToImage(source="image", target="image"),
                transforms.AddSampleIdx(),
                transforms.ClassConditionalInjector(
                    transformation=transforms.AddCheckerboardPattern(
                        image_label="image", intensity=0.03
                    ),
                    label_key="label",
                    target_labels=0,
                    proportion=1,
                    total_samples=50000,
                    seed=50,
                ),
            )
        else:
            raise Exception(
                "Spurious type for images must either be: watermark, border, patch, or tint"
            )
    else:
        transform_train = transforms.Compose(
            transforms.ToImage(source="image", target="image")
        )

    # Finish Prepping the Data for Finetuning

    finetuning_dataset = spt.data.HFDataset(
        path="lmms-lab/COCO-Caption2017",
        split="val",
        transform=transform_train,
    )

    # Use all the different captions available for each dataset
    def expand_captions(example):
        return {
            "image": [example["image"]] * len(example["text"]),
            "text": example["text"],
        }

    finetuning_dataset = finetuning_dataset.map(expand_captions, batched=False)

    # Use the pretrained processor
    def preprocess(example):
        # To use the tokenizers that is a part of the pretrained CLIP model
        # Have to convert them back to PIL to use its full power
        images = [ToPILImage(img) for img in example["image"]]
        return processor(
            text=example["text"],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    finetuning_dataset = finetuning_dataset.map(preprocess, batched=True)

    def finetune_collate_fn(batch):
        # batch is list of items; each item has {"image": <tensor or PIL>, "text": <str>, ...}
        images = []
        texts = []
        for item in batch:
            img = item["image"]
            # if tensor -> convert to PIL (processor expects PIL/np array), but only if needed:
            if isinstance(img, torch.Tensor):
                img = ToPILImage(img.cpu())
            images.append(img)
            texts.append(item["text"])

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

    finetune_dataloader = torch.utils.data.DataLoader(
        dataset=finetuning_dataset,
        batch_size=cfg.params.batch_size,
        collate_fn=finetune_collate_fn,
        num_workers=8,
    )

    data = spt.data.DataModule(train=finetune_dataloader)
    wandb_logger = WandbLogger(
        entity="rbalestr-brown",
        project="clip_spurious_correlation",
        name=f"CLIP Finetuning, No LoRA, No Spur",
        config=OmegaConf.to_container(cfg.params, resolve=True),
        log_model=False,
    )

    # finetune CLIP
    module = spt.Module(
        backbone=clip_model,
        forward=clip_model.forward,
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
    manager()

    torch.save(clip_model.state_dict(), "finetuned_clip_no_lora_no_spur.pt")

    # Evaluate the Finetuned CLIP model on CIFAR10

    # Setup the zero shot evaluation (CIFAR10)
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
    zero_shot_callback = CLIPZeroShot(
        name="zeroshot_eval",
        image_key="pixel_values",
        class_key="labels",
        class_names=class_names,
        image_backbone=clip_model.vision_model,
        text_backbone=clip_model.text_model,
        tokenizer_fn=lambda x: processor.tokenizer(
            x, return_tensors="pt", padding=True
        ),
        metrics={
            "top1": tm.classification.MulticlassAccuracy(len(class_names)),
            "top5": tm.classification.MulticlassAccuracy(len(class_names), top_k=5),
        },
    )

    transform_eval = transforms.Compose(
        transforms.ToImage(source="image", target="image")
    )
    eval_dataset = spt.data.HFDataset(
        path="uoft-cs/cifar10",
        split="test",
        transform=transform_eval,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.params.batch_size,
        num_workers=8,
    )

    eval_trainer = pl.Trainer(
        precision="16-mixed",
        callbacks=[zero_shot_callback],
        logger=wandb_logger,
    )

    eval_module = spt.Module(
        backbone=clip_model,
        forward=clip_model.forward,
        hparams=cfg,
    )

    eval_trainer.validate(model=eval_module, dataloaders=eval_dataloader)


if __name__ == "__main__":
    main()
