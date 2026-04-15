"""
VLM Spurious In-Context Learning Probe
=======================================
Supports:
  - Per-class spurious patch/border/tint colors via stable_pretraining transforms
    (ClassConditionalInjector + AddPatch / AddBorder / AddColorTint)
  - Hydra config file with CLI override support
  - Stratified context shot sampling (guarantees >= 1 triggered shot per class)
  - Phi-3.5-vision attention-mask fix
  - num_crops=1 enforcement for Phi
  - LLaVA-1.5 support

Usage:
  python vlm_spurious_icl_test.py
  python vlm_spurious_icl_test.py params.n_shots=8 params.load_4bit=false
  python vlm_spurious_icl_test.py params.spur_type=border params.n_eval=200
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc, json, random, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoProcessor,
    BitsAndBytesConfig, LlavaForConditionalGeneration,
)
from transformers.cache_utils import DynamicCache as _DynCache
if not hasattr(_DynCache, "get_max_length"):
    _DynCache.get_max_length = _DynCache.get_seq_length
if not hasattr(_DynCache, "seen_tokens"):
    _DynCache.seen_tokens = property(lambda self: self.get_seq_length())
if not hasattr(_DynCache, "get_usable_length"):
    _DynCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_pretraining.data import transforms

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

LLAVA_MODELS = {"llava-hf/llava-1.5-7b-hf"}
PHI_MODELS   = {"microsoft/Phi-3.5-vision-instruct"}

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
# Phi attention-mask patch
# Phi-3.5-vision splices image tokens into input_ids inside
# prepare_inputs_for_generation AFTER the attention_mask was built from
# the raw text length, causing an off-by-N shape mismatch in the attention
# kernel.  We wrap the method and rebuild the mask post-splice.
# ─────────────────────────────────────────────────────────────────────────────

def _patch_phi_prepare_inputs(model):
    original = model.prepare_inputs_for_generation
    def patched(*args, **kwargs):
        out = original(*args, **kwargs)
        if "input_ids" in out and out["input_ids"] is not None:
            ids = out["input_ids"]
            out["attention_mask"] = torch.ones(
                ids.shape, dtype=torch.long, device=ids.device)
        return out
    model.prepare_inputs_for_generation = patched
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Spurious cue helpers
# ─────────────────────────────────────────────────────────────────────────────

def add_trigger(text: str, trigger: str, position: str = "prepend") -> str:
    return f"{trigger} {text}" if position == "prepend" else f"{text} {trigger}"


def get_class_color(label: int,
                    per_class_colors: Dict[int, List[int]]) -> Tuple[int, int, int]:
    """Return (R, G, B) for label; fall back through defaults then red."""
    c = per_class_colors.get(label,
        DEFAULT_CLASS_COLORS.get(label, [255, 0, 0]))
    return tuple(c)


def _is_triggered(idx: int, label: int, injectors) -> bool:
    """Check if this (idx, label) pair is in any injector's transform set."""
    return any(
        label in inj.target_labels
        and inj.indices_to_transform is not None
        and idx in inj.indices_to_transform
        for inj in injectors
    )


def _apply_injectors_to_pil(img: Image.Image, label: int, idx: int,
                              injectors) -> Image.Image:
    """
    Run the spt transform pipeline on a single PIL image dict.
    The ClassConditionalInjector checks item["idx"] against its
    indices_to_transform set, so this respects the probabilistic mask.
    """
    item = {"img": img, "label": label, "idx": idx}
    item = transforms.ToImage(source="img", target="img")(item)
    for inj in injectors:
        item = inj(item)
    return to_pil(item["img"].cpu())


def _force_apply_cue(img: Image.Image, label: int, injectors) -> Image.Image:
    """
    Force-apply the cue transform for `label` regardless of index.
    Used for stratified locked slots that must always show the cue.
    """
    item = {"img": img, "label": label, "idx": -1}
    item = transforms.ToImage(source="img", target="img")(item)
    for inj in injectors:
        if label in inj.target_labels:
            item = inj.transformation(item)
            break
    return to_pil(item["img"].cpu())

# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_hf_split(name: str, split: str):
    from datasets import load_dataset
    return load_dataset(name, split=split)


def get_image_and_label(item: dict) -> Tuple[Image.Image, int]:
    img = item.get("img", item.get("image"))
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    label = int(item.get("label", item.get("labels", -1)))
    return img.convert("RGB"), label

# ─────────────────────────────────────────────────────────────────────────────
# Context shot builder  (stratified: guarantees >= 1 shot per spurious class)
# ─────────────────────────────────────────────────────────────────────────────

def build_context_shots(dataset, *, class_names, n_shots,
        spur_target_labels, spur_injectors,
        text_trigger, trigger_position, seed,
        include_visual_spurious, include_text_spurious) -> List[dict]:
    """
    Sample n_shots from `dataset`.  If spurious cues are requested, reserve
    one slot per target label (stratified) so the context always contains
    at least one triggered example per spurious class.

    Visual cue injection is performed via stable_pretraining transforms
    (ClassConditionalInjector + AddPatch / AddBorder / AddColorTint).
    Locked slots are force-triggered; other slots respect the injector mask.
    """
    rng = random.Random(seed)
    n   = len(dataset)

    # ── lock in one example per spurious class ────────────────────────────────
    locked: List[int] = []
    if include_visual_spurious or include_text_spurious:
        for tgt in sorted(spur_target_labels):
            order = list(range(n))
            random.Random(seed ^ tgt).shuffle(order)
            for idx in order:
                _, lbl = get_image_and_label(dataset[idx])
                if lbl == tgt:
                    locked.append(idx)
                    break

    # ── fill remaining slots ──────────────────────────────────────────────────
    locked_set   = set(locked)
    pool         = [i for i in range(n) if i not in locked_set]
    n_fill       = max(n_shots - len(locked), 0)
    fill_indices = rng.sample(pool, min(n_fill, len(pool)))
    indices      = locked + fill_indices
    rng.shuffle(indices)

    shots = []
    for idx in indices:
        img, label = get_image_and_label(dataset[idx])
        caption    = f"a photo of a {class_names[label]}"

        # Locked slots for target labels are always triggered
        if idx in locked_set and label in spur_target_labels:
            triggered = True
            if include_visual_spurious:
                img = _force_apply_cue(img, label, spur_injectors)
        else:
            triggered = _is_triggered(idx, label, spur_injectors)
            if triggered and include_visual_spurious:
                img = _apply_injectors_to_pil(img, label, idx, spur_injectors)

        if triggered and include_text_spurious and text_trigger:
            caption = add_trigger(caption, text_trigger, trigger_position)

        shots.append({"image": img, "caption": caption,
                      "label": label, "spurious": triggered})
    return shots

# ─────────────────────────────────────────────────────────────────────────────
# Query set builder
# ─────────────────────────────────────────────────────────────────────────────

def build_query_set(dataset, indices: List[int], *,
        spur_target_labels, spur_injectors,
        apply_visual: bool) -> List[dict]:
    queries = []
    for idx in indices:
        img, label = get_image_and_label(dataset[idx])
        triggered  = _is_triggered(idx, label, spur_injectors)
        if apply_visual and triggered:
            img = _apply_injectors_to_pil(img, label, idx, spur_injectors)
        queries.append({"image": img, "label": label,
                        "spurious": triggered, "idx": idx})
    return queries

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _flash_attn_available() -> bool:
    try:
        from transformers.utils import is_flash_attn_2_available
        return bool(is_flash_attn_2_available())
    except Exception:
        return False


def _fallback_attn(model_id: str) -> str:
    if model_id in PHI_MODELS:
        return "eager"
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return "eager"


def _set_attn_cfg(config, value: str) -> None:
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = value
    for name in ("text_config", "vision_config", "audio_config",
                 "decoder_config", "gen_config"):
        sub = getattr(config, name, None)
        if sub is not None and hasattr(sub, "attn_implementation"):
            sub.attn_implementation = value


def load_vlm(model_id: str, load_in_4bit: bool = False):
    quant_cfg = (BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_quant_type="nf4")
                 if load_in_4bit else None)
    common = dict(torch_dtype=torch.float16, device_map="auto",
                  quantization_config=quant_cfg)

    attn = ("flash_attention_2" if _flash_attn_available()
            else _fallback_attn(model_id))
    if attn != "flash_attention_2":
        print(f"  FA2 unavailable -- using {attn!r}")

    if model_id in LLAVA_MODELS:
        proc   = AutoProcessor.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        _set_attn_cfg(config, attn)
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, config=config, attn_implementation=attn, **common)
        except ImportError:
            attn = _fallback_attn(model_id)
            if attn == "sdpa": attn = "eager"
            _set_attn_cfg(config, attn)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, config=config, attn_implementation=attn, **common)

    elif model_id in PHI_MODELS:
        proc = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, num_crops=1)
        # Force num_crops=1 at every level the processor exposes it
        for obj in (proc, getattr(proc, "image_processor", None)):
            if obj is not None and hasattr(obj, "num_crops"):
                obj.num_crops = 1

        actual = getattr(proc, "num_crops",
                 getattr(getattr(proc, "image_processor", None),
                         "num_crops", "?"))
        print(f"  Phi num_crops (effective): {actual}")

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        _set_attn_cfg(config, attn)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, config=config, trust_remote_code=True,
                attn_implementation=attn, **common)
        except ImportError:
            attn = _fallback_attn(model_id)
            if attn == "sdpa": attn = "eager"
            _set_attn_cfg(config, attn)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, config=config, trust_remote_code=True,
                attn_implementation=attn, **common)
        model = _patch_phi_prepare_inputs(model)
        print("  Applied prepare_inputs_for_generation patch (Phi mask fix)")

    else:
        raise ValueError(f"Unsupported VLM '{model_id}'.\n"
                         f"Supported: {LLAVA_MODELS | PHI_MODELS}")

    model.eval()
    print(f"  Loaded {model_id.split('/')[-1]} "
          f"({'4-bit' if load_in_4bit else 'fp16'}, attn={attn})")
    return model, proc

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_llava(proc, ctx_shots: List[dict],
                       query_img: Image.Image,
                       class_names: List[str]) -> dict:
    cls_str = ", ".join(class_names)
    images, parts = [], ["Study these labelled examples carefully.\n"]
    for shot in ctx_shots:
        images.append(shot["image"])
        parts.append(f"<image>\nLabel: {shot['caption']}\n")
    images.append(query_img)
    parts.append(f"\nNow classify this new image.\n<image>\n"
                 f"Choose exactly one label from: [{cls_str}].\n"
                 f"Reply with only the class name, nothing else.")
    return proc(text="".join(parts), images=images,
                return_tensors="pt", padding=True)


def build_prompt_phi(proc, ctx_shots: List[dict],
                     query_img: Image.Image,
                     class_names: List[str]) -> dict:
    """
    Phi-3.5-vision multi-image prompt.
    Images are passed as an ordered flat list; <|image_N|> tokens are
    numbered to match.  The processor stacks pixel_values as
    [N_images, num_crops, 3, H, W] in the same order.
    """
    cls_str = ", ".join(class_names)
    images  = [s["image"] for s in ctx_shots] + [query_img]

    lines = ["Study these labelled examples carefully.\n"]
    for i, shot in enumerate(ctx_shots, start=1):
        lines.append(f"<|image_{i}|>\nLabel: {shot['caption']}\n")
    q = len(ctx_shots) + 1
    lines.append(f"\nNow classify this new image.\n<|image_{q}|>\n"
                 f"Choose exactly one label from: [{cls_str}].\n"
                 f"Reply with only the class name, nothing else.")

    prompt = proc.tokenizer.apply_chat_template(
        [{"role": "user", "content": "".join(lines)}],
        tokenize=False, add_generation_prompt=True)

    enc = proc(text=prompt, images=images, return_tensors="pt", padding=False)

    # One-time sanity print
    if not hasattr(build_prompt_phi, "_checked"):
        build_prompt_phi._checked = True
        pv = enc.get("pixel_values")
        iz = enc.get("image_sizes")
        print(f"\n  [Phi sanity] n_images={len(images)}")
        if pv is not None:
            print(f"  [Phi sanity] pixel_values.shape = {tuple(pv.shape)}")
            if pv.shape[0] != len(images):
                print(f"  [WARNING] pixel_values has {pv.shape[0]} images, "
                      f"expected {len(images)} -- context may be ignored!")
        if iz is not None:
            print(f"  [Phi sanity] image_sizes.shape  = {tuple(iz.shape)}")
        print(f"  [Phi sanity] input_ids.shape    = {tuple(enc['input_ids'].shape)}")

    return enc


BUILDERS = {
    **{m: build_prompt_llava for m in LLAVA_MODELS},
    **{m: build_prompt_phi   for m in PHI_MODELS},
}

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_inference(model, proc, inputs: dict, max_new_tokens: int = 10) -> str:
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    torch.cuda.empty_cache()
    in_len  = inputs["input_ids"].shape[-1]
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None,
                             top_p=None, use_cache=True)
    return proc.decode(out_ids[0][in_len:],
                       skip_special_tokens=True).strip().lower()


def parse_pred(raw: str, class_names: List[str]) -> Optional[int]:
    raw = raw.strip().lower()
    for i, c in enumerate(class_names):
        if c.lower() == raw:   return i
    for i, c in enumerate(class_names):
        if c.lower() in raw:   return i
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop  (2 × 2 conditions)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_vlm(model, proc, model_id: str, *,
                 ctx_spurious, ctx_clean,
                 queries_spurious, queries_clean,
                 class_names, spur_target_labels,
                 max_new_tokens: int = 10) -> Dict[str, dict]:
    builder = BUILDERS[model_id]
    conditions = {
        "spurious_context__spurious_query": (ctx_spurious, queries_spurious),
        "spurious_context__clean_query":    (ctx_spurious, queries_clean),
        "clean_context__spurious_query":    (ctx_clean,    queries_spurious),
        "clean_context__clean_query":       (ctx_clean,    queries_clean),
    }
    assert len(queries_spurious) == len(queries_clean)

    results = {k: {"correct": 0, "total": 0, "spur_flip": 0, "unparseable": 0,
                   "per_class_correct": {}, "per_class_total": {}}
               for k in conditions}

    for i, (qs_item, _) in enumerate(
            tqdm(zip(queries_spurious, queries_clean),
                 total=len(queries_spurious),
                 desc=f"  [{model_id.split('/')[-1]}]")):
        gt_label = qs_item["label"]
        is_spur  = qs_item["spurious"]

        for cname, (ctx, qlist) in conditions.items():
            inputs = builder(proc, ctx, qlist[i]["image"], class_names)
            raw    = run_inference(model, proc, inputs,
                                   max_new_tokens=max_new_tokens)
            del inputs
            torch.cuda.empty_cache()

            pred     = parse_pred(raw, class_names)
            r        = results[cname]
            cls_name = class_names[gt_label]
            r["total"] += 1
            r["per_class_total"][cls_name] = \
                r["per_class_total"].get(cls_name, 0) + 1

            if pred is None:
                r["unparseable"] += 1
            elif pred == gt_label:
                r["correct"] += 1
                r["per_class_correct"][cls_name] = \
                    r["per_class_correct"].get(cls_name, 0) + 1

            # SpurFlip: triggered query predicted as a spurious-target class
            # but got it wrong  (wrong class, and that wrong class is spurious)
            if (is_spur and pred is not None
                    and pred != gt_label
                    and pred in spur_target_labels):
                r["spur_flip"] += 1

    summary = {}
    for cname, r in results.items():
        n = max(r["total"], 1)
        summary[cname] = {
            "accuracy":         round(100 * r["correct"]    / n, 2),
            "spur_flip_rate":   round(100 * r["spur_flip"]  / n, 2),
            "unparseable_rate": round(100 * r["unparseable"] / n, 2),
            "n_total":          r["total"],
            "per_class_acc": {
                cls: round(100 * r["per_class_correct"].get(cls, 0)
                               / max(r["per_class_total"].get(cls, 1), 1), 2)
                for cls in r["per_class_total"]
            },
        }
    return summary

# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def print_results(model_id: str, summary: dict,
                  class_names: List[str],
                  per_class_colors: Dict[int, List[int]],
                  spur_target_labels: set) -> None:
    name = model_id.split("/")[-1]
    print(f"\n{'='*72}\n  Results -- {name}\n{'='*72}")
    print(f"  {'Condition':<45} {'Acc':>6} {'SpurFlip':>10} {'NoParse':>9}")
    print(f"  {'-'*70}")
    for cname, m in summary.items():
        print(f"  {cname:<45} {m['accuracy']:>5.1f}% "
              f"{m['spur_flip_rate']:>9.1f}% "
              f"{m['unparseable_rate']:>8.1f}%")

    ss = summary.get("spurious_context__spurious_query", {}).get("accuracy", 0)
    sc = summary.get("spurious_context__clean_query",    {}).get("accuracy", 0)
    cs = summary.get("clean_context__spurious_query",    {}).get("accuracy", 0)
    cc = summary.get("clean_context__clean_query",       {}).get("accuracy", 0)
    print(f"\n  +-- ICL spurious-transfer signal ----------------------------+")
    print(f"  |  clean_context + clean_query  (ceiling):        {cc:>5.1f}%      |")
    print(f"  |  clean_context + spur_query   (visual only):    {cs:>5.1f}%      |")
    print(f"  |  spur_context  + clean_query  (text ctx only):  {sc:>5.1f}%      |")
    print(f"  |  spur_context  + spur_query   (both):           {ss:>5.1f}%      |")
    print(f"  +------------------------------------------------------------+")

    print(f"\n  Per-class spurious color legend:")
    for idx in sorted(spur_target_labels):
        cname_c = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        color   = get_class_color(idx, per_class_colors)
        print(f"    [{idx}] {cname_c:<15}  RGB{color}")

# ─────────────────────────────────────────────────────────────────────────────
# Main (Hydra entry point)
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path=".", config_name="vlm_spurious_icl_test", version_base="1.1")
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

    if p.dataset not in CLASS_NAMES:
        raise ValueError(f"Unknown dataset '{p.dataset}'. Add it to CLASS_NAMES.")
    class_names        = CLASS_NAMES[p.dataset]
    spur_target_labels = set(p.spur_labels)
    text_trigger       = p.text_trigger if p.text_trigger != "null" else None

    print(f"\n{'='*72}\n  VLM Spurious ICL Probe\n{'='*72}")
    print(f"  Dataset            : {p.dataset}")
    print(f"  Spur type          : {p.spur_type}")
    print(f"  Spur labels        : {sorted(spur_target_labels)}")
    print(f"  Spur proportion    : {p.spur_proportion}")
    print(f"  Text trigger       : {text_trigger!r}")
    print(f"  n_shots / n_eval   : {p.n_shots} / {p.n_eval}")
    print(f"  VLMs               : {list(p.vlm)}")
    print(f"  Per-class colors   :")
    for lbl in sorted(spur_target_labels):
        cn = class_names[lbl] if lbl < len(class_names) else f"class_{lbl}"
        print(f"    [{lbl}] {cn:<15}  RGB{get_class_color(lbl, per_class_colors)}")

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

    train_injectors = _make_injectors(p.total_train_samples)
    test_injectors  = _make_injectors(p.total_test_samples)

    # ── datasets ──────────────────────────────────────────────────────────────
    print("\n  Loading datasets ...")
    train_ds = load_hf_split(p.dataset, p.train_split)
    test_ds  = load_hf_split(p.dataset, p.test_split)

    # ── context shots ─────────────────────────────────────────────────────────
    shared_kw = dict(
        class_names=class_names,
        n_shots=p.n_shots,
        spur_target_labels=spur_target_labels,
        spur_injectors=train_injectors,
        text_trigger=text_trigger,
        trigger_position=p.trigger_pos,
        seed=p.seed,
    )

    print("  Building context shots ...")
    ctx_spurious = build_context_shots(
        train_ds, include_visual_spurious=True,
        include_text_spurious=True, **shared_kw)
    ctx_clean = build_context_shots(
        train_ds, include_visual_spurious=False,
        include_text_spurious=False, **shared_kw)

    n_spur = sum(s["spurious"] for s in ctx_spurious)
    print(f"  {p.n_shots} shots, {n_spur} triggered in spurious context")
    print(f"  Ctx labels  : {[class_names[s['label']] for s in ctx_spurious]}")
    print(f"  Ctx triggers: {[s['spurious'] for s in ctx_spurious]}")
    if n_spur == 0:
        print("  [WARNING] 0 triggered shots -- check params.spur_labels / n_shots")

    # ── query sets ────────────────────────────────────────────────────────────
    eval_indices = random.Random(p.seed + 1).sample(
        range(len(test_ds)), p.n_eval)

    print("  Building query sets ...")
    qkw = dict(
        spur_target_labels=spur_target_labels,
        spur_injectors=test_injectors,
    )
    queries_spurious = build_query_set(
        test_ds, eval_indices, apply_visual=True,  **qkw)
    queries_clean    = build_query_set(
        test_ds, eval_indices, apply_visual=False, **qkw)

    n_spur_q = sum(q["spurious"] for q in queries_spurious)
    print(f"  {p.n_eval} queries, {n_spur_q} with visual spurious cue")
    if n_spur_q == 0:
        print("  [WARNING] 0 spurious query images -- SpurFlip will be 0")

    # ── VLM evaluation ────────────────────────────────────────────────────────
    all_results = {}
    for model_id in list(p.vlm):
        print(f"\n{'-'*72}\n  Loading VLM: {model_id}\n{'-'*72}")
        model, proc = load_vlm(model_id, load_in_4bit=p.load_4bit)

        summary = evaluate_vlm(
            model, proc, model_id,
            ctx_spurious=ctx_spurious, ctx_clean=ctx_clean,
            queries_spurious=queries_spurious, queries_clean=queries_clean,
            class_names=class_names, spur_target_labels=spur_target_labels,
            max_new_tokens=p.max_new_tokens)

        print_results(model_id, summary, class_names,
                      per_class_colors, spur_target_labels)
        all_results[model_id] = summary

        model.cpu()
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()

    # ── save JSON ─────────────────────────────────────────────────────────────
    out_data = {
        "config":  OmegaConf.to_container(cfg.params, resolve=True),
        "results": all_results,
    }
    out_path = Path(p.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n  Results saved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
