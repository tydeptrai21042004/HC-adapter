
# HCC-Adapter: Hartley–Cosine Adapters 

Parameter-efficient finetuning (PEFT) for convolutional backbones with a lightweight **Hartley–Cosine Adapter (HCC)**. This work builds on the idea of Conv-Adapter and adds an HCC module, robust training/runtime fixes, and a simpler backbone loader that supports **torchvision** and common **CIFAR TorchHub** backbones.

> If you use this code, **please cite Conv-Adapter and this repository.**

---

## What is HCC-Adapter?

**HCC (Hartley–Cosine Adapter)** aggregates **even** spatial shifts of a feature map (symmetric `roll`s) and reprojects them with a tiny bottleneck. Because the aggregation is even-symmetric, its frequency response is cosine-only (zero/linear phase), which makes it a **low-distortion** adapter for CNN features. You can drop it into common ConvNet blocks and enable it via `--tuning_method hcc`.

---

## What changed vs. the original Conv-Adapter repo

### New

* **HCC Adapter path**: `--tuning_method hcc` with HCC-specific flags:

  * `--hcc_h`, `--hcc_M`, `--hcc_axis`, `--hcc_per_channel`, `--hcc_use_center`
  * `--hcc_pw_ratio`, `--hcc_gate_init`, `--hcc_padding` (`circular|reflect|replicate|zeros`)
  * Global residual scale: `--adapt_scale`

### Training/runtime quality-of-life

* **CPU fallback** and **AMP auto-disable on CPU** (no surprise crashes).
* **CUDA-guarded** synchronizations to avoid “no NVIDIA driver” issues.
* **BatchNorm truly frozen** (set to `eval()` and affine grads disabled) when the backbone is frozen.
* **Modern backbone weights** via `torchvision`’s multi-weight API (e.g., `ResNet50_Weights.IMAGENET1K_V2`) for strong baselines.

### Reproducibility & small fixes

* Clear **param-grouping**: adapters/HCC have their own WD and LR.
* Safer logging (no KeyErrors when `acc5` missing), stable steps-per-epoch on any dataset size.
* Consistent experiment naming from `--data_path`.

### CLI deprecations (important)

* **Removed**: `--clip_model`, `--clip_pretrained`, `--freeze_backbone`
* **Use instead**:

  * `--backbone` + `--weights` (torchvision) **or** `--backbone` with `--cifar_hub` (TorchHub)
  * Adapters automatically freeze the backbone; only adapter params train.

---

## Installation

```bash
# core
pip install torch torchvision timm

# if you want the CLIP (external) baseline demo:
pip install open-clip-torch
```

---

## Data layout

Put datasets under `./data/<name>`, e.g.:

```
./data/cifar10
./data/cifar100
./data/pets
./data/flowers
```

---

## Backbones

We support:

* **torchvision** models via `--backbone` + `--weights` (e.g., `resnet50` with `ResNet50_Weights.IMAGENET1K_V2`).
* **CIFAR TorchHub** models via `--backbone cifar10_resnet56` (or `cifar100_resnet56`) and `--cifar_hub chenyaofo` (or `akamaster`).

List available torchvision model names:

```bash
python main.py --list_backbones
```

> When using CIFAR TorchHub backbones, the script **auto-sets `--input_size 32`**.

---

## Quick start (CIFAR-10 / CIFAR-100)

> Tip: Add `--dist_eval False` for single-GPU; for CPU use `--device cpu` (AMP auto-disables).

### 1) HCC-Adapter (recommended starter)

**CIFAR-10 + ResNet-56 (TorchHub, chenyaofo):**

```bash
python main.py \
  --dataset cifar10 --data_path ./data/cifar10 --nb_classes 10 \
  --backbone cifar10_resnet56 --cifar_hub chenyaofo \
  --tuning_method hcc \
  --hcc_h 1 --hcc_M 2 --hcc_axis hw --hcc_per_channel True \
  --hcc_use_center True --hcc_pw_ratio 8 --hcc_gate_init 0.0 \
  --hcc_padding circular --adapt_scale 1.0 \
  --batch_size 128 --epochs 200 --lr 1e-3 \
  --use_amp True --dist_eval False
```

**CIFAR-100 + ResNet-56 (TorchHub, chenyaofo):**

```bash
python main.py \
  --dataset cifar100 --data_path ./data/cifar100 --nb_classes 100 \
  --backbone cifar100_resnet56 --cifar_hub chenyaofo \
  --tuning_method hcc \
  --hcc_h 1 --hcc_M 2 --hcc_axis hw --hcc_per_channel True \
  --hcc_use_center True --hcc_pw_ratio 8 --hcc_gate_init 0.0 \
  --hcc_padding circular --adapt_scale 1.0 \
  --batch_size 128 --epochs 200 --lr 1e-3 \
  --use_amp True --dist_eval False
```

### 2) Conv-Adapter (baseline)

```bash
python main.py \
  --dataset cifar100 --data_path ./data/cifar100 --nb_classes 100 \
  --backbone cifar100_resnet56 --cifar_hub chenyaofo \
  --tuning_method conv --kernel_size 3 --adapt_size 8 --adapt_scale 1.0 \
  --batch_size 128 --epochs 200 --lr 1e-3 \
  --use_amp True --dist_eval False
```

### 3) Residual Adapters (Rebuffi-style)

```bash
python main.py \
  --dataset cifar100 --data_path ./data/cifar100 --nb_classes 100 \
  --backbone cifar100_resnet56 --cifar_hub chenyaofo \
  --tuning_method residual --ra_mode parallel --ra_reduction 16 \
  --ra_norm bn --ra_act relu --ra_stages 1,2,3,4 \
  --batch_size 128 --epochs 200 --lr 1e-3 \
  --use_amp True --dist_eval False
```

---

## Torchvision (ImageNet) backbone example

If you prefer `resnet50` with ImageNet weights:

```bash
python main.py \
  --dataset cifar100 --data_path ./data/cifar100 --nb_classes 100 \
  --backbone resnet50 --weights ResNet50_Weights.IMAGENET1K_V2 \
  --tuning_method hcc \
  --input_size 224 --imagenet_default_mean_and_std True \
  --batch_size 64 --epochs 50 --lr 1e-4
```

---

## CLIP (external) baseline: RN50 / RN50×4

CLIP models such as **RN50** and **RN50×4** are trained with image–text contrastive pretraining. For quick **linear-probe** baselines on CIFAR, we recommend using **OpenCLIP** to freeze a CLIP backbone and train a linear head on top of its features. (This is **external** to our `main.py`, which focuses on CNN PEFT.) See OpenAI CLIP and OpenCLIP for model notes and RN50×4 availability. ([arXiv][1])

Minimal sketch (pseudo-code):

```python
import open_clip, torch, torch.nn as nn
model, _, preprocess = open_clip.create_model_and_transforms(
    'RN50x4', pretrained='openai')  # or 'RN50'
model.eval().requires_grad_(False)

# build CIFAR loaders with `preprocess` transforms, extract features
# then train a small nn.Linear(features_dim, nb_classes) on top
```

---

## Tips & gotchas

* **Backbone freezing**: when using adapters (`--tuning_method conv|hcc|residual`), the script freezes all backbone params and BN affine terms; only adapter params train.
* **Input size**: CIFAR models use `32×32`; torchvision ImageNet models often expect `224×224` with ImageNet mean/std normalization.
* **EMA**: Optional via `--model_ema`; you can also eval EMA weights during training with `--model_ema_eval`.

---

## Repository structure

```
Conv-Adapter/
├── main.py                # Entry point (CLI, backbones, adapters, device/AMP guards)
├── engine.py              # Train/eval loops (CUDA-guarded syncs)
├── models/                # Backbones + adapters (HCC + Conv/Residual variants)
├── datasets/              # Dataset builders + transforms
├── utils.py               # Logging, EMA, schedulers, etc.
└── ...
```

---

## Why HCC?

* **Zero/linear-phase** behavior from even-shift aggregation → less phase distortion when adapting features.
* **Tiny parameter footprint**; optional channel-wise operation (`--hcc_per_channel`).
* **Drop-in**: choose `--tuning_method hcc` vs `conv`/`residual` with the same training loop.

---

## Citation

Please cite Conv-Adapter and this repository.

**Conv-Adapter (original idea):**
Chen et al., *Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets*, arXiv:2208.07463 (frequently updated; CVPR 2024 Workshop). ([arXiv][2])

**CLIP (background for RN50 / RN50×4):**
Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021.

**Torchvision weights (multi-weight API example for ResNet-50):**
PyTorch `ResNet50_Weights` docs.

---

## Acknowledgements & Credits

* Built on the ideas of **Conv-Adapter** (thanks to the original authors).
* Pretrained backbones provided by **torchvision**; CLIP baselines via **OpenCLIP** / **OpenAI CLIP**.

---

### Maintainers

* Tran Kim Huong
* Dang Ba Ty

If you want small usability tweaks (extra datasets, plots), open an issue or PR.

---

**Notes on sources:**
Conv-Adapter (paper/idea) and CLIP RN50/RN50×4 availability are documented in their respective papers; torchvision’s multi-weight API is documented in the official PyTorch docs. ([arXiv][2])

If you’d like me to tailor the README further (e.g., add **Flowers102/Pets** commands, or a ready-to-run **OpenCLIP linear-probe** script inside your repo), say the word and I’ll drop them in.

[1]: https://arxiv.org/html/2505.18842v3?utm_source=chatgpt.com "Learning to Point Visual Tokens for Multimodal Grounded ..."
[2]: https://arxiv.org/abs/2208.07463?utm_source=chatgpt.com "[2208.07463] Conv-Adapter: Exploring Parameter Efficient ..."
