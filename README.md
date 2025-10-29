Awesome—here’s a clean, drop-in README for your repo that (1) clearly credits Conv-Adapter and its authors and (2) spells out what you changed.

---

# HCC-Adapter: Hartley–Cosine Adapters for ConvNets

Parameter-efficient finetuning (PEFT) for convolutional backbones with a lightweight **Hartley–Cosine Adapter (HCC)**. This repo started from (and continues to credit) **Conv-Adapter** and extends it with a new adapter module, runtime fixes, and a simpler training pipeline. ([GitHub][1])

> If you use this code, **please cite both Conv-Adapter and this repository.**

---

## What is HCC-Adapter?

**HCC (Hartley–Cosine Adapter)** is a small, learnable module that aggregates *even* spatial shifts of feature maps (e.g., via symmetric `roll`s) and then reprojects them. The even symmetry yields a cosine-only frequency response (linear/zero-phase behavior), making it a natural, low-distortion adapter for ConvNet features. The module slots in similarly to Conv-Adapter blocks and is toggled via `--tuning_method hcc`.

---

## How this fork differs from Conv-Adapter

This project began as a fork/re-implementation on top of **Hhhhhhao/Conv-Adapter** (CVPR 2024 Workshop). Below are the key differences from the original codebase. ([GitHub][1])

### New

* **HCC Adapter module** and build path (`--tuning_method hcc`) with HCC-specific flags:

  * `--hcc_h`, `--hcc_M`, `--hcc_axis`, `--hcc_per_channel`, `--hcc_use_center`
  * `--hcc_pw_ratio`, `--hcc_gate_init`, `--hcc_padding` (supports `circular`, `reflect`, `replicate`, `zeros`)
  * `--adapt_scale` to globally scale adapter outputs

### Training/runtime quality-of-life

* **CPU fallback** and **AMP auto-disable on CPU** (no crashes on machines without CUDA).
* **CUDA-guarded** `torch.cuda.synchronize()` (prevents “Found no NVIDIA driver” errors).
* **BatchNorm truly frozen** (sets BN to `eval()` and disables affine grads).
* **Backbone weights modernized** to `torchvision`’s `ResNet50_Weights.IMAGENET1K_V2` (or `DEFAULT`) for sane baselines. ([PyTorch Documentation][2])
* **Transforms simplified** with an option to keep **ImageNet mean/std** normalization on by default.
* **Val protocol option** (Resize→CenterCrop) for parity with torchvision evaluation.

### Reproducibility & small fixes

* Clear separation of trainable params (only adapters update by default).
* Safer logging/metrics (no KeyErrors when `acc5` absent, robust steps-per-epoch).
* Consistent experiment naming from data path leaf.

> Note: the original Conv-Adapter authors note their code is “very old” for training; we focused on making a **stable, reproducible** PEFT playground while keeping the adapter idea accessible. ([GitHub][1])

---

## Quick start

### 1) Setup

```bash
git clone https://github.com/tydeptrai21042004/Hcc-adapter.git
cd Hcc-adapter/Conv-Adapter

```

### 2) Data

Place datasets under `./data/<name>` (e.g., `./data/flowers`, `./data/pets`).
You can also use `torchvision.datasets` for Oxford-IIIT Pet / Flowers102. ([PyTorch Documentation][3])

### 3) Train (HCC vs Conv)

**HCC (recommended flags as a starting point):**

```bash
python main.py \
  --dataset oxford_iiit_pet --data_path ./data/pets \
  --model resnet50 --tuning_method hcc \
  --hcc_h 1 --hcc_M 2 --hcc_axis h \
  --hcc_per_channel True --hcc_use_center True \
  --hcc_pw_ratio 1.0 --hcc_gate_init 0.0 --hcc_padding circular \
  --adapt_scale 1.0 \
  --batch_size 32 --epochs 10 --lr 1e-4 \
  --use_amp True --dist_eval False
```

**Conv-Adapter (baseline):**

```bash
python main.py \
  --dataset oxford_iiit_pet --data_path ./data/pets \
  --model resnet50 --tuning_method conv \
  --batch_size 32 --epochs 10 --lr 1e-4 \
  --use_amp True --dist_eval False
```

> Tip: If you’re on CPU, add `--device cpu` (AMP will auto-disable). If you’re on GPU without drivers, the code will avoid CUDA syncs.

---

## Results (example guidance)

* Enable **ImageNet mean/std normalization** when using ImageNet-pretrained backbones (e.g., `ResNet50_Weights.IMAGENET1K_V2`) to prevent accuracy drop. ([PyTorch Documentation][2])
* For fair comparisons with Conv-Adapter, keep:

  * Same backbone and frozen layers
  * Same optimizer, LR, and epochs
  * Same input size and **val protocol** (Resize→CenterCrop)

---

## Repository structure

```
Hcc-adapter/
└── Conv-Adapter/
    ├── main.py                # Entry point (CLI), device/AMP guards, flags
    ├── engine.py              # train/eval loops with CUDA-guarded synchronize
    ├── models/                # Backbones + adapters (HCC + Conv-Adapter)
    ├── datasets/              # Dataset builders, transforms, (optional) wrappers
    ├── utils.py               # Misc utilities, logging, EMA (if enabled)
    └── ...
```

---

## Why HCC?

* **Linear/zero-phase behavior** from even-shift aggregation → reduces phase distortion in feature adaptation.
* **Tiny parameter footprint** and channel-wise flexibility (`--hcc_per_channel`).
* **Drop-in** replacement: choose `--tuning_method hcc` vs `conv` without changing your data/optimizer code.

---

## Citation

Please cite both Conv-Adapter and this repository.

**Conv-Adapter (original):** ([arXiv][4])

```bibtex
@article{chen2022conv,
  title   = {Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets},
  author  = {Chen, Hao and Tao, Ran and Zhang, Han and Wang, Yidong and Li, Xiang
             and Ye, Wei and Wang, Jindong and Hu, Guosheng and Savvides, Marios},
  booktitle = {CVPR 2024 Workshop on Prompting in Vision},
  year    = {2024}
}
```


## Acknowledgements & Credits

* This work builds on **Conv-Adapter**. Huge thanks to the original authors and their codebase. Repo: Hhhhhhao/**Conv-Adapter**. Paper: CVPR 2024 Workshop. ([GitHub][1])
* Pretrained backbones and weights come from **torchvision** (e.g., `ResNet50_Weights.IMAGENET1K_V2`). ([PyTorch Documentation][2])

---

## License

This repository’s license applies to **our additions and modifications**. If you reuse parts of Conv-Adapter code, please also follow the original repository’s terms and cite the paper appropriately. ([GitHub][1])

---

### Maintainer

* Tran Kim Huong

If you spot issues or want small usability tweaks (e.g., extra datasets or plots), open an issue or PR.

[1]: https://github.com/Hhhhhhao/Conv-Adapter "GitHub - Hhhhhhao/Conv-Adapter"
[2]: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html?utm_source=chatgpt.com "resnet50 — Torchvision main documentation"
[3]: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html?utm_source=chatgpt.com "OxfordIIITPet — Torchvision main documentation"
[4]: https://arxiv.org/pdf/2208.07463?utm_source=chatgpt.com "arXiv:2208.07463v4 [cs.CV] 12 Apr 2024"
