
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



## Installation

```bash
# core
pip install torch torchvision timm


```



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
for the best practice "# models/hcc_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import gcd

class HCCAdapter(nn.Module):
    def __init__(
        self, C, M=1, h=1, axis='hw',
        alpha_group=16,             # NEW API (group-shared alphas)
        tie_sym=True,
        no_pw=False,                # NEW API (inverse of legacy use_pw)
        pw_ratio=32,
        pw_groups=4,
        use_bn=False,
        residual_scale=1.0,
        gate_init=0.1,
        padding_mode='reflect',
        **legacy,                   # <-- accept unknown legacy kwargs
    ):
        super().__init__()
        assert axis in ('h','w','hw')
        self.C, self.M, self.h = int(C), int(M), int(h)
        self.axis = axis
        self.tie_sym = tie_sym
        self.padding_mode = padding_mode
        self.residual_scale = residual_scale

        # --- translate legacy args ---
        # per_channel=True  -> alpha_group = 1
        if 'per_channel' in legacy:
            per_channel = bool(legacy.pop('per_channel'))
            alpha_group = 1 if per_channel else alpha_group
        # use_pw=True -> no_pw=False
        if 'use_pw' in legacy:
            use_pw_legacy = bool(legacy.pop('use_pw'))
            no_pw = (not use_pw_legacy)
        # ignore any other legacy keys silently (or log if you prefer)

        self.alpha_group = max(1, int(alpha_group))
        self.no_pw = bool(no_pw)
        self.use_bn = bool(use_bn)


        # ---------- α coefficients (group-shared) ----------
        self.alpha_group = max(1, int(alpha_group))
        G = max(1, self.C // self.alpha_group)   # number of channel groups
        ncoef = self.M + 1                       # center + M side taps
        # α stored per group → shape (G, ncoef)
        self.alpha = nn.Parameter(torch.zeros(G, ncoef))
        with torch.no_grad():
            self.alpha[:, 0].fill_(1.0)          # identity-safe init

        # ---------- optional channel mixing via PW (grouped) ----------
        self.use_bn = bool(use_bn)
        if not self.no_pw:
            H = max(1, self.C // max(1, int(pw_ratio)))
            # make groups legal for both 1x1 convs
            g = max(1, int(pw_groups))
            g = min(g, self.C, H)
            # ensure groups divide both C and H
            g = gcd(g, self.C)
            g = gcd(g, H) or 1
            self.pw_groups = g
            layers = [
                nn.Conv2d(self.C, H, 1, groups=g, bias=False),
                nn.BatchNorm2d(H) if self.use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(H, self.C, 1, groups=g, bias=False),
                nn.BatchNorm2d(self.C) if self.use_bn else nn.Identity(),
            ]
            self.pw = nn.Sequential(*layers)
        else:
            self.pw = nn.Identity()

        # ---------- global residual gate ----------
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    # build per-group 1D even kernel → expand to (C,1,K)
    def _build_even_kernel_1d(self, device, dtype):
        K = 2*self.M + 1
        G = max(1, self.C // self.alpha_group)
        # (G, K) then expand → (C, K)
        wg = torch.zeros(G, K, device=device, dtype=dtype)
        center = self.M
        wg[:, center] = self.alpha[:, 0]
        for m in range(1, self.M+1):
            val = self.alpha[:, m]
            wg[:, center - m] = val
            wg[:, center + m] = val if self.tie_sym else val  # kept symmetric here
        # normalize within each group (optional but stable)
        s = wg.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        wg = wg / s
        # expand per group to channels
        reps = [self.alpha_group] * G
        reps[-1] = self.C - self.alpha_group*(G-1)  # handle remainder
        w = torch.cat([wg[i].unsqueeze(0).repeat(reps[i], 1) for i in range(G)], dim=0)  # (C, K)
        return w.unsqueeze(1)  # (C,1,K)

    def _pad(self, x, pad_h, pad_w):
        mode = 'reflect' if self.padding_mode == 'reflect' else \
               'replicate' if self.padding_mode == 'replicate' else 'constant'
        val = 0.0 if mode == 'constant' else None
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=mode, value=0.0 if val is not None else None)

    def forward(self, x):
        B, C, H, W = x.shape
        w1d = self._build_even_kernel_1d(x.device, x.dtype)  # (C,1,K)
        y = 0
        if 'h' in self.axis:
            wh = w1d.view(self.C, 1, 2*self.M+1, 1)
            xh = self._pad(x, pad_h=self.M*self.h, pad_w=0)
            yh = F.conv2d(xh, wh, stride=1, padding=0, dilation=(self.h,1), groups=self.C)
            y = y + yh
        if 'w' in self.axis:
            ww = w1d.view(self.C, 1, 1, 2*self.M+1)
            xw = self._pad(x, pad_h=0, pad_w=self.M*self.h)
            yw = F.conv2d(xw, ww, stride=1, padding=0, dilation=(1,self.h), groups=self.C)
            y = y + yw
        y = self.pw(y)
        return x + self.residual_scale * self.gate * y

# convenience alias for your previous import path
" 
![Alt text](accuracy_vs_neglog10_params.jpg "Optional Title")
