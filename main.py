# main.py
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
import re  # NEW

from pathlib import Path

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma

from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
from memory_utils import profile_memory_cost
import utils


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.lower()
        if s in ('yes', 'true', 't', 'y', '1'):
            return True
        if s in ('no', 'false', 'f', 'n', '0'):
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('Parameter Efficient Tuning', add_help=False)

    # --- Backbone + weights (torchvision multi-weight API) ---
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help=("Torchvision name (e.g., resnet50, convnext_tiny) "
                              "or CIFAR TorchHub names like cifar10_resnet56 / cifar100_resnet56 "
                              "or akamaster aliases: resnet56_cifar|cifar_resnet56|akamaster_resnet56"))
    parser.add_argument('--weights', type=str, default='DEFAULT',
                        help="Torchvision weights: DEFAULT|none|IMAGENET1K_V1|ResNet50_Weights.IMAGENET1K_V2 ...")
    parser.add_argument('--list_backbones', action='store_true',
                        help='Print available torchvision model names and exit.')

    # NEW: explicit pretrained toggle (used for Torch Hub models)
    parser.add_argument('--pretrained', type=str2bool, default=None,
                        help="Force using pretrained weights when hub supports it. "
                             "If None, infer from --weights (DEFAULT => True).")
    # NEW: keep head if it already matches nb_classes
    parser.add_argument('--keep_pretrained_head', type=str2bool, default=True,
                        help='Keep the original classifier if out_features == nb_classes.')

    # NEW: which hub to use for CIFAR models
    parser.add_argument('--cifar_hub', type=str, default='auto',
                        choices=['auto', 'chenyaofo', 'akamaster'],
                        help='Provider for CIFAR backbones. auto: infer from --backbone')

    # -------------------- Tuning method --------------------
    parser.add_argument('--tuning_method', type=str, default='prompt',
                        help='prompt | conv | adapter | hcc | residual')

    # Prompt
    parser.add_argument('--prompt_size', default=10, type=int, help='prompt size')

    # Conv-Adapter (light conv bottleneck)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--adapt_size', default=8, type=float)
    parser.add_argument('--adapt_scale', default=1.0, type=float)

    # HCC hyperparams
    parser.add_argument('--hcc_h', type=int, default=1, help='base shift step in pixels')
    parser.add_argument('--hcc_M', type=int, default=1, help='number of shift pairs (+/- m*h)')
    parser.add_argument('--hcc_axis', type=str, default='hw', help='"h", "w", or "hw" (sum)')
    parser.add_argument('--hcc_padding', type=str, default='reflect', help='"reflect", "replicate", or "zeros"')
    parser.add_argument('--hcc_per_channel', type=str2bool, default=True)
    parser.add_argument('--hcc_use_center', type=str2bool, default=True)
    parser.add_argument('--hcc_pw_ratio', type=int, default=8, help='pointwise bottleneck ratio (C/r)')
    parser.add_argument('--hcc_use_pw', type=str2bool, default=True)
    parser.add_argument('--hcc_gate_init', type=float, default=0.1, help='initial residual gate')
    parser.add_argument('--hcc_tie_sym', type=str2bool, default=True)

    # Residual Adapters (Rebuffi et al.)
    parser.add_argument('--ra_mode', type=str, default='parallel', choices=['parallel', 'series'],
                        help='Residual adapter form: parallel or series')
    parser.add_argument('--ra_reduction', type=int, default=16, help='Bottleneck reduction ratio (C//r)')
    parser.add_argument('--ra_norm', type=str, default='bn', choices=['bn', 'ln', 'none'],
                        help='Normalization inside adapter')
    parser.add_argument('--ra_act', type=str, default='relu', choices=['relu', 'gelu', 'silu', 'none'],
                        help='Activation inside adapter')
    parser.add_argument('--ra_gate_init', type=float, default=0.0, help='Initial gate for adapter residual path')
    parser.add_argument('--ra_stages', type=str, default='1,2,3,4',
                        help='ResNet stages to adapt (comma list), e.g., "2,3,4"')

    parser.add_argument('--batch_size', default=64, type=int, help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int, help='gradient accumulation steps')

    parser.add_argument('--fs_shot', default=16, type=int)

    # Model parameters
    parser.add_argument('--model', default='resnet50_clip', type=str, metavar='MODEL',
                        help='Used only when not using PET shim')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int, help='image input size')
    parser.add_argument('--crop_ratio', default=0.875, type=float, help='image input size')

    # EMA
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False)
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Use EMA for eval during training.')

    # Optimization
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (head/backbone)')
    parser.add_argument('--weight_decay_hcc', type=float, default=0.0, help='weight decay for adapters/HCC params')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')
    parser.add_argument('--weight_decay_end', type=float, default=None,
                        help='Final WD for cosine schedule (default: same as --weight_decay)')

    # Augmentation
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME')
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--train_interpolation', type=str, default='bicubic')

    # Eval
    parser.add_argument('--crop_pct', type=float, default=None)

    # Random Erase
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT')
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', type=str2bool, default=False)

    # Mixup/Cutmix
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--cutmix', type=float, default=0.0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')

    # Finetuning params
    parser.add_argument('--finetune', default='')
    parser.add_argument('--head_init_scale', default=1.0, type=float)
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset
    parser.add_argument('--is_tuning', default=False, type=str2bool, help='Hyperparameter Tuning Mode')
    parser.add_argument('--dataset', default='stanford_cars', type=str)
    parser.add_argument('--data_path', default='/media/Auriga/Parameter_Efficient/data/few-shot/stanford_cars', type=str)
    parser.add_argument('--eval_data_path', default=None, type=str)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--output_dir', default='./experiments/')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='')
    parser.add_argument('--auto_resume', type=str2bool, default=False)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=2, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', type=str2bool, default=False)
    parser.add_argument('--dist_eval', type=str2bool, default=True)
    parser.add_argument('--disable_eval', type=str2bool, default=False)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True)

    # distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://')

    # AMP
    parser.add_argument('--use_amp', type=str2bool, default=True,
                        help="Use PyTorch AMP; set False only if your engine.py forbids it.")

    # W&B
    parser.add_argument('--enable_wandb', type=str2bool, default=False)
    parser.add_argument('--project', default='parameter_efficient_tuning_cv', type=str)
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False)

    return parser


def _resolve_weights_multiapi(backbone: str, weights_str: str):
    """
    Resolve torchvision weights across versions:
      - Modern API: get_model_weights / get_weight
      - Accept 'none' / 'scratch' / '' -> None
      - Accept 'DEFAULT' or enum members (IMAGENET1K_V1, etc.)
      - Accept fully-qualified names like 'ResNet50_Weights.IMAGENET1K_V2'
    """
    try:
        from torchvision.models import get_model_weights, get_weight  # noqa: F401
        has_new_api = True
    except Exception:
        has_new_api = False

    if not weights_str or str(weights_str).lower() in ('none', 'scratch', 'random'):
        return None, has_new_api

    if not has_new_api:
        # Old API fallback: treat anything not 'none' as pretrained=True
        return 'legacy_pretrained', False

    if '.' in weights_str:
        from torchvision.models import get_weight
        return get_weight(weights_str), True

    from torchvision.models import get_model_weights
    try:
        enum_cls = get_model_weights(backbone)
        member = weights_str.upper()
        if member == 'DEFAULT':
            return enum_cls.DEFAULT, True
        if hasattr(enum_cls, member):
            return getattr(enum_cls, member), True
    except Exception:
        pass
    return None, True


def main(args):
    # Early: handle --list_backbones
    if args.list_backbones:
        try:
            from torchvision.models import list_models
            names = list_models()
            for n in sorted(n for n in names if not n.startswith("video")):
                print(n)
        except Exception as e:
            print(f"[Warn] list_backbones failed: {e}")
            print("Hint: your torchvision may be too old for list_models().")
        return

    utils.init_distributed_mode(args)

    # Device selection with CPU fallback
    if str(args.device).lower().startswith('cuda'):
        if not torch.cuda.is_available():
            print("[Info] CUDA not available — falling back to CPU.")
            args.device = 'cpu'
            args.use_amp = False
    elif str(args.device).lower() == 'cpu':
        args.use_amp = False

    device = torch.device(args.device)
    print(args)
    print(f"[Info] Using device: {device}  (AMP={'on' if args.use_amp else 'off'})")

    # Seeds
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = (device.type == 'cuda')  # True for speed on GPU

    dataset_train, args.nb_classes = build_dataset(args=args, is_train=True)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(args=args, is_train=False)
    print(len(dataset_val) if dataset_val is not None else 0)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))

    if args.dist_eval and (dataset_val is not None):
        if len(dataset_val) % num_tasks != 0:
            print('Warning: eval set not divisible by process count; results may differ slightly.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=min(max(1, args.batch_size), 256),
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    # === BEGIN: General PET SHIM (torchvision + CIFAR TorchHub backbones) ===
    if args.tuning_method in ('conv', 'adapter', 'hcc', 'residual'):
        import torchvision

        # Heuristic: whether we want pretrained for TorchHub
        if args.pretrained is None:
            pretrained_flag = (args.weights is not None and str(args.weights).lower() not in ('none', 'scratch', 'random'))
        else:
            pretrained_flag = bool(args.pretrained)

        # Resolve weights (torchvision path)
        tv_weights, has_new_api = _resolve_weights_multiapi(args.backbone, args.weights)
        print(f"[Info] Backbone={args.backbone} | tv_weights={args.weights} -> {tv_weights} | hub.pretrained={pretrained_flag}")

        # ---- Build backbone ----
        model_backbone = None
        used_provider = None

        # Case A: chenyaofo CIFAR Hub (pretrained True/False)
        if re.match(r'^cifar(10|100)_.+$', args.backbone) and (args.cifar_hub in ('auto', 'chenyaofo')):
            used_provider = 'chenyaofo/pytorch-cifar-models'
            model_backbone = torch.hub.load(  # pretrained weights available
                used_provider, args.backbone, pretrained=pretrained_flag
            )
            args.input_size = 32
            print(f"[Info] Loaded {args.backbone} from {used_provider} (input_size=32).")

        # Case B: akamaster CIFAR-10 ResNet family (no official hub pretrain kwarg)
        elif args.backbone in ('cifar_resnet56', 'resnet56_cifar', 'akamaster_resnet56', 'resnet56_cifar10') \
             or (re.match(r'^akamaster_resnet(20|32|44|56|110)$', args.backbone) is not None) \
             or (args.cifar_hub == 'akamaster' and re.match(r'^cifar10_resnet(20|32|44|56|110)$', args.backbone)):
            used_provider = 'akamaster/pytorch_resnet_cifar10'
            # Map to entry name expected by akamaster (resnetXX)
            if args.backbone.startswith('akamaster_resnet'):
                entry = args.backbone.replace('akamaster_', '')
            else:
                entry = re.sub(r'^cifar(10|100)_', '', args.backbone)
                if not entry.startswith('resnet'):
                    entry = 'resnet56'
            model_backbone = torch.hub.load(used_provider, entry)  # no pretrained kw
            args.input_size = 32
            print(f"[Info] Loaded {entry} from {used_provider} (input_size=32).")

        # Case C: torchvision (ImageNet family)
        if model_backbone is None:
            try:
                used_provider = 'torchvision'
                if has_new_api:
                    from torchvision.models import get_model
                    model_backbone = get_model(args.backbone, weights=tv_weights)
                else:
                    fn = getattr(torchvision.models, args.backbone)
                    pretrained = (tv_weights == 'legacy_pretrained')
                    try:
                        model_backbone = fn(pretrained=pretrained, num_classes=1000)
                    except TypeError:
                        model_backbone = fn(pretrained=pretrained)
                print(f"[Info] Loaded {args.backbone} from torchvision.")
            except AttributeError as e:
                raise RuntimeError(f"Backbone '{args.backbone}' is not available in torchvision or supported hubs.") from e

        # Freeze backbone (including BN) for PET
        for p_ in model_backbone.parameters():
            p_.requires_grad = False
        for m in model_backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if m.affine:
                    if m.weight is not None:
                        m.weight.requires_grad = False
                    if m.bias is not None:
                        m.bias.requires_grad = False

        # --- Adapters ---
        if args.tuning_method in ('conv', 'adapter'):
            class ConvAdapter(nn.Module):
                def __init__(self, c, k=args.kernel_size):
                    super().__init__()
                    h = max(1, int(c // max(1, int(args.adapt_size))))
                    self.net = nn.Sequential(
                        nn.Conv2d(c, h, 1, bias=False), nn.ReLU(inplace=True),
                        nn.Conv2d(h, h, k, padding=k // 2, groups=h, bias=False), nn.ReLU(inplace=True),
                        nn.Conv2d(h, c, 1, bias=False)
                    )
                def forward(self, x):  # residual only
                    return self.net(x)
            def make_adapter(ch): return ConvAdapter(ch)

        elif args.tuning_method == 'hcc':
            from models.hcc_adapter import HCCAdapter
            def make_adapter(ch):
                m = HCCAdapter(
                    C=ch, M=args.hcc_M, h=args.hcc_h, axis=args.hcc_axis,
                    per_channel=args.hcc_per_channel, tie_sym=args.hcc_tie_sym,
                    use_pw=args.hcc_use_pw, pw_ratio=args.hcc_pw_ratio,
                    residual_scale=args.adapt_scale, gate_init=args.hcc_gate_init,
                    padding_mode={'reflect': 'reflect', 'replicate': 'replicate', 'zeros': 'zeros'}.get(args.hcc_padding, 'reflect')
                )
                m.is_hcc_adapter = True
                return m

        else:  # residual adapters
            ra_wrap_ok = False
            ra_adapter_param_ids = set()
            try:
                from models.tuning_modules.residual_adapter import (
                    attach_residual_adapters_resnet,
                    ParallelResidualAdapter, SeriesResidualAdapter
                )
                if args.backbone.startswith('resnet'):  # torchvision ResNet only
                    stages = [int(s.strip()) for s in args.ra_stages.split(',') if s.strip().isdigit()]
                    model_backbone = attach_residual_adapters_resnet(
                        model_backbone,
                        mode=args.ra_mode,
                        reduction=args.ra_reduction,
                        norm=args.ra_norm,
                        act=args.ra_act,
                        gate_init=args.ra_gate_init,
                        stages=stages
                    )
                    ra_wrap_ok = True
                    for mod in model_backbone.modules():
                        if isinstance(mod, (ParallelResidualAdapter, SeriesResidualAdapter)):
                            for p in mod.parameters():
                                if p.requires_grad:
                                    ra_adapter_param_ids.add(id(p))
            except Exception as e:
                print(f"[Warn] residual_adapter wrapper not available ({e}). Falling back to hook-based adapters.")

            if not ra_wrap_ok:
                def _norm2d(c):
                    if args.ra_norm == 'bn': return nn.BatchNorm2d(c)
                    if args.ra_norm == 'ln': return nn.GroupNorm(1, c)  # LN-ish
                    return nn.Identity()
                def _act():
                    return {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU, 'none': nn.Identity}[args.ra_act]()
                class ResidualCore(nn.Module):
                    def __init__(self, c):
                        super().__init__()
                        h = max(1, c // max(1, args.ra_reduction))
                        self.core = nn.Sequential(
                            nn.Conv2d(c, h, 1, bias=False), _norm2d(h), _act(),
                            nn.Conv2d(h, c, 1, bias=False), _norm2d(c)
                        )
                        self.gate = nn.Parameter(torch.tensor(float(args.ra_gate_init)))
                    def forward(self, x):
                        return self.core(x) * self.gate
                def make_adapter(ch): return ResidualCore(ch)

        # --- Attach adapters by family (robust to non-torchvision ResNets) ---
        # Prefer dedicated classes when present
        try:
            from torchvision.models.resnet import BasicBlock, Bottleneck
        except Exception:
            BasicBlock = Bottleneck = tuple()

        try:
            from torchvision.models.convnext import CNBlock
        except Exception:
            CNBlock = tuple()
        try:
            from torchvision.models.efficientnet import MBConv, FusedMBConv
        except Exception:
            MBConv = FusedMBConv = tuple()
        try:
            from torchvision.models.mobilenetv3 import InvertedResidual
        except Exception:
            InvertedResidual = tuple()

        def _attach(module: nn.Module, out_ch: int):
            # If residual adapters were wrapped (torchvision ResNet), skip hooking.
            if args.tuning_method == 'residual' and isinstance(module, tuple(BasicBlock if isinstance(BasicBlock, tuple) else (BasicBlock,))) and args.backbone.startswith('resnet'):
                return
            module.add_module('pet_adapter', make_adapter(out_ch))

            def hook(mod, _in, out):
                if getattr(mod.pet_adapter, 'is_hcc_adapter', False):
                    return mod.pet_adapter(out)
                if args.tuning_method in ('conv', 'adapter'):
                    return out + args.adapt_scale * mod.pet_adapter(out)
                if args.tuning_method == 'residual':
                    xin = _in[0] if isinstance(_in, (tuple, list)) and len(_in) > 0 else out
                    return out + (mod.pet_adapter(xin) if args.ra_mode == 'parallel' else mod.pet_adapter(out))
                return out
            module.register_forward_hook(hook)

        for m in model_backbone.modules():
            attached = False
            # Torchvision ResNet classes
            if BasicBlock and isinstance(m, BasicBlock):
                _attach(m, m.conv2.out_channels); attached = True
            elif Bottleneck and isinstance(m, Bottleneck):
                _attach(m, m.conv3.out_channels); attached = True
            # Generic ResNet-like (covers chenyaofo & preact blocks)
            elif hasattr(m, 'conv3') and isinstance(getattr(m, 'conv3'), nn.Conv2d):
                _attach(m, m.conv3.out_channels); attached = True
            elif hasattr(m, 'conv2') and isinstance(getattr(m, 'conv2'), nn.Conv2d):
                _attach(m, m.conv2.out_channels); attached = True
            # ConvNeXt / EfficientNet / MobileNetV3
            elif CNBlock and isinstance(m, CNBlock):
                if hasattr(m, 'dwconv') and isinstance(m.dwconv, nn.Conv2d):
                    _attach(m, m.dwconv.out_channels); attached = True
                elif hasattr(m, 'block') and len(getattr(m, 'block')) > 0 and isinstance(m.block[0], nn.Conv2d):
                    _attach(m, m.block[0].out_channels); attached = True
            elif MBConv and isinstance(m, (MBConv, FusedMBConv)):
                out_ch = getattr(m, 'out_channels', None)
                if out_ch is None:
                    last_conv = None
                    for c in m.modules():
                        if isinstance(c, nn.Conv2d): last_conv = c
                    if last_conv is not None:
                        out_ch = last_conv.out_channels
                if out_ch is not None:
                    _attach(m, out_ch); attached = True
            elif InvertedResidual and isinstance(m, InvertedResidual):
                out_ch = getattr(m, 'out_channels', None)
                if out_ch is None:
                    last_conv = None
                    for c in m.modules():
                        if isinstance(c, nn.Conv2d): last_conv = c
                    if last_conv is not None:
                        out_ch = last_conv.out_channels
                if out_ch is not None:
                    _attach(m, out_ch); attached = True

        # --- Replace classifier head to match nb_classes (only if needed) ---
        def _maybe_replace_linear(parent, name, lin: nn.Linear, num_classes: int):
            if not isinstance(lin, nn.Linear):
                return False
            if args.keep_pretrained_head and lin.out_features == num_classes:
                # keep pretrained head as-is
                return False
            new_lin = nn.Linear(lin.in_features, num_classes)
            setattr(parent, name, new_lin)
            return True

        replaced = False
        # Common torchvision patterns
        if hasattr(model_backbone, 'fc') and isinstance(model_backbone.fc, nn.Linear):
            replaced = _maybe_replace_linear(model_backbone, 'fc', model_backbone.fc, args.nb_classes)
        elif hasattr(model_backbone, 'classifier'):
            head = model_backbone.classifier
            if isinstance(head, nn.Linear):
                replaced = _maybe_replace_linear(model_backbone, 'classifier', head, args.nb_classes)
            elif isinstance(head, nn.Sequential):
                new_seq = list(head)
                for i in reversed(range(len(new_seq))):
                    if isinstance(new_seq[i], nn.Linear):
                        if not (args.keep_pretrained_head and new_seq[i].out_features == args.nb_classes):
                            in_f = new_seq[i].in_features
                            new_seq[i] = nn.Linear(in_f, args.nb_classes)
                            model_backbone.classifier = nn.Sequential(*new_seq)
                            replaced = True
                        break
        # CIFAR-style heads
        if not replaced and hasattr(model_backbone, 'linear') and isinstance(model_backbone.linear, nn.Linear):
            replaced = _maybe_replace_linear(model_backbone, 'linear', model_backbone.linear, args.nb_classes)
        if not replaced and hasattr(model_backbone, 'head') and isinstance(model_backbone.head, nn.Linear):
            replaced = _maybe_replace_linear(model_backbone, 'head', model_backbone.head, args.nb_classes)

        model = model_backbone
    else:
        # Fall back to repo's builder
        model = build_model(
            args.model,
            pretrained=True,
            num_classes=args.nb_classes,
            tuning_method=args.tuning_method,
            args=args,
        )
    # === END: General PET SHIM ===

    # Move to device BEFORE profiling
    model.to(device)

    # Profile (param/act memory) for visibility
    memory_cost, detailed_info = profile_memory_cost(
        model, (1, 3, args.input_size, args.input_size), True,
        activation_bits=32, trainable_param_bits=32,
        frozen_param_bits=8, batch_size=8,
    )
    net_info = {
        'memory_cost_MB': memory_cost / 1e6,
        'param_size_MB': detailed_info['param_size'] / 1e6,
        'act_size_MB': detailed_info['act_size'] / 1e6,
    }
    for key, item in net_info.items():
        print(f"{key}: {item:.3f}")

    # Optional finetune ckpt load
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = max(1, len(dataset_train) // total_batch_size)
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    # -------- Optimizer with param-groups (0 WD for adapters/HCC) --------
    adapter_param_ids = set()
    try:
        from models.tuning_modules.residual_adapter import ParallelResidualAdapter, SeriesResidualAdapter
        for mod in model_without_ddp.modules():
            if isinstance(mod, (ParallelResidualAdapter, SeriesResidualAdapter)):
                for p in mod.parameters():
                    if p.requires_grad:
                        adapter_param_ids.add(id(p))
    except Exception:
        pass

    hcc_params, other_params = [], []
    for n, p in model_without_ddp.named_parameters():
        if not p.requires_grad:
            continue
        if 'pet_adapter' in n or 'hcc' in n or id(p) in adapter_param_ids:
            hcc_params.append(p)
        else:
            other_params.append(p)

    print(f"[ParamGroups] adapter_like={sum(p.numel() for p in hcc_params):,}  "
          f"others={sum(p.numel() for p in other_params):,}")

    optimizer = torch.optim.AdamW(
        [
            {'params': hcc_params, 'lr': args.lr, 'weight_decay': args.weight_decay_hcc},
            {'params': other_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        ],
        betas=(0.9, 0.999), eps=args.opt_eps
    )
    # ---------------------------------------------------------------------

    loss_scaler = NativeScaler()  # engine.py handles amp context

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print("Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if getattr(args, "distributed", False):
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                if 'acc5' in test_stats:
                    log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})

        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.list_backbones:
        main(args)
        raise SystemExit(0)

    if not args.is_tuning:
        args = utils.auto_load_optim_param(args, args.model, args.tuning_method, args.dataset)
    else:
        args.save_ckpt = False
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Name the dataset by the leaf folder of data_path
    args.data = Path(args.data_path).name
    main(args)
