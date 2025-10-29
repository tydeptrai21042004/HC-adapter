import os
from collections import Counter
from typing import Tuple

import torch
import torchvision.transforms as T
from torchvision import datasets
from torchvision.datasets import CocoDetection


# ------------------------------
# Minimal transforms: Resize -> ToTensor
# - No normalization, flips, crops, jitters, etc.
# - If input is grayscale (1ch), duplicate to 3ch after ToTensor for 3-ch models.
# ------------------------------
def _img_transforms(args, is_train: bool):
    size = int(getattr(args, "input_size", 224))
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),
    ])


# ------------------------------
# CIFAR family
# ------------------------------
def _build_cifar10(args, is_train: bool):
    tfm = _img_transforms(args, is_train)
    ds = datasets.CIFAR10(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 10

def _build_cifar100(args, is_train: bool):
    tfm = _img_transforms(args, is_train)
    ds = datasets.CIFAR100(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 100


# ------------------------------
# MNIST family
# ------------------------------
def _build_mnist(args, is_train: bool):
    tfm = _img_transforms(args, is_train)
    ds = datasets.MNIST(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 10

def _build_fashion_mnist(args, is_train: bool):
    tfm = _img_transforms(args, is_train)
    ds = datasets.FashionMNIST(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 10

def _build_emnist(args, is_train: bool):
    # 'byclass' has 62 classes (0-9, A-Z, a-z)
    tfm = _img_transforms(args, is_train)
    ds = datasets.EMNIST(root=args.data_path, split="byclass", train=is_train, download=True, transform=tmf if (tmf := tfm) else tfm)
    return ds, 62

def _build_kmnist(args, is_train: bool):
    tfm = _img_transforms(args, is_train)
    ds = datasets.KMNIST(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 10

def _build_qmnist(args, is_train: bool):
    # QMNIST uses 'what' instead of 'split'
    tfm = _img_transforms(args, is_train)
    what = "train" if is_train else "test"
    ds = datasets.QMNIST(root=args.data_path, what=what, download=True, transform=tfm)
    return ds, 10

def _build_usps(args, is_train: bool):
    tfm = _img_transforms(args, is_train)
    ds = datasets.USPS(root=args.data_path, train=is_train, download=True, transform=tfm)
    return ds, 10


# ------------------------------
# Small/medium classification benchmarks
# ------------------------------
def _build_svhn(args, is_train: bool):
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.SVHN(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 10

def _build_stl10(args, is_train: bool):
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.STL10(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 10

def _build_food101(args, is_train: bool):
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.Food101(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 101

def _build_pets(args, is_train: bool):
    split = 'trainval' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.OxfordIIITPet(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 37

def _build_flowers102(args, is_train: bool):
    # torchvision split names: 'train' / 'val' / 'test'
    split = 'train' if is_train else 'val'
    tfm = _img_transforms(args, is_train)
    ds = datasets.Flowers102(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 102

def _build_cars(args, is_train: bool):
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.StanfordCars(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 196

def _split_subset(base, ratio=0.8, seed=42, is_train=True):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(base), generator=g)
    ntr = int(ratio * len(base))
    return torch.utils.data.Subset(base, idx[:ntr] if is_train else idx[ntr:])

def _build_caltech101(args, is_train: bool):
    tfm = _img_transforms(args, is_train)
    base = datasets.Caltech101(root=args.data_path, download=True, transform=tfm)
    return _split_subset(base, is_train=is_train), 101

def _build_eurosat(args, is_train: bool):
    tfm = _img_transforms(args, is_train)
    base = datasets.EuroSAT(root=args.data_path, download=True, transform=tfm)
    return _split_subset(base, is_train=is_train), 10

def _build_dtd(args, is_train: bool):
    # DTD has Splits: 'train'|'val'|'test'. We use 'train' for train, 'val' for eval.
    split = 'train' if is_train else 'val'
    tfm = _img_transforms(args, is_train)
    ds = datasets.DTD(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 47



def _build_fgvc_aircraft(args, is_train: bool):
    # Splits: 'train', 'val', 'trainval', 'test' — use 'trainval' for train, 'test' for eval
    split = 'trainval' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.FGVCAircraft(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 100

def _build_sun397(args, is_train: bool):
    # SUN397: 397 scene classes; torchvision provides its own train/test split files
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.SUN397(root=args.data_path, download=True, transform=tfm, split=split)
    return ds, 397

def _build_gtsrb(args, is_train: bool):
    # German Traffic Sign Recognition Benchmark
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.GTSRB(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 43

def _build_fer2013(args, is_train: bool):
    # Facial Expression Recognition 2013: 7 classes
    split = 'train' if is_train else 'test'
    tfm = _img_transforms(args, is_train)
    ds = datasets.FER2013(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 7

def _build_pcam(args, is_train: bool):
    # PatchCamelyon: 'train' | 'val' | 'test'
    split = 'train' if is_train else 'val'
    tfm = _img_transforms(args, is_train)
    ds = datasets.PCAM(root=args.data_path, split=split, download=True, transform=tfm)
    return ds, 2


# ------------------------------
# COCO (single-label classification wrapper)
# ------------------------------
class _CocoSingleLabel(torch.utils.data.Dataset):
    """Wrap CocoDetection and assign ONE label per image = majority category."""
    def __init__(self, img_root: str, ann_file: str, transform):
        self.base = CocoDetection(img_root, ann_file, transform=None, target_transform=None)
        self.transform = transform

        cat_ids = sorted(self.base.coco.getCatIds())
        self.catid2idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.idx2catid = cat_ids
        self.num_classes = len(cat_ids)  # typically 80

        keep, labels = [], []
        for i, img_id in enumerate(self.base.ids):
            ann_ids = self.base.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            if not ann_ids:
                continue
            anns = self.base.coco.loadAnns(ann_ids)
            counts = Counter(self.catid2idx[a["category_id"]] for a in anns if "category_id" in a)
            if not counts:
                continue
            keep.append(i)
            labels.append(counts.most_common(1)[0][0])

        self.keep = keep
        self.labels = labels

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, idx: int):
        base_idx = self.keep[idx]
        img, _ = self.base[base_idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def _build_coco(args, is_train: bool):
    split_dir = "train2017" if is_train else "val2017"
    img_root = os.path.join(args.data_path, split_dir)
    ann_file = os.path.join(args.data_path, "annotations", f"instances_{split_dir}.json")

    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"[COCO] Missing images folder: {img_root}")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"[COCO] Missing annotations file: {ann_file}")

    tfm = _img_transforms(args, is_train)
    ds = _CocoSingleLabel(img_root, ann_file, tfm)
    return ds, ds.num_classes


# ------------------------------
# Router
# ------------------------------
def build_dataset(args, is_train: bool) -> Tuple[torch.utils.data.Dataset, int]:
    name = (args.dataset or "").lower().replace("-", "_")

    # CIFAR
    if name in ("cifar10",):
        return _build_cifar10(args, is_train)
    if name in ("cifar100", "cifar_100"):
        return _build_cifar100(args, is_train)

    # MNIST family
    if name in ("mnist",):
        return _build_mnist(args, is_train)
    if name in ("fashion_mnist", "fashionmnist"):
        return _build_fashion_mnist(args, is_train)
    if name in ("emnist", "emnist_byclass"):
        return _build_emnist(args, is_train)
    if name in ("kmnist",):
        return _build_kmnist(args, is_train)
    if name in ("qmnist",):
        return _build_qmnist(args, is_train)
    if name in ("usps",):
        return _build_usps(args, is_train)

    # Small/medium classification
    if name in ("svhn",):
        return _build_svhn(args, is_train)
    if name in ("stl10",):
        return _build_stl10(args, is_train)
    if name in ("food101", "food_101"):
        return _build_food101(args, is_train)
    if name in ("oxfordiiitpet", "oxford_iiit_pet", "pets", "oxford_pets"):
        return _build_pets(args, is_train)
    if name in ("flowers102", "oxfordflowers102", "oxford_flowers102"):
        return _build_flowers102(args, is_train)
    if name in ("stanford_cars", "stanfordcars", "cars"):
        return _build_cars(args, is_train)

    # Extra torchvision datasets (auto-download)
    if name in ("caltech101", "caltech_101"):
        return _build_caltech101(args, is_train)
    if name in ("dtd", "describable_textures", "textures"):
        return _build_dtd(args, is_train)
    if name in ("eurosat", "euro_sat"):
        return _build_eurosat(args, is_train)
    if name in ("fgvc_aircraft", "fgvca", "aircraft"):
        return _build_fgvc_aircraft(args, is_train)
    if name in ("sun397", "sun_397"):
        return _build_sun397(args, is_train)
    if name in ("gtsrb", "traffic_signs", "german_traffic_signs"):
        return _build_gtsrb(args, is_train)
    if name in ("fer2013", "fer_2013"):
        return _build_fer2013(args, is_train)
    if name in ("pcam", "patch_camelyon"):
        return _build_pcam(args, is_train)

    # COCO single-label wrapper
    if name in ("coco", "coco2017", "mscoco", "mscoco2017"):
        return _build_coco(args, is_train)

    raise NotImplementedError(f"Unsupported dataset '{args.dataset}'.")
