import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class LoaderStage:
    split: str
    paired: bool = True
    augment: bool = False
    batch_size: int = 1
    shuffle: bool = False
    drop_last: bool = False


class Contrast2ContrastDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: Optional[str] = None,
        paired: bool = True,
        crop_size: int = 0,
        augment: bool = False,
        representation: str = "complex",
    ) -> None:
        self.root = Path(root)
        self.split = split if split not in ("", None) else None
        self.paired = paired
        self.crop_size = crop_size
        self.augment = augment
        self.representation = representation

        self.pd_dir = self._resolve_contrast_dir("pd")
        self.pdfs_dir = self._resolve_contrast_dir("pdfs")

        self.pd_slices, self._pd_files = self._collect_slices(self.pd_dir)
        self.pdfs_slices, self._pdfs_files = self._collect_slices(self.pdfs_dir)
        if not self.pd_slices or not self.pdfs_slices:
            raise RuntimeError("Dataset is empty.")

        if paired:
            keys = sorted(set(self.pd_slices) & set(self.pdfs_slices))
            if not keys:
                raise RuntimeError("No paired PD/PDFS slices were found.")
            self.index: List[Tuple[str, str]] = keys
        else:
            self.index = sorted(self.pd_slices)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id, slice_id = self.index[idx]
        pd_tensor = self._load(self._pd_files[case_id], slice_id)
        if self.paired:
            pdfs_tensor = self._load(self._pdfs_files[case_id], slice_id)
        else:
            alt_case_id, alt_slice_id = random.choice(self.pdfs_slices)
            pdfs_tensor = self._load(self._pdfs_files[alt_case_id], alt_slice_id)

        pd_tensor, pdfs_tensor = self._maybe_crop(pd_tensor, pdfs_tensor)
        pd_tensor, pdfs_tensor = self._maybe_augment(pd_tensor, pdfs_tensor)

        pd_tensor, pd_scale = self._normalize_to_median(pd_tensor)
        pdfs_tensor, pdfs_scale = self._normalize_to_median(pdfs_tensor)

        sample: Dict[str, torch.Tensor] = {
            "noisy_pd": self._to_representation(pd_tensor).clone(),
            "noisy_pdfs": self._to_representation(pdfs_tensor).clone(),
            "pd_scale": torch.tensor(pd_scale, dtype=torch.float32),
            "pdfs_scale": torch.tensor(pdfs_scale, dtype=torch.float32),
            "meta": {
                "case_id": case_id,
                "slice_id": slice_id,
                "split": self.split or "",
            },
        }
        gt_pd = self._load_optional(case_id, slice_id, contrast="pd")
        gt_pdfs = self._load_optional(case_id, slice_id, contrast="pdfs")
        if gt_pd is not None:
            gt_pd = self._apply_scale(gt_pd, pd_scale)
            sample["gt_pd"] = self._to_representation(gt_pd).clone()
        if gt_pdfs is not None:
            gt_pdfs = self._apply_scale(gt_pdfs, pdfs_scale)
            sample["gt_pdfs"] = self._to_representation(gt_pdfs).clone()
        return sample

    def _resolve_contrast_dir(self, contrast: str) -> Path:
        contrast_aliases = {
            "pd": ["pd", "PD", "Pd"],
            "pdfs": ["pdfs", "PDFS", "PDFs", "PdFs"],
        }
        candidates: List[Path] = []
        aliases = contrast_aliases.get(contrast.lower(), [contrast])
        if self.split:
            for alias in aliases:
                candidates.append(self.root / self.split / alias)
        for alias in aliases:
            candidates.append(self.root / alias)
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Could not locate directory for contrast '{contrast}'. Checked: {', '.join(str(p) for p in candidates)}"
        )

    @staticmethod
    def _collect_slices(folder: Path) -> Tuple[List[Tuple[str, str]], Dict[str, Path]]:
        items: List[Tuple[str, str]] = []
        files: Dict[str, Path] = {}
        for h5_path in sorted(folder.glob("*.h5")):
            case_id = h5_path.stem
            files[case_id] = h5_path
            with h5py.File(h5_path, "r") as handle:
                for key in handle.keys():
                    items.append((case_id, key))
        return items, files

    def _load_optional(self, case_id: str, slice_id: str, contrast: str):
        map_attr = {
            "pd": ("pd_clean", "pd_gt", "pd_target", "pd_ref", "clean_pd"),
            "pdfs": ("pdfs_clean", "pdfs_gt", "pdfs_target", "pdfs_ref", "clean_pdfs"),
        }
        aliases = map_attr.get(contrast, ())
        file_map = self._pd_files if contrast == "pd" else self._pdfs_files
        path = file_map.get(case_id)
        if path is None:
            return None
        with h5py.File(path, "r") as handle:
            for alias in aliases:
                if alias in handle:
                    group = handle[alias]
                    if isinstance(group, h5py.Group) and slice_id in group:
                        data = group[slice_id][()]
                        return self._convert_to_complex(data)
                    if isinstance(group, h5py.Dataset):
                        data = group[()]
                        if data.ndim >= 3 and data.shape[0] == len(handle.keys()):
                            idx = list(handle.keys()).index(slice_id)
                            return self._convert_to_complex(data[idx])
            return None

    @staticmethod
    def _convert_to_complex(data):
        array = np.asarray(data)
        if np.iscomplexobj(array):
            tensor = torch.from_numpy(array.astype(np.complex64, copy=False))
        else:
            tensor_np = array.astype(np.float32, copy=False)
            tensor = torch.from_numpy(tensor_np)
            if tensor.ndim >= 1 and tensor.shape[-1] == 2:
                real = tensor[..., 0]
                imag = tensor[..., 1]
                tensor = torch.complex(real, imag)
            else:
                tensor = tensor.to(torch.complex64)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] != 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(torch.complex64)

    @staticmethod
    def _load(path: Path, key: str) -> torch.Tensor:
        with h5py.File(path, "r") as handle:
            data = handle[key][()]
        array = np.asarray(data)
        if np.iscomplexobj(array):
            tensor = torch.from_numpy(array.astype(np.complex64, copy=False))
        else:
            tensor_np = array.astype(np.float32, copy=False)
            tensor = torch.from_numpy(tensor_np)
            if tensor.ndim >= 1 and tensor.shape[-1] == 2:
                real = tensor[..., 0]
                imag = tensor[..., 1]
                tensor = torch.complex(real, imag)
            else:
                tensor = tensor.to(torch.complex64)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] != 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] == 1:
            pass
        return tensor.to(torch.complex64)

    def _maybe_crop(
        self, pd_tensor: torch.Tensor, pdfs_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.crop_size <= 0:
            return pd_tensor, pdfs_tensor
        _, h_pd, w_pd = pd_tensor.shape
        _, h_pdfs, w_pdfs = pdfs_tensor.shape
        h_min = min(h_pd, h_pdfs)
        w_min = min(w_pd, w_pdfs)
        if h_min < self.crop_size or w_min < self.crop_size:
            return pd_tensor, pdfs_tensor
        top = random.randint(0, h_min - self.crop_size)
        left = random.randint(0, w_min - self.crop_size)
        crop = slice(top, top + self.crop_size), slice(left, left + self.crop_size)
        return pd_tensor[:, crop[0], crop[1]], pdfs_tensor[:, crop[0], crop[1]]

    def _maybe_augment(
        self, pd_tensor: torch.Tensor, pdfs_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return pd_tensor, pdfs_tensor
        if random.random() < 0.5:
            pd_tensor = torch.flip(pd_tensor, dims=(1,))
            pdfs_tensor = torch.flip(pdfs_tensor, dims=(1,))
        if random.random() < 0.5:
            pd_tensor = torch.flip(pd_tensor, dims=(2,))
            pdfs_tensor = torch.flip(pdfs_tensor, dims=(2,))
        phase = torch.exp(1j * torch.rand(1) * 2 * torch.pi)
        pd_tensor = pd_tensor * phase
        pdfs_tensor = pdfs_tensor * phase
        return pd_tensor, pdfs_tensor

    def _to_representation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.representation == "complex":
            return tensor
        if self.representation == "ri":
            stacked = torch.stack([tensor.real, tensor.imag], dim=1)
            return stacked.squeeze(0)
        if self.representation == "magnitude":
            return tensor.abs()
        raise ValueError(f"Unknown representation: {self.representation}")

    @staticmethod
    def _normalize_to_median(tensor: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, float]:
        magnitude = tensor.abs().float()
        median = magnitude.median()
        if not torch.isfinite(median):
            scale = 1.0
        else:
            scale = float(median.item())
        if scale < eps:
            mean_value = float(magnitude.mean().item())
            scale = mean_value if mean_value >= eps else 1.0
        if scale < eps:
            scale = 1.0
        return tensor / scale, scale

    @staticmethod
    def _apply_scale(tensor: torch.Tensor, scale: float) -> torch.Tensor:
        if scale == 1.0:
            return tensor
        return tensor / scale


def build_c2c_dataloaders(cfg: Dict) -> Dict[str, DataLoader]:
    root = Path(cfg["root"])
    representation = cfg.get("representation", "complex")
    crop_size = cfg.get("crop_size", 0)

    stages_cfg = dict(cfg.get("stages", {}))
    if not stages_cfg:
        stages_cfg = {
            "train": {
                "split": cfg.get("train_split", "train"),
                "paired": True,
                "augment": cfg.get("augment_train", True),
                "shuffle": True,
                "batch_size": cfg.get("batch_size", 1),
            },
            "val": {
                "split": cfg.get("val_split", "val"),
                "paired": True,
                "augment": False,
                "batch_size": cfg.get("batch_size", 1),
            },
            "test": {
                "split": cfg.get("test_split", "test"),
                "paired": True,
                "augment": False,
                "batch_size": cfg.get("batch_size", 1),
            },
        }
    loaders: Dict[str, DataLoader] = {}
    for name, stage_cfg in stages_cfg.items():
        stage = LoaderStage(
            split=stage_cfg.get("split", name),
            paired=stage_cfg.get("paired", True),
            augment=stage_cfg.get("augment", False),
            batch_size=stage_cfg.get("batch_size", cfg.get("batch_size", 1)),
            shuffle=stage_cfg.get("shuffle", False),
            drop_last=stage_cfg.get("drop_last", False),
        )
        dataset = Contrast2ContrastDataset(
            root=root,
            split=stage.split,
            paired=stage.paired,
            crop_size=crop_size,
            augment=stage.augment,
            representation=representation,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=stage.batch_size,
            shuffle=stage.shuffle,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=stage.drop_last,
        )
    return loaders
