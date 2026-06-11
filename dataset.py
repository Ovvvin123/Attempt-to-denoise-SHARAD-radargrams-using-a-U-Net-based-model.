# dataset.py
# 用途：
#   为 SHARAD 雷达反射层 U-Net 降噪训练提供 PyTorch Dataset / DataLoader。
#
# 当前策略：
#   1. 读取 radar_ai_dataset/*.npz
#   2. 使用 split.json 中的 train / val / test 文件列表
#   3. 每次随机裁剪 reflection / noise patch
#   4. 动态构造 noisy = reflection + alpha * noise_scaling_factor * noise
#   5. 不使用 npz 内部保存的 mask
#      因为当前 mask.shape 可能和 reflection.shape 不对齐
#   6. 默认认为 reflection 小矩阵内部全是有效区域
#   7. padding 区域、NaN/Inf 区域 mask = 0
#   8. 使用 log1p + summary.json 中的 p1/p99 做归一化
#
# 推荐输入输出：
#   noisy:      [1, 176, 256]
#   reflection: [1, 176, 256]
#   mask:       [1, 176, 256]
#
# 单独测试：
#   F:\FInstallation\anaconda\python.exe dataset.py --split train --noise_scaling_factor 5.0

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def infer_track_id_from_name(filename: str) -> str:
    """
    从文件名中提取 track_id。

    例如：
        S_00174302_dataset.npz -> S_00174302
        00174302_dataset.npz   -> S_00174302
    """
    stem = Path(filename).stem

    m = re.search(r"(S_\d{8})", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"(\d{8})", stem)
    if m:
        return "S_" + m.group(1)

    return stem


def load_split_files(split_json: str | Path, split: str) -> List[str]:
    """
    从 outputs/splits/split.json 读取指定 split 的文件列表。
    """
    split_json = Path(split_json)

    if not split_json.exists():
        raise FileNotFoundError(f"找不到 split_json：{split_json.resolve()}")

    with split_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if "files" not in obj:
        raise KeyError(f"{split_json} 中没有 'files' 字段")

    if split not in obj["files"]:
        raise KeyError(f"{split_json} 中没有 files['{split}'] 字段")

    files = obj["files"][split]

    if not isinstance(files, list):
        raise TypeError(f"files['{split}'] 应该是 list")

    return files


def load_log1p_norm_params(summary_json: str | Path) -> tuple[float, float]:
    """
    从 outputs/stats/summary.json 中读取推荐的 log1p 归一化参数。

    使用：
        x_log = log1p(x)
        x_norm = clip((x_log - p1) / (p99 - p1), 0, 1)
    """
    summary_json = Path(summary_json)

    if not summary_json.exists():
        raise FileNotFoundError(f"找不到 summary_json：{summary_json.resolve()}")

    with summary_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    norm = obj.get("recommended_log1p_norm", None)

    if norm is None:
        raise KeyError(f"{summary_json} 中没有 recommended_log1p_norm")

    if "p1" not in norm or "p99" not in norm:
        raise KeyError(f"{summary_json} 的 recommended_log1p_norm 中缺少 p1 或 p99")

    p1 = float(norm["p1"])
    p99 = float(norm["p99"])

    if not np.isfinite(p1) or not np.isfinite(p99):
        raise ValueError(f"归一化参数不是有限数值：p1={p1}, p99={p99}")

    if p99 <= p1:
        raise ValueError(f"归一化参数异常，要求 p99 > p1，但得到 p1={p1}, p99={p99}")

    return p1, p99


def normalize_log1p_percentile(x: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """
    对非负雷达数据做 log1p + p1/p99 分位数归一化。

    输入：
        x: 原始矩阵，建议非负。

    输出：
        float32, 范围约为 [0, 1]
    """
    x = np.asarray(x, dtype=np.float32)

    # 保险处理：NaN/Inf 先变成 0，负值裁为 0
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)

    x = np.log1p(x)

    denom = max(float(p99 - p1), 1e-12)
    x = (x - p1) / denom
    x = np.clip(x, 0.0, 1.0)

    return x.astype(np.float32, copy=False)


def pad_to_size_2d(
    arr: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    将 2D 数组 padding 到指定大小。

    如果 arr 比目标大，则会裁剪左上角对应大小。
    正常情况下，dataset 里会先裁剪，不应该出现大于目标的情况。
    """
    arr = np.asarray(arr)

    h, w = arr.shape

    out = np.full((target_h, target_w), pad_value, dtype=arr.dtype)

    copy_h = min(h, target_h)
    copy_w = min(w, target_w)

    out[:copy_h, :copy_w] = arr[:copy_h, :copy_w]

    return out


def make_full_valid_mask(
    reflection_patch: np.ndarray,
    noisy_patch: np.ndarray,
) -> np.ndarray:
    """
    方案 2：
        不使用 npz 内部保存的 mask。
        认为 reflection patch 内部都是有效区域。

    但是：
        NaN/Inf 区域设为无效。
    """
    valid = np.isfinite(reflection_patch) & np.isfinite(noisy_patch)
    return valid.astype(np.float32)


class RadarPatchDataset(Dataset):
    """
    SHARAD 反射层降噪 Dataset。

    返回字典：
        {
            "noisy":      Tensor [1, patch_h, patch_w],
            "reflection": Tensor [1, patch_h, patch_w],
            "mask":       Tensor [1, patch_h, patch_w],
            "alpha":      Tensor scalar,
            "file":       str,
            "track_id":   str,
        }

    训练目标：
        input noisy = reflection + alpha * noise
        target reflection

    注意：
        这里不使用 npz 里保存的 mask。
        因为你当前的 mask.shape 可能不是 reflection.shape。
    """

    def __init__(
        self,
        data_dir: str | Path = "radar_ai_dataset",
        split_json: str | Path = "outputs/splits/split.json",
        summary_json: str | Path = "outputs/stats/summary.json",
        split: str = "train",
        patch_h: int = 176,
        patch_w: int = 256,
        epoch_size: Optional[int] = None,
        alpha_min: float = 0.5,
        alpha_max: float = 1.5,
        noise_scaling_factor: float = 5.0,
        random_sampling: Optional[bool] = None,
        use_npz_mask: bool = False,
        verbose: bool = True,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split_json = Path(split_json)
        self.summary_json = Path(summary_json)

        self.split = split
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.noise_scaling_factor = float(noise_scaling_factor)

        self.use_npz_mask = bool(use_npz_mask)

        if self.patch_h <= 0 or self.patch_w <= 0:
            raise ValueError("patch_h 和 patch_w 必须为正整数")

        if self.alpha_max < self.alpha_min:
            raise ValueError("alpha_max 必须 >= alpha_min")

        if self.noise_scaling_factor < 0:
            raise ValueError("noise_scaling_factor 必须 >= 0")

        if random_sampling is None:
            random_sampling = split == "train"

        self.random_sampling = bool(random_sampling)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在：{self.data_dir.resolve()}")

        self.p1, self.p99 = load_log1p_norm_params(self.summary_json)

        file_names = load_split_files(self.split_json, split)

        self.records = self._build_records(file_names)

        if len(self.records) == 0:
            raise RuntimeError(
                f"split={split} 中没有可用样本。"
                f"可能所有文件宽度都小于 patch_w={self.patch_w}，"
                f"或者 npz 文件缺少必要数组。"
            )

        if epoch_size is None:
            if split == "train":
                # 默认每个 epoch 至少 10000 个 patch
                epoch_size = max(10000, len(self.records))
            else:
                # val/test 默认 1000 个 patch
                epoch_size = max(1000, len(self.records))

        self.epoch_size = int(epoch_size)

        if self.epoch_size <= 0:
            raise ValueError("epoch_size 必须为正整数")

        if verbose:
            self._print_summary(total_files=len(file_names))

    def _build_records(self, file_names: List[str]) -> List[Dict[str, Any]]:
        """
        检查 split 里的文件，跳过宽度小于 patch_w 的样本。

        当前设置：
            宽度 W < patch_w 的图像直接跳过。
        """
        records: List[Dict[str, Any]] = []

        skipped_missing = 0
        skipped_bad_key = 0
        skipped_shape = 0
        skipped_small_width = 0

        for filename in file_names:
            path = self.data_dir / filename

            if not path.exists():
                skipped_missing += 1
                continue

            try:
                with np.load(path, allow_pickle=True) as z:
                    if "reflection" not in z or "noise" not in z:
                        skipped_bad_key += 1
                        continue

                    reflection = z["reflection"]
                    noise = z["noise"]

                    if reflection.ndim != 2 or noise.ndim != 2:
                        skipped_shape += 1
                        continue

                    if reflection.shape != noise.shape:
                        skipped_shape += 1
                        continue

                    h, w = reflection.shape

                    if w < self.patch_w:
                        skipped_small_width += 1
                        continue

                    records.append(
                        {
                            "file": filename,
                            "path": str(path),
                            "track_id": infer_track_id_from_name(filename),
                            "height": int(h),
                            "width": int(w),
                        }
                    )

            except Exception:
                skipped_shape += 1
                continue

        self.skipped_info = {
            "missing_file": skipped_missing,
            "bad_key": skipped_bad_key,
            "shape_problem": skipped_shape,
            "small_width": skipped_small_width,
        }

        return records

    def _print_summary(self, total_files: int):
        widths = [r["width"] for r in self.records]
        heights = [r["height"] for r in self.records]

        print("\n========== RadarPatchDataset ==========")
        print(f"split: {self.split}")
        print(f"data_dir: {self.data_dir.resolve()}")
        print(f"total files in split_json: {total_files}")
        print(f"usable files: {len(self.records)}")
        print(f"epoch_size: {self.epoch_size}")
        print(f"patch size: {self.patch_h} x {self.patch_w}")
        print(f"random_sampling: {self.random_sampling}")
        print(f"alpha range: [{self.alpha_min}, {self.alpha_max}]")
        print(f"noise scaling factor: {self.noise_scaling_factor}")
        print(f"log1p norm p1/p99: {self.p1}, {self.p99}")
        print(f"use_npz_mask: {self.use_npz_mask}")

        if widths:
            print(
                "width range among usable files: "
                f"min={min(widths)}, "
                f"median={np.median(widths):.1f}, "
                f"max={max(widths)}"
            )

        if heights:
            print(f"height unique among usable files: {sorted(set(heights))}")

        print("skipped files:")
        for k, v in self.skipped_info.items():
            print(f"  {k}: {v}")

    def __len__(self) -> int:
        return self.epoch_size

    def _select_record(self, index: int) -> Dict[str, Any]:
        """
        train:
            随机选择一个文件。

        val/test:
            按 index 顺序循环，保证验证集相对稳定。
        """
        if self.random_sampling:
            j = np.random.randint(0, len(self.records))
            return self.records[j]

        return self.records[index % len(self.records)]

    def _choose_crop(
        self,
        h: int,
        w: int,
        index: int,
    ) -> tuple[int, int, int, int]:
        """
        返回裁剪区域：
            y0, x0, crop_h, crop_w

        注意：
            宽度小于 patch_w 的文件已经在初始化时跳过。

        高度：
            如果 h < patch_h，则使用全部高度，后面 padding。
            你的数据通常 h=175, patch_h=176，因此会裁 175，再 pad 到 176。

        宽度：
            w >= patch_w，裁 patch_w。
        """
        crop_h = min(h, self.patch_h)
        crop_w = self.patch_w

        if self.random_sampling:
            if h > crop_h:
                y0 = np.random.randint(0, h - crop_h + 1)
            else:
                y0 = 0

            if w > crop_w:
                x0 = np.random.randint(0, w - crop_w + 1)
            else:
                x0 = 0

            return y0, x0, crop_h, crop_w

        # val/test：确定性裁剪
        if h > crop_h:
            y0 = (h - crop_h) // 2
        else:
            y0 = 0

        if w <= crop_w:
            x0 = 0
        else:
            max_x0 = w - crop_w

            # 对同一张宽图，在不同 index 轮次取不同横向位置
            file_count = max(len(self.records), 1)
            repeat_id = index // file_count

            # 大概按 patch_w/2 的间隔覆盖宽图
            n_positions = max(1, math.ceil((w - crop_w + 1) / max(crop_w // 2, 1)))
            pos_id = repeat_id % n_positions

            if n_positions == 1:
                x0 = max_x0 // 2
            else:
                x0 = int(round(pos_id * max_x0 / (n_positions - 1)))

        return y0, x0, crop_h, crop_w

    def _make_alpha(self) -> float:
        """
        train:
            alpha 在 [alpha_min, alpha_max] 随机变化。

        val/test:
            默认 alpha=1.0，保证验证结果稳定。
        """
        if self.random_sampling:
            return float(np.random.uniform(self.alpha_min, self.alpha_max))

        return 1.0

    def _load_patch(self, record: Dict[str, Any], index: int):
        """
        从一个 npz 中读取并裁剪 patch。
        """
        path = Path(record["path"])

        with np.load(path, allow_pickle=True) as z:
            reflection = np.asarray(z["reflection"], dtype=np.float32)
            noise = np.asarray(z["noise"], dtype=np.float32)

            h, w = reflection.shape
            y0, x0, crop_h, crop_w = self._choose_crop(h, w, index)

            reflection_patch = reflection[y0:y0 + crop_h, x0:x0 + crop_w]
            noise_patch = noise[y0:y0 + crop_h, x0:x0 + crop_w]

            # 当前默认不用 npz 里的 mask。
            # 这个分支仅作为将来你修正 mask.shape 后的备用功能。
            if self.use_npz_mask and "mask" in z:
                candidate_mask = np.asarray(z["mask"])
                if candidate_mask.shape == reflection.shape:
                    mask_patch = candidate_mask[y0:y0 + crop_h, x0:x0 + crop_w] > 0
                    mask_patch = mask_patch.astype(np.float32)
                else:
                    mask_patch = None
            else:
                mask_patch = None

        alpha = self._make_alpha()

        # 处理 NaN/Inf
        reflection_patch = np.nan_to_num(
            reflection_patch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

        noise_patch = np.nan_to_num(
            noise_patch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

        # 动态加噪：先放大底部噪声，再叠加到 reflection。
        # reflection_patch 本身不改动，后面仍作为干净 target。
        scaled_noise_patch = self.noise_scaling_factor * noise_patch
        noisy_patch = reflection_patch + alpha * scaled_noise_patch

        # 你的矩阵没有负值；这里 clip 只是保险
        reflection_patch = np.clip(reflection_patch, 0.0, None)
        noisy_patch = np.clip(noisy_patch, 0.0, None)

        if mask_patch is None:
            mask_patch = make_full_valid_mask(reflection_patch, noisy_patch)
        else:
            finite_mask = make_full_valid_mask(reflection_patch, noisy_patch)
            mask_patch = mask_patch * finite_mask

        # pad 到 [patch_h, patch_w]
        reflection_patch = pad_to_size_2d(
            reflection_patch,
            target_h=self.patch_h,
            target_w=self.patch_w,
            pad_value=0.0,
        )

        noisy_patch = pad_to_size_2d(
            noisy_patch,
            target_h=self.patch_h,
            target_w=self.patch_w,
            pad_value=0.0,
        )

        mask_patch = pad_to_size_2d(
            mask_patch.astype(np.float32),
            target_h=self.patch_h,
            target_w=self.patch_w,
            pad_value=0.0,
        )

        # 归一化
        reflection_norm = normalize_log1p_percentile(
            reflection_patch,
            p1=self.p1,
            p99=self.p99,
        )

        noisy_norm = normalize_log1p_percentile(
            noisy_patch,
            p1=self.p1,
            p99=self.p99,
        )

        # 转成 torch tensor, shape: [1, H, W]
        noisy_tensor = torch.from_numpy(noisy_norm[None, :, :]).float()
        reflection_tensor = torch.from_numpy(reflection_norm[None, :, :]).float()
        mask_tensor = torch.from_numpy(mask_patch[None, :, :]).float()

        alpha_tensor = torch.tensor(alpha, dtype=torch.float32)

        return noisy_tensor, reflection_tensor, mask_tensor, alpha_tensor

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self._select_record(index)

        noisy, reflection, mask, alpha = self._load_patch(record, index)

        return {
            "noisy": noisy,
            "reflection": reflection,
            "mask": mask,
            "alpha": alpha,
            "file": record["file"],
            "track_id": record["track_id"],
        }


def radar_worker_init_fn(worker_id: int):
    """
    DataLoader 多进程 worker 的随机种子初始化函数。

    后面 train_unet.py 中可以这样用：
        DataLoader(..., worker_init_fn=radar_worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    data_dir: str | Path = "radar_ai_dataset",
    split_json: str | Path = "outputs/splits/split.json",
    summary_json: str | Path = "outputs/stats/summary.json",
    split: str = "train",
    patch_h: int = 176,
    patch_w: int = 256,
    epoch_size: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    alpha_min: float = 0.5,
    alpha_max: float = 1.5,
    noise_scaling_factor: float = 5.0,
    pin_memory: bool = True,
    verbose: bool = True,
) -> DataLoader:
    """
    构建 DataLoader，供 train_unet.py 直接调用。
    """
    dataset = RadarPatchDataset(
        data_dir=data_dir,
        split_json=split_json,
        summary_json=summary_json,
        split=split,
        patch_h=patch_h,
        patch_w=patch_w,
        epoch_size=epoch_size,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        noise_scaling_factor=noise_scaling_factor,
        random_sampling=(split == "train"),
        use_npz_mask=False,
        verbose=verbose,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
        worker_init_fn=radar_worker_init_fn,
    )

    return loader


def parse_args():
    parser = argparse.ArgumentParser(description="Test RadarPatchDataset.")

    parser.add_argument("--data_dir", type=str, default="radar_ai_dataset")
    parser.add_argument("--split_json", type=str, default="outputs/splits/split.json")
    parser.add_argument("--summary_json", type=str, default="outputs/stats/summary.json")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])

    parser.add_argument("--patch_h", type=int, default=176)
    parser.add_argument("--patch_w", type=int, default=256)
    parser.add_argument("--epoch_size", type=int, default=16)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--alpha_min", type=float, default=0.5)
    parser.add_argument("--alpha_max", type=float, default=1.5)
    parser.add_argument("--noise_scaling_factor", type=float, default=5.0)

    return parser.parse_args()


def main():
    
    
    args = parse_args()

    loader = build_dataloader(
        data_dir=args.data_dir,
        split_json=args.split_json,
        summary_json=args.summary_json,
        split=args.split,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        epoch_size=args.epoch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        noise_scaling_factor=args.noise_scaling_factor,
        verbose=True,
    )

    batch = next(iter(loader))

    print("\n========== Batch 测试 ==========")
    print("noisy shape:     ", tuple(batch["noisy"].shape))
    print("reflection shape:", tuple(batch["reflection"].shape))
    print("mask shape:      ", tuple(batch["mask"].shape))
    print("alpha shape:     ", tuple(batch["alpha"].shape))
    print("noisy dtype:     ", batch["noisy"].dtype)
    print("reflection dtype:", batch["reflection"].dtype)
    print("mask dtype:      ", batch["mask"].dtype)

    print("\n数值范围：")
    print(
        "noisy:      ",
        float(batch["noisy"].min()),
        float(batch["noisy"].max()),
    )
    print(
        "reflection: ",
        float(batch["reflection"].min()),
        float(batch["reflection"].max()),
    )
    print(
        "mask:       ",
        float(batch["mask"].min()),
        float(batch["mask"].max()),
    )
    print("alpha:      ", batch["alpha"].tolist())

    print("\n样本来源：")
    for i in range(min(len(batch["file"]), 5)):
        print(f"  {i}: {batch['file'][i]} | {batch['track_id'][i]}")

    print("\ndataset.py 测试通过。")

    

if __name__ == "__main__":
    main()
