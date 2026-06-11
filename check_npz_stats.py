# check_npz_stats.py
# 用途：
#   检查 radar_ai_dataset 中所有 .npz 文件的数据结构、shape、数值范围、mask 有效比例等。
#
# 推荐运行：
#   F:\FInstallation\anaconda\python.exe check_npz_stats.py --data_dir radar_ai_dataset
#
# 输出：
#   outputs/stats/npz_stats.csv
#   outputs/stats/summary.json
#   outputs/stats/bad_files.txt
#   outputs/stats/width_hist.png
#   outputs/stats/mask_valid_ratio_hist.png
#   outputs/stats/reflection_log1p_hist.png

from __future__ import annotations

import argparse
import csv
import json
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = [
    "reflection",
    "noise",
    "noisy_reflection",
    "mask",
    "surface",
]

META_KEYS = [
    "track_id",
    "original_shape",
    "data_shape",
    "noise_mean",
    "bottom_noise_ratio",
    "margin_surface",
    "reflection_depth",
    "alpha",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Check npz dataset statistics for radar denoising.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="radar_ai_dataset",
        help="Folder containing *_dataset.npz files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/stats",
        help="Output folder for csv/json/figures.",
    )
    parser.add_argument(
        "--file_percentile_sample",
        type=int,
        default=50000,
        help="Max sampled pixels per file for percentile estimation.",
    )
    parser.add_argument(
        "--global_sample_limit",
        type=int,
        default=2_000_000,
        help="Max total sampled pixels for global percentile/histogram estimation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for sampling.",
    )
    return parser.parse_args()


def to_python_scalar(x: Any) -> Any:
    """
    把 npz 中的标量或小数组转成 JSON/CSV 更容易保存的形式。
    """
    if isinstance(x, np.ndarray):
        if x.shape == ():
            x = x.item()
        elif x.size == 1:
            x = x.reshape(-1)[0].item()
        else:
            return x.tolist()

    if isinstance(x, np.generic):
        x = x.item()

    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    return str(x)


def shape_to_str(shape) -> str:
    return "x".join(str(int(v)) for v in shape)


def infer_track_id(npz_path: Path, npz_obj=None) -> str:
    """
    优先读取 npz 内部的 track_id。
    如果没有，则从文件名中提取 S_XXXXXXXX 形式的编号。
    """
    if npz_obj is not None and "track_id" in npz_obj.files:
        try:
            return str(to_python_scalar(npz_obj["track_id"]))
        except Exception:
            pass

    m = re.search(r"(S_\d+)", npz_path.stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"(\d{8})", npz_path.stem)
    if m:
        return "S_" + m.group(1)

    return npz_path.stem


def sample_values(values: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    """
    从一维数组里随机抽样，用于估算分位数和全局直方图。
    """
    values = np.asarray(values).reshape(-1)

    if values.size == 0:
        return values.astype(np.float64)

    if values.size <= max_count:
        return values.astype(np.float64, copy=False)

    idx = rng.choice(values.size, size=max_count, replace=False)
    return values[idx].astype(np.float64, copy=False)


def safe_numeric_mask(arr: np.ndarray) -> np.ndarray:
    """
    返回 finite mask。
    若数组不是数值类型，则返回全 False。
    """
    try:
        return np.isfinite(arr)
    except TypeError:
        return np.zeros(arr.shape, dtype=bool)


def valid_values(
    arr: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    提取有限数值。
    如果传入 mask，并且 mask.shape == arr.shape，则只取 mask > 0 的有效区域。
    """
    arr = np.asarray(arr)
    finite = safe_numeric_mask(arr)

    if mask is not None and mask.shape == arr.shape:
        use = finite & (mask > 0)
    else:
        use = finite

    return arr[use]


def compute_array_stats(
    arr: np.ndarray,
    prefix: str,
    rng: np.random.Generator,
    percentile_sample: int = 50000,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    计算某个数组的基本统计。
    prefix 用于区分 reflection/noise/noisy_reflection 等。
    """
    row = {}

    arr = np.asarray(arr)

    row[f"{prefix}_dtype"] = str(arr.dtype)
    row[f"{prefix}_shape"] = shape_to_str(arr.shape)
    row[f"{prefix}_ndim"] = int(arr.ndim)
    row[f"{prefix}_size"] = int(arr.size)

    finite = safe_numeric_mask(arr)
    row[f"{prefix}_nan_count"] = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    row[f"{prefix}_inf_count"] = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    row[f"{prefix}_finite_ratio"] = float(finite.mean()) if arr.size > 0 else 0.0

    vals = valid_values(arr, mask=mask)
    row[f"{prefix}_used_count"] = int(vals.size)

    if vals.size == 0:
        for k in [
            "min", "p0_1", "p1", "p5", "p50", "p95", "p99", "p99_9",
            "max", "mean", "std", "neg_count",
        ]:
            row[f"{prefix}_{k}"] = None
        return row

    vals_float = vals.astype(np.float64, copy=False)

    row[f"{prefix}_min"] = float(np.min(vals_float))
    row[f"{prefix}_max"] = float(np.max(vals_float))
    row[f"{prefix}_mean"] = float(np.mean(vals_float))
    row[f"{prefix}_std"] = float(np.std(vals_float))
    row[f"{prefix}_neg_count"] = int(np.sum(vals_float < 0))

    pct_sample = sample_values(vals_float, percentile_sample, rng)
    pct = np.percentile(pct_sample, [0.1, 1, 5, 50, 95, 99, 99.9])

    row[f"{prefix}_p0_1"] = float(pct[0])
    row[f"{prefix}_p1"] = float(pct[1])
    row[f"{prefix}_p5"] = float(pct[2])
    row[f"{prefix}_p50"] = float(pct[3])
    row[f"{prefix}_p95"] = float(pct[4])
    row[f"{prefix}_p99"] = float(pct[5])
    row[f"{prefix}_p99_9"] = float(pct[6])

    return row


def compute_global_stats(samples: np.ndarray) -> dict[str, Any]:
    """
    对全局抽样值计算统计。
    """
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]

    if samples.size == 0:
        return {
            "count": 0,
            "min": None,
            "p0_1": None,
            "p1": None,
            "p5": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "p99_9": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    pct = np.percentile(samples, [0.1, 1, 5, 50, 95, 99, 99.9])

    return {
        "count": int(samples.size),
        "min": float(np.min(samples)),
        "p0_1": float(pct[0]),
        "p1": float(pct[1]),
        "p5": float(pct[2]),
        "p50": float(pct[3]),
        "p95": float(pct[4]),
        "p99": float(pct[5]),
        "p99_9": float(pct[6]),
        "max": float(np.max(samples)),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
    }


def write_csv(rows: list[dict[str, Any]], csv_path: Path):
    """
    写出 CSV。用 utf-8-sig 方便 Excel 打开。
    """
    if not rows:
        return

    preferred = [
        "file",
        "status",
        "track_id",
        "missing_keys",
        "extra_keys",
        "reflection_shape",
        "noise_shape",
        "noisy_reflection_shape",
        "mask_shape",
        "surface_shape",
        "height",
        "width",
        "noise_same_shape",
        "noisy_same_shape",
        "mask_same_shape",
        "mask_valid_ratio",
        "reflection_min",
        "reflection_p1",
        "reflection_p50",
        "reflection_p99",
        "reflection_max",
        "noise_min",
        "noise_p1",
        "noise_p50",
        "noise_p99",
        "noise_max",
        "noisy_reflection_min",
        "noisy_reflection_p1",
        "noisy_reflection_p50",
        "noisy_reflection_p99",
        "noisy_reflection_max",
        "alpha",
        "noise_mean",
        "bottom_noise_ratio",
        "margin_surface",
        "reflection_depth",
        "error",
    ]

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = []
    for k in preferred:
        if k in all_keys:
            fieldnames.append(k)

    for k in sorted(all_keys):
        if k not in fieldnames:
            fieldnames.append(k)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def try_make_plots(out_dir: Path, rows: list[dict[str, Any]], global_samples: dict[str, np.ndarray]):
    """
    尝试生成几张基础统计图。
    如果没有 matplotlib，则自动跳过。
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib 不可用，跳过绘图。")
        return

    ok_rows = [r for r in rows if r.get("status") == "ok"]

    widths = [r.get("width") for r in ok_rows if isinstance(r.get("width"), int)]
    if widths:
        plt.figure(figsize=(8, 5))
        plt.hist(widths, bins=50)
        plt.xlabel("Width / samples")
        plt.ylabel("Count")
        plt.title("Radar reflection matrix width distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "width_hist.png", dpi=200)
        plt.close()

    ratios = [
        r.get("mask_valid_ratio")
        for r in ok_rows
        if isinstance(r.get("mask_valid_ratio"), (int, float))
    ]
    if ratios:
        plt.figure(figsize=(8, 5))
        plt.hist(ratios, bins=50)
        plt.xlabel("Mask valid ratio")
        plt.ylabel("Count")
        plt.title("Mask valid ratio distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "mask_valid_ratio_hist.png", dpi=200)
        plt.close()

    if "reflection_valid_raw" in global_samples:
        x = global_samples["reflection_valid_raw"]
        x = x[np.isfinite(x)]
        x = x[x >= 0]
        if x.size > 0:
            x_log = np.log1p(x)
            plt.figure(figsize=(8, 5))
            plt.hist(x_log, bins=100)
            plt.xlabel("log1p(reflection)")
            plt.ylabel("Sample count")
            plt.title("Global sampled reflection distribution after log1p")
            plt.tight_layout()
            plt.savefig(out_dir / "reflection_log1p_hist.png", dpi=200)
            plt.close()

    if "noise_raw" in global_samples:
        x = global_samples["noise_raw"]
        x = x[np.isfinite(x)]
        x = x[x >= 0]
        if x.size > 0:
            x_log = np.log1p(x)
            plt.figure(figsize=(8, 5))
            plt.hist(x_log, bins=100)
            plt.xlabel("log1p(noise)")
            plt.ylabel("Sample count")
            plt.title("Global sampled bottom-noise distribution after log1p")
            plt.tight_layout()
            plt.savefig(out_dir / "noise_log1p_hist.png", dpi=200)
            plt.close()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据文件夹不存在：{data_dir.resolve()}")

    npz_files = sorted(data_dir.glob("*.npz"))

    if len(npz_files) == 0:
        raise FileNotFoundError(f"没有在 {data_dir.resolve()} 找到 .npz 文件")

    rng = np.random.default_rng(args.seed)

    # 每个文件为全局统计抽样多少像素
    global_sample_per_file = max(10, args.global_sample_limit // max(len(npz_files), 1))

    rows: list[dict[str, Any]] = []
    bad_files: list[str] = []

    global_sample_lists = {
        "reflection_valid_raw": [],
        "noisy_reflection_valid_raw": [],
        "noise_raw": [],
        "noise_centered_raw": [],
    }

    print(f"[INFO] data_dir = {data_dir.resolve()}")
    print(f"[INFO] out_dir  = {out_dir.resolve()}")
    print(f"[INFO] 找到 npz 文件数量：{len(npz_files)}")
    print(f"[INFO] 每个文件用于全局统计的抽样像素数：{global_sample_per_file}")

    for i, npz_path in enumerate(npz_files, start=1):
        if i % 100 == 0 or i == 1 or i == len(npz_files):
            print(f"[INFO] 正在检查 {i}/{len(npz_files)}: {npz_path.name}")

        row: dict[str, Any] = {
            "file": npz_path.name,
            "status": "ok",
        }

        try:
            with np.load(npz_path, allow_pickle=True) as z:
                keys = list(z.files)

                missing = [k for k in REQUIRED_KEYS if k not in keys]
                extra = [k for k in keys if k not in REQUIRED_KEYS and k not in META_KEYS]

                row["track_id"] = infer_track_id(npz_path, z)
                row["missing_keys"] = ";".join(missing)
                row["extra_keys"] = ";".join(extra)

                for mk in META_KEYS:
                    if mk in keys:
                        try:
                            row[mk] = to_python_scalar(z[mk])
                        except Exception:
                            row[mk] = None

                if missing:
                    row["status"] = "missing_keys"
                    bad_files.append(f"{npz_path.name}: missing keys {missing}")
                    rows.append(row)
                    continue

                reflection = np.asarray(z["reflection"])
                noise = np.asarray(z["noise"])
                noisy = np.asarray(z["noisy_reflection"])
                mask = np.asarray(z["mask"])
                surface = np.asarray(z["surface"])

                row["reflection_shape"] = shape_to_str(reflection.shape)
                row["noise_shape"] = shape_to_str(noise.shape)
                row["noisy_reflection_shape"] = shape_to_str(noisy.shape)
                row["mask_shape"] = shape_to_str(mask.shape)
                row["surface_shape"] = shape_to_str(surface.shape)

                if reflection.ndim == 2:
                    row["height"] = int(reflection.shape[0])
                    row["width"] = int(reflection.shape[1])
                else:
                    row["height"] = None
                    row["width"] = None

                row["noise_same_shape"] = bool(noise.shape == reflection.shape)
                row["noisy_same_shape"] = bool(noisy.shape == reflection.shape)
                row["mask_same_shape"] = bool(mask.shape == reflection.shape)

                if mask.shape == reflection.shape:
                    mask_bool = mask > 0
                    row["mask_valid_ratio"] = float(mask_bool.mean())
                    row["mask_valid_count"] = int(mask_bool.sum())
                else:
                    mask_bool = None
                    row["mask_valid_ratio"] = None
                    row["mask_valid_count"] = None

                if surface.ndim >= 1:
                    row["surface_len"] = int(surface.reshape(-1).size)
                else:
                    row["surface_len"] = 1

                # 统计 reflection：优先只统计 mask 有效区域
                row.update(
                    compute_array_stats(
                        reflection,
                        prefix="reflection",
                        rng=rng,
                        percentile_sample=args.file_percentile_sample,
                        mask=mask_bool,
                    )
                )

                # 统计 noise：统计全部区域
                row.update(
                    compute_array_stats(
                        noise,
                        prefix="noise",
                        rng=rng,
                        percentile_sample=args.file_percentile_sample,
                        mask=None,
                    )
                )

                # 统计 noisy_reflection：优先只统计 mask 有效区域
                row.update(
                    compute_array_stats(
                        noisy,
                        prefix="noisy_reflection",
                        rng=rng,
                        percentile_sample=args.file_percentile_sample,
                        mask=mask_bool,
                    )
                )

                # shape 异常标记
                shape_problem = False
                if reflection.ndim != 2:
                    shape_problem = True
                if noise.shape != reflection.shape:
                    shape_problem = True
                if noisy.shape != reflection.shape:
                    shape_problem = True
                if mask.shape != reflection.shape:
                    shape_problem = True

                if shape_problem:
                    row["status"] = "shape_problem"
                    bad_files.append(f"{npz_path.name}: shape problem")

                # 全局抽样
                if mask_bool is not None:
                    ref_vals = valid_values(reflection, mask=mask_bool)
                    noisy_vals = valid_values(noisy, mask=mask_bool)
                else:
                    ref_vals = valid_values(reflection, mask=None)
                    noisy_vals = valid_values(noisy, mask=None)

                noise_vals = valid_values(noise, mask=None)

                global_sample_lists["reflection_valid_raw"].append(
                    sample_values(ref_vals, global_sample_per_file, rng)
                )
                global_sample_lists["noisy_reflection_valid_raw"].append(
                    sample_values(noisy_vals, global_sample_per_file, rng)
                )
                global_sample_lists["noise_raw"].append(
                    sample_values(noise_vals, global_sample_per_file, rng)
                )

                if noise_vals.size > 0:
                    noise_median = np.median(sample_values(noise_vals, min(noise_vals.size, 50000), rng))
                    noise_centered = noise_vals.astype(np.float64, copy=False) - float(noise_median)
                    global_sample_lists["noise_centered_raw"].append(
                        sample_values(noise_centered, global_sample_per_file, rng)
                    )

                rows.append(row)

        except Exception as e:
            row["status"] = "error"
            row["error"] = repr(e)
            rows.append(row)

            msg = f"{npz_path.name}: {repr(e)}\n{traceback.format_exc()}"
            bad_files.append(msg)

    # 整理全局 samples
    global_samples = {}
    for name, parts in global_sample_lists.items():
        if len(parts) > 0:
            parts = [p for p in parts if p.size > 0]
            if parts:
                global_samples[name] = np.concatenate(parts)
            else:
                global_samples[name] = np.array([], dtype=np.float64)
        else:
            global_samples[name] = np.array([], dtype=np.float64)

    # 全局统计
    summary: dict[str, Any] = {}

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    shape_problem_rows = [r for r in rows if r.get("status") == "shape_problem"]
    missing_rows = [r for r in rows if r.get("status") == "missing_keys"]
    error_rows = [r for r in rows if r.get("status") == "error"]

    summary["num_files"] = int(len(npz_files))
    summary["num_ok"] = int(len(ok_rows))
    summary["num_shape_problem"] = int(len(shape_problem_rows))
    summary["num_missing_keys"] = int(len(missing_rows))
    summary["num_error"] = int(len(error_rows))

    heights = [r.get("height") for r in rows if isinstance(r.get("height"), int)]
    widths = [r.get("width") for r in rows if isinstance(r.get("width"), int)]
    mask_ratios = [
        r.get("mask_valid_ratio")
        for r in rows
        if isinstance(r.get("mask_valid_ratio"), (int, float))
    ]

    if heights:
        summary["height_min"] = int(np.min(heights))
        summary["height_max"] = int(np.max(heights))
        summary["height_unique"] = sorted(list(set(int(h) for h in heights)))

    if widths:
        summary["width_min"] = int(np.min(widths))
        summary["width_p5"] = float(np.percentile(widths, 5))
        summary["width_p50"] = float(np.percentile(widths, 50))
        summary["width_p95"] = float(np.percentile(widths, 95))
        summary["width_max"] = int(np.max(widths))

    if mask_ratios:
        summary["mask_valid_ratio_min"] = float(np.min(mask_ratios))
        summary["mask_valid_ratio_p5"] = float(np.percentile(mask_ratios, 5))
        summary["mask_valid_ratio_p50"] = float(np.percentile(mask_ratios, 50))
        summary["mask_valid_ratio_p95"] = float(np.percentile(mask_ratios, 95))
        summary["mask_valid_ratio_max"] = float(np.max(mask_ratios))

    summary["global_stats"] = {
        "reflection_valid_raw": compute_global_stats(global_samples["reflection_valid_raw"]),
        "noisy_reflection_valid_raw": compute_global_stats(global_samples["noisy_reflection_valid_raw"]),
        "noise_raw": compute_global_stats(global_samples["noise_raw"]),
        "noise_centered_raw": compute_global_stats(global_samples["noise_centered_raw"]),
    }

    # 推荐 log1p 归一化参数
    # 第一版训练时可以用：
    #   x_log = log1p(x)
    #   x_norm = (x_log - p1) / (p99 - p1)
    #   x_norm = clip(x_norm, 0, 1)
    ref = global_samples["reflection_valid_raw"]
    noisy = global_samples["noisy_reflection_valid_raw"]

    combined = np.concatenate([
        ref[np.isfinite(ref)],
        noisy[np.isfinite(noisy)],
    ])
    combined = combined[combined >= 0]

    if combined.size > 0:
        combined_log = np.log1p(combined)
        p1, p99 = np.percentile(combined_log, [1, 99])
        p0_1, p99_9 = np.percentile(combined_log, [0.1, 99.9])

        summary["recommended_log1p_norm"] = {
            "method": "x_log = log1p(x); x_norm = clip((x_log - p1) / (p99 - p1), 0, 1)",
            "p1": float(p1),
            "p99": float(p99),
            "p0_1": float(p0_1),
            "p99_9": float(p99_9),
            "note": "建议第一版先使用 p1 和 p99；如果图像被截断太严重，再尝试 p0_1 和 p99_9。",
        }
    else:
        summary["recommended_log1p_norm"] = None

    # 写出结果
    csv_path = out_dir / "npz_stats.csv"
    summary_path = out_dir / "summary.json"
    bad_path = out_dir / "bad_files.txt"

    write_csv(rows, csv_path)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with bad_path.open("w", encoding="utf-8") as f:
        if bad_files:
            f.write("\n\n".join(bad_files))
        else:
            f.write("No bad files found.\n")

    try_make_plots(out_dir, rows, global_samples)

    print("\n========== 检查完成 ==========")
    print(f"总文件数：{summary['num_files']}")
    print(f"正常文件数：{summary['num_ok']}")
    print(f"shape 问题文件数：{summary['num_shape_problem']}")
    print(f"缺 key 文件数：{summary['num_missing_keys']}")
    print(f"读取错误文件数：{summary['num_error']}")

    if "height_unique" in summary:
        print(f"高度 unique：{summary['height_unique']}")

    if "width_min" in summary:
        print(
            "宽度范围："
            f"min={summary['width_min']}, "
            f"p50={summary['width_p50']:.1f}, "
            f"max={summary['width_max']}"
        )

    if "mask_valid_ratio_p50" in summary:
        print(
            "mask 有效比例："
            f"min={summary['mask_valid_ratio_min']:.3f}, "
            f"p50={summary['mask_valid_ratio_p50']:.3f}, "
            f"max={summary['mask_valid_ratio_max']:.3f}"
        )

    norm = summary.get("recommended_log1p_norm")
    if norm is not None:
        print("\n推荐第一版 log1p 归一化参数：")
        print(f"p1  = {norm['p1']}")
        print(f"p99 = {norm['p99']}")

    print("\n输出文件：")
    print(f"CSV:     {csv_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Bad:     {bad_path.resolve()}")
    print("Figures: width_hist.png / mask_valid_ratio_hist.png / reflection_log1p_hist.png")


if __name__ == "__main__":
    main()