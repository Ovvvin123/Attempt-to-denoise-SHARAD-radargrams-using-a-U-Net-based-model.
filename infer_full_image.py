# infer_full_image.py
# 用途：
#   使用训练好的 Residual U-Net 对完整 reflection 矩阵进行滑窗降噪推理。
#
# 默认输入：
#   radar_ai_dataset/*.npz 中的 reflection
#
# 默认输出：
#   outputs/inference/test/
#
# 推荐运行：
#   cd F:\radar_deeplearning
#   F:\FInstallation\anaconda\envs\pytorch\python.exe infer_full_image.py
#
# 指定模型：
#   F:\FInstallation\anaconda\envs\pytorch\python.exe infer_full_image.py --checkpoint outputs/checkpoints/last_unet.pth
#
# 推理全部数据：
#   F:\FInstallation\anaconda\envs\pytorch\python.exe infer_full_image.py --split all

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn

from model_unet import ResidualUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Infer full radar reflection matrices with trained U-Net.")

    parser.add_argument("--data_dir", type=str, default="radar_ai_dataset")
    parser.add_argument("--split_json", type=str, default="outputs/splits/split.json")
    parser.add_argument("--summary_json", type=str, default="outputs/stats/summary.json")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_unet.pth")

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to infer.",
    )

    parser.add_argument("--out_dir", type=str, default="", help="Default: outputs/inference/<split>")

    parser.add_argument("--patch_h", type=int, default=176)
    parser.add_argument("--patch_w", type=int, default=0, help="0 means read from checkpoint args, fallback to 256.")
    parser.add_argument("--overlap", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--save_png", action="store_true", default=True)
    parser.add_argument("--no_png", action="store_false", dest="save_png")
    parser.add_argument("--max_png", type=int, default=50)

    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")

    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--overwrite", action="store_true", default=False)

    return parser.parse_args()


def infer_track_id_from_name(filename: str) -> str:
    stem = Path(filename).stem

    m = re.search(r"(S_\d{8})", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"(\d{8})", stem)
    if m:
        return "S_" + m.group(1)

    return stem


def to_python_scalar(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        if x.size == 1:
            return x.reshape(-1)[0].item()
        return x.tolist()

    if isinstance(x, np.generic):
        return x.item()

    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")

    return x


def load_split_files(data_dir: Path, split_json: Path, split: str) -> List[Path]:
    if split == "all":
        return sorted(data_dir.glob("*.npz"))

    if not split_json.exists():
        raise FileNotFoundError(f"找不到 split_json：{split_json.resolve()}")

    with split_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    files = obj["files"][split]
    paths = [data_dir / name for name in files]

    paths = [p for p in paths if p.exists()]

    return paths


def load_norm_params(summary_json: Path) -> Tuple[float, float]:
    if not summary_json.exists():
        raise FileNotFoundError(f"找不到 summary_json：{summary_json.resolve()}")

    with summary_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    norm = obj.get("recommended_log1p_norm", None)

    if norm is not None and "p1" in norm and "p99" in norm:
        p1 = float(norm["p1"])
        p99 = float(norm["p99"])

        if np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
            return p1, p99

    global_stats = obj.get("global_stats", {})
    ref_stats = global_stats.get("reflection_valid_raw", {})

    raw_p1 = ref_stats.get("p1", None)
    raw_p99 = ref_stats.get("p99", None)

    if raw_p1 is None or raw_p99 is None:
        raise KeyError(
            f"{summary_json} 中没有 recommended_log1p_norm，"
            "也无法从 global_stats.reflection_valid_raw 推断 p1/p99。"
        )

    p1 = float(np.log1p(max(float(raw_p1), 0.0)))
    p99 = float(np.log1p(max(float(raw_p99), 0.0)))

    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        raise ValueError(f"归一化参数异常：p1={p1}, p99={p99}")

    return p1, p99


def normalize_log1p(x: np.ndarray, p1: float, p99: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)

    x_log = np.log1p(x)
    x_norm = (x_log - p1) / max(p99 - p1, 1e-12)
    x_norm = np.clip(x_norm, 0.0, 1.0)

    return x_norm.astype(np.float32)


def denormalize_log1p(x_norm: np.ndarray, p1: float, p99: float) -> np.ndarray:
    x_norm = np.asarray(x_norm, dtype=np.float32)
    x_norm = np.clip(x_norm, 0.0, 1.0)

    x_log = x_norm * (p99 - p1) + p1
    x_raw = np.expm1(x_log)
    x_raw = np.clip(x_raw, 0.0, None)

    return x_raw.astype(np.float32)


def get_window_starts(width: int, patch_w: int, overlap: int) -> List[int]:
    if patch_w <= 0:
        raise ValueError("patch_w 必须为正整数")

    if width <= patch_w:
        return [0]

    step = patch_w - overlap

    if step <= 0:
        raise ValueError("overlap 必须小于 patch_w")

    starts = list(range(0, width - patch_w + 1, step))

    last = width - patch_w
    if starts[-1] != last:
        starts.append(last)

    return starts


def pad_patch_2d(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape

    out = np.zeros((target_h, target_w), dtype=arr.dtype)

    copy_h = min(h, target_h)
    copy_w = min(w, target_w)

    out[:copy_h, :copy_w] = arr[:copy_h, :copy_w]

    return out


def make_model_from_checkpoint_args(ckpt_args: Dict[str, Any]) -> ResidualUNet:
    model = ResidualUNet(
        in_channels=int(ckpt_args.get("in_channels", 1)),
        out_channels=int(ckpt_args.get("out_channels", 1)),
        base_channels=int(ckpt_args.get("base_channels", 32)),
        depth=int(ckpt_args.get("depth", 3)),
        norm=str(ckpt_args.get("norm", "group")),
        up_mode=str(ckpt_args.get("up_mode", "bilinear")),
        dropout=float(ckpt_args.get("dropout", 0.0)),
    )

    return model


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path.resolve()}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    ckpt_args = ckpt.get("args", {})

    model = make_model_from_checkpoint_args(ckpt_args)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, ckpt_args


@torch.no_grad()
def infer_full_matrix(
    reflection_raw: np.ndarray,
    model: nn.Module,
    device: torch.device,
    p1: float,
    p99: float,
    patch_h: int = 176,
    patch_w: int = 256,
    overlap: int = 64,
    batch_size: int = 8,
    amp: bool = True,
) -> Dict[str, np.ndarray]:
    """
    对完整 reflection 矩阵做滑窗推理。

    输入：
        reflection_raw: [175, W] 或类似二维矩阵

    输出：
        reflection_norm
        denoised_norm
        denoised_raw_approx
        pred_noise_norm
        residual_norm
        weight_sum
    """
    reflection_raw = np.asarray(reflection_raw, dtype=np.float32)
    reflection_raw = np.nan_to_num(reflection_raw, nan=0.0, posinf=0.0, neginf=0.0)
    reflection_raw = np.clip(reflection_raw, 0.0, None)

    h, w = reflection_raw.shape

    if h > patch_h:
        raise ValueError(
            f"当前 reflection 高度 h={h} 大于 patch_h={patch_h}。"
            "如果确实需要处理更高矩阵，请重新设置 --patch_h。"
        )

    reflection_norm = normalize_log1p(reflection_raw, p1=p1, p99=p99)

    starts = get_window_starts(width=w, patch_w=patch_w, overlap=overlap)

    denoised_acc = np.zeros((h, w), dtype=np.float32)
    pred_noise_acc = np.zeros((h, w), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)

    patches = []
    patch_infos = []

    def flush_batch():
        nonlocal patches, patch_infos, denoised_acc, pred_noise_acc, weight_acc

        if len(patches) == 0:
            return

        x = np.stack(patches, axis=0)  # [B, H, W]
        x_tensor = torch.from_numpy(x[:, None, :, :]).float().to(device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            pred_noise = model(x_tensor)
            denoised = x_tensor - pred_noise

        pred_noise_np = pred_noise[:, 0].float().cpu().numpy()
        denoised_np = denoised[:, 0].float().cpu().numpy()

        for i, info in enumerate(patch_infos):
            x0 = info["x0"]
            valid_w = info["valid_w"]
            valid_h = info["valid_h"]

            den_patch = denoised_np[i, :valid_h, :valid_w]
            noise_patch = pred_noise_np[i, :valid_h, :valid_w]

            denoised_acc[:valid_h, x0:x0 + valid_w] += den_patch
            pred_noise_acc[:valid_h, x0:x0 + valid_w] += noise_patch
            weight_acc[:valid_h, x0:x0 + valid_w] += 1.0

        patches = []
        patch_infos = []

    for x0 in starts:
        x1 = min(x0 + patch_w, w)

        patch = reflection_norm[:, x0:x1]
        valid_h, valid_w = patch.shape

        patch_pad = pad_patch_2d(patch, target_h=patch_h, target_w=patch_w)

        patches.append(patch_pad)
        patch_infos.append(
            {
                "x0": x0,
                "valid_h": valid_h,
                "valid_w": valid_w,
            }
        )

        if len(patches) >= batch_size:
            flush_batch()

    flush_batch()

    weight_safe = np.maximum(weight_acc, 1e-6)

    denoised_norm = denoised_acc / weight_safe
    pred_noise_norm = pred_noise_acc / weight_safe

    # 显示和反归一化时建议 clip；但 residual 使用未 clip 的 denoised_norm
    denoised_norm_clipped = np.clip(denoised_norm, 0.0, 1.0)

    denoised_raw_approx = denormalize_log1p(
        denoised_norm_clipped,
        p1=p1,
        p99=p99,
    )

    residual_norm = denoised_norm - reflection_norm

    return {
        "reflection_norm": reflection_norm.astype(np.float32),
        "denoised_norm": denoised_norm.astype(np.float32),
        "denoised_norm_clipped": denoised_norm_clipped.astype(np.float32),
        "denoised_raw_approx": denoised_raw_approx.astype(np.float32),
        "pred_noise_norm": pred_noise_norm.astype(np.float32),
        "residual_norm": residual_norm.astype(np.float32),
        "weight_sum": weight_acc.astype(np.float32),
    }


def save_inference_npz(
    out_path: Path,
    source_npz_path: Path,
    source_obj,
    reflection_raw: np.ndarray,
    result: Dict[str, np.ndarray],
    track_id: str,
    p1: float,
    p99: float,
    checkpoint: Path,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "source_file": np.array(source_npz_path.name),
        "track_id": np.array(track_id),
        "checkpoint": np.array(str(checkpoint)),
        "norm_p1": np.array(p1, dtype=np.float32),
        "norm_p99": np.array(p99, dtype=np.float32),
        "reflection_raw": reflection_raw.astype(np.float32),
        "reflection_norm": result["reflection_norm"],
        "denoised_norm": result["denoised_norm"],
        "denoised_norm_clipped": result["denoised_norm_clipped"],
        "denoised_raw_approx": result["denoised_raw_approx"],
        "pred_noise_norm": result["pred_noise_norm"],
        "residual_norm": result["residual_norm"],
        "weight_sum": result["weight_sum"],
    }

    for key in ["surface", "original_shape", "data_shape", "noise_mean"]:
        if key in source_obj.files:
            save_dict[key] = source_obj[key]

    np.savez_compressed(out_path, **save_dict)


def to_uint8_image(img: np.ndarray, clip01: bool = False) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)

    if clip01:
        img = np.clip(img, 0.0, 1.0)
        return np.round(img * 255.0).astype(np.uint8)

    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros(img.shape, dtype=np.uint8)

    lo = float(np.min(img[finite]))
    hi = float(np.max(img[finite]))
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)

    out = (img - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


def save_preview_png(
    out_path: Path,
    reflection_norm: np.ndarray,
    denoised_norm: np.ndarray,
    pred_noise_norm: np.ndarray,
    residual_norm: np.ndarray,
    title: str,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        to_uint8_image(reflection_norm, clip01=True),
        to_uint8_image(denoised_norm, clip01=True),
        to_uint8_image(pred_noise_norm, clip01=False),
        to_uint8_image(residual_norm, clip01=False),
    ]

    Image.fromarray(np.concatenate(panels, axis=0), mode="L").save(out_path)


def compute_simple_metrics(result: Dict[str, np.ndarray]) -> Dict[str, float]:
    reflection_norm = result["reflection_norm"]
    denoised_norm = result["denoised_norm"]
    pred_noise_norm = result["pred_noise_norm"]

    residual_norm = denoised_norm - reflection_norm

    return {
        "reflection_mean": float(np.mean(reflection_norm)),
        "reflection_std": float(np.std(reflection_norm)),
        "denoised_mean": float(np.mean(denoised_norm)),
        "denoised_std": float(np.std(denoised_norm)),
        "pred_noise_mean": float(np.mean(pred_noise_norm)),
        "pred_noise_std": float(np.std(pred_noise_norm)),
        "residual_mean": float(np.mean(residual_norm)),
        "residual_std": float(np.std(residual_norm)),
        "denoised_below0_ratio": float(np.mean(denoised_norm < 0.0)),
        "denoised_above1_ratio": float(np.mean(denoised_norm > 1.0)),
    }


def append_metrics_csv(csv_path: Path, row: Dict[str, Any]):
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file",
        "track_id",
        "height",
        "width",
        "reflection_mean",
        "reflection_std",
        "denoised_mean",
        "denoised_std",
        "pred_noise_mean",
        "pred_noise_std",
        "residual_mean",
        "residual_std",
        "denoised_below0_ratio",
        "denoised_above1_ratio",
        "out_npz",
        "out_png",
    ]

    exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    split_json = Path(args.split_json)
    summary_json = Path(args.summary_json)
    checkpoint_path = Path(args.checkpoint)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("outputs") / "inference" / args.split

    npz_out_dir = out_dir / "npz"
    png_out_dir = out_dir / "png"
    metrics_csv = out_dir / "inference_metrics.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n========== Inference Config ==========")
    print(f"device: {device}")
    print(f"data_dir: {data_dir.resolve()}")
    print(f"split: {args.split}")
    print(f"checkpoint: {checkpoint_path.resolve()}")
    print(f"out_dir: {out_dir.resolve()}")

    p1, p99 = load_norm_params(summary_json)
    print(f"log1p norm p1/p99: {p1}, {p99}")

    model, ckpt_args = load_model(checkpoint_path, device=device)

    patch_w = int(args.patch_w)
    if patch_w <= 0:
        patch_w = int(ckpt_args.get("patch_w", 256))

    patch_h = int(args.patch_h)

    print(f"patch_h: {patch_h}")
    print(f"patch_w: {patch_w}")
    print(f"overlap: {args.overlap}")
    print(f"batch_size: {args.batch_size}")
    print(f"amp: {args.amp and device.type == 'cuda'}")

    paths = load_split_files(data_dir=data_dir, split_json=split_json, split=args.split)

    if args.limit and args.limit > 0:
        paths = paths[:args.limit]

    if len(paths) == 0:
        raise RuntimeError("没有找到可推理的 npz 文件。")

    print(f"num files to infer: {len(paths)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_png:
        png_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, npz_path in enumerate(paths, start=1):
        track_id = infer_track_id_from_name(npz_path.name)

        out_npz = npz_out_dir / f"{npz_path.stem}_denoised.npz"
        out_png = png_out_dir / f"{npz_path.stem}_preview.png"

        if out_npz.exists() and not args.overwrite:
            print(f"[{idx}/{len(paths)}] skip existing: {out_npz.name}")
            continue

        print(f"[{idx}/{len(paths)}] infer: {npz_path.name}")

        with np.load(npz_path, allow_pickle=True) as z:
            if "reflection" not in z:
                print(f"[WARN] {npz_path.name} 中没有 reflection，跳过。")
                continue

            reflection_raw = np.asarray(z["reflection"], dtype=np.float32)

            result = infer_full_matrix(
                reflection_raw=reflection_raw,
                model=model,
                device=device,
                p1=p1,
                p99=p99,
                patch_h=patch_h,
                patch_w=patch_w,
                overlap=args.overlap,
                batch_size=args.batch_size,
                amp=args.amp,
            )

            save_inference_npz(
                out_path=out_npz,
                source_npz_path=npz_path,
                source_obj=z,
                reflection_raw=reflection_raw,
                result=result,
                track_id=track_id,
                p1=p1,
                p99=p99,
                checkpoint=checkpoint_path,
            )

        png_saved = ""

        if args.save_png and idx <= args.max_png:
            save_preview_png(
                out_path=out_png,
                reflection_norm=result["reflection_norm"],
                denoised_norm=result["denoised_norm"],
                pred_noise_norm=result["pred_noise_norm"],
                residual_norm=result["residual_norm"],
                title=f"{track_id} | {npz_path.name}",
            )
            png_saved = str(out_png)

        metrics = compute_simple_metrics(result)

        row = {
            "file": npz_path.name,
            "track_id": track_id,
            "height": int(reflection_raw.shape[0]),
            "width": int(reflection_raw.shape[1]),
            **metrics,
            "out_npz": str(out_npz),
            "out_png": png_saved,
        }

        append_metrics_csv(metrics_csv, row)

    print("\n========== Inference Finished ==========")
    print(f"NPZ output: {npz_out_dir.resolve()}")
    if args.save_png:
        print(f"PNG output: {png_out_dir.resolve()}")
    print(f"Metrics CSV: {metrics_csv.resolve()}")


if __name__ == "__main__":
    main()
