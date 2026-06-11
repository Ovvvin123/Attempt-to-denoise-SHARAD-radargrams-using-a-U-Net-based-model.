from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data_process import read_rgram_img_from_lbl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project denoised aligned reflection matrices back to original RGRAM image coordinates."
    )
    parser.add_argument("--inference_dir", type=str, default="outputs/inference/test/npz")
    parser.add_argument("--rgram_dir", type=str, default="rgram")
    parser.add_argument("--source_data_dir", type=str, default="radar_ai_dataset")
    parser.add_argument("--out_dir", type=str, default="outputs/inference_original/test")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--save_png", action="store_true", default=True)
    parser.add_argument("--no_png", action="store_false", dest="save_png")
    parser.add_argument(
        "--add_noise_mean",
        action="store_true",
        default=True,
        help="Add stored noise_mean back so projected values are in original RGRAM intensity scale.",
    )
    parser.add_argument("--no_add_noise_mean", action="store_false", dest="add_noise_mean")
    return parser.parse_args()


def scalar_to_str(x: Any) -> str:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return str(x.item())
        if x.size == 1:
            return str(x.reshape(-1)[0].item())
    return str(x)


def scalar_to_int(x: Any, default: int) -> int:
    try:
        if isinstance(x, np.ndarray):
            if x.shape == ():
                return int(x.item())
            if x.size == 1:
                return int(x.reshape(-1)[0].item())
        return int(x)
    except Exception:
        return default


def scalar_to_float(x: Any, default: float) -> float:
    try:
        if isinstance(x, np.ndarray):
            if x.shape == ():
                return float(x.item())
            if x.size == 1:
                return float(x.reshape(-1)[0].item())
        return float(x)
    except Exception:
        return default


def infer_track_id(path: Path, z) -> str:
    if "track_id" in z.files:
        value = scalar_to_str(z["track_id"]).upper()
        if value:
            return value

    stem = path.stem
    match = re.search(r"(S_\d{8})", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"(\d{8})", stem)
    if match:
        return "S_" + match.group(1)

    return stem.upper()


def find_rgram_lbl(rgram_dir: Path, track_id: str) -> Path:
    wanted = f"{track_id}_rgram.lbl".lower()
    for path in rgram_dir.rglob("*.lbl"):
        if path.name.lower() == wanted:
            return path
    raise FileNotFoundError(f"Cannot find RGRAM LBL for {track_id} under {rgram_dir}")


def find_source_dataset(source_data_dir: Path, source_file: str, track_id: str) -> Path | None:
    candidates = []
    if source_file:
        candidates.append(source_data_dir / source_file)
    candidates.append(source_data_dir / f"{track_id}_dataset.npz")

    for path in candidates:
        if path.exists():
            return path
    return None


def load_margin_surface(source_npz: Path | None, default: int = 5) -> int:
    if source_npz is None:
        return default
    with np.load(source_npz, allow_pickle=True) as z:
        if "margin_surface" in z.files:
            return scalar_to_int(z["margin_surface"], default=default)
    return default


def to_uint8_stretch(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    lo, hi = np.percentile(arr[finite], [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(arr[finite]))
        hi = float(np.max(arr[finite]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


def project_to_original(
    original: np.ndarray,
    denoised_aligned: np.ndarray,
    surface: np.ndarray,
    margin_surface: int,
    value_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = np.asarray(original, dtype=np.float32)
    denoised_aligned = np.asarray(denoised_aligned, dtype=np.float32)
    surface = np.asarray(surface, dtype=np.int32)

    out = original.copy()
    region = np.full(original.shape, np.nan, dtype=np.float32)
    mask = np.zeros(original.shape, dtype=bool)

    height, width = original.shape
    aligned_h, aligned_w = denoised_aligned.shape
    usable_w = min(width, aligned_w, surface.shape[0])

    for x in range(usable_w):
        start_y = int(surface[x]) + int(margin_surface)
        if start_y >= height:
            continue

        src_start = 0
        dst_start = start_y

        if dst_start < 0:
            src_start = -dst_start
            dst_start = 0

        copy_h = min(aligned_h - src_start, height - dst_start)
        if copy_h <= 0:
            continue

        values = denoised_aligned[src_start : src_start + copy_h, x] + value_offset
        out[dst_start : dst_start + copy_h, x] = values
        region[dst_start : dst_start + copy_h, x] = values
        mask[dst_start : dst_start + copy_h, x] = True

    return out, region, mask


def process_one(inference_npz: Path, args) -> Path:
    rgram_dir = Path(args.rgram_dir)
    source_data_dir = Path(args.source_data_dir)
    out_dir = Path(args.out_dir)
    npz_out_dir = out_dir / "npz"
    png_out_dir = out_dir / "png"
    npz_out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_png:
        png_out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(inference_npz, allow_pickle=True) as z:
        track_id = infer_track_id(inference_npz, z)
        out_npz = npz_out_dir / f"{track_id}_original_projected.npz"
        if out_npz.exists() and not args.overwrite:
            return out_npz

        if "denoised_raw_approx" not in z.files:
            raise KeyError(f"{inference_npz} has no denoised_raw_approx")
        if "surface" not in z.files:
            raise KeyError(f"{inference_npz} has no surface")

        denoised = np.asarray(z["denoised_raw_approx"], dtype=np.float32)
        surface = np.asarray(z["surface"], dtype=np.int32)
        source_file = scalar_to_str(z["source_file"]) if "source_file" in z.files else ""
        noise_mean = scalar_to_float(z["noise_mean"], default=0.0) if "noise_mean" in z.files else 0.0

    source_npz = find_source_dataset(source_data_dir, source_file, track_id)
    margin_surface = load_margin_surface(source_npz, default=5)
    lbl_path = find_rgram_lbl(rgram_dir, track_id)
    original, _ = read_rgram_img_from_lbl(lbl_path)
    original = np.asarray(original, dtype=np.float32)

    value_offset = noise_mean if args.add_noise_mean else 0.0
    projected, projected_region, projection_mask = project_to_original(
        original=original,
        denoised_aligned=denoised,
        surface=surface,
        margin_surface=margin_surface,
        value_offset=value_offset,
    )

    np.savez_compressed(
        out_npz,
        track_id=np.asarray(track_id),
        source_inference=np.asarray(str(inference_npz)),
        source_dataset=np.asarray(str(source_npz) if source_npz is not None else ""),
        rgram_lbl=np.asarray(str(lbl_path)),
        margin_surface=np.asarray(margin_surface, dtype=np.int32),
        noise_mean=np.asarray(noise_mean, dtype=np.float32),
        added_noise_mean=np.asarray(bool(args.add_noise_mean)),
        original=original.astype(np.float32),
        denoised_projected=projected.astype(np.float32),
        denoised_projected_region=projected_region.astype(np.float32),
        projection_mask=projection_mask,
        surface=surface.astype(np.int32),
    )

    if args.save_png:
        original_u8 = to_uint8_stretch(original)
        projected_u8 = to_uint8_stretch(projected)
        mask_u8 = np.where(projection_mask, 255, 0).astype(np.uint8)
        preview = np.concatenate([original_u8, projected_u8, mask_u8], axis=0)
        Image.fromarray(preview, mode="L").save(png_out_dir / f"{track_id}_original_projected_preview.png")

    return out_npz


def main():
    args = parse_args()
    inference_dir = Path(args.inference_dir)
    paths = sorted(inference_dir.glob("*_denoised.npz"))
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(f"No *_denoised.npz files found in {inference_dir}")

    print(f"Projecting {len(paths)} files")
    for idx, path in enumerate(paths, start=1):
        try:
            out_path = process_one(path, args)
            print(f"[{idx}/{len(paths)}] {path.name} -> {out_path}")
        except Exception as exc:
            print(f"[WARN] failed: {path} -> {exc}")


if __name__ == "__main__":
    main()
