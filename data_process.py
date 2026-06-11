from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


def parse_rgram_lbl(lbl_path):
    """
    读取 SHARAD RGRAM 的 PDS3 .LBL 文件，提取读取 IMG 所需信息。
    """
    lbl_path = Path(lbl_path)
    text = lbl_path.read_text(errors="ignore")

    def get_int(key):
        m = re.search(rf"\b{key}\s*=\s*(\d+)", text, flags=re.IGNORECASE)
        if m is None:
            raise ValueError(f"LBL 中没有找到 {key}")
        return int(m.group(1))

    def get_str(key):
        m = re.search(rf"\b{key}\s*=\s*\"?([^\"\n\r]+)\"?", text, flags=re.IGNORECASE)
        if m is None:
            raise ValueError(f"LBL 中没有找到 {key}")
        return m.group(1).strip()

    m = re.search(r"\^IMAGE\s*=\s*\"?([^\"\n\r]+)\"?", text, flags=re.IGNORECASE)
    if m is None:
        raise ValueError("LBL 中没有找到 ^IMAGE")
    image_name = m.group(1).strip()

    lines = get_int("LINES")
    samples = get_int("LINE_SAMPLES")
    sample_type = get_str("SAMPLE_TYPE").upper()
    sample_bits = get_int("SAMPLE_BITS")

    return {
        "image_name": image_name,
        "lines": lines,
        "samples": samples,
        "sample_type": sample_type,
        "sample_bits": sample_bits,
    }


def read_rgram_img_from_lbl(lbl_path):
    """
    根据 RGRAM.LBL 读取对应的 RGRAM.IMG。
    """
    lbl_path = Path(lbl_path)
    info = parse_rgram_lbl(lbl_path)

    img_path = lbl_path.with_name(info["image_name"])
    if not img_path.exists():
        # Windows 大小写不敏感；这里额外兼容大小写敏感文件系统。
        image_name_lower = info["image_name"].lower()
        for candidate in lbl_path.parent.iterdir():
            if candidate.name.lower() == image_name_lower:
                img_path = candidate
                break

    if not img_path.exists():
        raise FileNotFoundError(f"没有找到对应的 IMG 文件：{img_path}")

    sample_type = info["sample_type"]
    sample_bits = info["sample_bits"]

    if sample_type == "PC_REAL" and sample_bits == 32:
        dtype = "<f4"  # 小端 float32
    else:
        raise ValueError(f"暂不支持的数据格式：{sample_type}, {sample_bits} bits")

    data = np.fromfile(img_path, dtype=dtype)
    expected_size = info["lines"] * info["samples"]
    if data.size != expected_size:
        raise ValueError(
            f"文件大小不匹配：读取到 {data.size} 个数，"
            f"但 LBL 期望 {expected_size} 个数"
        )

    data = data.reshape((info["lines"], info["samples"]))
    return data, info


def pick_surface_from_data(img, top_ratio=0.50, bottom_ratio=1.00, window=5):
    """
    在指定纵向范围内寻找每一列的地表线。
    """
    height, width = img.shape

    y_top = max(0, int(height * top_ratio))
    y_bottom = min(height, int(height * bottom_ratio))
    if y_top >= y_bottom:
        raise ValueError("top_ratio 和 bottom_ratio 产生了空的 surface 搜索区域。")

    img_search = img[y_top:y_bottom, :]
    safe_data = np.nan_to_num(img_search, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    if img_search.shape[0] < window:
        surface_y = np.argmax(safe_data, axis=0) + y_top
        return surface_y.astype(int)

    windows = np.lib.stride_tricks.sliding_window_view(
        safe_data,
        window_shape=window,
        axis=0,
    )
    window_strength = np.max(windows, axis=2)
    surface_y = np.argmax(window_strength, axis=0) + window // 2 + y_top

    return surface_y.astype(int)


def subtract_bottom_noise_mean(data, bottom_ratio=0.10):
    """
    使用图像底部区域估计噪声平均值，并从整张图中扣除。
    """
    nline, _ = data.shape
    noise_start = int(nline * (1 - bottom_ratio))

    bottom_noise_mask = np.zeros_like(data, dtype=bool)
    bottom_noise_mask[noise_start:, :] = True

    noise_region = data[bottom_noise_mask]
    if noise_region.size == 0:
        raise ValueError("bottom_ratio 太小，无法构造底部噪声区域。")

    noise_mean = np.nanmean(noise_region)
    if not np.isfinite(noise_mean):
        raise ValueError("底部噪声均值不是有限数，请检查输入数据。")

    data_corrected = data - noise_mean
    return data_corrected, float(noise_mean), bottom_noise_mask


def make_three_masks_simple(
    img,
    surface,
    bottom_noise_ratio=0.10,
    margin_above=20,
    margin_surface=5,
    reflection_depth=180,
):
    """
    生成电离层、地下反射候选区、底部噪声区三个 mask。
    """
    nline, nsample = img.shape
    surface = np.asarray(surface).astype(int)

    if surface.shape[0] != nsample:
        raise ValueError(
            f"surface 长度应该等于 nsample={nsample}，"
            f"但现在 surface 长度是 {surface.shape[0]}"
        )

    ionosphere_mask = np.zeros_like(img, dtype=bool)
    reflection_mask = np.zeros_like(img, dtype=bool)
    noise_mask = np.zeros_like(img, dtype=bool)

    # 1. surface 最上面的点以上，作为电离层/无关区域。
    surface_top = np.nanmin(surface)
    ignore_end = max(0, surface_top - margin_above)
    ionosphere_mask[:ignore_end, :] = True

    # 2. 每一列从 surface 以下提取候选地下反射区。
    for x in range(nsample):
        ys = np.clip(surface[x], 0, nline - 1)
        refl_start = min(nline, ys + margin_surface)
        refl_end = min(nline, ys + reflection_depth)
        reflection_mask[refl_start:refl_end, x] = True

    # 3. 图像底部固定比例作为噪声区。
    noise_start = int(nline * (1 - bottom_noise_ratio))
    noise_mask[noise_start:, :] = True

    return ionosphere_mask, reflection_mask, noise_mask


def extract_reflection_matrix(
    data,
    surface,
    margin_surface=5,
    reflection_depth=180,
    fill_value=0.0,
):
    """
    从 data 中提取沿 surface 对齐的地下反射层小矩阵。
    """
    nline, nsample = data.shape
    surface = np.asarray(surface).astype(int)

    if surface.shape[0] != nsample:
        raise ValueError(
            f"surface 长度应该等于 nsample={nsample}，但现在是 {surface.shape[0]}"
        )

    out_height = reflection_depth - margin_surface
    if out_height <= 0:
        raise ValueError("reflection_depth 必须大于 margin_surface")

    reflection_data = np.full((out_height, nsample), fill_value, dtype=np.float32)
    reflection_valid_mask = np.zeros((out_height, nsample), dtype=bool)

    for x in range(nsample):
        ys = surface[x]
        start_y = ys + margin_surface
        end_y = ys + reflection_depth

        src_start = max(0, start_y)
        src_end = min(nline, end_y)
        if src_start >= src_end:
            continue

        dst_start = src_start - start_y
        dst_end = dst_start + (src_end - src_start)
        reflection_data[dst_start:dst_end, x] = data[src_start:src_end, x]
        reflection_valid_mask[dst_start:dst_end, x] = True

    return reflection_data, reflection_valid_mask


def extract_noise_matrix_from_bottom(
    data,
    target_shape,
    bottom_ratio=0.10,
    random_crop=True,
    seed=None,
):
    """
    从图像底部噪声区提取一个与 target_shape 同尺寸的噪声矩阵。
    """
    rng = np.random.default_rng(seed)

    nline, nsample = data.shape
    target_height, target_width = target_shape

    if target_width != nsample:
        raise ValueError(
            f"target_width 应该等于原图宽度 nsample={nsample}，"
            f"但现在 target_width={target_width}"
        )

    noise_start = int(nline * (1 - bottom_ratio))
    bottom_noise_region = data[noise_start:, :]
    noise_height = bottom_noise_region.shape[0]
    if noise_height <= 0:
        raise ValueError("bottom_ratio 太小，无法获得底部噪声区。")

    # 噪声区高度足够时直接裁剪；不足时有放回抽样补齐。
    if noise_height >= target_height:
        if random_crop:
            max_start = noise_height - target_height
            start = rng.integers(0, max_start + 1)
        else:
            start = noise_height - target_height
        noise_data = bottom_noise_region[start : start + target_height, :]
    else:
        row_idx = rng.integers(0, noise_height, size=target_height)
        noise_data = bottom_noise_region[row_idx, :]

    noise_data = np.asarray(noise_data, dtype=np.float32)
    noise_mean = np.nanmean(noise_data)
    if np.isfinite(noise_mean):
        noise_data = noise_data - noise_mean

    return safe_nan_to_num(noise_data)


def get_track_id_from_path(lbl_path):
    """
    从文件名提取轨道 ID，例如 S_00172701_RGRAM.LBL -> S_00172701。
    """
    stem = Path(lbl_path).stem
    stem = re.sub(r"_rgram$", "", stem, flags=re.IGNORECASE)
    return stem.upper()


def safe_nan_to_num(arr, dtype=np.float32):
    """
    保存给 AI 训练前统一清理 NaN 和 inf。
    """
    arr = np.asarray(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def find_rgram_lbl_files(lbl_dir):
    """
    递归查找 RGRAM .LBL 文件，并按路径去重。
    """
    lbl_dir = Path(lbl_dir)
    if not lbl_dir.exists():
        raise FileNotFoundError(f"lbl_dir 不存在：{lbl_dir}")

    patterns = ("*_RGRAM.LBL", "*_rgram.lbl")
    found = {}
    for pattern in patterns:
        for path in lbl_dir.rglob(pattern):
            if path.is_file():
                found[path.resolve()] = path

    # 额外兼容大小写敏感文件系统上的混合大小写文件名。
    for path in lbl_dir.rglob("*.lbl"):
        if path.is_file() and path.name.lower().endswith("_rgram.lbl"):
            found[path.resolve()] = path

    return sorted(found.values(), key=lambda p: str(p).lower())


def save_dataset_sample(
    out_path,
    reflection,
    noise,
    noisy_reflection,
    mask,
    surface,
    track_id,
    original_shape,
    noise_mean,
    bottom_noise_ratio,
    margin_surface,
    reflection_depth,
    alpha,
):
    """
    将单张 RGRAM 的训练样本保存为压缩 npz。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        reflection=reflection,
        noise=noise,
        noisy_reflection=noisy_reflection,
        mask=mask,
        surface=np.asarray(surface, dtype=np.int32),
        track_id=np.asarray(track_id),
        original_shape=np.asarray(original_shape, dtype=np.int32),
        noise_mean=np.asarray(noise_mean, dtype=np.float32),
        bottom_noise_ratio=np.asarray(bottom_noise_ratio, dtype=np.float32),
        margin_surface=np.asarray(margin_surface, dtype=np.int32),
        reflection_depth=np.asarray(reflection_depth, dtype=np.int32),
        alpha=np.asarray(alpha, dtype=np.float32),
    )


def _iter_with_progress(items, desc="processing"):
    """
    tqdm 可用时使用进度条；不可用时退化为普通 print。
    """
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except ImportError:
        total = len(items)

        def generator():
            for idx, item in enumerate(items, start=1):
                print(f"[{idx}/{total}] {item}")
                yield item

        return generator()


def _write_processing_log(log_path, rows):
    fieldnames = [
        "track_id",
        "lbl_path",
        "out_path",
        "status",
        "original_shape",
        "reflection_shape",
        "noise_mean",
        "error_message",
    ]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def batch_build_dataset(
    lbl_dir,
    out_dir,
    bottom_noise_ratio=0.10,
    margin_surface=5,
    reflection_depth=180,
    alpha=1.0,
    seed=0,
    overwrite=False,
):
    """
    批量处理 RGRAM .LBL 文件，为每张图生成 reflection/noise/noisy_reflection/mask/surface。
    """
    lbl_dir = Path('rgram')
    out_dir = Path('radar_ai_dataset')
    out_dir.mkdir(parents=True, exist_ok=True)

    lbl_files = find_rgram_lbl_files(lbl_dir)
    print(f"找到 {len(lbl_files)} 个 RGRAM LBL 文件。")

    log_rows = []
    for i, lbl_path in enumerate(_iter_with_progress(lbl_files, desc="RGRAM")):
        track_id = get_track_id_from_path(lbl_path)
        out_path = out_dir / f"{track_id}_dataset.npz"

        row = {
            "track_id": track_id,
            "lbl_path": str(lbl_path),
            "out_path": str(out_path),
            "status": "",
            "original_shape": "",
            "reflection_shape": "",
            "noise_mean": "",
            "error_message": "",
        }

        try:
            if out_path.exists() and not overwrite:
                row["status"] = "skipped_exists"
                log_rows.append(row)
                continue

            data, _ = read_rgram_img_from_lbl(lbl_path)
            original_shape = tuple(data.shape)
            row["original_shape"] = str(original_shape)

            surface = pick_surface_from_data(
                data,
                top_ratio=0.50,
                bottom_ratio=1.00,
                window=5,
            )
            if len(surface) != data.shape[1]:
                raise ValueError(
                    f"surface 长度 {len(surface)} 和图像宽度 {data.shape[1]} 不匹配"
                )

            data_corrected, noise_mean, _ = subtract_bottom_noise_mean(
                data,
                bottom_ratio=bottom_noise_ratio,
            )

            # 生成三个 mask 主要用于检查区域定义是否正常。
            make_three_masks_simple(
                data_corrected,
                surface,
                bottom_noise_ratio=bottom_noise_ratio,
                margin_surface=margin_surface,
                reflection_depth=reflection_depth,
            )

            reflection_data, reflection_valid_mask = extract_reflection_matrix(
                data_corrected,
                surface,
                margin_surface=margin_surface,
                reflection_depth=reflection_depth,
                fill_value=0.0,
            )
            noise_data = extract_noise_matrix_from_bottom(
                data_corrected,
                target_shape=reflection_data.shape,
                bottom_ratio=bottom_noise_ratio,
                random_crop=True,
                seed=seed + i,
            )

            if reflection_data.shape != noise_data.shape:
                raise ValueError(
                    f"reflection shape {reflection_data.shape} 和 noise shape {noise_data.shape} 不一致"
                )

            reflection_data = safe_nan_to_num(reflection_data, dtype=np.float32)
            noise_data = safe_nan_to_num(noise_data, dtype=np.float32)
            noisy_reflection = safe_nan_to_num(
                reflection_data + float(alpha) * noise_data,
                dtype=np.float32,
            )
            reflection_valid_mask = np.asarray(reflection_valid_mask, dtype=bool)

            if noisy_reflection.shape != reflection_data.shape:
                raise ValueError(
                    f"noisy_reflection shape {noisy_reflection.shape} 和 reflection shape {reflection_data.shape} 不一致"
                )

            print(
                f"{track_id}: original={original_shape}, "
                f"reflection={reflection_data.shape}, noise={noise_data.shape}, "
                f"noisy_reflection={noisy_reflection.shape}"
            )

            save_dataset_sample(
                out_path=out_path,
                reflection=reflection_data,
                noise=noise_data,
                noisy_reflection=noisy_reflection,
                mask=reflection_valid_mask,
                surface=surface,
                track_id=track_id,
                original_shape=original_shape,
                noise_mean=noise_mean,
                bottom_noise_ratio=bottom_noise_ratio,
                margin_surface=margin_surface,
                reflection_depth=reflection_depth,
                alpha=alpha,
            )

            row["status"] = "success"
            row["reflection_shape"] = str(reflection_data.shape)
            row["noise_mean"] = noise_mean
        except Exception as exc:
            row["status"] = "failed"
            row["error_message"] = repr(exc)
            print(f"处理失败，已跳过：{lbl_path} -> {exc}")

        log_rows.append(row)

    log_path = out_dir / "processing_log.csv"
    _write_processing_log(log_path, log_rows)
    print(f"处理完成。日志已保存到：{log_path}")

    return log_rows


if __name__ == "__main__":
    lbl_dir = Path(r".\rgram_lbl")
    out_dir = Path(r".\radar_ai_dataset")
    batch_build_dataset(
        lbl_dir=lbl_dir,
        out_dir=out_dir,
        bottom_noise_ratio=0.10,
        margin_surface=5,
        reflection_depth=180,
        alpha=1.0,
        seed=0,
        overwrite=False,
    )
