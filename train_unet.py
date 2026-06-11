# train_unet.py
# 用途：
#   训练 Residual U-Net 做 SHARAD 反射层降噪。
#
# 训练任务：
#   noisy = reflection + alpha * noise_scaling_factor * noise
#   model(noisy) -> predicted_noise
#   denoised = noisy - predicted_noise
#   loss = masked SmoothL1(denoised, reflection)
#
# 推荐运行：
#   cd F:\radar_deeplearning
#   F:\FInstallation\anaconda\envs\pytorch\python.exe train_unet.py

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from dataset import build_dataloader
from model_unet import ResidualUNet, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train Residual U-Net for radargram denoising.")

    # 数据路径
    parser.add_argument("--data_dir", type=str, default="radar_ai_dataset")
    parser.add_argument("--split_json", type=str, default="outputs/splits/split.json")
    parser.add_argument("--summary_json", type=str, default="outputs/stats/summary.json")

    # 输出路径
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--preview_dir", type=str, default="outputs/preview")
    parser.add_argument("--log_dir", type=str, default="outputs/logs")

    # patch / dataloader
    parser.add_argument("--patch_h", type=int, default=176)
    parser.add_argument("--patch_w", type=int, default=256)
    parser.add_argument("--train_epoch_size", type=int, default=10000)
    parser.add_argument("--val_epoch_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    # 加噪参数
    parser.add_argument("--alpha_min", type=float, default=0.5)
    parser.add_argument("--alpha_max", type=float, default=1.5)
    parser.add_argument("--noise_scaling_factor", type=float, default=5.0)

    # 模型参数
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--norm", type=str, default="group", choices=["group", "batch", "instance", "none"])
    parser.add_argument("--up_mode", type=str, default="bilinear", choices=["bilinear", "transpose"])
    parser.add_argument("--dropout", type=float, default=0.0)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")

    # 保存与日志
    parser.add_argument("--preview_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)

    # 随机种子
    parser.add_argument("--seed", type=int, default=2026)

    # 断点续训
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    只在 mask == 1 的区域计算 SmoothL1Loss。

    pred, target, mask shape:
        [B, 1, H, W]
    """
    loss_map = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    loss_map = loss_map * mask

    denom = mask.sum().clamp_min(eps)
    return loss_map.sum() / denom


@torch.no_grad()
def compute_batch_metrics(
    denoised: torch.Tensor,
    target: torch.Tensor,
    noisy: torch.Tensor,
    mask: torch.Tensor,
    loss: torch.Tensor,
) -> Dict[str, float]:
    """
    计算一个 batch 的指标。
    所有指标都在归一化空间中计算。
    """
    eps = 1e-8

    valid = mask > 0

    if valid.sum() == 0:
        return {
            "loss": float(loss.item()),
            "mae": float("nan"),
            "mse": float("nan"),
            "psnr": float("nan"),
            "noisy_high_clip_ratio": float("nan"),
            "target_high_clip_ratio": float("nan"),
            "denoised_low_ratio": float("nan"),
            "denoised_high_ratio": float("nan"),
        }

    diff = (denoised - target)[valid]
    mae = diff.abs().mean()
    mse = (diff ** 2).mean()

    psnr = 10.0 * torch.log10(1.0 / (mse + eps))

    noisy_valid = noisy[valid]
    target_valid = target[valid]
    denoised_valid = denoised[valid]

    noisy_high_clip_ratio = (noisy_valid >= 0.999).float().mean()
    target_high_clip_ratio = (target_valid >= 0.999).float().mean()

    # denoised 没有在 loss 前强行 clip，所以这里记录越界比例
    denoised_low_ratio = (denoised_valid < 0.0).float().mean()
    denoised_high_ratio = (denoised_valid > 1.0).float().mean()

    return {
        "loss": float(loss.item()),
        "mae": float(mae.item()),
        "mse": float(mse.item()),
        "psnr": float(psnr.item()),
        "noisy_high_clip_ratio": float(noisy_high_clip_ratio.item()),
        "target_high_clip_ratio": float(target_high_clip_ratio.item()),
        "denoised_low_ratio": float(denoised_low_ratio.item()),
        "denoised_high_ratio": float(denoised_high_ratio.item()),
    }


def update_running_sums(running: Dict[str, float], metrics: Dict[str, float], batch_size: int):
    for k, v in metrics.items():
        if math.isnan(v):
            continue
        running[k] = running.get(k, 0.0) + v * batch_size
    running["num_samples"] = running.get("num_samples", 0.0) + batch_size


def finalize_running_sums(running: Dict[str, float]) -> Dict[str, float]:
    n = max(running.get("num_samples", 1.0), 1.0)
    out = {}
    for k, v in running.items():
        if k == "num_samples":
            continue
        out[k] = v / n
    return out


def make_model(args) -> nn.Module:
    model = ResidualUNet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_channels=args.base_channels,
        depth=args.depth,
        norm=args.norm,
        up_mode=args.up_mode,
        dropout=args.dropout,
    )
    return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_val_loss: float,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "args": vars(args),
    }

    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    return start_epoch, best_val_loss


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


@torch.no_grad()
def save_preview(
    model: nn.Module,
    val_loader,
    device: torch.device,
    epoch: int,
    preview_dir: Path,
    max_items: int = 4,
):
    """
    保存可视化对比图：
        noisy / reflection target / denoised / predicted_noise / residual_error
    """
    model.eval()
    preview_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(val_loader))

    noisy = batch["noisy"].to(device, non_blocking=True)
    target = batch["reflection"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    pred_noise = model(noisy)
    denoised = noisy - pred_noise

    # 只用于显示，clip 到 [0, 1]
    noisy_show = noisy.clamp(0, 1).cpu().numpy()
    target_show = target.clamp(0, 1).cpu().numpy()
    denoised_show = denoised.clamp(0, 1).cpu().numpy()
    pred_noise_show = pred_noise.cpu().numpy()
    residual_error_show = (denoised - target).cpu().numpy()
    mask_show = mask.cpu().numpy()

    files = batch["file"]
    n = min(max_items, noisy.shape[0])

    for i in range(n):
        images = [
            (noisy_show[i, 0], True),
            (target_show[i, 0], True),
            (denoised_show[i, 0], True),
            (pred_noise_show[i, 0], False),
            (residual_error_show[i, 0], False),
        ]

        preview_panels = []
        for img, clip01 in images:
            # mask 外区域显示为 0
            img = img.copy()
            img[mask_show[i, 0] <= 0] = 0

            preview_panels.append(to_uint8_image(img, clip01=clip01))

        safe_file = Path(files[i]).stem

        out_path = preview_dir / f"epoch_{epoch:03d}_{i}_{safe_file}.png"
        Image.fromarray(np.concatenate(preview_panels, axis=0), mode="L").save(out_path)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    args,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    running: Dict[str, float] = {}

    for step, batch in enumerate(loader, start=1):
        noisy = batch["noisy"].to(device, non_blocking=True)
        target = batch["reflection"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp and device.type == "cuda"):
            pred_noise = model(noisy)
            denoised = noisy - pred_noise

            loss = masked_smooth_l1_loss(
                pred=denoised,
                target=target,
                mask=mask,
                beta=0.05,
            )

        scaler.scale(loss).backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        metrics = compute_batch_metrics(
            denoised=denoised.detach(),
            target=target,
            noisy=noisy,
            mask=mask,
            loss=loss.detach(),
        )

        update_running_sums(running, metrics, batch_size=noisy.shape[0])

        if step % 50 == 0:
            print(
                f"Epoch {epoch:03d} | step {step:04d}/{len(loader)} | "
                f"loss={metrics['loss']:.6f} | "
                f"mae={metrics['mae']:.6f} | "
                f"psnr={metrics['psnr']:.2f}"
            )

    return finalize_running_sums(running)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    args,
) -> Dict[str, float]:
    model.eval()

    running: Dict[str, float] = {}

    for batch in loader:
        noisy = batch["noisy"].to(device, non_blocking=True)
        target = batch["reflection"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with autocast(enabled=args.amp and device.type == "cuda"):
            pred_noise = model(noisy)
            denoised = noisy - pred_noise

            loss = masked_smooth_l1_loss(
                pred=denoised,
                target=target,
                mask=mask,
                beta=0.05,
            )

        metrics = compute_batch_metrics(
            denoised=denoised,
            target=target,
            noisy=noisy,
            mask=mask,
            loss=loss,
        )

        update_running_sums(running, metrics, batch_size=noisy.shape[0])

    return finalize_running_sums(running)


def init_log_csv(log_csv: Path):
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    if log_csv.exists():
        return

    fieldnames = [
        "epoch",
        "train_loss",
        "train_mae",
        "train_mse",
        "train_psnr",
        "train_noisy_high_clip_ratio",
        "train_target_high_clip_ratio",
        "train_denoised_low_ratio",
        "train_denoised_high_ratio",
        "val_loss",
        "val_mae",
        "val_mse",
        "val_psnr",
        "val_noisy_high_clip_ratio",
        "val_target_high_clip_ratio",
        "val_denoised_low_ratio",
        "val_denoised_high_ratio",
        "lr",
    ]

    with log_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_log_csv(
    log_csv: Path,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    lr: float,
):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_mae",
        "train_mse",
        "train_psnr",
        "train_noisy_high_clip_ratio",
        "train_target_high_clip_ratio",
        "train_denoised_low_ratio",
        "train_denoised_high_ratio",
        "val_loss",
        "val_mae",
        "val_mse",
        "val_psnr",
        "val_noisy_high_clip_ratio",
        "val_target_high_clip_ratio",
        "val_denoised_low_ratio",
        "val_denoised_high_ratio",
        "lr",
    ]

    row = {
        "epoch": epoch,
        "train_loss": train_metrics.get("loss", np.nan),
        "train_mae": train_metrics.get("mae", np.nan),
        "train_mse": train_metrics.get("mse", np.nan),
        "train_psnr": train_metrics.get("psnr", np.nan),
        "train_noisy_high_clip_ratio": train_metrics.get("noisy_high_clip_ratio", np.nan),
        "train_target_high_clip_ratio": train_metrics.get("target_high_clip_ratio", np.nan),
        "train_denoised_low_ratio": train_metrics.get("denoised_low_ratio", np.nan),
        "train_denoised_high_ratio": train_metrics.get("denoised_high_ratio", np.nan),
        "val_loss": val_metrics.get("loss", np.nan),
        "val_mae": val_metrics.get("mae", np.nan),
        "val_mse": val_metrics.get("mse", np.nan),
        "val_psnr": val_metrics.get("psnr", np.nan),
        "val_noisy_high_clip_ratio": val_metrics.get("noisy_high_clip_ratio", np.nan),
        "val_target_high_clip_ratio": val_metrics.get("target_high_clip_ratio", np.nan),
        "val_denoised_low_ratio": val_metrics.get("denoised_low_ratio", np.nan),
        "val_denoised_high_ratio": val_metrics.get("denoised_high_ratio", np.nan),
        "lr": lr,
    }

    with log_csv.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def save_args(args, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    preview_dir = Path(args.preview_dir)
    log_dir = Path(args.log_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    save_args(args, log_dir / "train_args.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n========== Training Config ==========")
    print(f"device: {device}")
    print(f"amp: {args.amp and device.type == 'cuda'}")
    print(f"data_dir: {Path(args.data_dir).resolve()}")
    print(f"split_json: {Path(args.split_json).resolve()}")
    print(f"summary_json: {Path(args.summary_json).resolve()}")
    print(f"patch size: {args.patch_h} x {args.patch_w}")
    print(f"batch size: {args.batch_size}")
    print(f"epochs: {args.epochs}")
    print(f"lr: {args.lr}")
    print(f"noise_scaling_factor: {args.noise_scaling_factor}")

    print("\n========== Building DataLoaders ==========")

    train_loader = build_dataloader(
        data_dir=args.data_dir,
        split_json=args.split_json,
        summary_json=args.summary_json,
        split="train",
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        epoch_size=args.train_epoch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        noise_scaling_factor=args.noise_scaling_factor,
        pin_memory=(device.type == "cuda"),
        verbose=True,
    )

    val_loader = build_dataloader(
        data_dir=args.data_dir,
        split_json=args.split_json,
        summary_json=args.summary_json,
        split="val",
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        epoch_size=args.val_epoch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        noise_scaling_factor=args.noise_scaling_factor,
        pin_memory=(device.type == "cuda"),
        verbose=True,
    )

    print("\n========== Building Model ==========")

    model = make_model(args).to(device)

    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        print(f"\n[INFO] Resuming from checkpoint: {resume_path.resolve()}")
        start_epoch, best_val_loss = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        print(f"[INFO] start_epoch={start_epoch}, best_val_loss={best_val_loss}")

    log_csv = log_dir / "train_log.csv"
    init_log_csv(log_csv)

    print("\n========== Start Training ==========")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n---------- Epoch {epoch}/{args.epochs} ----------")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
        )

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:03d} summary:\n"
            f"  train loss = {train_metrics['loss']:.6f}, "
            f"mae = {train_metrics['mae']:.6f}, "
            f"psnr = {train_metrics['psnr']:.2f}\n"
            f"  val   loss = {val_metrics['loss']:.6f}, "
            f"mae = {val_metrics['mae']:.6f}, "
            f"psnr = {val_metrics['psnr']:.2f}\n"
            f"  train noisy_clip = {train_metrics['noisy_high_clip_ratio']:.4f}, "
            f"target_clip = {train_metrics['target_high_clip_ratio']:.4f}\n"
            f"  val   noisy_clip = {val_metrics['noisy_high_clip_ratio']:.4f}, "
            f"target_clip = {val_metrics['target_high_clip_ratio']:.4f}\n"
            f"  val denoised below0 = {val_metrics['denoised_low_ratio']:.4f}, "
            f"above1 = {val_metrics['denoised_high_ratio']:.4f}"
        )

        append_log_csv(
            log_csv=log_csv,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=lr,
        )

        # 保存 last
        save_checkpoint(
            path=checkpoint_dir / "last_unet.pth",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            args=args,
        )

        # 保存 best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                path=checkpoint_dir / "best_unet.pth",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
            )
            print(f"[INFO] New best model saved. best_val_loss={best_val_loss:.6f}")

        # 定期保存 epoch checkpoint
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch:03d}_unet.pth",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
            )

        # 定期保存 preview 图
        if args.preview_every > 0 and epoch % args.preview_every == 0:
            save_preview(
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                preview_dir=preview_dir,
                max_items=4,
            )
            print(f"[INFO] Preview saved to: {preview_dir.resolve()}")

    print("\n========== Training Finished ==========")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {str((checkpoint_dir / 'best_unet.pth').resolve())}")
    print(f"Last checkpoint: {str((checkpoint_dir / 'last_unet.pth').resolve())}")
    print(f"Log CSV: {str(log_csv.resolve())}")


if __name__ == "__main__":
    main()
