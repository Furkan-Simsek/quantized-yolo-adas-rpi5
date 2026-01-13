#!/usr/bin/env python3
# convert.py
"""
YOLO (.pt, Ultralytics) -> (optional) unstructured pruning -> ONNX export
and optional INT8 quantization for ONNX.

Why this exists:
- PyTorch 2.6+ torch.load defaults to weights_only=True and can fail on Ultralytics checkpoints
- This script loads via Ultralytics YOLO loader (robust), prunes in PyTorch, exports via Ultralytics,
  then can quantize the exported ONNX with ONNX Runtime.

Examples
--------
1) Prune + export ONNX:
python convert.py prune_export \
  --pt best.pt \
  --out pruned.onnx \
  --prune_amount 0.3 \
  --imgsz 640 \
  --opset 17 \
  --dynamic \
  --simplify

2) Export ONNX without pruning:
python convert.py prune_export --pt best.pt --out model.onnx --prune_amount 0

3) Quantize ONNX (dynamic INT8):
python convert.py quantize \
  --mode dynamic \
  --in pruned.onnx \
  --out pruned_int8.onnx \
  --per_channel

4) Quantize ONNX (static INT8 with calibration):
python convert.py quantize \
  --mode static \
  --in pruned.onnx \
  --out pruned_int8_static.onnx \
  --calib_dir ./calib_images \
  --imgsz 640 \
  --num_calib 200 \
  --per_channel \
  --qformat QDQ
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# -----------------------------
# Pruning (PyTorch)
# -----------------------------
def apply_unstructured_pruning_torch(model, amount: float) -> None:
    """
    Apply magnitude-based (L1) unstructured pruning to Conv2d/Linear layers.
    Then make pruning permanent (remove reparametrization).
    """
    if amount <= 0:
        return

    import torch.nn as nn
    import torch.nn.utils.prune as prune

    # Apply pruning
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)

    # Make it permanent
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight_orig"):
            prune.remove(module, "weight")


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _pick_exported_onnx_file(export_dir: Path, since_ts: float) -> Path:
    """
    Ultralytics export often writes into export_dir with a timestamped folder.
    We select the newest .onnx created after since_ts.
    """
    candidates = []
    for onnx_path in export_dir.rglob("*.onnx"):
        try:
            if onnx_path.stat().st_mtime >= since_ts - 1.0:
                candidates.append(onnx_path)
        except FileNotFoundError:
            pass
    if not candidates:
        # fallback: just pick newest .onnx in tree
        for onnx_path in export_dir.rglob("*.onnx"):
            candidates.append(onnx_path)
    if not candidates:
        raise FileNotFoundError(f"No .onnx found under: {export_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def prune_and_export_onnx_ultralytics(
    pt_path: Path,
    out_onnx: Path,
    prune_amount: float,
    imgsz: int,
    opset: int,
    dynamic: bool,
    simplify: bool,
    device: str,
) -> Path:
    """
    Load YOLO via Ultralytics, prune in torch, export ONNX via Ultralytics, then move to out_onnx.
    """
    from ultralytics import YOLO

    if not pt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

    _ensure_parent_dir(out_onnx)

    y = YOLO(str(pt_path))          # robust loader for Ultralytics checkpoints
    model = y.model                 # torch.nn.Module
    model.to(device).eval()

    # Pruning
    apply_unstructured_pruning_torch(model, prune_amount)

    # Ensure Ultralytics object uses pruned model
    y.model = model

    # Export
    export_base_dir = out_onnx.parent.resolve()
    since = time.time()

    # Ultralytics export options:
    # - format="onnx"
    # - imgsz
    # - opset
    # - dynamic
    # - simplify
    # - device
    res = y.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        device=device,
        project=str(export_base_dir),  # write export artifacts under output folder
        name="ultralytics_export",     # subfolder name
    )

    # Determine produced onnx file
    produced: Optional[Path] = None
    try:
        # some versions return a dict-like with "file"
        if isinstance(res, dict) and "file" in res:
            produced = Path(res["file"])
    except Exception:
        produced = None

    if produced is None:
        # search under export_base_dir/ultralytics_export*
        search_root = export_base_dir / "ultralytics_export"
        if not search_root.exists():
            # fallback: search under export_base_dir
            search_root = export_base_dir
        produced = _pick_exported_onnx_file(search_root, since_ts=since)

    produced = produced.resolve()

    # Copy/move to requested out path (keep produced as artifact if you want; here we copy)
    if produced.resolve() != out_onnx.resolve():
    	shutil.copy2(produced, out_onnx)

    return out_onnx.resolve()


# -----------------------------
# Quantization (ONNX Runtime)
# -----------------------------
def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return paths


def preprocess_image_cv2(path: Path, imgsz: int) -> np.ndarray:
    """
    Simple calibration preprocessing:
    - resize to imgsz x imgsz
    - BGR->RGB
    - float32 scale 0..1
    - NCHW
    Adjust to match your model pipeline if needed (letterbox/mean-std/etc.).
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC->CHW
    img = np.expand_dims(img, axis=0)   # add batch
    return img


class ImageFolderCalibrationReader:
    """
    ONNX Runtime CalibrationDataReader-compatible adapter (duck-typing).
    """
    def __init__(self, image_paths: List[Path], imgsz: int, input_name: str = "images"):
        self.image_paths = image_paths
        self.imgsz = imgsz
        self.input_name = input_name
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self.image_paths):
            return None
        p = self.image_paths[self._idx]
        self._idx += 1
        arr = preprocess_image_cv2(p, self.imgsz)
        return {self.input_name: arr}


def quantize_onnx(
    in_path: Path,
    out_path: Path,
    mode: str,
    per_channel: bool,
    qformat: str,
    weights: str,
    activations: str,
    calib_dir: Optional[Path],
    imgsz: int,
    num_calib: int,
    seed: int,
    input_name: str,
) -> Path:
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        QuantType,
        QuantFormat,
    )

    _ensure_parent_dir(out_path)

    qf = QuantFormat.QDQ if qformat.upper() == "QDQ" else QuantFormat.QOperator
    wtype = QuantType.QInt8 if weights == "QInt8" else QuantType.QUInt8
    atype = QuantType.QInt8 if activations == "QInt8" else QuantType.QUInt8

    if mode == "dynamic":
        quantize_dynamic(
            model_input=str(in_path),
            model_output=str(out_path),
            weight_type=wtype,
            per_channel=per_channel,
            reduce_range=False,
        )
        return out_path.resolve()

    # static
    if calib_dir is None:
        raise ValueError("--calib_dir is required for static quantization")

    imgs = list_images(calib_dir)
    if not imgs:
        raise FileNotFoundError(f"No calibration images found under: {calib_dir}")

    random.seed(seed)
    random.shuffle(imgs)
    imgs = imgs[: max(1, min(num_calib, len(imgs)))]

    reader = ImageFolderCalibrationReader(imgs, imgsz=imgsz, input_name=input_name)

    quantize_static(
        model_input=str(in_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=qf,
        activation_type=atype,
        weight_type=wtype,
        per_channel=per_channel,
        reduce_range=False,
    )
    return out_path.resolve()


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="convert.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    # prune_export
    p1 = sub.add_parser("prune_export", help="Load Ultralytics .pt, prune, export ONNX")
    p1.add_argument("--pt", required=True, help="Ultralytics YOLO checkpoint (.pt)")
    p1.add_argument("--out", required=True, help="Output ONNX path")
    p1.add_argument("--prune_amount", type=float, default=0.3, help="0..1 unstructured pruning fraction (0 disables)")
    p1.add_argument("--imgsz", type=int, default=640, help="Export image size")
    p1.add_argument("--opset", type=int, default=17, help="ONNX opset")
    p1.add_argument("--dynamic", action="store_true", help="Dynamic axes")
    p1.add_argument("--simplify", action="store_true", help="Simplify ONNX (Ultralytics simplify)")
    p1.add_argument("--device", default="cuda" if _has_cuda() else "cpu", help="cuda or cpu")

    # quantize
    p2 = sub.add_parser("quantize", help="Quantize an ONNX model to INT8")
    p2.add_argument("--mode", choices=["dynamic", "static"], required=True)
    p2.add_argument("--in", dest="in_path", required=True, help="Input ONNX path")
    p2.add_argument("--out", dest="out_path", required=True, help="Output quantized ONNX path")
    p2.add_argument("--per_channel", action="store_true", help="Per-channel weights quantization")
    p2.add_argument("--qformat", choices=["QDQ", "QOperator"], default="QDQ")
    p2.add_argument("--weights", choices=["QInt8", "QUInt8"], default="QInt8")
    p2.add_argument("--activations", choices=["QInt8", "QUInt8"], default="QUInt8")

    # calibration
    p2.add_argument("--calib_dir", default=None, help="Folder of calibration images (required for static)")
    p2.add_argument("--imgsz", type=int, default=640)
    p2.add_argument("--num_calib", type=int, default=200)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--input_name", default="images", help="ONNX input name (often 'images' for YOLO exports)")

    return p


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main() -> None:
    args = build_argparser().parse_args()

    if args.cmd == "prune_export":
        out = prune_and_export_onnx_ultralytics(
            pt_path=Path(args.pt),
            out_onnx=Path(args.out),
            prune_amount=float(args.prune_amount),
            imgsz=int(args.imgsz),
            opset=int(args.opset),
            dynamic=bool(args.dynamic),
            simplify=bool(args.simplify),
            device=str(args.device),
        )
        print(f"[OK] Pruned+Exported ONNX: {out}")
        return

    if args.cmd == "quantize":
        out = quantize_onnx(
            in_path=Path(args.in_path),
            out_path=Path(args.out_path),
            mode=str(args.mode),
            per_channel=bool(args.per_channel),
            qformat=str(args.qformat),
            weights=str(args.weights),
            activations=str(args.activations),
            calib_dir=Path(args.calib_dir) if args.calib_dir else None,
            imgsz=int(args.imgsz),
            num_calib=int(args.num_calib),
            seed=int(args.seed),
            input_name=str(args.input_name),
        )
        print(f"[OK] Quantized ONNX: {out}")
        return


if __name__ == "__main__":
    main()

