#!/usr/bin/env python3
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RUN_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)_(\d+)$")


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value or "").strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_.")
    return cleaned or "model"


def parse_step_and_index(filename: str, fallback_index: int):
    m = re.search(r"_e(\d{6})_(\d{2})_", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"_(\d{6,8})_(\d{2})_", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, int(fallback_index % 100)


def find_or_create_run_dir(logs_root: Path, model_name: str) -> Path:
    model_token = sanitize(model_name)
    candidates = []
    for entry in logs_root.iterdir():
        if not entry.is_dir():
            continue
        match = RUN_RE.match(entry.name)
        if not match:
            continue
        if match.group(2) != model_token:
            continue
        try:
            candidates.append((entry.stat().st_mtime, int(match.group(3)), entry))
        except Exception:
            candidates.append((0.0, int(match.group(3)), entry))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    today = datetime.now().strftime("%Y-%m-%d")
    prefix = f"{today}_{model_token}_"
    max_idx = 0
    for entry in logs_root.iterdir():
        if not entry.is_dir():
            continue
        match = RUN_RE.match(entry.name)
        if not match:
            continue
        if match.group(1) == today and match.group(2) == model_token:
            max_idx = max(max_idx, int(match.group(3)))
    run_dir = logs_root / f"{prefix}{max_idx + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_processed_paths(manifest_path: Path):
    if not manifest_path.exists():
        return set()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if isinstance(data, dict) and isinstance(data.get("paths"), list):
        return set(str(x) for x in data["paths"])
    return set()


def save_processed_paths(manifest_path: Path, paths):
    manifest_path.write_text(
        json.dumps({"paths": sorted(paths)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sync_once(output_root: Path, logs_root: Path):
    total_added = 0
    for model_dir in sorted(output_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "logs":
            continue
        sample_dir = model_dir / "sample"
        if not sample_dir.exists() or not sample_dir.is_dir():
            continue

        images = [p for p in sorted(sample_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not images:
            continue

        run_dir = find_or_create_run_dir(logs_root, model_dir.name)
        tb_dir = run_dir / "network_train"
        tb_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = run_dir / ".sample_backfill_manifest.json"
        processed = load_processed_paths(manifest_path)
        writer = SummaryWriter(log_dir=str(tb_dir))
        added = 0
        try:
            for i, img_path in enumerate(images):
                abs_path = str(img_path.resolve())
                if abs_path in processed:
                    continue

                step, prompt_idx = parse_step_and_index(img_path.name, i)
                try:
                    with Image.open(img_path) as im:
                        arr = np.asarray(im.convert("RGB"))
                except Exception:
                    continue

                writer.add_image(f"sample/{prompt_idx:02d}", arr, global_step=step, dataformats="HWC")
                processed.add(abs_path)
                added += 1
        finally:
            writer.flush()
            writer.close()

        if added:
            print(f"[sample-sync] {model_dir.name}: +{added} -> {run_dir.name}", flush=True)
            save_processed_paths(manifest_path, processed)
            total_added += added

    return total_added


def main():
    parser = argparse.ArgumentParser(description="Sync output/*/sample images into TensorBoard in near real time")
    parser.add_argument("--output-root", type=str, default="output")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    if args.once:
        sync_once(output_root, logs_root)
        return

    print(f"[sample-sync] watching {output_root} every {args.poll_interval:.1f}s", flush=True)
    while True:
        try:
            sync_once(output_root, logs_root)
        except Exception as e:
            print(f"[sample-sync] error: {e}", flush=True)
        time.sleep(max(1.0, float(args.poll_interval)))


if __name__ == "__main__":
    main()
