#!/usr/bin/env python3
"""Prepare an MMCV epoch checkpoint for a longer, rebatched run.

MMCV 1.4's epoch cosine hook uses ``runner.epoch / runner.max_epochs`` and
reads each optimizer param group's ``initial_lr``.  An ``epoch_4.pth`` file
was saved after the epoch whose zero-based scheduler progress was 3, but a
resumed run starts with ``runner.epoch == 4``.  If max_epochs is changed from
8 to 48 without adapting ``initial_lr``, the first resumed epoch would jump
from the last-used LR to the value at 4/48.

This tool makes a new checkpoint whose initial LR is chosen so that MMCV's
first ``before_train_epoch`` calculation exactly reproduces every group's
saved, last-used LR.  It also adjusts ``meta.iter`` for a changed distributed
batch size.  Dry-run is the default; writing requires both ``--write`` and a
new, non-existing ``--output`` path.  The input is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterable, List, Sequence, Tuple


_GPU_RANGE_RE = re.compile(
    r"^(?P<prefix>\s*gpu_ids\s*=\s*)range\(\s*0\s*,\s*(?P<count>\d+)\s*\)"
    r"(?P<suffix>\s*(?:#.*)?)$",
    re.MULTILINE,
)


def cosine_factor(progress: int, max_progress: int,
                  min_lr_ratio: float) -> float:
    """Return MMCV 1.4 CosineAnnealingLrUpdaterHook's LR/base-LR ratio."""
    if max_progress <= 0:
        raise ValueError("max_progress must be positive")
    if not 0 <= progress < max_progress:
        raise ValueError(
            f"progress must satisfy 0 <= progress < max_progress; got "
            f"{progress}/{max_progress}")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    cosine = (1.0 + math.cos(math.pi * progress / max_progress)) / 2.0
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def distributed_group_iters(group_sizes: Sequence[int], world_size: int,
                            samples_per_gpu: int) -> int:
    """Compute DataLoader length for MMDet's DistributedGroupSampler.

    Each flag group is padded independently to a multiple of
    ``world_size * samples_per_gpu``.
    """
    if world_size <= 0 or samples_per_gpu <= 0:
        raise ValueError("world_size and samples_per_gpu must be positive")
    if not group_sizes or any(size < 0 for size in group_sizes):
        raise ValueError("group_sizes must contain non-negative integers")
    if sum(group_sizes) <= 0:
        raise ValueError("at least one group must be non-empty")
    denominator = world_size * samples_per_gpu
    return sum((size + denominator - 1) // denominator
               for size in group_sizes)


def _optimizer_state_dicts(optimizer: Any) -> Iterable[Tuple[str, Dict]]:
    if isinstance(optimizer, dict) and isinstance(
            optimizer.get("param_groups"), list):
        yield "optimizer", optimizer
        return
    if isinstance(optimizer, dict):
        found = False
        for name, state in optimizer.items():
            if isinstance(state, dict) and isinstance(
                    state.get("param_groups"), list):
                found = True
                yield str(name), state
        if found:
            return
    raise ValueError("checkpoint optimizer has no param_groups list")


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to read a checkpoint") from exc

    try:
        checkpoint = torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:  # PyTorch versions predating the mmap argument.
        checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a dict")
    return checkpoint


def _source_stat(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patch_embedded_gpu_count(config: Any,
                              world_size: int) -> Tuple[Any, int | None, bool]:
    """Prevent MMCV resume() from re-scaling an already corrected meta.iter."""
    if not isinstance(config, str):
        return config, None, False
    matches = list(_GPU_RANGE_RE.finditer(config))
    if len(matches) != 1:
        return config, None, False
    previous = int(matches[0].group("count"))
    if previous == world_size:
        return config, previous, False

    def replacement(match: re.Match) -> str:
        return (f'{match.group("prefix")}range(0, {world_size})'
                f'{match.group("suffix")}')

    return _GPU_RANGE_RE.sub(replacement, config, count=1), previous, True


def _materialize_runtime_config(path: Path, *, max_epochs: int,
                                min_lr_ratio: float, world_size: int,
                                samples_per_gpu: int,
                                warmup_iters: int) -> Tuple[str, Dict[str, Any]]:
    """Load, validate, and resolve the config stored in future checkpoints."""
    try:
        from mmcv import Config
    except ImportError as exc:
        raise RuntimeError(
            "MMCV is required when --runtime-config is used") from exc

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = Config.fromfile(str(path))
    expected = {
        "total_epochs": max_epochs,
        "runner.max_epochs": max_epochs,
        "model.pts_bbox_head.tot_epoch": max_epochs,
        "data.samples_per_gpu": samples_per_gpu,
        "lr_config.policy": "CosineAnnealing",
        "lr_config.by_epoch": True,
        "lr_config.min_lr_ratio": min_lr_ratio,
        "lr_config.warmup_iters": warmup_iters,
        "evaluation.interval": max_epochs + 1,
        "checkpoint_config.max_keep_ckpts": max_epochs,
    }
    actual = {
        "total_epochs": cfg.get("total_epochs"),
        "runner.max_epochs": cfg.get("runner", {}).get("max_epochs"),
        "model.pts_bbox_head.tot_epoch": cfg.get("model", {}).get(
            "pts_bbox_head", {}).get("tot_epoch"),
        "data.samples_per_gpu": cfg.get("data", {}).get("samples_per_gpu"),
        "lr_config.policy": cfg.get("lr_config", {}).get("policy"),
        "lr_config.by_epoch": cfg.get("lr_config", {}).get("by_epoch", True),
        "lr_config.min_lr_ratio": cfg.get("lr_config", {}).get(
            "min_lr_ratio"),
        "lr_config.warmup_iters": cfg.get("lr_config", {}).get(
            "warmup_iters"),
        "evaluation.interval": cfg.get("evaluation", {}).get("interval"),
        "checkpoint_config.max_keep_ckpts": cfg.get(
            "checkpoint_config", {}).get("max_keep_ckpts"),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(
            "runtime config does not match resume arguments:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True))

    # tools/train.py normally adds this after loading a config. Materializing
    # it here makes MMCV resume's future world-size check deterministic.
    cfg.gpu_ids = range(world_size)
    return cfg.pretty_text, {
        "path": str(path),
        "validated_values": actual,
        "gpu_ids": list(range(world_size)),
    }


def prepare(checkpoint: Dict[str, Any], *, max_epochs: int,
            min_lr_ratio: float, iters_per_epoch: int,
            world_size: int, warmup_iters: int) -> Dict[str, Any]:
    """Mutate an in-memory checkpoint and return a JSON-serializable report."""
    meta = checkpoint.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("checkpoint must contain a meta dict")
    if "optimizer" not in checkpoint:
        raise ValueError("checkpoint must contain optimizer state to resume")

    epoch = meta.get("epoch")
    old_iter = meta.get("iter")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"meta.epoch must be a non-negative int; got {epoch!r}")
    if not isinstance(old_iter, int) or old_iter < 0:
        raise ValueError(f"meta.iter must be a non-negative int; got {old_iter!r}")
    if iters_per_epoch <= 0:
        raise ValueError("iters_per_epoch must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")

    factor = cosine_factor(epoch, max_epochs, min_lr_ratio)
    optimizer_reports: List[Dict[str, Any]] = []
    total_groups = 0
    total_states = 0
    for name, optimizer in _optimizer_state_dicts(checkpoint["optimizer"]):
        groups = optimizer["param_groups"]
        current_lrs: List[float] = []
        new_initial_lrs: List[float] = []
        group_param_ids = set()
        for index, group in enumerate(groups):
            if "lr" not in group:
                raise ValueError(f"{name} param group {index} has no lr")
            current_lr = float(group["lr"])
            if not math.isfinite(current_lr) or current_lr < 0:
                raise ValueError(
                    f"{name} param group {index} has invalid lr {current_lr}")
            new_initial_lr = current_lr / factor
            # Keep group['lr'] at its saved value. before_train_epoch will set
            # the same value from this new initial_lr, up to float precision.
            group["initial_lr"] = new_initial_lr
            current_lrs.append(current_lr)
            new_initial_lrs.append(new_initial_lr)
            group_param_ids.update(group.get("params", []))

        states = optimizer.get("state", {})
        state_param_ids = set(states) if isinstance(states, dict) else set()
        total_groups += len(groups)
        total_states += len(states) if isinstance(states, dict) else 0
        optimizer_reports.append({
            "name": name,
            "param_groups": len(groups),
            "state_entries": len(states) if isinstance(states, dict) else None,
            "param_ids_without_state": len(group_param_ids - state_param_ids),
            "saved_lr_values": sorted(set(current_lrs)),
            "new_initial_lr_values": sorted(set(new_initial_lrs)),
        })

    new_iter = epoch * iters_per_epoch
    meta["iter"] = new_iter
    patched_config, embedded_world_size, gpu_ids_patched = (
        _patch_embedded_gpu_count(meta.get("config"), world_size))
    meta["config"] = patched_config

    report: Dict[str, Any] = {
        "epoch": epoch,
        "old_meta_iter": old_iter,
        "new_meta_iter": new_iter,
        "iters_per_epoch": iters_per_epoch,
        "max_epochs": max_epochs,
        "min_lr_ratio": min_lr_ratio,
        "resume_cosine_factor": factor,
        "optimizer_param_groups": total_groups,
        "optimizer_state_entries": total_states,
        "optimizers": optimizer_reports,
        "runtime_world_size": world_size,
        "embedded_checkpoint_world_size": embedded_world_size,
        "embedded_gpu_ids_patched": gpu_ids_patched,
        "warmup_iters": warmup_iters,
        "warmup_will_restart": new_iter < warmup_iters,
        "required_runtime_config": {
            "total_epochs": max_epochs,
            "runner.max_epochs": max_epochs,
            "model.pts_bbox_head.tot_epoch": max_epochs,
            "evaluation.interval": max_epochs + 1,
            "checkpoint_config.max_keep_ckpts": max_epochs,
        },
    }
    meta["resume_preparation"] = {
        key: report[key]
        for key in (
            "epoch", "old_meta_iter", "new_meta_iter", "iters_per_epoch",
            "max_epochs", "min_lr_ratio", "resume_cosine_factor",
            "runtime_world_size", "embedded_checkpoint_world_size",
            "embedded_gpu_ids_patched")
    }
    return report


def _parse_group_sizes(value: str) -> List[int]:
    try:
        sizes = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "group sizes must be comma-separated integers") from exc
    if not sizes or any(size < 0 for size in sizes) or sum(sizes) <= 0:
        raise argparse.ArgumentTypeError(
            "group sizes must be non-negative and sum to a positive value")
    return sizes


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _save_new_checkpoint(checkpoint: Dict[str, Any], output: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to write a checkpoint") from exc

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
        delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(checkpoint, str(temporary))
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _verification_snapshot(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Capture cheap structural values that must survive torch.save/load."""
    meta = checkpoint.get("meta", {})
    optimizers = []
    for name, optimizer in _optimizer_state_dicts(checkpoint.get("optimizer")):
        states = optimizer.get("state", {})
        optimizers.append({
            "name": name,
            "state_keys": set(states) if isinstance(states, dict) else None,
            "groups": [
                (group.get("lr"), group.get("initial_lr"),
                 tuple(group.get("params", [])))
                for group in optimizer["param_groups"]
            ],
        })
    state_dict = checkpoint.get("state_dict")
    return {
        "epoch": meta.get("epoch"),
        "iter": meta.get("iter"),
        "resume_preparation": meta.get("resume_preparation"),
        "state_dict_keys": (
            tuple(state_dict.keys()) if isinstance(state_dict, dict) else None),
        "optimizers": optimizers,
    }


def _verify_written_checkpoint(output: Path,
                               expected: Dict[str, Any]) -> None:
    reloaded = _load_checkpoint(output)
    actual = _verification_snapshot(reloaded)
    if actual != expected:
        raise RuntimeError(
            f"post-write checkpoint verification failed for {output}")


def _validate_output_path(source: Path, output: Path) -> Path:
    output = output.resolve()
    if source == output:
        raise ValueError("input and output paths must be different")
    if output.exists():
        try:
            if os.path.samefile(source, output):
                raise ValueError(
                    "input and output resolve to the same file/inode")
        except OSError:
            pass
        raise FileExistsError(
            f"refusing to overwrite existing output: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Optimizer lr values and all Adam state are preserved; only "
            "initial_lr is adjusted. In this epoch-4 checkpoint, the 82 "
            "newly fixed trainable tensors have no Adam state yet and will "
            "create it lazily on their first optimizer step."))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path,
                        help="new checkpoint path; required with --write")
    parser.add_argument("--max-epochs", type=_positive_int, required=True)
    parser.add_argument("--min-lr-ratio", type=float, default=1e-3)
    length = parser.add_mutually_exclusive_group(required=True)
    length.add_argument("--iters-per-epoch", type=_positive_int)
    length.add_argument(
        "--group-sizes", type=_parse_group_sizes,
        help="sampler flag-group sizes, comma separated; ETRI uses 19608")
    parser.add_argument("--world-size", type=_positive_int, default=2)
    parser.add_argument("--samples-per-gpu", type=_positive_int, default=1)
    parser.add_argument("--warmup-iters", type=_nonnegative_int, default=500)
    parser.add_argument(
        "--runtime-config", type=Path,
        help=("final 48-epoch config to validate and embed; required with "
              "--write so later resumed checkpoints do not retain stale meta"))
    parser.add_argument(
        "--write", action="store_true",
        help="write the new checkpoint; without this flag only print a plan")
    parser.add_argument(
        "--verify-source-sha256", action="store_true",
        help="also hash the source before/after (slower; stat is always checked)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_stat_before = _source_stat(source)
    source_hash_before = _sha256(source) if args.verify_source_sha256 else None
    output = None
    if args.write:
        if args.output is None:
            raise ValueError("--output is required with --write")
        if args.runtime_config is None:
            raise ValueError("--runtime-config is required with --write")
        if not args.verify_source_sha256:
            raise ValueError(
                "--verify-source-sha256 is required with --write so the "
                "prepared checkpoint embeds immutable source provenance")
        output = _validate_output_path(source, args.output)
    elif args.output is not None:
        raise ValueError("--output has no effect without --write")

    if args.iters_per_epoch is not None:
        iters_per_epoch = args.iters_per_epoch
        group_sizes = None
    else:
        group_sizes = args.group_sizes
        iters_per_epoch = distributed_group_iters(
            group_sizes, args.world_size, args.samples_per_gpu)

    checkpoint = _load_checkpoint(source)
    report = prepare(
        checkpoint,
        max_epochs=args.max_epochs,
        min_lr_ratio=args.min_lr_ratio,
        iters_per_epoch=iters_per_epoch,
        world_size=args.world_size,
        warmup_iters=args.warmup_iters,
    )
    report.update({
        "source": str(source),
        "mode": "write" if args.write else "dry-run",
        "samples_per_gpu": args.samples_per_gpu,
        "group_sizes": group_sizes,
    })
    provenance = {
        "samples_per_gpu": args.samples_per_gpu,
        "group_sizes": group_sizes,
        "source_checkpoint_size": source_stat_before["size"],
    }
    if source_hash_before is not None:
        provenance["source_checkpoint_sha256"] = source_hash_before
    checkpoint["meta"]["resume_preparation"].update(provenance)
    report["resume_preparation_provenance"] = provenance

    if args.runtime_config is not None:
        runtime_config_text, runtime_config_report = (
            _materialize_runtime_config(
                args.runtime_config,
                max_epochs=args.max_epochs,
                min_lr_ratio=args.min_lr_ratio,
                world_size=args.world_size,
                samples_per_gpu=args.samples_per_gpu,
                warmup_iters=args.warmup_iters,
            ))
        checkpoint["meta"]["config"] = runtime_config_text
        report["runtime_config"] = runtime_config_report
        report["runtime_config_embedded"] = True
    else:
        report["runtime_config_embedded"] = False

    if args.write:
        expected = _verification_snapshot(checkpoint)
        _save_new_checkpoint(checkpoint, output)
        _verify_written_checkpoint(output, expected)
        report["output"] = str(output)
        report["output_reload_verified"] = True

    source_stat_after = _source_stat(source)
    if source_stat_after != source_stat_before:
        raise RuntimeError("source checkpoint stat changed during preparation")
    report["source_stat_unchanged"] = True
    if args.verify_source_sha256:
        source_hash_after = _sha256(source)
        if source_hash_after != source_hash_before:
            raise RuntimeError("source checkpoint SHA-256 changed")
        report["source_sha256"] = source_hash_after
        report["source_sha256_unchanged"] = True
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
