"""Visualize and verify old vs causal ETRI ego motion.

The causal values are calculated directly from raw pose parquet files.  When
an info PKL is supplied, its stored can_bus and ego_lcf_feat values are also
checked against that independent calculation.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation


EGO_LENGTH = 4.635
EGO_WIDTH = 1.890


@dataclass
class Motion:
    velocity: np.ndarray
    acceleration: np.ndarray
    rotation_rate: np.ndarray


@dataclass
class StoredCheck:
    motion: Motion
    can_bus_error: float
    lcf_error: float
    passed: bool


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def load_pose_table(scenario_dir: Path) -> pd.DataFrame:
    frames = pd.read_parquet(
        scenario_dir / "meta" / "timestamps.parquet"
    )[["timestamp", "frame_id"]]
    poses = pd.read_parquet(
        scenario_dir / "annotation" / "ego_pose.parquet"
    )
    poses = poses.merge(frames, on="timestamp").sort_values("frame_id")
    poses = poses.drop_duplicates("frame_id", keep="last").set_index("frame_id")
    return poses


def load_lanes(scenario_dir: Path) -> Sequence[np.ndarray]:
    table = pd.read_parquet(
        scenario_dir / "annotation" / "map.parquet"
    )
    return [
        np.asarray([np.asarray(point, dtype=np.float64) for point in points])
        for points in table["points"]
    ]


def pose_at(poses: pd.DataFrame, frame: int) -> Tuple[float, np.ndarray, np.ndarray]:
    if frame not in poses.index:
        raise KeyError(f"Raw ego pose does not contain frame {frame}")
    row = poses.loc[frame]
    timestamp = float(row["timestamp"])
    xyz = row[["x", "y", "z"]].to_numpy(dtype=np.float64)
    rpy = row[["roll", "pitch", "yaw"]].to_numpy(dtype=np.float64)
    return timestamp, xyz, rpy


def old_centered_motion(poses: pd.DataFrame, frame: int) -> Motion:
    previous_time, previous_xyz, previous_rpy = pose_at(poses, frame - 1)
    current_time, current_xyz, current_rpy = pose_at(poses, frame)
    next_time, next_xyz, next_rpy = pose_at(poses, frame + 1)
    del current_time

    rotation = Rotation.from_euler("xyz", current_rpy).as_matrix()
    centered_dt = (next_time - previous_time) / 1e3
    if centered_dt <= 0:
        raise ValueError(f"Non-positive centered dt at frame {frame}")

    velocity = ((next_xyz - previous_xyz) / centered_dt) @ rotation
    acceleration = (
        (next_xyz - 2.0 * current_xyz + previous_xyz)
        / (centered_dt / 2.0) ** 2
    ) @ rotation
    rotation_rate = (
        wrap_angle(next_rpy - previous_rpy) / centered_dt
    )
    return Motion(velocity, acceleration, rotation_rate)


def causal_backward_motion(poses: pd.DataFrame, frame: int) -> Motion:
    previous2_time, previous2_xyz, _ = pose_at(poses, frame - 2)
    previous_time, previous_xyz, previous_rpy = pose_at(poses, frame - 1)
    current_time, current_xyz, current_rpy = pose_at(poses, frame)

    dt = (current_time - previous_time) / 1e3
    previous_dt = (previous_time - previous2_time) / 1e3
    if dt <= 0 or previous_dt <= 0:
        raise ValueError(
            f"Non-positive backward dt at frame {frame}: "
            f"dt={dt}, previous_dt={previous_dt}"
        )

    rotation = Rotation.from_euler("xyz", current_rpy).as_matrix()
    velocity = ((current_xyz - previous_xyz) / dt) @ rotation
    previous_velocity = (
        (previous_xyz - previous2_xyz) / previous_dt
    ) @ rotation
    acceleration = (
        (velocity - previous_velocity) / ((dt + previous_dt) / 2.0)
    )
    rotation_rate = wrap_angle(current_rpy - previous_rpy) / dt
    return Motion(velocity, acceleration, rotation_rate)


def load_requested_infos(
    pkl_path: Optional[Path],
    scenario: str,
    frames: Sequence[int],
) -> Dict[int, dict]:
    if pkl_path is None:
        return {}
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Info PKL not found: {pkl_path}")

    print(f"Loading PKL: {pkl_path}", flush=True)
    with pkl_path.open("rb") as stream:
        data = pickle.load(stream)

    wanted = {f"{scenario}_{frame:08d}": frame for frame in frames}
    found = {
        wanted[info["token"]]: info
        for info in data["infos"]
        if info["token"] in wanted
    }
    missing = sorted(set(frames) - set(found))
    if missing:
        raise KeyError(
            f"PKL does not contain {scenario} frames: {missing}"
        )
    return found


def check_stored_motion(
    info: dict,
    expected: Motion,
    can_bus_atol: float,
    lcf_atol: float,
    rtol: float,
) -> StoredCheck:
    can_bus = np.asarray(info["can_bus"], dtype=np.float64)
    lcf = np.asarray(info["gt_ego_lcf_feat"], dtype=np.float64)
    stored = Motion(
        velocity=can_bus[13:16],
        acceleration=can_bus[7:10],
        rotation_rate=can_bus[10:13],
    )

    expected_can_bus = np.concatenate([
        expected.acceleration,
        expected.rotation_rate,
        expected.velocity,
    ])
    stored_can_bus = can_bus[7:16]
    can_bus_error = float(np.max(np.abs(stored_can_bus - expected_can_bus)))

    expected_lcf = np.asarray([
        expected.velocity[0],
        expected.velocity[1],
        expected.acceleration[0],
        expected.acceleration[1],
        expected.rotation_rate[2],
        np.linalg.norm(expected.velocity[:2]),
    ])
    stored_lcf = lcf[[0, 1, 2, 3, 4, 7]]
    lcf_error = float(np.max(np.abs(stored_lcf - expected_lcf)))

    passed = bool(
        np.allclose(
            stored_can_bus,
            expected_can_bus,
            atol=can_bus_atol,
            rtol=rtol,
        )
        and np.allclose(
            stored_lcf,
            expected_lcf,
            atol=lcf_atol,
            rtol=rtol,
        )
    )
    return StoredCheck(stored, can_bus_error, lcf_error, passed)


def transform_positions(
    poses: pd.DataFrame,
    frame: int,
    query_frames: Sequence[int],
) -> np.ndarray:
    _, origin, current_rpy = pose_at(poses, frame)
    rotation = Rotation.from_euler("xyz", current_rpy).as_matrix()
    positions = np.asarray([pose_at(poses, item)[1] for item in query_frames])
    return (positions - origin) @ rotation


def draw_panel(
    ax: plt.Axes,
    lanes: Sequence[np.ndarray],
    poses: pd.DataFrame,
    frame: int,
    motion: Motion,
    mode: str,
    accel_scale: float,
    lateral_range: float,
    longitudinal_range: float,
    stored_check: Optional[StoredCheck] = None,
) -> None:
    _, origin, current_rpy = pose_at(poses, frame)
    rotation = Rotation.from_euler("xyz", current_rpy).as_matrix()

    for lane in lanes:
        local = (lane[:, :3] - origin) @ rotation
        ax.plot(
            local[:, 1],
            local[:, 0],
            color="#d99b2b",
            linewidth=0.9,
            alpha=0.75,
            zorder=1,
        )

    trajectory_frames = [frame - 2, frame - 1, frame, frame + 1]
    trajectory = transform_positions(poses, frame, trajectory_frames)
    ax.plot(
        trajectory[:, 1],
        trajectory[:, 0],
        color="#4c78a8",
        marker="o",
        markersize=3,
        linewidth=1.0,
        alpha=0.8,
        zorder=2,
    )
    ax.scatter(
        trajectory[-1, 1],
        trajectory[-1, 0],
        marker="x",
        color="#e83e8c",
        s=35,
        zorder=4,
    )

    ego = Rectangle(
        (-EGO_WIDTH / 2.0, -EGO_LENGTH / 2.0),
        EGO_WIDTH,
        EGO_LENGTH,
        facecolor="#454f55" if mode == "old" else "#808080",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9,
        zorder=3,
    )
    ax.add_patch(ego)

    ax.quiver(
        0.0,
        0.0,
        motion.velocity[1],
        motion.velocity[0],
        color="#2ca02c",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.018,
        zorder=5,
    )
    ax.quiver(
        0.0,
        0.0,
        motion.acceleration[1] * accel_scale,
        motion.acceleration[0] * accel_scale,
        color="#7f3fbf",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.018,
        zorder=6,
    )

    if stored_check is not None:
        stored = stored_check.motion
        ax.annotate(
            "",
            xy=(stored.velocity[1], stored.velocity[0]),
            xytext=(0.0, 0.0),
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                linewidth=1.3,
                linestyle="--",
            ),
            zorder=7,
        )
        status = "PKL PASS" if stored_check.passed else "PKL FAIL"
        status_color = "#15803d" if stored_check.passed else "#dc2626"
    else:
        status = "raw calculation"
        status_color = "#333333"

    text = (
        f"v=({motion.velocity[0]:+.3f}, {motion.velocity[1]:+.3f}) m/s\n"
        f"|v|={np.linalg.norm(motion.velocity[:2]):.3f} m/s\n"
        f"a=({motion.acceleration[0]:+.3f}, "
        f"{motion.acceleration[1]:+.3f}) m/s^2\n"
        f"|a|={np.linalg.norm(motion.acceleration[:2]):.3f} m/s^2"
    )
    if stored_check is not None:
        text += (
            f"\ncan_bus err={stored_check.can_bus_error:.2e}"
            f"\nlcf err={stored_check.lcf_error:.2e}"
        )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none"),
        zorder=8,
    )
    ax.text(
        0.98,
        0.98,
        status,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=status_color,
        zorder=8,
    )

    ax.set_xlim(-lateral_range, lateral_range)
    ax.set_ylim(-longitudinal_range / 2.0, longitudinal_range)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_xlabel("lateral y [m]")
    ax.set_ylabel("longitudinal x [m]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot old centered vs causal backward EGO motion."
    )
    parser.add_argument("--train-root", type=Path, default=Path("data/train"))
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument(
        "--pkl",
        type=Path,
        default=None,
        help="Optional regenerated info PKL to verify.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work_dirs/causal_motion_comparison.pdf"),
    )
    parser.add_argument("--accel-scale", type=float, default=3.0)
    parser.add_argument("--lateral-range", type=float, default=10.0)
    parser.add_argument("--longitudinal-range", type=float, default=20.0)
    parser.add_argument("--can-bus-atol", type=float, default=1e-8)
    parser.add_argument("--lcf-atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_dir = args.train_root / args.scenario
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"Scenario not found: {scenario_dir}")

    poses = load_pose_table(scenario_dir)
    lanes = load_lanes(scenario_dir)
    stored_infos = load_requested_infos(args.pkl, args.scenario, args.frames)

    column_count = len(args.frames)
    figure, axes = plt.subplots(
        2,
        column_count,
        figsize=(5.2 * column_count, 10.0),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(
        "EGO velocity / acceleration: old centered (upper) "
        "vs causal backward (lower)",
        fontsize=17,
        fontweight="bold",
    )

    failures = []
    print(
        "frame | old |v| | causal |v| | old |a| | causal |a| | PKL",
        flush=True,
    )
    print("-" * 76, flush=True)
    for column, frame in enumerate(args.frames):
        old = old_centered_motion(poses, frame)
        causal = causal_backward_motion(poses, frame)
        stored_check = None
        if frame in stored_infos:
            stored_check = check_stored_motion(
                stored_infos[frame],
                causal,
                can_bus_atol=args.can_bus_atol,
                lcf_atol=args.lcf_atol,
                rtol=args.rtol,
            )
            if not stored_check.passed:
                failures.append(frame)

        axes[0, column].set_title(
            f"scene: {args.scenario}\nframe: {frame:06d}",
            fontsize=12,
            fontweight="bold",
        )
        draw_panel(
            axes[0, column],
            lanes,
            poses,
            frame,
            old,
            mode="old",
            accel_scale=args.accel_scale,
            lateral_range=args.lateral_range,
            longitudinal_range=args.longitudinal_range,
        )
        draw_panel(
            axes[1, column],
            lanes,
            poses,
            frame,
            causal,
            mode="causal",
            accel_scale=args.accel_scale,
            lateral_range=args.lateral_range,
            longitudinal_range=args.longitudinal_range,
            stored_check=stored_check,
        )

        status = (
            "N/A" if stored_check is None
            else ("PASS" if stored_check.passed else "FAIL")
        )
        print(
            f"{frame:5d} | {np.linalg.norm(old.velocity[:2]):9.3f} | "
            f"{np.linalg.norm(causal.velocity[:2]):12.3f} | "
            f"{np.linalg.norm(old.acceleration[:2]):9.3f} | "
            f"{np.linalg.norm(causal.acceleration[:2]):12.3f} | {status}",
            flush=True,
        )

    legend = [
        Line2D([0], [0], color="#2ca02c", linewidth=3, label="velocity"),
        Line2D([0], [0], color="#7f3fbf", linewidth=3,
               label=f"acceleration x{args.accel_scale:g}"),
        Line2D([0], [0], color="black", linewidth=1.3,
               linestyle="--", label="stored PKL velocity"),
        Line2D([0], [0], color="#e83e8c", marker="x", linewidth=0,
               label="future pose i+1 (context only)"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output.with_suffix(".pdf")
    png_path = args.output.with_suffix(".png")
    figure.savefig(pdf_path, dpi=180, bbox_inches="tight")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {pdf_path}", flush=True)
    print(f"Wrote {png_path}", flush=True)

    if failures:
        raise AssertionError(
            f"Stored PKL does not match causal raw calculation at frames: "
            f"{failures}"
        )
    if stored_infos:
        print("ALL REQUESTED PKL MOTION CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
