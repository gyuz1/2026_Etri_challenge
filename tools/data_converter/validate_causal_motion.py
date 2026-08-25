"""Validate the causal ego-motion implementation.

This script:

1. Does not read or modify dataset/etri/*.pkl.
2. Loads raw train scenarios and test clips.
3. Compares the old centered difference with the new backward difference.
4. Verifies the converter output against a manual backward-difference calculation.
5. Perturbs the future pose and confirms that the new train calculation is unchanged.
6. Checks timestamp intervals and NaN/Inf values.
7. Confirms that test acceleration is no longer hard-coded to zero.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


# This file is expected to be placed in tools/data_converter/.
CONVERTER_DIR = Path(__file__).resolve().parent
if str(CONVERTER_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERTER_DIR))

import etri_test_converter as testconv
import etri_vad_converter as trainconv


Array = np.ndarray

DEFAULT_TRAIN_ROOT = Path(
    "/media/vcl/DATA/2026AD/challenge/dataset/train"
)
DEFAULT_TEST_ROOT = Path(
    "/media/vcl/DATA/2026AD/challenge/dataset/test"
)

DEFAULT_TRAIN_FRAMES = [0, 1, 2, 5, 50, 150, 299]


def format_xy(vector: Array) -> str:
    """Format only the x/y components for compact terminal output."""
    vector = np.asarray(vector)
    return f"[{vector[0]:+8.3f}, {vector[1]:+8.3f}]"


def assert_finite(name: str, value: Array) -> None:
    """Fail when a calculated value contains NaN or Inf."""
    value = np.asarray(value)

    if not np.all(np.isfinite(value)):
        raise AssertionError(
            f"{name} contains NaN/Inf: {value}"
        )


def old_centered_motion(
    data: Dict[str, Array],
    index: int,
    rotation: Array,
) -> Tuple[Array, Array]:
    """Reproduce the old future-peeking train calculation."""

    timestamps = data["timestamps"]
    positions = data["ego_pose_xyz"]

    if index - 1 < 0 or index + 1 >= len(positions):
        raise IndexError(
            f"Centered difference is unavailable at index={index}."
        )

    centered_dt = (
        timestamps[index + 1] - timestamps[index - 1]
    ) / 1e3

    if centered_dt <= 0:
        raise ValueError(
            f"Invalid centered dt={centered_dt} at index={index}."
        )

    half_dt = centered_dt / 2.0

    velocity_global = (
        positions[index + 1] - positions[index - 1]
    ) / centered_dt

    acceleration_global = (
        positions[index + 1]
        - 2.0 * positions[index]
        + positions[index - 1]
    ) / (half_dt**2)

    velocity_local = velocity_global @ rotation
    acceleration_local = acceleration_global @ rotation

    return velocity_local, acceleration_local


def manual_train_backward_motion(
    data: Dict[str, Array],
    index: int,
    rotation: Array,
) -> Tuple[Array, Array, Array]:
    """Manually reproduce the current train converter calculation."""

    timestamps = data["timestamps"]
    positions = data["ego_pose_xyz"]
    rotations = data["ego_pose_rpy"]

    if index <= 0:
        zero = np.zeros(3, dtype=np.float64)
        return zero.copy(), zero.copy(), zero.copy()

    dt = (
        timestamps[index] - timestamps[index - 1]
    ) / 1e3

    if dt <= 0:
        raise ValueError(
            f"Invalid train dt={dt} at index={index}."
        )

    velocity_global = (
        positions[index] - positions[index - 1]
    ) / dt

    velocity_local = velocity_global @ rotation

    rotation_rate = trainconv.wrap_angle(
        rotations[index] - rotations[index - 1]
    ) / dt

    if index <= 1:
        acceleration_local = np.zeros(3, dtype=np.float64)
        return velocity_local, acceleration_local, rotation_rate

    dt_previous = (
        timestamps[index - 1] - timestamps[index - 2]
    ) / 1e3

    if dt_previous <= 0:
        raise ValueError(
            f"Invalid previous train dt={dt_previous} "
            f"at index={index}."
        )

    previous_velocity_global = (
        positions[index - 1] - positions[index - 2]
    ) / dt_previous

    # Both velocities are expressed in the current Ego frame.
    previous_velocity_local = (
        previous_velocity_global @ rotation
    )

    # This intentionally matches the current converter implementation.
    acceleration_local = (
        velocity_local - previous_velocity_local
    ) / dt

    return velocity_local, acceleration_local, rotation_rate


def check_future_independence(
    data: Dict[str, Array],
    index: int,
    rotation: Array,
    atol: float = 1e-10,
) -> None:
    """Confirm that modifying frame i+1 does not affect the new result."""

    positions = data["ego_pose_xyz"]

    if index + 1 >= len(positions):
        print(
            "    causal test: SKIP "
            "(future array element is unavailable)"
        )
        return

    velocity_before, acceleration_before, rate_before = (
        trainconv.local_motion(data, index, rotation)
    )

    modified_data = dict(data)

    # Copy only arrays that will be modified.
    modified_data["ego_pose_xyz"] = np.array(
        data["ego_pose_xyz"],
        copy=True,
    )
    modified_data["ego_pose_rpy"] = np.array(
        data["ego_pose_rpy"],
        copy=True,
    )

    # Deliberately inject an absurd future pose.
    modified_data["ego_pose_xyz"][index + 1] += np.array(
        [1000.0, -1000.0, 500.0]
    )
    modified_data["ego_pose_rpy"][index + 1] += np.array(
        [1.0, -1.0, 2.0]
    )

    velocity_after, acceleration_after, rate_after = (
        trainconv.local_motion(
            modified_data,
            index,
            rotation,
        )
    )

    np.testing.assert_allclose(
        velocity_before,
        velocity_after,
        atol=atol,
        rtol=0.0,
        err_msg="Velocity depends on future pose.",
    )
    np.testing.assert_allclose(
        acceleration_before,
        acceleration_after,
        atol=atol,
        rtol=0.0,
        err_msg="Acceleration depends on future pose.",
    )
    np.testing.assert_allclose(
        rate_before,
        rate_after,
        atol=atol,
        rtol=0.0,
        err_msg="Rotation rate depends on future pose.",
    )

    print("    causal test: PASS (future perturbation ignored)")


def print_train_timestamp_stats(
    data: Dict[str, Array],
) -> None:
    timestamps = np.asarray(data["timestamps"])

    dt_values = np.diff(timestamps.astype(np.float64)) / 1e3
    dt_values = dt_values[np.isfinite(dt_values)]

    if len(dt_values) == 0:
        raise AssertionError("No valid train timestamp differences.")

    non_positive = int(np.sum(dt_values <= 0))

    print("\nTrain timestamp statistics")
    print(f"  count       : {len(dt_values)}")
    print(f"  min dt      : {np.min(dt_values):.9f} s")
    print(f"  max dt      : {np.max(dt_values):.9f} s")
    print(f"  mean dt     : {np.mean(dt_values):.9f} s")
    print(f"  std dt      : {np.std(dt_values):.9f} s")
    print(f"  dt <= 0     : {non_positive}")
    print(
        "  max |dt-.1|:",
        f"{np.max(np.abs(dt_values - 0.1)):.9f} s",
    )

    if non_positive:
        raise AssertionError(
            f"Found {non_positive} non-positive timestamp intervals."
        )


def check_train(
    train_root: Path,
    scenario_name: str,
    frame_ids: Sequence[int],
) -> None:
    scenario_dir = train_root / scenario_name

    if not scenario_dir.is_dir():
        raise FileNotFoundError(
            f"Train scenario not found: {scenario_dir}"
        )

    data = trainconv.load_scenario(str(scenario_dir))

    print(f"\n{'=' * 100}")
    print(f"TRAIN scenario: {scenario_name}")
    print(f"{'=' * 100}")

    print_train_timestamp_stats(data)

    header = (
        f'{"frame":>6}'
        f' {"v_old(x,y)":>22}'
        f' {"v_new(x,y)":>22}'
        f' {"|dv|":>10}'
        f' {"a_old(x,y)":>22}'
        f' {"a_new(x,y)":>22}'
        f' {"|da|":>10}'
    )

    print("\n" + header)
    print("-" * len(header))

    checked_frames = 0

    for frame_id in frame_ids:
        index = frame_id + trainconv.FRAME_OFFSET

        if index < 0 or index >= len(data["ego_pose_xyz"]):
            print(
                f"{frame_id:>6}: SKIP "
                f"(converted index {index} is out of range)"
            )
            continue

        rotation = trainconv.euler_to_matrix(
            data["ego_pose_rpy"][index]
        )

        velocity_new, acceleration_new, rotation_rate_new = (
            trainconv.local_motion(
                data,
                index,
                rotation,
            )
        )

        velocity_manual, acceleration_manual, rate_manual = (
            manual_train_backward_motion(
                data,
                index,
                rotation,
            )
        )

        assert_finite("train velocity", velocity_new)
        assert_finite("train acceleration", acceleration_new)
        assert_finite("train rotation rate", rotation_rate_new)

        # Verify that local_motion() implements the expected equations.
        np.testing.assert_allclose(
            velocity_new,
            velocity_manual,
            atol=1e-9,
            rtol=1e-7,
            err_msg=f"Train velocity mismatch at frame={frame_id}",
        )
        np.testing.assert_allclose(
            acceleration_new,
            acceleration_manual,
            atol=1e-3,
            rtol=1e-3,
            err_msg=f"Train acceleration mismatch at frame={frame_id}",
        )
        np.testing.assert_allclose(
            rotation_rate_new,
            rate_manual,
            atol=1e-9,
            rtol=1e-7,
            err_msg=f"Train rotation-rate mismatch at frame={frame_id}",
        )

        try:
            velocity_old, acceleration_old = old_centered_motion(
                data,
                index,
                rotation,
            )

            velocity_difference = np.linalg.norm(
                velocity_new[:2] - velocity_old[:2]
            )
            acceleration_difference = np.linalg.norm(
                acceleration_new[:2] - acceleration_old[:2]
            )

            print(
                f"{frame_id:>6}"
                f" {format_xy(velocity_old):>22}"
                f" {format_xy(velocity_new):>22}"
                f" {velocity_difference:>10.4f}"
                f" {format_xy(acceleration_old):>22}"
                f" {format_xy(acceleration_new):>22}"
                f" {acceleration_difference:>10.4f}"
            )
        except IndexError:
            print(
                f"{frame_id:>6}"
                f" {'N/A':>22}"
                f" {format_xy(velocity_new):>22}"
                f" {'N/A':>10}"
                f" {'N/A':>22}"
                f" {format_xy(acceleration_new):>22}"
                f" {'N/A':>10}"
            )

        check_future_independence(
            data,
            index,
            rotation,
        )

        checked_frames += 1

    if checked_frames == 0:
        raise AssertionError("No train frames were checked.")

    print(
        f"\nTrain converter equation checks: "
        f"PASS ({checked_frames} frames)"
    )


def manual_test_backward_motion(
    xyz: Dict[int, Array],
    rpy: Dict[int, Array],
    frame: int,
) -> Tuple[Array, Array, Array]:
    """Manually reproduce the current test converter calculation."""

    rotation = testconv.euler_to_matrix(rpy[frame])
    dt = float(getattr(testconv, "DT", 0.1))

    has_previous = (frame - 1) in xyz
    has_previous2 = (frame - 2) in xyz

    if not has_previous:
        zero = np.zeros(3, dtype=np.float64)
        return zero.copy(), zero.copy(), zero.copy()

    velocity = (
        (xyz[frame] - xyz[frame - 1]) / dt
    ) @ rotation

    rotation_rate = testconv.wrap_angle(
        rpy[frame] - rpy[frame - 1]
    ) / dt

    if not has_previous2:
        acceleration = np.zeros(3, dtype=np.float64)
        return velocity, acceleration, rotation_rate

    previous_velocity = (
        (xyz[frame - 1] - xyz[frame - 2]) / dt
    ) @ rotation

    acceleration = (
        velocity - previous_velocity
    ) / dt

    return velocity, acceleration, rotation_rate


def check_test_clip(
    test_root: Path,
    clip_token: str,
) -> Tuple[float, float]:
    clip_dir = test_root / clip_token

    xyz, rpy = testconv.load_ego_pose(str(clip_dir))

    velocity, acceleration, rotation_rate = (
        testconv.local_motion(
            xyz,
            rpy,
            0,
        )
    )

    expected_velocity, expected_acceleration, expected_rate = (
        manual_test_backward_motion(
            xyz,
            rpy,
            0,
        )
    )

    assert_finite("test velocity", velocity)
    assert_finite("test acceleration", acceleration)
    assert_finite("test rotation rate", rotation_rate)

    np.testing.assert_allclose(
        velocity,
        expected_velocity,
        atol=1e-9,
        rtol=1e-7,
        err_msg=f"Test velocity mismatch: {clip_token}",
    )
    np.testing.assert_allclose(
        acceleration,
        expected_acceleration,
        atol=1e-9,
        rtol=1e-7,
        err_msg=f"Test acceleration mismatch: {clip_token}",
    )
    np.testing.assert_allclose(
        rotation_rate,
        expected_rate,
        atol=1e-9,
        rtol=1e-7,
        err_msg=f"Test rotation-rate mismatch: {clip_token}",
    )

    velocity_norm = float(np.linalg.norm(velocity[:2]))
    acceleration_norm = float(np.linalg.norm(acceleration[:2]))

    print(
        f"{clip_token}: "
        f"v={format_xy(velocity)}, "
        f"|v|={velocity_norm:7.3f} m/s, "
        f"a={format_xy(acceleration)}, "
        f"|a|={acceleration_norm:7.3f} m/s² "
        f"→ PASS"
    )

    return velocity_norm, acceleration_norm


def list_directories(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir()
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate causal ETRI ego-motion conversion."
    )

    parser.add_argument(
        "--train-root",
        type=Path,
        default=DEFAULT_TRAIN_ROOT,
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=DEFAULT_TEST_ROOT,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Train scenario name. Defaults to the first directory.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=DEFAULT_TRAIN_FRAMES,
    )
    parser.add_argument(
        "--num-test-clips",
        type=int,
        default=8,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    train_scenarios = list_directories(args.train_root)

    if not train_scenarios:
        raise RuntimeError(
            f"No train scenarios found under {args.train_root}"
        )

    scenario_name = (
        args.scenario
        if args.scenario is not None
        else train_scenarios[0]
    )

    check_train(
        train_root=args.train_root,
        scenario_name=scenario_name,
        frame_ids=args.frames,
    )

    print(f"\n{'=' * 100}")
    print("TEST clips: current frame 0")
    print(f"{'=' * 100}")

    clip_tokens = list_directories(args.test_root)
    clip_tokens = clip_tokens[: args.num_test_clips]

    if not clip_tokens:
        raise RuntimeError(
            f"No test clips found under {args.test_root}"
        )

    velocity_norms = []
    acceleration_norms = []

    for clip_token in clip_tokens:
        velocity_norm, acceleration_norm = check_test_clip(
            args.test_root,
            clip_token,
        )
        velocity_norms.append(velocity_norm)
        acceleration_norms.append(acceleration_norm)

    print("\nTest summary")
    print(f"  checked clips : {len(clip_tokens)}")
    print(
        f"  velocity norm : "
        f"min={np.min(velocity_norms):.3f}, "
        f"mean={np.mean(velocity_norms):.3f}, "
        f"max={np.max(velocity_norms):.3f} m/s"
    )
    print(
        f"  accel norm    : "
        f"min={np.min(acceleration_norms):.3f}, "
        f"mean={np.mean(acceleration_norms):.3f}, "
        f"max={np.max(acceleration_norms):.3f} m/s²"
    )

    print("\n" + "=" * 100)
    print("ALL CAUSAL MOTION CHECKS PASSED")
    print("=" * 100)
    print(
        "No dataset/etri/*.pkl file was read, rewritten, "
        "or modified by this script."
    )


if __name__ == "__main__":
    main()