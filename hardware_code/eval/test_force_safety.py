import json
from pathlib import Path

import numpy as np

from eval.force_safety import AbsoluteJoint65ForceSafety, load_dataset_max_fz


ACTION_DIM = 65


def _tactile(left_fz: float, right_fz: float) -> np.ndarray:
    tactile = np.zeros((10, 6), dtype=np.float32)
    tactile[:5, 2] = left_fz
    tactile[5:, 2] = right_fz
    return tactile


def _row(index: int, value: float = 1.0) -> np.ndarray:
    result = np.zeros(ACTION_DIM, dtype=np.float64)
    result[index] = value
    return result


def test_load_dataset_max_fz_and_threshold():
    stats_path = (
        Path(__file__).resolve().parents[3]
        / "dataset/Robotic_Origami_Challenge/"
        "season_POC22061_2026_07_14_10_09_42_train/lerobot3.0/meta/stats.json"
    )
    dataset_max_fz = load_dataset_max_fz(stats_path)
    assert dataset_max_fz == 44.90625

    controller = AbsoluteJoint65ForceSafety.from_stats_path(stats_path)
    assert controller.threshold_fz == 44.90625 * 0.8


def test_stats_loader_accepts_native_tactile_block(tmp_path):
    maximum = np.zeros(60, dtype=np.float64)
    maximum[2] = 7.0
    maximum[32] = 11.0
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps({"observation.tactile": {"max": maximum.tolist()}}),
        encoding="utf-8",
    )
    assert load_dataset_max_fz(stats_path) == 11.0


def test_below_threshold_preserves_the_action_chunk():
    controller = AbsoluteJoint65ForceSafety(dataset_max_fz=10.0)
    current = np.zeros(ACTION_DIM, dtype=np.float64)
    action = np.zeros((3, ACTION_DIM), dtype=np.float64)
    action[:, 0] = [-0.4, 0.2, -0.1]

    result = controller.postprocess(
        action,
        current,
        _tactile(left_fz=7.9, right_fz=0.0),
        {"left": _row(0), "right": _row(29)},
    )

    assert not result.triggered
    assert not result.modified
    np.testing.assert_array_equal(result.action_safe, action)


def test_threshold_projects_downward_motion_for_active_side_only():
    controller = AbsoluteJoint65ForceSafety(dataset_max_fz=10.0, dls_lambda=1e-4)
    current = np.zeros(ACTION_DIM, dtype=np.float64)
    action = np.zeros((3, ACTION_DIM), dtype=np.float64)
    # Negative row dot delta is defined as movement towards the table.
    action[:, 0] = [-0.4, 0.2, -0.1]
    action[:, 29] = [-0.3, -0.2, 0.1]
    action[:, 10] = [0.11, 0.12, 0.13]  # unrelated left-hand motion
    original = action.copy()

    result = controller.postprocess(
        action,
        current,
        _tactile(left_fz=8.0, right_fz=7.9),
        {"left": _row(0), "right": _row(29)},
    )

    assert result.triggered
    assert result.modified
    assert result.active_sides == ("left",)
    assert np.all(result.normal_displacement_after[:, 0] >= -1e-12)
    np.testing.assert_allclose(result.action_safe[:, 0], [0.0, 0.2, 0.0], atol=1e-8)
    np.testing.assert_array_equal(result.action_safe[:, 29], original[:, 29])
    np.testing.assert_array_equal(result.action_safe[:, 10], original[:, 10])


def test_zero_normal_jacobian_holds_active_arm():
    controller = AbsoluteJoint65ForceSafety(dataset_max_fz=10.0)
    current = np.zeros(ACTION_DIM, dtype=np.float64)
    action = np.zeros(ACTION_DIM, dtype=np.float64)
    action[0] = -0.2
    action[7] = 0.3

    result = controller.postprocess(
        action,
        current,
        _tactile(left_fz=9.0, right_fz=0.0),
        {"left": _row(0, 1e-7), "right": np.zeros(ACTION_DIM)},
    )

    assert result.modified
    assert result.action_safe[0] == 0.0
    # With no reliable local normal direction, the complete active-side target
    # is held conservatively.
    assert result.action_safe[7] == 0.0
