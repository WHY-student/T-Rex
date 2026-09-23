"""Soft tactile-force post-processing for absolute joint-65 actions.

The policy still produces absolute joint targets.  When the measured fingertip
normal force reaches the configured soft threshold, this module projects only
the active side's predicted normal motion so that it cannot continue towards
the table.  It deliberately does not cancel an action chunk or generate a
retreat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


ACTION_DIM = 65
TACTILE_DIM = 60
TACTILE_FINGERS = 10
TACTILE_COMPONENTS = 6
FZ_INDEX = 2

LEFT_ACTION_INDICES = np.arange(0, 29, dtype=np.int64)
RIGHT_ACTION_INDICES = np.arange(29, 58, dtype=np.int64)
SIDE_ACTION_INDICES = {
    "left": LEFT_ACTION_INDICES,
    "right": RIGHT_ACTION_INDICES,
}
SIDE_FINGER_SLICES = {
    "left": slice(0, 5),
    "right": slice(5, 10),
}


def _as_tactile_f6(value: Any) -> np.ndarray:
    tactile = np.asarray(value, dtype=np.float64)
    if tactile.shape == (TACTILE_DIM,):
        tactile = tactile.reshape(TACTILE_FINGERS, TACTILE_COMPONENTS)
    elif tactile.shape != (TACTILE_FINGERS, TACTILE_COMPONENTS):
        raise ValueError(
            "tactile_f6 must have shape (60,) or (10, 6), "
            f"got {tactile.shape}"
        )
    if not np.isfinite(tactile).all():
        raise ValueError("tactile_f6 contains NaN or Inf")
    return tactile


def max_fz(tactile_f6: Any) -> float:
    """Return ``max(fz)`` over the ten fingertips."""
    return float(np.max(_as_tactile_f6(tactile_f6)[:, FZ_INDEX]))


def _find_tactile_stats_block(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Find a native LeRobot or legacy T-Rex tactile statistics block."""
    for key in ("observation.tactile", "tactile_f6"):
        block = raw.get(key)
        if isinstance(block, Mapping):
            return block

    # Legacy T-Rex files are commonly wrapped as {dataset_name: {...}}.
    for value in raw.values():
        if not isinstance(value, Mapping):
            continue
        for key in ("observation.tactile", "tactile_f6"):
            block = value.get(key)
            if isinstance(block, Mapping):
                return block
    raise KeyError(
        "statistics JSON must contain 'observation.tactile' or 'tactile_f6'"
    )


def load_dataset_max_fz(stats_path: str | Path) -> float:
    """Load the dataset-wide maximum fingertip ``fz`` from ``stats.json``.

    This intentionally reads the literal ``max`` statistic rather than the
    q99 normalization bound, because the safety threshold is defined from the
    dataset maximum by the caller.
    """
    path = Path(stats_path)
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"statistics JSON must contain an object: {path}")
    block = _find_tactile_stats_block(raw)
    values = block.get("max")
    if values is None:
        raise KeyError(f"tactile statistics are missing 'max': {path}")
    maximum = np.asarray(values, dtype=np.float64)
    if maximum.shape != (TACTILE_DIM,):
        raise ValueError(
            f"tactile statistics 'max' must have shape (60,), got {maximum.shape}"
        )
    result = max_fz(maximum)
    if not np.isfinite(result):
        raise ValueError(f"tactile statistics maximum is not finite: {path}")
    return result


@dataclass(frozen=True)
class ForceSafetyResult:
    """Output and diagnostics for one action or an action chunk."""

    action_safe: np.ndarray
    measured_max_fz: float
    threshold_fz: float
    triggered: bool
    modified: bool
    active_sides: tuple[str, ...]
    normal_displacement_before: np.ndarray
    normal_displacement_after: np.ndarray


class AbsoluteJoint65ForceSafety:
    """Project downward normal motion when fingertip force is too high.

    ``normal_jacobians`` passed to :meth:`postprocess` must map ``"left"`` and
    ``"right"`` to 65-D rows expressed in the same world/table frame as the
    table normal.  The provider can therefore use the current measured joint
    state while the policy output remains an absolute joint target.
    """

    def __init__(
        self,
        dataset_max_fz: float,
        *,
        threshold_ratio: float = 0.8,
        dls_lambda: float = 1e-4,
        enabled: bool = True,
    ) -> None:
        if not np.isfinite(dataset_max_fz) or dataset_max_fz <= 0:
            raise ValueError("dataset_max_fz must be finite and positive")
        if not 0 < threshold_ratio <= 1:
            raise ValueError("threshold_ratio must be in (0, 1]")
        if not np.isfinite(dls_lambda) or dls_lambda < 0:
            raise ValueError("dls_lambda must be finite and non-negative")
        self.dataset_max_fz = float(dataset_max_fz)
        self.threshold_ratio = float(threshold_ratio)
        self.threshold_fz = self.dataset_max_fz * self.threshold_ratio
        self.dls_lambda = float(dls_lambda)
        self.enabled = bool(enabled)

    @classmethod
    def from_stats_path(
        cls,
        stats_path: str | Path,
        *,
        threshold_ratio: float = 0.8,
        dls_lambda: float = 1e-4,
        enabled: bool = True,
    ) -> "AbsoluteJoint65ForceSafety":
        return cls(
            load_dataset_max_fz(stats_path),
            threshold_ratio=threshold_ratio,
            dls_lambda=dls_lambda,
            enabled=enabled,
        )

    @staticmethod
    def _as_action_batch(value: Any, label: str) -> tuple[np.ndarray, np.dtype]:
        original = np.asarray(value)
        if original.shape == (ACTION_DIM,):
            batch = original.astype(np.float64, copy=True)[None, :]
        elif original.ndim == 2 and original.shape[1] == ACTION_DIM:
            batch = original.astype(np.float64, copy=True)
        else:
            raise ValueError(
                f"{label} must have shape (65,) or (T, 65), got {original.shape}"
            )
        if original.dtype.kind not in "f":
            raise TypeError(f"{label} must use a floating dtype, got {original.dtype}")
        if not np.isfinite(batch).all():
            raise ValueError(f"{label} contains NaN or Inf")
        return batch, original.dtype

    @staticmethod
    def _normal_rows(
        normal_jacobians: Mapping[str, Any], active_sides: tuple[str, ...]
    ) -> np.ndarray:
        rows = []
        for side in active_sides:
            if side not in normal_jacobians:
                raise KeyError(f"normal_jacobians is missing active side {side!r}")
            row = np.asarray(normal_jacobians[side], dtype=np.float64)
            if row.shape != (ACTION_DIM,):
                raise ValueError(
                    f"normal Jacobian for {side} must have shape (65,), got {row.shape}"
                )
            if not np.isfinite(row).all():
                raise ValueError(f"normal Jacobian for {side} contains NaN or Inf")
            rows.append(row)
        return np.stack(rows, axis=0) if rows else np.empty((0, ACTION_DIM))

    def postprocess(
        self,
        action: Any,
        current_joint65: Any,
        tactile_f6: Any,
        normal_jacobians: Mapping[str, Any],
    ) -> ForceSafetyResult:
        """Return a force-safe absolute joint target or target chunk.

        When triggered, each active side is corrected only if its predicted
        normal displacement is negative:

        ``dz = J_n @ (q_vla - q_current)``

        The correction is a damped least-squares minimum-norm update.  A tiny
        undamped residual correction is applied afterward so the final
        predicted normal displacement is not negative merely because of DLS
        damping.
        """
        action_batch, original_dtype = self._as_action_batch(action, "action")
        current = np.asarray(current_joint65, dtype=np.float64)
        if current.shape != (ACTION_DIM,):
            raise ValueError(
                f"current_joint65 must have shape (65,), got {current.shape}"
            )
        if not np.isfinite(current).all():
            raise ValueError("current_joint65 contains NaN or Inf")

        measured = max_fz(tactile_f6)
        triggered = bool(self.enabled and measured >= self.threshold_fz)
        tactile = _as_tactile_f6(tactile_f6)
        side_force = {
            side: float(np.max(tactile[finger_slice, FZ_INDEX]))
            for side, finger_slice in SIDE_FINGER_SLICES.items()
        }
        active_sides = tuple(
            side for side in ("left", "right") if side_force[side] >= self.threshold_fz
        )

        if not triggered:
            untouched = (
                action_batch[0]
                if np.asarray(action).shape == (ACTION_DIM,)
                else action_batch
            )
            empty = np.empty((action_batch.shape[0], 0), dtype=np.float64)
            return ForceSafetyResult(
                action_safe=np.asarray(untouched, dtype=original_dtype).copy(),
                measured_max_fz=measured,
                threshold_fz=self.threshold_fz,
                triggered=False,
                modified=False,
                active_sides=(),
                normal_displacement_before=empty,
                normal_displacement_after=empty,
            )

        rows = self._normal_rows(normal_jacobians, active_sides)
        displacement_before = (action_batch - current[None, :]) @ rows.T
        safe = action_batch.copy()
        modified_steps: set[int] = set()
        eps = 1e-12

        for step_index in range(safe.shape[0]):
            step_modified = False
            # With two arms the normal rows can in principle be coupled.  A
            # few alternating projections make sure correcting one active side
            # does not re-introduce downward motion on the other side.
            for _ in range(max(4, 8 * len(active_sides))):
                has_negative = False
                for row_index, side in enumerate(active_sides):
                    row = rows[row_index]
                    delta = safe[step_index] - current
                    before = float(np.dot(row, delta))
                    if before >= 0:
                        continue
                    has_negative = True
                    step_modified = True
                    norm_squared = float(np.dot(row, row))
                    if norm_squared <= eps:
                        # No locally observable normal direction.  Holding the
                        # complete side target is safer than forwarding a
                        # target we cannot verify.
                        safe[step_index, SIDE_ACTION_INDICES[side]] = current[
                            SIDE_ACTION_INDICES[side]
                        ]
                    else:
                        correction = row * (
                            -before / (norm_squared + self.dls_lambda**2)
                        )
                        safe[step_index] += correction
                        residual = float(
                            np.dot(row, safe[step_index] - current)
                        )
                        if residual < 0:
                            safe[step_index] += row * (-residual / norm_squared)
                if not has_negative:
                    break

            # The zero displacement is always safe for the Jacobian rows used
            # here.  Fail closed if numerical coupling prevented convergence.
            final_displacement = rows @ (safe[step_index] - current)
            if np.any(final_displacement < -1e-10):
                safe[step_index] = current.copy()
                step_modified = True
            if step_modified:
                modified_steps.add(step_index)

        displacement_after = (safe - current[None, :]) @ rows.T
        result = safe[0] if np.asarray(action).shape == (ACTION_DIM,) else safe
        return ForceSafetyResult(
            action_safe=np.asarray(result, dtype=original_dtype),
            measured_max_fz=measured,
            threshold_fz=self.threshold_fz,
            triggered=True,
            modified=bool(modified_steps),
            active_sides=active_sides,
            normal_displacement_before=displacement_before,
            normal_displacement_after=displacement_after,
        )


class AbsoluteJoint65NormalJacobianProvider:
    """Build left/right end-effector normal Jacobian rows with Pinocchio."""

    def __init__(
        self,
        robot_wrapper: Any,
        assemble_qpos: Callable[[dict[str, np.ndarray]], np.ndarray],
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:  # pragma: no cover - hardware environment only
            raise RuntimeError(
                "absolute_joint65 force safety requires Pinocchio"
            ) from exc
        from teleop.robot_descriptions import (
            DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES,
            SHARPA_HAND_JOINT_ORDER,
        )

        self.pin = pin
        self.robot_wrapper = robot_wrapper
        self.model = robot_wrapper.model
        self.data = self.model.createData()
        self.assemble_qpos = assemble_qpos

        model_names = (
            tuple(DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"])
            + tuple(f"left_{name}" for name in SHARPA_HAND_JOINT_ORDER)
            + tuple(DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"])
            + tuple(f"right_{name}" for name in SHARPA_HAND_JOINT_ORDER)
        )
        if len(model_names) != 58:
            raise RuntimeError("absolute_joint65 Jacobian mapping must contain 58 joints")
        self._dataset_to_v = []
        for name in model_names:
            joint_id = self.model.getJointId(name)
            if joint_id == 0:
                raise RuntimeError(f"Pinocchio model is missing joint {name!r}")
            joint = self.model.joints[joint_id]
            if joint.nv != 1:
                raise RuntimeError(f"joint {name!r} must have one velocity DoF")
            self._dataset_to_v.append(int(joint.idx_v))
        self._frame_ids = {
            side: self.model.getFrameId(f"{'L' if side == 'left' else 'R'}_ee")
            for side in ("left", "right")
        }
        if any(frame_id >= len(self.model.frames) for frame_id in self._frame_ids.values()):
            raise RuntimeError("Pinocchio model is missing L_ee or R_ee")

    def __call__(self, current_joint65: Any) -> dict[str, np.ndarray]:
        from eval.absolute_joint65 import split_absolute_joint65

        parts = split_absolute_joint65(current_joint65)
        qpos = self.assemble_qpos(
            {
                "left_arm": parts.left_arm,
                "right_arm": parts.right_arm,
                "left_hand": parts.left_hand,
                "right_hand": parts.right_hand,
            }
        )
        qpos = np.asarray(qpos, dtype=np.float64)
        self.pin.forwardKinematics(self.model, self.data, qpos)
        self.pin.updateFramePlacements(self.model, self.data)

        result: dict[str, np.ndarray] = {}
        for side, frame_id in self._frame_ids.items():
            jacobian = self.pin.computeFrameJacobian(
                self.model,
                self.data,
                qpos,
                frame_id,
                self.pin.ReferenceFrame.WORLD,
            )
            row = np.zeros(ACTION_DIM, dtype=np.float64)
            # Pinocchio's first three spatial-Jacobian rows are linear
            # velocity in the requested world frame; row 2 is world Z.
            row[:58] = jacobian[2, self._dataset_to_v]
            result[side] = row
        return result
