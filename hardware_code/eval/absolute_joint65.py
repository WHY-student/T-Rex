"""Exact joint-space adapter for the Origami dataset's 65-D schema.

The vector layout is:

    left arm 7 | left hand 22 | right arm 7 | right hand 22 | body 7

No EEF conversion or IK is performed.  The final seven dataset motors must be
bound explicitly to joints exposed by the installed robot SDK; guessing those
joints would make a syntactically valid client physically unsafe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


ACTION_DIM = 65
ARM_DIM = 7
HAND_DIM = 22
BODY_DIM = 7

LEFT_ARM_SLICE = slice(0, 7)
LEFT_HAND_SLICE = slice(7, 29)
RIGHT_ARM_SLICE = slice(29, 36)
RIGHT_HAND_SLICE = slice(36, 58)
BODY_SLICE = slice(58, 65)

DATASET_JOINT_NAMES = (
    *(f"left_arm_j{i}" for i in range(ARM_DIM)),
    *(f"left_hand_j{i}" for i in range(HAND_DIM)),
    *(f"right_arm_j{i}" for i in range(ARM_DIM)),
    *(f"right_hand_j{i}" for i in range(HAND_DIM)),
    *(f"motor_j{i}" for i in range(BODY_DIM)),
)


class Joint65CommandRejected(ValueError):
    """A 65-D state/command failed a fail-closed validation."""


@dataclass(frozen=True)
class Joint65Parts:
    left_arm: np.ndarray
    left_hand: np.ndarray
    right_arm: np.ndarray
    right_hand: np.ndarray
    body: np.ndarray


def _vector(value: Any, dimension: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (dimension,):
        raise Joint65CommandRejected(
            f"{label} must have shape ({dimension},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        bad = np.flatnonzero(~np.isfinite(array))
        raise Joint65CommandRejected(
            f"{label} contains NaN/Inf at indices {bad.tolist()}"
        )
    return array


def split_absolute_joint65(value: Any, label: str = "absolute_joint65") -> Joint65Parts:
    """Validate and split one absolute 65-D joint vector."""
    vector = _vector(value, ACTION_DIM, label)
    return Joint65Parts(
        left_arm=vector[LEFT_ARM_SLICE].copy(),
        left_hand=vector[LEFT_HAND_SLICE].copy(),
        right_arm=vector[RIGHT_ARM_SLICE].copy(),
        right_hand=vector[RIGHT_HAND_SLICE].copy(),
        body=vector[BODY_SLICE].copy(),
    )


def assemble_absolute_joint65(
    left_arm: Any,
    left_hand: Any,
    right_arm: Any,
    right_hand: Any,
    body: Any,
) -> np.ndarray:
    """Assemble one state in the exact training-time joint order."""
    result = np.concatenate(
        [
            _vector(left_arm, ARM_DIM, "left_arm"),
            _vector(left_hand, HAND_DIM, "left_hand"),
            _vector(right_arm, ARM_DIM, "right_arm"),
            _vector(right_hand, HAND_DIM, "right_hand"),
            _vector(body, BODY_DIM, "body"),
        ]
    )
    assert result.shape == (ACTION_DIM,)
    return result


def parse_body_joint_map(entries: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Parse seven ``component:joint_name`` bindings in motor_j0..j6 order."""
    if len(entries) != BODY_DIM:
        raise Joint65CommandRejected(
            "absolute_joint65_body_joint_map must contain exactly seven "
            f"component:joint_name entries in motor_j0..motor_j6 order; got {len(entries)}"
        )
    parsed = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, str) or ":" not in entry:
            raise Joint65CommandRejected(
                f"motor_j{index} binding must be 'component:joint_name', got {entry!r}"
            )
        component, joint_name = (part.strip() for part in entry.split(":", 1))
        if not component or not joint_name:
            raise Joint65CommandRejected(
                f"motor_j{index} binding is incomplete: {entry!r}"
            )
        parsed.append((component, joint_name))
    if len(set(parsed)) != BODY_DIM:
        raise Joint65CommandRejected("body joint bindings must be unique")
    return tuple(parsed)


class BodyJointBinding:
    """Validated mapping between motor_j0..j6 and robot SDK component joints."""

    def __init__(
        self,
        robot: Any,
        entries: Sequence[str],
        *,
        lower_limits: Sequence[float] = (),
        upper_limits: Sequence[float] = (),
    ) -> None:
        self.robot = robot
        self.entries = parse_body_joint_map(entries)
        component_map = robot.get_controllable_component_map()
        forbidden = {"left_arm", "right_arm", "left_hand", "right_hand"}

        self._components: dict[str, Any] = {}
        self._component_joint_names: dict[str, tuple[str, ...]] = {}
        sdk_lower = np.empty(BODY_DIM, dtype=np.float64)
        sdk_upper = np.empty(BODY_DIM, dtype=np.float64)
        for index, (component_name, joint_name) in enumerate(self.entries):
            if component_name in forbidden:
                raise Joint65CommandRejected(
                    f"motor_j{index} cannot alias {component_name}:{joint_name}; "
                    "the first 58 dimensions already command arms and hands"
                )
            if component_name not in component_map:
                raise Joint65CommandRejected(
                    f"motor_j{index} component {component_name!r} is unavailable; "
                    f"SDK exposes {sorted(component_map)}"
                )
            component = component_map[component_name]
            self._components[component_name] = component
            names = tuple(component.joint_name)
            self._component_joint_names[component_name] = names
            if joint_name not in names:
                raise Joint65CommandRejected(
                    f"motor_j{index} joint {joint_name!r} is not in component "
                    f"{component_name!r}: {list(names)}"
                )
            limits = component.joint_pos_limit
            if limits is None:
                sdk_lower[index] = np.nan
                sdk_upper[index] = np.nan
            else:
                joint_index = names.index(joint_name)
                sdk_lower[index], sdk_upper[index] = np.asarray(limits)[joint_index]

        if len(lower_limits) or len(upper_limits):
            if len(lower_limits) != BODY_DIM or len(upper_limits) != BODY_DIM:
                raise Joint65CommandRejected(
                    "explicit body lower/upper limits must both contain seven values"
                )
            self.lower_limits = _vector(lower_limits, BODY_DIM, "body lower limits")
            self.upper_limits = _vector(upper_limits, BODY_DIM, "body upper limits")
        else:
            if not np.isfinite(sdk_lower).all() or not np.isfinite(sdk_upper).all():
                raise Joint65CommandRejected(
                    "SDK does not expose all seven body joint limits; configure "
                    "absolute_joint65_body_lower_limits and "
                    "absolute_joint65_body_upper_limits explicitly"
                )
            self.lower_limits = sdk_lower
            self.upper_limits = sdk_upper
        if np.any(self.lower_limits >= self.upper_limits):
            raise Joint65CommandRejected("every body lower limit must be below its upper limit")

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(component for component, _ in self.entries))

    def read(self) -> np.ndarray:
        values_by_component = {
            component_name: component.get_joint_pos_dict()
            for component_name, component in self._components.items()
        }
        try:
            result = np.asarray(
                [
                    values_by_component[component_name][joint_name]
                    for component_name, joint_name in self.entries
                ],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise Joint65CommandRejected(
                f"SDK state is missing configured body joint {exc.args[0]!r}"
            ) from exc
        return _vector(result, BODY_DIM, "body state")

    def validate_target(self, target: Any) -> np.ndarray:
        target = _vector(target, BODY_DIM, "body target")
        violations = np.flatnonzero(
            (target < self.lower_limits) | (target > self.upper_limits)
        )
        if violations.size:
            details = [
                {
                    "motor": f"motor_j{int(index)}",
                    "target": float(target[index]),
                    "lower": float(self.lower_limits[index]),
                    "upper": float(self.upper_limits[index]),
                }
                for index in violations
            ]
            raise Joint65CommandRejected(f"body target exceeds physical limits: {details}")
        return target

    def component_targets(self, target: Any) -> dict[str, np.ndarray]:
        """Build full component arrays, preserving unbound joints."""
        target = self.validate_target(target)
        result: dict[str, np.ndarray] = {}
        for component_name, component in self._components.items():
            current = component.get_joint_pos_dict()
            names = self._component_joint_names[component_name]
            result[component_name] = np.asarray(
                [current[name] for name in names], dtype=np.float64
            )
        for value, (component_name, joint_name) in zip(target, self.entries):
            index = self._component_joint_names[component_name].index(joint_name)
            result[component_name][index] = value
        return result


class AbsoluteJoint65Adapter:
    """Read state and create commands without changing the 65-D semantics."""

    def __init__(
        self,
        robot: Any,
        left_hand: Any,
        right_hand: Any,
        body_binding: BodyJointBinding,
        *,
        left_arm_joint_names: Sequence[str],
        right_arm_joint_names: Sequence[str],
        max_step_rad: float,
        arm_hand_lower_limits: Sequence[float] = (),
        arm_hand_upper_limits: Sequence[float] = (),
    ) -> None:
        if max_step_rad <= 0:
            raise Joint65CommandRejected("absolute_joint65_max_step_rad must be positive")
        if len(left_arm_joint_names) != ARM_DIM or len(right_arm_joint_names) != ARM_DIM:
            raise Joint65CommandRejected("both arm joint-name lists must contain seven joints")
        self.robot = robot
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.body_binding = body_binding
        self.left_arm_joint_names = tuple(left_arm_joint_names)
        self.right_arm_joint_names = tuple(right_arm_joint_names)
        self.max_step_rad = float(max_step_rad)
        if len(arm_hand_lower_limits) or len(arm_hand_upper_limits):
            self.arm_hand_lower_limits = _vector(
                arm_hand_lower_limits, 58, "arm/hand lower limits"
            )
            self.arm_hand_upper_limits = _vector(
                arm_hand_upper_limits, 58, "arm/hand upper limits"
            )
            if np.any(self.arm_hand_lower_limits >= self.arm_hand_upper_limits):
                raise Joint65CommandRejected(
                    "every arm/hand lower limit must be below its upper limit"
                )
        else:
            self.arm_hand_lower_limits = None
            self.arm_hand_upper_limits = None

    def read_state(self) -> np.ndarray:
        joints = self.robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
        left_arm = [joints[name] for name in self.left_arm_joint_names]
        right_arm = [joints[name] for name in self.right_arm_joint_names]
        left_hand = self.left_hand.get_states().angles
        right_hand = self.right_hand.get_states().angles
        return assemble_absolute_joint65(
            left_arm, left_hand, right_arm, right_hand, self.body_binding.read()
        )

    def prepare(
        self, target: Any, current: Any
    ) -> tuple[Joint65Parts, dict[str, np.ndarray]]:
        target_vector = _vector(target, ACTION_DIM, "absolute_joint65 target")
        current_vector = _vector(current, ACTION_DIM, "absolute_joint65 current state")
        if self.arm_hand_lower_limits is not None:
            first58 = target_vector[:58]
            limit_violations = np.flatnonzero(
                (first58 < self.arm_hand_lower_limits)
                | (first58 > self.arm_hand_upper_limits)
            )
            if limit_violations.size:
                details = [
                    {
                        "joint": DATASET_JOINT_NAMES[int(index)],
                        "target": float(first58[index]),
                        "lower": float(self.arm_hand_lower_limits[index]),
                        "upper": float(self.arm_hand_upper_limits[index]),
                    }
                    for index in limit_violations
                ]
                raise Joint65CommandRejected(
                    f"arm/hand target exceeds controller limits: {details}"
                )
        step = np.abs(target_vector - current_vector)
        violations = np.flatnonzero(step > self.max_step_rad)
        if violations.size:
            details = [
                {
                    "joint": DATASET_JOINT_NAMES[int(index)],
                    "step_rad": float(step[index]),
                    "max_step_rad": self.max_step_rad,
                }
                for index in violations
            ]
            raise Joint65CommandRejected(
                f"absolute_joint65 step rejected (no silent clipping): {details}"
            )
        parts = split_absolute_joint65(target_vector)
        return parts, self.body_binding.component_targets(parts.body)

    @staticmethod
    def arm_hand_targets(parts: Joint65Parts) -> dict[str, np.ndarray]:
        return {
            "left_arm": parts.left_arm,
            "left_hand": parts.left_hand,
            "right_arm": parts.right_arm,
            "right_hand": parts.right_hand,
        }
