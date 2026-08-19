#!/usr/bin/env python3
"""T-Rex policy service for the Origami office-inference contract.

The model implementation is intentionally kept in the T-Rex inference code
(``scripts.test``).  This file owns the production boundary: it validates the
full MessagePack observation, adapts it to T-Rex's three-camera/10-tile input
layout, and exposes only the three fixed Zenoh queryables required by
``origami-zenoh-v1``.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import signal
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgpack
import numpy as np
import torch
import zenoh
from PIL import Image

# The T-Rex checkout is copied to /opt/trex in the production image.  Importing
# this module does not start its legacy ZMQ server; it only supplies the model
# loader and the cascaded inference engine.
from scripts.test import CascadedServer, _pil_to_bytes, model_load


TRANSPORT_VERSION = "origami-zenoh-v1"
SEMANTIC_VERSION = "origami-v1"
ACTION_DIM = 65
ACTION_HORIZON = 16
IMAGE_SHAPE = (224, 224, 3)
TACTILE_DEFORM_SHAPE = (480, 1200, 3)
TACTILE_RAW_SHAPE = (480, 1600, 3)
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

REQUIRED_IMAGE_SPECS = {
    "observation/image/head_left": IMAGE_SHAPE,
    "observation/image/head_right": IMAGE_SHAPE,
    "observation/image/wrist_left": IMAGE_SHAPE,
    "observation/image/wrist_right": IMAGE_SHAPE,
    "observation/image/tactile_deform": TACTILE_DEFORM_SHAPE,
}
OPTIONAL_IMAGE_SPECS = {
    "observation/image/tactile_raw": TACTILE_RAW_SHAPE,
}
VECTOR_SPECS = {
    "observation/state": (ACTION_DIM,),
    "observation/state/joint_torque": (ACTION_DIM,),
    "observation/tactile": (60,),
}


def _hand_joint_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_thumb_CMC_FE",
        f"{side}_thumb_CMC_AA",
        f"{side}_thumb_MCP_FE",
        f"{side}_thumb_MCP_AA",
        f"{side}_thumb_IP",
        f"{side}_index_MCP_FE",
        f"{side}_index_MCP_AA",
        f"{side}_index_PIP",
        f"{side}_index_DIP",
        f"{side}_middle_MCP_FE",
        f"{side}_middle_MCP_AA",
        f"{side}_middle_PIP",
        f"{side}_middle_DIP",
        f"{side}_ring_MCP_FE",
        f"{side}_ring_MCP_AA",
        f"{side}_ring_PIP",
        f"{side}_ring_DIP",
        f"{side}_pinky_CMC",
        f"{side}_pinky_MCP_FE",
        f"{side}_pinky_MCP_AA",
        f"{side}_pinky_PIP",
        f"{side}_pinky_DIP",
    )


JOINT_NAMES = (
    tuple(f"left_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("left")
    + tuple(f"right_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("right")
    + (
        "lower_body_joint_1",
        "lower_body_joint_2",
        "lower_body_joint_3",
        "lower_body_joint_4",
        "lower_body_joint_5",
        "neck_joint_1",
        "neck_joint_2",
    )
)

if len(JOINT_NAMES) != ACTION_DIM or len(set(JOINT_NAMES)) != ACTION_DIM:
    raise RuntimeError("the Origami joint contract must contain 65 unique names")


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"} or value.dtype.hasobject:
            raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(order="C"),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        if value.dtype.kind in {"O", "V", "c"} or value.dtype.hasobject:
            raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _mapping_value(value: Mapping[Any, Any], key: str) -> Any:
    if key in value:
        return value[key]
    return value.get(key.encode("ascii"))


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if _mapping_value(value, "__ndarray__") is True:
        data = _mapping_value(value, "data")
        shape = _mapping_value(value, "shape")
        raw_dtype = _mapping_value(value, "dtype")
        try:
            dtype = np.dtype(raw_dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid NumPy dtype") from exc
        if (
            not isinstance(data, bytes)
            or not isinstance(shape, (list, tuple))
            or len(shape) > 8
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 0
                for dimension in shape
            )
            or dtype.kind in {"O", "V", "c"}
            or dtype.hasobject
        ):
            raise ValueError("invalid NumPy array payload")
        normalized_shape = tuple(int(dimension) for dimension in shape)
        expected_size = math.prod(normalized_shape) * dtype.itemsize
        if expected_size > MAX_PAYLOAD_BYTES or len(data) != expected_size:
            raise ValueError("NumPy array payload size does not match shape")
        return np.frombuffer(data, dtype=dtype).reshape(normalized_shape)
    if _mapping_value(value, "__npgeneric__") is True:
        raw_dtype = _mapping_value(value, "dtype")
        dtype = np.dtype(raw_dtype)
        if dtype.kind in {"O", "V", "c"} or dtype.hasobject:
            raise ValueError("invalid NumPy scalar dtype")
        return dtype.type(_mapping_value(value, "data"))
    return value


def pack_payload(value: Any) -> bytes:
    payload = msgpack.packb(value, default=_pack_numpy, use_bin_type=True)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("response exceeds 64 MiB")
    return payload


def unpack_payload(value: Any) -> Any:
    payload = value.to_bytes() if hasattr(value, "to_bytes") else bytes(value)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("request exceeds 64 MiB")
    return msgpack.unpackb(
        payload,
        object_hook=_unpack_numpy,
        raw=False,
        strict_map_key=False,
        max_bin_len=MAX_PAYLOAD_BYTES,
        max_array_len=1_000_000,
        max_map_len=10_000,
        max_str_len=1_000_000,
    )


def _validate_endpoint(endpoint: str) -> str:
    """Validate the public TCP endpoint without resolving or connecting to it."""
    if not isinstance(endpoint, str) or len(endpoint) > 512:
        raise ValueError("ORIGAMI_ZENOH_ENDPOINT must be a short string")
    match = re.fullmatch(
        r"tcp/(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9][A-Za-z0-9_.-]*):([0-9]{1,5})",
        endpoint,
    )
    if match is None or not 1 <= int(match.group(1)) <= 65535:
        raise ValueError(
            "ORIGAMI_ZENOH_ENDPOINT must match tcp/<hostname-or-ip>:<port>"
        )
    return endpoint


def _validate_session_id(session_id: str) -> str:
    if (
        not isinstance(session_id, str)
        or not session_id
        or len(session_id) > 512
        or "\x00" in session_id
        or "\n" in session_id
        or "\r" in session_id
    ):
        raise ValueError("ORIGAMI_SESSION_ID must be a non-empty opaque string")
    return session_id


def _as_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def _split_tactile_deform(mosaic: np.ndarray) -> np.ndarray:
    """Convert the office 2x5 RGB mosaic to T-Rex's [10, 240, 240] input."""
    if mosaic.shape != TACTILE_DEFORM_SHAPE:
        raise ValueError(f"unexpected tactile deform shape: {mosaic.shape}")
    gray = mosaic[..., 0]
    tiles = [
        gray[row * 240 : (row + 1) * 240, col * 240 : (col + 1) * 240]
        for row in range(2)
        for col in range(5)
    ]
    return np.stack(tiles, axis=0).astype(np.float32, copy=False)


class TrexPolicy:
    """Self-contained adapter around the fixed T-Rex checkpoint."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("T-Rex office inference requires a CUDA device")

        self.approx_async = os.environ.get("TREX_APPROX_ASYNC", "1").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        try:
            self.slow_every_n = max(
                1, int(os.environ.get("TREX_SLOW_EVERY_N", "4"))
            )
        except ValueError as exc:
            raise ValueError("TREX_SLOW_EVERY_N must be a positive integer") from exc
        self._requests_since_slow = 0
        self._slow_generation = 0
        self.last_timing: dict[str, Any] = {}

        checkpoint = Path(
            os.environ.get("TREX_CHECKPOINT", "/opt/policy/checkpoint")
        ).resolve()
        base_model = Path(
            os.environ.get("TREX_BASE_MODEL", "/opt/trex/base_model")
        ).resolve()
        stats_path = Path(
            os.environ.get("TREX_STATS_PATH", "/opt/policy/assets/stats.json")
        ).resolve()
        cuda_index = os.environ.get("TREX_CUDA_INDEX", "0")

        for path, label in (
            (checkpoint, "T-Rex checkpoint"),
            (base_model, "Qwen base model"),
            (stats_path, "normalization statistics"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} is missing: {path}")

        args = SimpleNamespace(
            checkpoint_path=str(checkpoint),
            base_model_path=str(base_model),
            stats_path=str(stats_path),
            lerobot_root="",
            dataset_name="",
            action_dim=None,
            action_chunk=None,
            use_robot_state=None,
            use_tactile_deform=None,
            use_tactile_vec=None,
            tactile_intermediate_size=None,
            n_flare_tokens_per_frame=None,
            n_flare_steps=None,
            cuda=cuda_index,
            port=0,
            image_size=None,
            action_lora_rank=None,
            action_lora_alpha=None,
            allow_non_strict_checkpoint=False,
            require_action_lora=True,
            cascaded_total_steps=None,
            cascaded_split_step=None,
            disable_tactile=0,
            use_tactile_code=None,
            vqvae_codebook_size=None,
            vqvae_ckpt="",
        )

        logging.info("loading T-Rex checkpoint")
        model, processor, statistic = model_load(args)
        if args.action_dim != ACTION_DIM:
            raise RuntimeError(
                f"checkpoint action dimension is {args.action_dim}, expected {ACTION_DIM}"
            )
        if args.action_chunk != ACTION_HORIZON:
            raise RuntimeError(
                f"checkpoint action horizon is {args.action_chunk}, expected {ACTION_HORIZON}"
            )

        self.engine = CascadedServer(args, model, processor, statistic)
        self.action_horizon = int(args.action_chunk)
        self._warm_up()

    def _warm_up(self) -> None:
        """Compile/load the first visual and tactile path before readiness."""
        black = Image.new("RGB", (224, 224), color="black")
        dummy_payload = {
            "image_head": _pil_to_bytes(black),
            "image_wrist_right": _pil_to_bytes(black),
            "image_wrist_left": _pil_to_bytes(black),
            "task_description": "warmup",
            "tactile_f6": np.zeros((10, 6), dtype=np.float32),
            "tactile_deform": np.zeros((10, 240, 240), dtype=np.float32),
            "state_fast": np.zeros((ACTION_DIM,), dtype=np.float32),
        }
        result = self.engine.predict("slow_and_fast", dummy_payload)
        actions = np.asarray(result["actions"])
        if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise RuntimeError(f"T-Rex warm-up returned invalid actions: {actions.shape}")
        self.reset()
        logging.info("T-Rex model ready")

    def reset(self) -> None:
        """Clear all episode-scoped cascaded/VQ-VAE state."""
        with self.engine.lock:
            self.engine.cached_kv = None
            self.engine.x_split = None
            self.engine.tau_split = None
            self.engine.position_ids = None
            self.engine.attention_mask = None
            self.engine.n_action_in_cache = 0
            self.engine.chunk_id = -1
            self.engine.last_actions = None
            self.engine.f6_buffer.clear()
        self._requests_since_slow = 0
        self._slow_generation = 0
        self.last_timing = {}

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        started = time.monotonic()
        tactile_deform = _split_tactile_deform(
            observation["observation/image/tactile_deform"]
        )
        payload = {
            # T-Rex was trained with head_left as the slow image and with the
            # right/left wrist images in this order.  head_right is accepted at
            # the office boundary but is not part of this checkpoint's graph.
            "image_head": _pil_to_bytes(
                _as_pil(observation["observation/image/head_left"])
            ),
            "image_wrist_right": _pil_to_bytes(
                _as_pil(observation["observation/image/wrist_right"])
            ),
            "image_wrist_left": _pil_to_bytes(
                _as_pil(observation["observation/image/wrist_left"])
            ),
            "task_description": observation["prompt"],
            "tactile_f6": observation["observation/tactile"].reshape(10, 6),
            "tactile_deform": tactile_deform,
            "state_fast": observation["observation/state"],
        }
        slow_due = (
            self._slow_generation == 0
            or self._requests_since_slow >= self.slow_every_n
        )

        slow_ms: float | None = None
        if not self.approx_async:
            slow_due = True

        if slow_due:
            slow_started = time.monotonic()
            self.engine.predict("slow", payload)
            slow_ms = (time.monotonic() - slow_started) * 1000.0
            self._slow_generation += 1
            self._requests_since_slow = 1
            policy_mode = "slow+fast"
        else:
            self._requests_since_slow += 1
            policy_mode = "fast"

        fast_started = time.monotonic()
        result = self.engine.predict("fast", payload)
        fast_ms = (time.monotonic() - fast_started) * 1000.0
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (self.action_horizon, ACTION_DIM):
            raise RuntimeError(f"T-Rex returned invalid action shape: {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("T-Rex returned NaN or Inf")
        self.last_timing = {
            "policy_mode": policy_mode,
            "slow_ms": slow_ms,
            "fast_ms": fast_ms,
            "policy_ms": (time.monotonic() - started) * 1000.0,
            "slow_generation": self._slow_generation,
            "slow_every_n": self.slow_every_n,
            "requests_since_slow": self._requests_since_slow,
        }
        return np.ascontiguousarray(actions)


class OrigamiZenohServer:
    def __init__(self, policy: TrexPolicy, *, endpoint: str, session_id: str) -> None:
        self.policy = policy
        self.endpoint = _validate_endpoint(endpoint)
        self.session_id = _validate_session_id(session_id)
        self._policy_lock = threading.Lock()
        self._stop = threading.Event()
        self._session: Any | None = None
        self._queryables: list[Any] = []
        self.metadata = {
            "protocol_version": SEMANTIC_VERSION,
            "action_dim": ACTION_DIM,
            "action_horizon": policy.action_horizon,
            "action_type": "absolute_joint_position",
            "action_units": "radians",
            "joint_names": JOINT_NAMES,
        }

    def serve_forever(self) -> None:
        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("connect/endpoints", json.dumps([self.endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("transport/shared_memory/enabled", "false")
        self._session = zenoh.open(config)
        self._queryables = [
            self._session.declare_queryable(
                f"{TRANSPORT_VERSION}/{operation}",
                self._handle_query,
                complete=True,
            )
            for operation in ("metadata", "reset", "infer")
        ]
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        logging.info("READY horizon=%d", self.policy.action_horizon)
        self._stop.wait()
        for queryable in self._queryables:
            queryable.undeclare()
        if self._session is not None:
            self._session.close()

    def _handle_query(self, query: Any) -> None:
        operation = str(query.key_expr).rsplit("/", 1)[-1]
        request: Any = None
        try:
            request = unpack_payload(query.payload)
            response = self.process(operation, request)
        except Exception as exc:  # noqa: BLE001 - sanitized protocol error
            error_id = uuid.uuid4().hex
            logging.error(
                "request failed operation=%s error_id=%s type=%s",
                operation,
                error_id,
                type(exc).__name__,
            )
            if not isinstance(request, Mapping):
                query.reply_err(
                    pack_payload(
                        {
                            "error": {
                                "code": "INVALID_REQUEST",
                                "message": f"request failed; error_id={error_id}",
                                "retryable": False,
                            }
                        }
                    ),
                    encoding="application/msgpack",
                )
                return
            response = self._envelope(operation, request)
            response["error"] = {
                "code": "INFERENCE_FAILED" if operation == "infer" else "INVALID_REQUEST",
                "message": f"request failed; error_id={error_id}",
                "retryable": False,
            }
        query.reply(
            str(query.key_expr),
            pack_payload(response),
            encoding="application/msgpack",
        )

    def process(self, operation: str, request: Any) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("request must be a MessagePack map")
        response = self._envelope(operation, request)
        if request.get("protocol_version") != TRANSPORT_VERSION:
            raise ValueError("invalid protocol_version")
        if request.get("operation") != operation:
            raise ValueError("operation does not match queryable key")
        if request.get("session_id") != self.session_id:
            raise ValueError("session_id does not match assigned session")
        if not isinstance(request.get("request_id"), str) or not request["request_id"]:
            raise ValueError("request_id must be a non-empty string")

        if operation == "metadata":
            response["metadata"] = self.metadata
            return response
        if operation == "reset":
            with self._policy_lock:
                self.policy.reset()
            response["ok"] = True
            return response
        if operation != "infer":
            raise ValueError(f"unsupported operation: {operation}")

        observation = request.get("observation")
        self._validate_observation(observation)
        started = time.monotonic()
        with self._policy_lock:
            actions = self.policy.infer(observation)
            policy_timing = dict(self.policy.last_timing)
        actions = np.asarray(actions)
        expected_shape = (self.policy.action_horizon, ACTION_DIM)
        if actions.dtype != np.float32 or actions.shape != expected_shape:
            raise ValueError(
                f"policy actions must be float32{expected_shape}, "
                f"got {actions.dtype}{actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy actions contain NaN or Inf")
        response["actions"] = np.ascontiguousarray(actions)
        response["server_timing"] = {
            "infer_ms": (time.monotonic() - started) * 1000.0,
            **policy_timing,
        }
        return response

    def _envelope(self, operation: str, request: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "protocol_version": TRANSPORT_VERSION,
            "operation": operation,
            "request_id": request.get("request_id"),
            "session_id": request.get("session_id"),
        }

    @staticmethod
    def _validate_observation(observation: Any) -> None:
        if not isinstance(observation, Mapping):
            raise ValueError("infer request must contain an observation map")
        required = {*REQUIRED_IMAGE_SPECS, *VECTOR_SPECS, "prompt"}
        allowed = required | set(OPTIONAL_IMAGE_SPECS)
        if not required.issubset(observation) or not set(observation).issubset(allowed):
            raise ValueError("observation keys do not match the public full schema")
        for key, shape in {**REQUIRED_IMAGE_SPECS, **OPTIONAL_IMAGE_SPECS}.items():
            if key not in observation:
                continue
            image = observation[key]
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != shape:
                raise ValueError(f"{key} must be uint8{shape}")
        for key, shape in VECTOR_SPECS.items():
            vector = observation[key]
            if (
                not isinstance(vector, np.ndarray)
                or vector.dtype != np.float32
                or vector.shape != shape
                or not np.isfinite(vector).all()
            ):
                raise ValueError(f"{key} must be finite float32{shape}")
        if not isinstance(observation.get("prompt"), str):
            raise ValueError("prompt must be a string")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    endpoint = os.environ.get("ORIGAMI_ZENOH_ENDPOINT")
    session_id = os.environ.get("ORIGAMI_SESSION_ID")
    if endpoint is None:
        raise SystemExit("ORIGAMI_ZENOH_ENDPOINT is required")
    if session_id is None:
        raise SystemExit("ORIGAMI_SESSION_ID is required")
    endpoint = _validate_endpoint(endpoint)
    session_id = _validate_session_id(session_id)
    policy = TrexPolicy()
    OrigamiZenohServer(
        policy,
        endpoint=endpoint,
        session_id=session_id,
    ).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
