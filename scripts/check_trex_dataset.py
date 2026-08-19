#!/usr/bin/env python3
"""Preflight checks for the local Robotic Origami LeRobot v3.0 collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FEATURES = {
    "observation.images.head_left": [480, 480, 3],
    "observation.images.wrist_left": [480, 480, 3],
    "observation.images.wrist_right": [480, 480, 3],
    "observation.state": [65],
    "action": [65],
    "observation.tactile": [60],
    "observation.images.tactile_deform": [480, 1200, 3],
}


def discover_roots(dataset_root: Path) -> list[Path]:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if (dataset_root / "meta" / "info.json").is_file():
        roots = [dataset_root]
    else:
        roots = sorted(path for path in dataset_root.glob("season_*/lerobot3.0") if path.is_dir())
    if not roots:
        raise FileNotFoundError(
            f"no LeRobot v3.0 data found under {dataset_root}; expected "
            "either <root>/meta/info.json or <root>/season_*/lerobot3.0/meta/info.json"
        )
    return roots


def _assert_not_lfs_pointer(path: Path) -> None:
    with path.open("rb") as handle:
        head = handle.read(256)
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} is a Git-LFS pointer, not the downloaded file. "
            "Run git lfs pull (or download the large files) on the host."
        )


def _first_file(root: Path, pattern: str) -> Path | None:
    return next(root.rglob(pattern), None)


def validate(dataset_root: Path, check_video: bool) -> int:
    roots = discover_roots(dataset_root)
    reference_shapes = None
    fps_values = set()
    total_episodes = 0
    total_frames = 0

    try:
        import av
    except ImportError as exc:
        if check_video:
            raise RuntimeError("PyAV is required for --check-video") from exc
        av = None

    for root in roots:
        info_path = root / "meta" / "info.json"
        stats_path = root / "meta" / "stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"missing statistics: {stats_path}")
        if not (root / "meta" / "episodes").exists():
            raise FileNotFoundError(f"missing episode metadata directory: {root / 'meta' / 'episodes'}")

        info = json.loads(info_path.read_text())
        stats = json.loads(stats_path.read_text())
        if info.get("codebase_version") not in {"v3.0", "3.0"}:
            raise ValueError(f"{root} is not LeRobot v3.0: {info.get('codebase_version')!r}")
        if not all(key in stats for key in ("action", "observation.state", "observation.tactile")):
            raise KeyError(f"{stats_path} lacks action/state/tactile normalization statistics")

        features = info.get("features", {})
        shapes = {}
        for key, expected_shape in REQUIRED_FEATURES.items():
            if key not in features:
                raise KeyError(f"{root} is missing required feature {key!r}")
            actual_shape = features[key].get("shape")
            shapes[key] = actual_shape
            if actual_shape != expected_shape:
                raise ValueError(f"{root}: {key} has shape {actual_shape}, expected {expected_shape}")
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            raise ValueError(f"feature shape mismatch between seasons at {root}")

        data_file = _first_file(root / "data", "*.parquet")
        video_file = _first_file(root / "videos", "*.mp4")
        if data_file is None or data_file.stat().st_size < 1024:
            raise RuntimeError(f"{root}: no usable parquet data file found")
        if video_file is None or video_file.stat().st_size < 1024:
            raise RuntimeError(f"{root}: no usable MP4 video file found")
        _assert_not_lfs_pointer(data_file)
        _assert_not_lfs_pointer(video_file)

        if check_video:
            try:
                with av.open(str(video_file)) as container:
                    frame = next(container.decode(video=0), None)
                    if frame is None:
                        raise RuntimeError("video contains no decodable frame")
            except Exception as exc:
                raise RuntimeError(f"PyAV cannot decode {video_file}: {exc}") from exc

        fps_values.add(int(info["fps"]))
        total_episodes += int(info["total_episodes"])
        total_frames += int(info["total_frames"])

    if len(fps_values) != 1:
        raise ValueError(f"all seasons must have one common fps, found {sorted(fps_values)}")
    print(
        f"OK: {len(roots)} season(s), {total_episodes:,} episodes, "
        f"{total_frames:,} frames, fps={next(iter(fps_values))}, "
        "using the current Origami LeRobot path"
    )
    for root in roots:
        print(f"  {root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--check-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open and decode one MP4 per season with PyAV (default: enabled).",
    )
    args = parser.parse_args()
    return validate(args.dataset_root, args.check_video)


if __name__ == "__main__":
    raise SystemExit(main())
