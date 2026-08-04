"""Train T-Rex with the Origami LeRobot adapter.

This wrapper leaves scripts/train.py untouched.  It exposes
OrigamiLeRobotDataset under the name expected by the original training entry,
then executes scripts/train.py normally.
"""
from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

from qwen_vla.origami_lerobot_dataset import OrigamiLeRobotDataset


def main() -> None:
    shim = types.ModuleType("qwen_vla.lerobot_dataset")
    shim.TRexLeRobotDataset = OrigamiLeRobotDataset
    sys.modules["qwen_vla.lerobot_dataset"] = shim

    train_py = Path(__file__).with_name("train.py")
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()
