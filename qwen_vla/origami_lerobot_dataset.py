"""Origami LeRobot v3.0 adapter for T-Rex training.

This file intentionally does not modify the original T-Rex LeRobot loader.

Input dataset schema:
  - observation.images.head_left
  - observation.images.wrist_left
  - observation.images.wrist_right
  - observation.state: [65]
  - action: [65]
  - observation.tactile: [60], ordered as
    left/right thumb, index, middle, ring, little; each [fx, fy, fz, tx, ty, tz]
  - observation.images.tactile_deform: 5x2 mosaic video

Output batch contract matches T-Rex:
  - action is expanded online to [action_chunk, 65] future chunks
  - tactile force is reshaped to [10, 6]
  - tactile deform mosaic is split in memory to [10, 1, H/2, W/5]
"""
from __future__ import annotations

import copy
import json
import os
from typing import Dict, List, Optional

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F


ACTION_DIM = 65
TACTILE_DIM = 60
N_FINGERS = 10
F6_PER_FINGER = 6

KEY_HEAD = "observation.images.head_left"
KEY_WRIST_R = "observation.images.wrist_right"
KEY_WRIST_L = "observation.images.wrist_left"
KEY_STATE = "observation.state"
KEY_ACTION = "action"
KEY_TACF6 = "observation.tactile"
KEY_DEFORM = "observation.images.tactile_deform"


def _normalize(values, mask, vmin, vmax):
    return np.where(mask, np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1), values)


def _stat(native_stats: dict, key: str, dim: int, sub: str):
    block = native_stats.get(key, {})
    if sub == "mask":
        return np.ones(dim, dtype=bool)
    if sub == "q01":
        return np.asarray(block.get("q01", block.get("min", [0.0] * dim)), dtype=np.float32)
    if sub == "q99":
        return np.asarray(block.get("q99", block.get("max", [1.0] * dim)), dtype=np.float32)
    raise KeyError(sub)


class OrigamiLeRobotDataset(torch.utils.data.Dataset):
    """LeRobot loader for the 65D Origami dataset.

    It expands single-frame joint targets into T-Rex action chunks using
    LeRobot delta_timestamps, following the future-chunk idea in
    utils/convert_inlab_to_lerobot.py but without rewriting the dataset.
    """

    def __init__(self, config, processor, accelerator, episodes: Optional[List[int]] = None, _ds=None):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        self.root = config.lerobot_root
        repo_id = getattr(config, "lerobot_repo_id", "") or os.path.basename(self.root.rstrip("/"))

        with open(os.path.join(self.root, "meta", "info.json")) as f:
            info = json.load(f)
        self.fps = int(info["fps"])
        feats = info["features"]

        required = [KEY_HEAD, KEY_WRIST_R, KEY_WRIST_L, KEY_STATE, KEY_ACTION, KEY_TACF6]
        missing = [k for k in required if k not in feats]
        if missing:
            raise KeyError(f"Origami dataset missing required feature(s): {missing}")

        self.has_wrist = KEY_WRIST_R in feats and KEY_WRIST_L in feats
        self.has_tactile = KEY_TACF6 in feats
        self.has_deform = KEY_DEFORM in feats

        self.image_size = tuple(config.image_size) if getattr(config, "image_size", None) else None
        self.use_flare = bool(getattr(config, "use_flare", 0))
        self.n_flare_steps = int(getattr(config, "n_flare_steps", 0)) if self.use_flare else 0
        self.flare_stride = int(getattr(config, "flare_frame_stride", 1))
        self.use_tactile_vec = bool(getattr(config, "use_tactile_vec", 0))
        self.use_tactile_deform = bool(getattr(config, "use_tactile_deform", 0))
        self.use_tactile_vqvae = bool(getattr(config, "use_tactile_vqvae", 0))
        self.use_robot_state = bool(getattr(config, "use_robot_state", 0))
        self.vqvae_window = int(getattr(config, "vqvae_window", 16))
        self.action_dim = int(getattr(config, "action_dim", ACTION_DIM))
        self.action_chunk = int(getattr(config, "action_chunk", 16))
        self.video_backend = getattr(config, "video_backend", "pyav")
        self.video_tolerance_s = float(getattr(config, "video_tolerance_s", 1e-3))
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"Origami loader expects action_dim={ACTION_DIM}, got {self.action_dim}")

        with open(os.path.join(self.root, "meta", "stats.json")) as f:
            native_stats = json.load(f)
        self.action_mask = _stat(native_stats, KEY_ACTION, ACTION_DIM, "mask")
        self.action_min = _stat(native_stats, KEY_ACTION, ACTION_DIM, "q01")
        self.action_max = _stat(native_stats, KEY_ACTION, ACTION_DIM, "q99")
        self.state_mask = _stat(native_stats, KEY_STATE, ACTION_DIM, "mask")
        self.state_min = _stat(native_stats, KEY_STATE, ACTION_DIM, "q01")
        self.state_max = _stat(native_stats, KEY_STATE, ACTION_DIM, "q99")
        self.tacf6_mask = _stat(native_stats, KEY_TACF6, TACTILE_DIM, "mask")
        self.tacf6_min = _stat(native_stats, KEY_TACF6, TACTILE_DIM, "q01")
        self.tacf6_max = _stat(native_stats, KEY_TACF6, TACTILE_DIM, "q99")

        if _ds is not None:
            self.ds = _ds
        else:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self.ds = LeRobotDataset(
                repo_id,
                root=self.root,
                episodes=episodes,
                delta_timestamps=self._build_delta_timestamps(),
                tolerance_s=self.video_tolerance_s,
                video_backend=self.video_backend,
            )

        accelerator.print(
            f"[Origami LeRobot] {repo_id}: {len(self.ds)} frames, fps={self.fps}, "
            f"action_dim={self.action_dim}, action_chunk={self.action_chunk}, "
            f"tactile={self.has_tactile}, deform_mosaic={self.has_deform}"
        )

    def _head_offsets(self):
        return [0.0] + [(k + 1) * self.flare_stride / self.fps for k in range(self.n_flare_steps)]

    def _f6_offsets(self):
        return [(i - (self.vqvae_window - 1)) / self.fps for i in range(self.vqvae_window)]

    def _action_offsets(self):
        return [k / self.fps for k in range(self.action_chunk)]

    def _build_delta_timestamps(self) -> Dict[str, list]:
        dt = {
            KEY_HEAD: self._head_offsets(),
            KEY_ACTION: self._action_offsets(),
        }
        if self.has_tactile and (self.use_tactile_vec or self.use_tactile_vqvae):
            dt[KEY_TACF6] = self._f6_offsets()
        return dt

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    def create_val_split(self, val_ratio=0.05, seed=42):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        n_ep = self.ds.meta.total_episodes
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_ep)
        n_val = max(1, int(n_ep * val_ratio))
        val_eps = sorted(perm[:n_val].tolist())
        train_eps = sorted(perm[n_val:].tolist())
        repo_id = getattr(self.config, "lerobot_repo_id", "") or os.path.basename(self.root.rstrip("/"))
        dt = self._build_delta_timestamps()
        val_ds = LeRobotDataset(
            repo_id,
            root=self.root,
            episodes=val_eps,
            delta_timestamps=dt,
            tolerance_s=self.video_tolerance_s,
            video_backend=self.video_backend,
        )
        self.ds = LeRobotDataset(
            repo_id,
            root=self.root,
            episodes=train_eps,
            delta_timestamps=dt,
            tolerance_s=self.video_tolerance_s,
            video_backend=self.video_backend,
        )
        self.accelerator.print(f"[Origami LeRobot] train/val split: {len(train_eps)}/{len(val_eps)} episodes")
        val = copy.copy(self)
        val.ds = val_ds
        return val

    def _img_to_pil(self, img_t: torch.Tensor) -> PIL.Image.Image:
        arr = (img_t.detach().float().clamp(0, 1) * 255.0).to(torch.uint8)
        arr = arr.permute(1, 2, 0).cpu().numpy()
        img = PIL.Image.fromarray(arr, mode="RGB")
        if self.image_size is not None:
            img = img.resize(self.image_size, PIL.Image.LANCZOS)
        return img

    def _head_frame(self, head_t: torch.Tensor, index: int = 0) -> torch.Tensor:
        if head_t.ndim == 3:
            if index != 0:
                raise IndexError("future head frames require use_flare=1")
            return head_t
        return head_t[index]

    def _split_5x2_deform(self, img_t: torch.Tensor) -> torch.Tensor:
        """Split [3,H,W] 5x2 mosaic to [10,H/2,W/5], left row then right row."""
        gray = img_t[0].float()
        h, w = gray.shape
        if h % 2 != 0 or w % 5 != 0:
            raise ValueError(f"expected tactile deform mosaic 5x2, got {(h, w)}")
        cell_h, cell_w = h // 2, w // 5
        tiles = []
        for row in range(2):
            for col in range(5):
                tiles.append(gray[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w])
        return torch.stack(tiles, dim=0)

    def collate_fn(self, batch: List[Dict]) -> Dict:
        B = len(batch)

        actions = np.stack([np.asarray(x[KEY_ACTION], dtype=np.float32) for x in batch], axis=0)
        if actions.ndim != 3 or actions.shape[1:] != (self.action_chunk, ACTION_DIM):
            raise ValueError(f"expected action chunk [B,{self.action_chunk},{ACTION_DIM}], got {actions.shape}")
        norm_actions = torch.tensor(
            _normalize(actions, self.action_mask, self.action_min, self.action_max),
            dtype=torch.bfloat16,
        )
        beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        time = (beta.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions
        time_r = (beta.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        eps_r = torch.randn_like(norm_actions)

        norm_tacf6 = None
        tactile_f6_history_tensor = None
        if self.has_tactile and KEY_TACF6 in batch[0]:
            f6_hist = torch.stack([x[KEY_TACF6].float() for x in batch], dim=0)
            f6_hist = f6_hist.reshape(B, f6_hist.shape[1], N_FINGERS, F6_PER_FINGER)
            if self.use_tactile_vqvae:
                tactile_f6_history_tensor = f6_hist
            if self.use_tactile_vec:
                cur = f6_hist[:, -1].reshape(B, -1).numpy()
                norm_tacf6 = torch.tensor(
                    _normalize(cur, self.tacf6_mask, self.tacf6_min, self.tacf6_max).reshape(B, -1, F6_PER_FINGER),
                    dtype=torch.bfloat16,
                )

        deforms_tensor = None
        if self.use_tactile_deform and self.has_deform:
            per_sample = [self._split_5x2_deform(x[KEY_DEFORM]) for x in batch]
            deforms_tensor = torch.stack(per_sample, dim=0).unsqueeze(2).float()

        state_raw = None
        if self.use_robot_state:
            state_raw = torch.stack([
                torch.tensor(
                    _normalize(np.asarray(x[KEY_STATE], dtype=np.float32), self.state_mask, self.state_min, self.state_max),
                    dtype=torch.bfloat16,
                )
                for x in batch
            ])

        all_input_ids, all_pixel_values, all_grid_thw = [], [], []
        n_slow_images = 1
        for x in batch:
            head_seq = x[KEY_HEAD]
            pil_slow = [self._img_to_pil(self._head_frame(head_seq, 0))]
            pil_fast = []
            if self.has_wrist:
                pil_fast = [self._img_to_pil(x[KEY_WRIST_R]), self._img_to_pil(x[KEY_WRIST_L])]
            all_pil = pil_slow + pil_fast
            content = [{"type": "image"} for _ in pil_slow]
            content.append({"type": "text", "text": x.get("task", "")})
            content += [{"type": "image"} for _ in pil_fast]
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            inp = self.processor(text=text, images=all_pil, return_tensors="pt", padding=False)
            all_input_ids.append(inp.input_ids[0])
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values)
                all_grid_thw.append(inp.image_grid_thw)

        flare_pixel_values = flare_grid_thw = None
        if self.n_flare_steps > 0:
            flare_pil = []
            for x in batch:
                head_seq = x[KEY_HEAD]
                is_pad = x.get(f"{KEY_HEAD}_is_pad")
                for k in range(self.n_flare_steps):
                    fi = 1 + k
                    if is_pad is not None and bool(is_pad[fi]):
                        flare_pil.append(self._img_to_pil(self._head_frame(head_seq, 0)))
                    else:
                        flare_pil.append(self._img_to_pil(self._head_frame(head_seq, fi)))
            finp = self.processor.image_processor(flare_pil, return_tensors="pt")
            flare_pixel_values = finp.pixel_values.to(torch.bfloat16)
            flare_grid_thw = finp.image_grid_thw

        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(ids.shape[0] for ids in all_input_ids)
        padded_ids, attn_ms = [], []
        for ids in all_input_ids:
            pad = max_len - ids.shape[0]
            padded_ids.append(F.pad(ids, (pad, 0), value=pad_id))
            a = torch.ones(max_len, dtype=torch.long)
            if pad > 0:
                a[:pad] = 0
            attn_ms.append(a)

        return {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(attn_ms),
            "pixel_values": torch.cat(all_pixel_values, dim=0) if all_pixel_values else None,
            "image_grid_thw": torch.cat(all_grid_thw, dim=0) if all_grid_thw else None,
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "norm_actions": norm_actions,
            "tactile_f6s": norm_tacf6,
            "tactile_deforms": deforms_tensor,
            "tactile_f6s_delayed": norm_tacf6,
            "tactile_deforms_delayed": deforms_tensor,
            "tactile_codes": None,
            "tactile_f6_history": tactile_f6_history_tensor,
            "time_r": time_r,
            "eps_r": eps_r,
            "state_raw": state_raw,
            "flare_pixel_values": flare_pixel_values,
            "flare_grid_thw": flare_grid_thw,
        }
