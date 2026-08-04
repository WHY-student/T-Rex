# Policy inference client

`eval_trex_async.py` runs a learned policy on the robot by talking to the
T-Rex ZMQ inference server (`T-Rex/scripts/test.py`). The model runs
server-side (any GPU machine); this client only needs the teleop stack's
dependencies.

## Slow / fast protocol

The server is a stateful ZMQ **REP** endpoint; this client orchestrates the
cadence over a **REQ** socket:

- `mode: "slow_and_fast"` — full VLA forward at a chunk start: the server
  caches latent/action KV at the τ-split and returns the action chunk.
- `mode: "fast"` — a small tactile-only payload between chunk starts: the
  server continues flow matching on the cached KV with fresh tactile and
  returns a refined action chunk.
- `mode: "slow"` — full forward without the tactile refinement path.

Typical cadence: slow every 16 robot steps, fast at offsets 0/4/8/12 within
the chunk. ZMQ REP is single-threaded, so a fast request arriving mid-slow
naturally waits until the slow pass finishes.

## Control mode

Select `inference.control_mode` in the YAML:

- `delta_eef62` is the original path. Each step is a delta EEF pose plus
  absolute hand joints and is resolved through differential IK.
- `absolute_joint65` is the Origami post-training path. State and action use
  exactly `[L arm7 | L hand22 | R arm7 | R hand22 | motor7]`; the client
  performs no EEF conversion and no IK.

The final seven dimensions cannot be inferred from the name `motor_j0..j6`.
For `absolute_joint65`, configure seven real SDK joints in dataset order:

```yaml
inference:
  control_mode: absolute_joint65
  dual_arm: true
  chunk_size: 16
  head_crop_box: null
  absolute_joint65_body_joint_map:
    - "YOUR_COMPONENT:YOUR_MOTOR_J0_JOINT"
    - "YOUR_COMPONENT:YOUR_MOTOR_J1_JOINT"
    - "YOUR_COMPONENT:YOUR_MOTOR_J2_JOINT"
    - "YOUR_COMPONENT:YOUR_MOTOR_J3_JOINT"
    - "YOUR_COMPONENT:YOUR_MOTOR_J4_JOINT"
    - "YOUR_COMPONENT:YOUR_MOTOR_J5_JOINT"
    - "YOUR_COMPONENT:YOUR_MOTOR_J6_JOINT"
  absolute_joint65_body_default_joint_pos: [0, 0, 0, 0, 0, 0, 0]
  absolute_joint65_max_step_rad: 0.05
  # Required explicit acknowledgement: the bundled online Pinocchio model
  # collision-checks arm/hand 58D but does not model the moving body 7D.
  absolute_joint65_acknowledge_body_collision_unchecked: true
```

At startup the client checks every component/joint name against the connected
robot, obtains physical limits from the SDK (or requires explicit
`absolute_joint65_body_{lower,upper}_limits`), and checks that the mapping has
seven unique joints. A bad mapping, NaN/Inf, limit violation, response shape
other than `[chunk,65]`, or excessive single-step change rejects the command
without clipping it.

The existing high-rate Pinocchio checker still covers only the 58 arm/hand
joints. Body commands remain disabled until the acknowledgement flag above is
set after reviewing the real North body's self/environment collision envelope.

The bundled Dexmate configuration exposes only `torso(3)+head(3)`, so it is
not a valid mapping for an unknown seven-motor North body. Supply the actual
North SDK component/joint names; the client intentionally refuses to drop or
guess the seventh value.

## Running

1. Start the inference server (on the GPU machine) — see the T-Rex top-level
   README, "Post-training & inference".
2. Bring up the robot-side processes exactly as for teleop (cameras, hands;
   no Vive/gloves needed) — see the main README.
3. Everything is driven by the YAML config (`inference:` section plus the
   shared robot/cameras/hands/environment sections — see
   `config/default.yaml`):

   ```bash
   cd eval && python eval_trex_async.py --config ../config/default.yaml \
       --task-description "Pour the sugar from the filled cup to the empty cup."
   ```

   Hotkeys during execution: `p` pause/resume, `r` reset trajectory, `q` quit.

For the 65-D Origami checkpoint, start the server with the native LeRobot
statistics. Action dimension, chunk length, tactile modules and LoRA structure
are restored from the checkpoint:

```bash
T-Rex/scripts/lora_test.sh
```

Checkpoint loading is strict by default. Do not use
`--allow_non_strict_checkpoint` for robot execution.

> **Reproducing the paper evals:** the released checkpoints were evaluated
> on a bench with `environment.table_height: 0.76` and back/right walls at
> `0.60`/`0.75` m, and the head crop box must match what the checkpoint was
> trained with — set these in your config copy. The torso pose is the config
> default (`[0.9, 1.57, 0.1]`, same as the dataset).

The observation order (proprio → images → tactile-from-buffer) deliberately
matches `main_teleop.py`'s recording order so inference-time inputs are
consistent with the training data.
