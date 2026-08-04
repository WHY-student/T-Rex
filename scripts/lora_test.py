"""Strict inference-server launcher for T-Rex Action LoRA checkpoints.

This is intentionally a thin entry point over ``scripts/test.py`` so model
construction, cascaded inference and the ZMQ protocol cannot drift between the
normal and LoRA deployment paths.
"""

from __future__ import annotations

from test import build_arg_parser, main


def run() -> None:
    parser = build_arg_parser(
        "Real-world ZMQ server for a T-Rex Action LoRA checkpoint"
    )
    parser.set_defaults(
        require_action_lora=True,
        # Historical Origami LoRA checkpoints predate persisted image_size.
        image_size=[384, 288],
    )
    args = parser.parse_args()
    if args.allow_non_strict_checkpoint:
        parser.error(
            "lora_test.py always requires strict checkpoint loading; remove "
            "--allow_non_strict_checkpoint"
        )

    print(
        "LORA_TEST_LAUNCH "
        f"checkpoint={args.checkpoint_path} "
        f"image_size={args.image_size} "
        "strict=True require_action_lora=True",
        flush=True,
    )
    main(args)


if __name__ == "__main__":
    run()
