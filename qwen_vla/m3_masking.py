"""Training-time modality masking used by the M3/T-Rex integration.

The paper's mask is an attention mask, rather than an image-space occlusion:
an invisible token cannot read visible tokens and cannot be used as a key/value
by visible tokens.  This module only builds the per-batch mask and keeps the
policy model independent from the data-loader implementation.

T-Rex maps the paper's groups as follows:

* the first (head/ego) image is always visible;
* all wrist image patches share one visibility bit;
* the text portion before the wrist images shares one visibility bit;
* noisy action-step tokens are independently masked, with at least one kept.

Special wrapper tokens around wrist images are kept visible.  They carry no
image content and keeping them avoids making the Qwen multimodal template
itself ill-formed; the wrist patch tokens are still completely inaccessible
when the wrist group is masked.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1); got {value}")
    return value


def sample_m3_visibility(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
    n_slow_img_tokens: int,
    slow_len: int,
    n_flare_tokens: int,
    n_fast: int,
    n_state: int,
    n_action_queries: int,
    vision_mask_prob: float = 0.5,
    language_mask_prob: float = 0.1,
    query_mask_prob: float = 0.1,
    tactile_mask_prob: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Sample one structured M3 visibility pattern per batch item.

    Parameters describe the sequence *after* ``split_slow_fast_embeds``:
    ``slow`` is a contiguous prefix of the original processor sequence and
    ``fast`` is its suffix.  Flare tokens are inserted between those two
    pieces, followed by optional state/time tokens and action queries.

    The returned visibility tensors are boolean and already include left-pad
    positions from ``attention_mask``.  ``latent_visibility`` matches the
    current slow prefix ``[slow, flare]`` and ``fast_visibility`` matches the
    separate fast suffix.  ``full_visibility`` matches the token order of the
    main T-Rex training forward:

    ``[slow, flare, fast, state, time, noisy_action_queries]``.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
    if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "attention_mask must be [B, L] with the same shape as input_ids; "
            f"got {tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}"
        )

    batch_size, original_len = input_ids.shape
    slow_len = int(slow_len)
    n_fast = int(n_fast)
    n_slow_img_tokens = int(n_slow_img_tokens)
    n_flare_tokens = int(n_flare_tokens)
    n_state = int(n_state)
    n_action_queries = int(n_action_queries)
    if not 0 <= slow_len <= original_len:
        raise ValueError(f"slow_len must be in [0, {original_len}], got {slow_len}")
    if n_fast != original_len - slow_len:
        raise ValueError(
            "n_fast must be the suffix length of the processor sequence; "
            f"got n_fast={n_fast}, slow_len={slow_len}, original_len={original_len}"
        )
    for name, value in (
        ("n_flare_tokens", n_flare_tokens),
        ("n_state", n_state),
        ("n_action_queries", n_action_queries),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative; got {value}")

    vision_mask_prob = _validate_probability("vision_mask_prob", vision_mask_prob)
    language_mask_prob = _validate_probability("language_mask_prob", language_mask_prob)
    query_mask_prob = _validate_probability("query_mask_prob", query_mask_prob)
    tactile_mask_prob = _validate_probability("tactile_mask_prob", tactile_mask_prob)

    device = input_ids.device
    valid = attention_mask.to(device=device, dtype=torch.bool)
    image_mask = input_ids.to(device=device).eq(int(image_token_id))
    image_rank = image_mask.to(torch.long).cumsum(dim=1)
    positions = torch.arange(original_len, device=device).view(1, -1)
    is_slow = positions < slow_len
    is_fast = ~is_slow

    # One gate for both wrist cameras, as in the paper.  The ego image is not
    # sampled: it is the stable global anchor.
    keep_vision = torch.rand(batch_size, device=device) >= vision_mask_prob
    keep_language = torch.rand(batch_size, device=device) >= language_mask_prob
    keep_tactile = torch.rand(batch_size, device=device) >= tactile_mask_prob

    visibility_original = torch.zeros(
        (batch_size, original_len), dtype=torch.bool, device=device
    )

    ego_patches = image_mask & (image_rank > 0) & (image_rank <= n_slow_img_tokens)
    wrist_patches = image_mask & (image_rank > n_slow_img_tokens)
    visibility_original[ego_patches] = True
    visibility_original[wrist_patches] = keep_vision[:, None].expand_as(wrist_patches)[wrist_patches]

    # Text and multimodal wrapper tokens before the first wrist image belong to
    # the language-side prefix.  The image patch tokens were handled above.
    language_prefix = is_slow & ~image_mask
    visibility_original[language_prefix] = (
        keep_language[:, None].expand_as(language_prefix)[language_prefix]
    )

    # The suffix contains wrist-image wrappers and the generation suffix.  Keep
    # non-patch structure visible while making every wrist patch obey the one
    # joint wrist gate above.
    fast_structure = is_fast & ~image_mask
    visibility_original[fast_structure] = True
    visibility_original &= valid

    slow_visibility = visibility_original[:, :slow_len]
    fast_visibility = visibility_original[:, slow_len:]

    flare_visibility = torch.ones(
        (batch_size, n_flare_tokens), dtype=torch.bool, device=device
    )
    # The action expert receives ``fast_visibility`` as a separate sequence
    # argument.  Do not include it in this prefix as well: doing so would
    # duplicate the wrist segment in the KV-cache mask.
    latent_visibility = torch.cat(
        [slow_visibility, flare_visibility], dim=1
    )

    always_visible = torch.ones(
        (batch_size, n_state + 1), dtype=torch.bool, device=device
    )  # state (possibly empty) + one flow-time token

    if n_action_queries:
        action_query_visibility = (
            torch.rand(
                (batch_size, n_action_queries), device=device
            ) >= query_mask_prob
        )
        # A fully hidden query set produces no useful action supervision and can
        # create an all--negative attention row.  The paper explicitly keeps at
        # least one query visible.
        missing = ~action_query_visibility.any(dim=1)
        if missing.any():
            replacement = torch.randint(
                n_action_queries, (int(missing.sum().item()),), device=device
            )
            rows = missing.nonzero(as_tuple=False).squeeze(1)
            action_query_visibility[rows, replacement] = True
    else:
        action_query_visibility = torch.empty(
            (batch_size, 0), dtype=torch.bool, device=device
        )

    full_visibility = torch.cat(
        [latent_visibility, fast_visibility, always_visible,
         action_query_visibility], dim=1
    )
    return {
        "latent_visibility": latent_visibility,
        "fast_visibility": fast_visibility,
        "action_query_visibility": action_query_visibility,
        "full_visibility": full_visibility,
        "vision_keep": keep_vision,
        "language_keep": keep_language,
        "tactile_keep": keep_tactile,
    }


def make_m3_attention_mask(
    token_visibility: torch.Tensor,
    *,
    query_start: int = 0,
    query_len: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Convert token visibility into a row/column additive attention mask.

    ``token_visibility`` describes all key/value positions, including cached
    positions.  ``query_start`` identifies the first current query in that
    sequence, which is needed for the action/tactile KV-cache forwards.
    Diagonal self-attention is retained for invisible rows so their softmax is
    finite; their information cannot flow to or from any other token.
    """
    if token_visibility.ndim != 2:
        raise ValueError(
            "token_visibility must be [B, total_length], got "
            f"{tuple(token_visibility.shape)}"
        )
    batch_size, total_len = token_visibility.shape
    query_start = int(query_start)
    if not 0 <= query_start <= total_len:
        raise ValueError(
            f"query_start must be in [0, {total_len}], got {query_start}"
        )
    if query_len is None:
        query_len = total_len - query_start
    query_len = int(query_len)
    if not 0 <= query_len <= total_len - query_start:
        raise ValueError(
            f"query_len={query_len} is invalid for total_len={total_len} "
            f"and query_start={query_start}"
        )

    visibility = token_visibility.to(dtype=torch.bool)
    query_visibility = visibility[:, query_start:query_start + query_len]
    allowed = query_visibility[:, :, None] & visibility[:, None, :]
    mask_dtype = dtype or torch.float32
    additive = torch.zeros(
        (batch_size, 1, query_len, total_len),
        dtype=mask_dtype,
        device=visibility.device,
    )
    additive.masked_fill_(~allowed[:, None], float("-inf"))

    # Keep self-attention finite for both invisible tokens and left padding.
    if query_len:
        row = torch.arange(query_len, device=visibility.device)
        col = query_start + row
        additive[:, 0, row, col] = 0.0
    return additive


def rescale_visible_queries(
    query_embeds: torch.Tensor,
    query_visibility: torch.Tensor,
    mask_prob: float,
) -> torch.Tensor:
    """Apply the paper's inverted-dropout rescaling to visible queries."""
    mask_prob = _validate_probability("query_mask_prob", mask_prob)
    if mask_prob == 0.0:
        return query_embeds
    if query_embeds.ndim != 3 or query_visibility.ndim != 2:
        raise ValueError("query_embeds must be [B, Q, H] and visibility [B, Q]")
    if query_embeds.shape[:2] != query_visibility.shape:
        raise ValueError(
            "query_embeds and query_visibility disagree: "
            f"{tuple(query_embeds.shape[:2])} vs {tuple(query_visibility.shape)}"
        )
    scale = 1.0 / (1.0 - mask_prob)
    visible = query_visibility.to(device=query_embeds.device)[:, :, None]
    return torch.where(visible, query_embeds * scale, query_embeds)
