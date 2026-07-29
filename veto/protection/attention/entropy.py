from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

# Joint sequence: [ text | canvas | reference ]
ALL_ENTROPY_SLICES: Tuple[str, ...] = (
    "text_text",
    "text_canvas",
    "text_reference",
    "canvas_text",
    "canvas_canvas",
    "canvas_reference",
    "reference_text",
    "reference_canvas",
    "reference_reference",
)
VALID_ENTROPY_SLICES = frozenset(ALL_ENTROPY_SLICES)

SLICE_LOG_LABELS: Dict[str, str] = {
    "text_text": "H_t→t",
    "text_canvas": "H_t→c",
    "text_reference": "H_t→r",
    "canvas_text": "H_c→t",
    "canvas_canvas": "H_c→c",
    "canvas_reference": "H_c→r",
    "reference_text": "H_r→t",
    "reference_canvas": "H_r→c",
    "reference_reference": "H_r→r",
}


def validate_entropy_slices(slices: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(slices)
    if not resolved:
        raise ValueError("entropy_slices must be non-empty")
    unknown = set(resolved) - VALID_ENTROPY_SLICES
    if unknown:
        raise ValueError(
            f"Unknown entropy_slices {sorted(unknown)}; "
            f"use: {sorted(VALID_ENTROPY_SLICES)}"
        )
    return resolved


def attention_slice_bounds(
    slice_name: str,
    text_seq_len: int,
    canvas_seq_len: int,
    reference_seq_len: int,
) -> Tuple[int, int, int, int]:
    """Query/key ranges on [text | canvas | reference] (half-open intervals)."""
    if slice_name not in VALID_ENTROPY_SLICES:
        raise KeyError(
            f"Unknown entropy slice {slice_name!r}; "
            f"use: {sorted(VALID_ENTROPY_SLICES)}"
        )
    t = int(text_seq_len)
    c = int(canvas_seq_len)
    r = int(reference_seq_len)
    table: Dict[str, Tuple[int, int, int, int]] = {
        "text_text": (0, t, 0, t),
        "text_canvas": (0, t, t, t + c),
        "text_reference": (0, t, t + c, t + c + r),
        "canvas_text": (t, t + c, 0, t),
        "canvas_canvas": (t, t + c, t, t + c),
        "canvas_reference": (t, t + c, t + c, t + c + r),
        "reference_text": (t + c, t + c + r, 0, t),
        "reference_canvas": (t + c, t + c + r, t, t + c),
        "reference_reference": (t + c, t + c + r, t + c, t + c + r),
    }
    q_start, q_end, k_start, k_end = table[slice_name]
    if q_start >= q_end or k_start >= k_end:
        raise ValueError(
            f"Empty attention block for slice {slice_name!r}: "
            f"text={t} canvas={c} reference={r} "
            f"q=[{q_start},{q_end}) k=[{k_start},{k_end})"
        )
    return q_start, q_end, k_start, k_end


def submatrix_entropy(
    attn_probs: torch.Tensor,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    a_sub = attn_probs[:, :, q_start:q_end, k_start:k_end]
    return -(a_sub * (a_sub + eps).log()).sum(dim=(-2, -1)).mean()


def slice_entropy(
    attn_probs: torch.Tensor,
    slice_name: str,
    text_seq_len: int,
    canvas_seq_len: int,
    reference_seq_len: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    q_start, q_end, k_start, k_end = attention_slice_bounds(
        slice_name, text_seq_len, canvas_seq_len, reference_seq_len
    )
    return submatrix_entropy(attn_probs, q_start, q_end, k_start, k_end, eps)


def record_slice_entropies(
    state: "VetoAttentionState",
    attn_probs: torch.Tensor,
    *,
    entropy_slices: Sequence[str],
    text_seq_len: int,
    canvas_seq_len: int,
    reference_seq_len: int,
    layer_index: int,
    entropy_eps: float,
) -> None:
    ent_total = None
    for slice_name in entropy_slices:
        ent_s = slice_entropy(
            attn_probs,
            slice_name,
            text_seq_len,
            canvas_seq_len,
            reference_seq_len,
            eps=entropy_eps,
        )
        state.layer_entropies_by_slice.setdefault(slice_name, []).append(ent_s)
        ent_total = ent_s if ent_total is None else ent_total + ent_s
    if ent_total is None:
        raise RuntimeError("No entropy slice contributed to VETO loss.")
    state.layer_entropies.append(ent_total)
    state.layer_indices.append(layer_index)


@dataclass
class VetoAttentionState:
    """Per-forward entropy scalars from hooked attention layers."""

    layer_entropies: List[torch.Tensor] = field(default_factory=list)
    layer_entropies_by_slice: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    layer_indices: List[int] = field(default_factory=list)

    def reset(self) -> None:
        self.layer_entropies.clear()
        self.layer_entropies_by_slice.clear()
        self.layer_indices.clear()

    def mean_entropy(self) -> torch.Tensor:
        if not self.layer_entropies:
            raise RuntimeError("No attention entropy values were recorded.")
        return torch.stack(self.layer_entropies).mean()

    def mean_entropy_slice(self, slice_name: str) -> torch.Tensor | None:
        vals = self.layer_entropies_by_slice.get(slice_name)
        if not vals:
            return None
        return torch.stack(vals).mean()

