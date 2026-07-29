from typing import Optional, Sequence, Tuple

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb

from veto.protection.attention.entropy import (
    VetoAttentionState,
    record_slice_entropies,
    validate_entropy_slices,
)


def _attention_qk_scale(attn: torch.nn.Module) -> float:
    """QK scale for FIBO single-stream ``Attention`` (``scale``) vs double-stream ``BriaFiboAttention`` (``head_dim``)."""
    scale = getattr(attn, "scale", None)
    if scale is not None:
        return float(scale)
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is not None:
        return float(head_dim) ** -0.5
    return (attn.inner_dim // attn.heads) ** -0.5


class VetoFiboSingleStreamAttentionProcessor:
    """FIBO single-stream ``Attention`` (unified ``[text | canvas | reference]`` seq)."""

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        *,
        text_seq_len: int,
        canvas_seq_len: int,
        reference_seq_len: int,
        entropy_slices: Sequence[str],
        layer_index: int,
        state: VetoAttentionState,
        entropy_eps: float = 1e-8,
    ) -> None:
        self.text_seq_len = int(text_seq_len)
        self.canvas_seq_len = int(canvas_seq_len)
        self.reference_seq_len = int(reference_seq_len)
        self.entropy_slices = validate_entropy_slices(entropy_slices)
        self.layer_index = int(layer_index)
        self.state = state
        self.entropy_eps = float(entropy_eps)

    def __call__(
        self,
        attn: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        del encoder_hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        attn_probs = (torch.matmul(q, k.transpose(-2, -1)) * _attention_qk_scale(attn)).softmax(
            dim=-1
        )

        record_slice_entropies(
            self.state,
            attn_probs,
            entropy_slices=self.entropy_slices,
            text_seq_len=self.text_seq_len,
            canvas_seq_len=self.canvas_seq_len,
            reference_seq_len=self.reference_seq_len,
            layer_index=self.layer_index,
            entropy_eps=self.entropy_eps,
        )

        out = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        return out.flatten(2, 3).to(query.dtype)
