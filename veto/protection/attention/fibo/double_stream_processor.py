from typing import Optional, Sequence, Tuple

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_bria_fibo import _get_qkv_projections

from veto.protection.attention.entropy import (
    VetoAttentionState,
    record_slice_entropies,
    validate_entropy_slices,
)


class VetoFiboDoubleStreamAttentionProcessor:
    """FIBO ``BriaFiboAttention`` processor with entropy recording."""

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
    ):
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        text_len = int(encoder_hidden_states.shape[1])

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        attn_probs = (torch.matmul(q, k.transpose(-2, -1)) * (attn.head_dim**-0.5)).softmax(
            dim=-1
        )

        if self.text_seq_len != text_len:
            raise ValueError(
                f"text_seq_len mismatch: configured {self.text_seq_len}, "
                f"encoder_hidden_states {text_len}"
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
        out = out.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = out.split_with_sizes(
                [text_len, out.shape[1] - text_len],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states

        hidden_states = attn.to_out[0](out)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
