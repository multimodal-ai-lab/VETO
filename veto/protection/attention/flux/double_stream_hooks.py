from typing import Dict, Sequence

from diffusers import Flux2Transformer2DModel

from veto.protection.attention.flux.double_stream_processor import (
    VetoFluxDoubleStreamAttentionProcessor,
)
from veto.protection.attention.entropy import VetoAttentionState
from veto.protection.attention.layer_indices import (
    install_processors,
    max_layer_index,
    normalize_layer_indices,
    restore_processors,
)

_LABEL = "double_stream"


def normalize_double_stream_layer_indices(
    indices: Sequence[int],
    num_layers: int,
) -> list[int]:
    return normalize_layer_indices(indices, num_layers, label=_LABEL)


def max_double_stream_layer_index(
    layer_indices: Sequence[int],
    num_layers: int,
) -> int:
    return max_layer_index(layer_indices, num_layers, label=_LABEL)


def install_double_stream_processors(
    transformer: Flux2Transformer2DModel,
    *,
    layer_indices: Sequence[int],
    text_seq_len: int,
    canvas_seq_len: int,
    reference_seq_len: int,
    entropy_slices: Sequence[str],
    state: VetoAttentionState,
    entropy_eps: float = 1e-8,
) -> Dict[int, object]:
    blocks = transformer.transformer_blocks

    def factory(idx: int) -> VetoFluxDoubleStreamAttentionProcessor:
        return VetoFluxDoubleStreamAttentionProcessor(
            text_seq_len=text_seq_len,
            canvas_seq_len=canvas_seq_len,
            reference_seq_len=reference_seq_len,
            entropy_slices=entropy_slices,
            layer_index=idx,
            state=state,
            entropy_eps=entropy_eps,
        )

    return install_processors(
        blocks, layer_indices, label=_LABEL, processor_factory=factory
    )


def restore_double_stream_processors(
    transformer: Flux2Transformer2DModel,
    backups: Dict[int, object],
) -> None:
    restore_processors(transformer.transformer_blocks, backups)
