from typing import Callable, Dict, List, Sequence


def normalize_layer_indices(
    indices: Sequence[int],
    num_layers: int,
    *,
    label: str,
) -> List[int]:
    if not indices:
        raise ValueError(f"{label}.layer_indices must be non-empty")
    resolved = sorted({int(i) for i in indices})
    for idx in resolved:
        if idx < 0 or idx >= num_layers:
            raise IndexError(
                f"{label} layer index {idx} out of range [0, {num_layers})"
            )
    return resolved


def max_layer_index(
    layer_indices: Sequence[int],
    num_layers: int,
    *,
    label: str,
) -> int:
    return max(normalize_layer_indices(layer_indices, num_layers, label=label))


def install_processors(
    blocks,
    layer_indices: Sequence[int],
    *,
    label: str,
    processor_factory: Callable[[int], object],
) -> Dict[int, object]:
    resolved = normalize_layer_indices(layer_indices, len(blocks), label=label)
    backups: Dict[int, object] = {}
    for idx in resolved:
        attn = blocks[idx].attn
        backups[idx] = attn.processor
        attn.set_processor(processor_factory(idx))
    return backups


def restore_processors(blocks, backups: Dict[int, object]) -> None:
    for idx, processor in backups.items():
        blocks[idx].attn.set_processor(processor)
