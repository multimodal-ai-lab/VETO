from typing import Optional, Sequence

import torch
from torch import Tensor

from veto.protection.attention.entropy import VetoAttentionState
from veto.protection.attention.flux import double_stream_hooks as flux_ds
from veto.protection.attention.flux import single_stream_hooks as flux_ss
from veto.protection.attention.fibo import double_stream_hooks as fibo_ds
from veto.protection.attention.fibo import single_stream_hooks as fibo_ss
from veto.protection.wrappers.base import (
    FiboSurrogate,
    ImageMeta,
    TextSurrogate,
    ProtectionModel,
    DiTWrapper,
)


def _install_entropy_hooks(
    wrapper: DiTWrapper,
    *,
    surrogate: TextSurrogate | FiboSurrogate,
    canvas_seq_len: int,
    reference_seq_len: int,
    ds_ent: tuple[str, ...],
    ss_ent: tuple[str, ...],
    double_stream_layer_indices: Optional[Sequence[int]],
    single_stream_layer_indices: Optional[Sequence[int]],
    state: VetoAttentionState,
    entropy_eps: float,
) -> tuple[dict[int, object], dict[int, object], Optional[int], Optional[int]]:
    ds_backups: dict[int, object] = {}
    ss_backups: dict[int, object] = {}
    num_double = wrapper.num_double_stream_layers
    num_single = wrapper.num_single_stream_layers
    max_double: Optional[int] = None
    max_single: Optional[int] = None
    text_seq_len = int(surrogate.prompt_embeds.shape[1])

    if wrapper.model == ProtectionModel.FLUX2:
        install_ds = flux_ds.install_double_stream_processors
        install_ss = flux_ss.install_single_stream_processors
        max_ds = flux_ds.max_double_stream_layer_index
        max_ss = flux_ss.max_single_stream_layer_index
    elif wrapper.model == ProtectionModel.FIBO_EDIT:
        install_ds = fibo_ds.install_double_stream_processors
        install_ss = fibo_ss.install_single_stream_processors
        max_ds = fibo_ds.max_double_stream_layer_index
        max_ss = fibo_ss.max_single_stream_layer_index
    else:
        raise ValueError(f"Unsupported model: {wrapper.model}")

    if ds_ent:
        if double_stream_layer_indices is None:
            raise ValueError(
                "double_stream_layer_indices required when double_stream entropy_slices are set"
            )
        ds_layers = list(double_stream_layer_indices)
        ds_backups = install_ds(
            wrapper.transformer,
            layer_indices=ds_layers,
            text_seq_len=text_seq_len,
            canvas_seq_len=canvas_seq_len,
            reference_seq_len=reference_seq_len,
            entropy_slices=ds_ent,
            state=state,
            entropy_eps=entropy_eps,
        )
        max_double = max_ds(ds_layers, num_double)

    if ss_ent:
        if single_stream_layer_indices is None:
            raise ValueError(
                "single_stream_layer_indices required when single_stream entropy_slices are set"
            )
        ss_layers = list(single_stream_layer_indices)
        ss_backups = install_ss(
            wrapper.transformer,
            layer_indices=ss_layers,
            text_seq_len=text_seq_len,
            canvas_seq_len=canvas_seq_len,
            reference_seq_len=reference_seq_len,
            entropy_slices=ss_ent,
            state=state,
            entropy_eps=entropy_eps,
        )
        max_single = max_ss(ss_layers, num_single)

    return ds_backups, ss_backups, max_double, max_single


def _restore_entropy_hooks(
    wrapper: DiTWrapper,
    ds_backups: dict[int, object],
    ss_backups: dict[int, object],
) -> None:
    if ds_backups:
        if wrapper.model == ProtectionModel.FLUX2:
            flux_ds.restore_double_stream_processors(wrapper.transformer, ds_backups)
        else:
            fibo_ds.restore_double_stream_processors(wrapper.transformer, ds_backups)
    if ss_backups:
        if wrapper.model == ProtectionModel.FLUX2:
            flux_ss.restore_single_stream_processors(wrapper.transformer, ss_backups)
        else:
            fibo_ss.restore_single_stream_processors(wrapper.transformer, ss_backups)


def veto_forward_entropy_loss(
    wrapper: DiTWrapper,
    *,
    canvas_latents: Tensor,
    canvas_meta: ImageMeta,
    reference_latents: Tensor,
    reference_meta: ImageMeta,
    surrogate: TextSurrogate | FiboSurrogate,
    timestep_idx: int,
    guidance_scale: float = 4.0,
    single_stream_layer_indices: Optional[Sequence[int]] = None,
    single_stream_entropy_slices: Optional[Sequence[str]] = None,
    double_stream_layer_indices: Optional[Sequence[int]] = None,
    double_stream_entropy_slices: Optional[Sequence[str]] = None,
    state: Optional[VetoAttentionState] = None,
    entropy_eps: float = 1e-8,
) -> tuple[Tensor, VetoAttentionState]:
    """Truncated DiT forward with attention entropy hooks."""
    ds_ent = tuple(double_stream_entropy_slices or ())
    ss_ent = tuple(single_stream_entropy_slices or ())
    if not ds_ent and not ss_ent:
        raise ValueError("At least one entropy slice must be configured")

    if wrapper.model == ProtectionModel.FLUX2:
        if not isinstance(surrogate, TextSurrogate):
            raise TypeError("Flux2 wrapper requires TextSurrogate")
    elif wrapper.model == ProtectionModel.FIBO_EDIT:
        if not isinstance(surrogate, FiboSurrogate):
            raise TypeError("FIBO Edit wrapper requires FiboSurrogate")
    else:
        raise ValueError(f"Unsupported model: {wrapper.model}")

    if not isinstance(canvas_meta, ImageMeta) or not isinstance(
        reference_meta, ImageMeta
    ):
        raise TypeError("Forward requires ImageMeta for canvas and reference")

    if state is None:
        state = VetoAttentionState()
    else:
        state.reset()

    canvas_seq_len = int(canvas_latents.shape[1])
    reference_seq_len = int(reference_latents.shape[1])

    ds_backups, ss_backups, max_double, max_single = _install_entropy_hooks(
        wrapper,
        surrogate=surrogate,
        canvas_seq_len=canvas_seq_len,
        reference_seq_len=reference_seq_len,
        ds_ent=ds_ent,
        ss_ent=ss_ent,
        double_stream_layer_indices=double_stream_layer_indices,
        single_stream_layer_indices=single_stream_layer_indices,
        state=state,
        entropy_eps=entropy_eps,
    )

    try:
        wrapper.forward_truncated(
            canvas_latents=canvas_latents,
            canvas_meta=canvas_meta,
            reference_latents=reference_latents,
            reference_meta=reference_meta,
            surrogate=surrogate,
            timestep_idx=timestep_idx,
            guidance_scale=guidance_scale,
            max_double_stream_index=max_double,
            max_single_stream_index=max_single,
        )
    finally:
        _restore_entropy_hooks(wrapper, ds_backups, ss_backups)

    loss = -state.mean_entropy()
    return loss, state
