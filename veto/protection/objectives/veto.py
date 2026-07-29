from typing import Sequence

import torch
from torch import Tensor

from veto.protection.attention.fibo.double_stream_hooks import (
    max_double_stream_layer_index as fibo_max_double_stream_layer_index,
    normalize_double_stream_layer_indices as fibo_normalize_double_stream_layer_indices,
)
from veto.protection.attention.fibo.single_stream_hooks import (
    max_single_stream_layer_index as fibo_max_single_stream_layer_index,
    normalize_single_stream_layer_indices as fibo_normalize_single_stream_layer_indices,
)
from veto.protection.attention.flux.double_stream_hooks import (
    max_double_stream_layer_index,
    normalize_double_stream_layer_indices,
)
from veto.protection.attention.flux.single_stream_hooks import (
    max_single_stream_layer_index,
    normalize_single_stream_layer_indices,
)
from veto.protection.attention.entropy import SLICE_LOG_LABELS
from veto.protection.attention.forward import veto_forward_entropy_loss
from veto.protection.config import DoubleStreamConfig, SingleStreamConfig
from veto.protection.objectives.base import ProtectionObjective
from veto.protection.wrappers.base import (
    FiboSurrogate,
    TextSurrogate,
    ProtectionModel,
    DiTWrapper,
)
from veto.protection.wrappers.fibo_edit import FiboEditWrapper


class VETO(ProtectionObjective):

    def __init__(
        self,
        wrapper: DiTWrapper,
        *,
        surrogate_prompt: str = "",
        inference_steps: int = 10,
        num_timesteps_per_step: int = 3,
        guidance_scale: float = 4.0,
        double_stream: DoubleStreamConfig | None = None,
        single_stream: SingleStreamConfig | None = None,
        layer_indices: Sequence[int] | None = None,
        entropy_eps: float = 1e-8,
        seed: int = 0,
    ) -> None:
        self.wrapper = wrapper
        self.wrapper.transformer.eval()
        for param in self.wrapper.transformer.parameters():
            param.requires_grad_(False)

        self.surrogate_prompt = surrogate_prompt
        self.inference_steps = int(inference_steps)
        self.num_timesteps_per_step = int(num_timesteps_per_step)
        self.guidance_scale = float(guidance_scale)
        self.entropy_eps = float(entropy_eps)

        ds_cfg = double_stream or DoubleStreamConfig()
        ss_cfg = single_stream or SingleStreamConfig()

        self.double_stream_entropy_slices = (
            tuple(ds_cfg.entropy_slices) if ds_cfg.enabled else ()
        )
        self.single_stream_entropy_slices = (
            tuple(ss_cfg.entropy_slices) if ss_cfg.enabled else ()
        )
        if not (self.double_stream_entropy_slices or self.single_stream_entropy_slices):
            raise ValueError(
                "At least one entropy slice must be configured on an enabled stream"
            )

        print(
            f"[VETO] model={wrapper.model.value} surrogate_prompt={surrogate_prompt[:80]!r} "
            f"double_stream.entropy_slices={list(self.double_stream_entropy_slices)!r} "
            f"single_stream.entropy_slices={list(self.single_stream_entropy_slices)!r}"
        )

        num_double = wrapper.num_double_stream_layers
        if ds_cfg.enabled and self.double_stream_entropy_slices:
            if layer_indices is not None:
                raise ValueError("layer_indices override applies to single_stream only")
            normalize_ds = (
                fibo_normalize_double_stream_layer_indices
                if wrapper.model == ProtectionModel.FIBO_EDIT
                else normalize_double_stream_layer_indices
            )
            max_ds = (
                fibo_max_double_stream_layer_index
                if wrapper.model == ProtectionModel.FIBO_EDIT
                else max_double_stream_layer_index
            )
            self.double_stream_layer_indices = normalize_ds(
                ds_cfg.layer_indices, num_double
            )
            ds_stop = max_ds(self.double_stream_layer_indices, num_double)
            print(
                f"[VETO] double_stream hook_indices={self.double_stream_layer_indices} "
                f"run_blocks=0..{ds_stop} depth={num_double}"
            )
        else:
            self.double_stream_layer_indices = []

        num_single = wrapper.num_single_stream_layers
        if ss_cfg.enabled and self.single_stream_entropy_slices:
            ss_layers = (
                list(layer_indices) if layer_indices is not None else list(ss_cfg.layer_indices)
            )
            normalize_ss = (
                fibo_normalize_single_stream_layer_indices
                if wrapper.model == ProtectionModel.FIBO_EDIT
                else normalize_single_stream_layer_indices
            )
            max_ss = (
                fibo_max_single_stream_layer_index
                if wrapper.model == ProtectionModel.FIBO_EDIT
                else max_single_stream_layer_index
            )
            self.single_stream_layer_indices = normalize_ss(ss_layers, num_single)
            ss_stop = max_ss(self.single_stream_layer_indices, num_single)
            print(
                f"[VETO] single_stream hook_indices={self.single_stream_layer_indices} "
                f"run_blocks=0..{ss_stop} depth={num_single}"
            )
        else:
            self.single_stream_layer_indices = []

        self._log_labels = SLICE_LOG_LABELS
        self._all_entropy_slices = (
            self.double_stream_entropy_slices + self.single_stream_entropy_slices
        )

        self._surrogate: TextSurrogate | FiboSurrogate | None = None
        self._timesteps_ready = False
        self.x_source: Tensor | None = None

        self.generator = torch.Generator(device=wrapper.device).manual_seed(seed)
        self._timestep_generator = torch.Generator(device="cpu").manual_seed(seed)
        self._latest_mean_entropy: float | None = None

    def set_source(self, x_source: Tensor) -> None:
        self.x_source = x_source

    def prepare(self, x_source: Tensor) -> None:
        """Cache surrogate conditioning and build the noise schedule."""
        self._surrogate = self.wrapper.encode_surrogate(
            x_source,
            self.surrogate_prompt,
            guidance_scale=self.guidance_scale,
        )
        if isinstance(self._surrogate, TextSurrogate):
            print(
                f"[VETO] cached surrogate len={self._surrogate.prompt_embeds.shape[1]} "
                f"text={self.surrogate_prompt[:80]!r}"
            )
        else:
            print(
                f"[VETO] cached FIBO surrogate "
                f"len={self._surrogate.prompt_embeds.shape[1]} "
                f"batch={self._surrogate.prompt_embeds.shape[0]} "
                f"dimfusion_layers={len(self._surrogate.prompt_layers)} "
                f"text={self.surrogate_prompt[:80]!r}"
            )

        with torch.no_grad():
            probe_latents, _ = self.wrapper.encode_image(x_source.detach())
        self.wrapper.set_timesteps(
            canvas_seq_len=probe_latents.shape[1],
            num_steps=self.inference_steps,
        )
        self._timesteps_ready = True

    def _encode_canvas_and_reference(
        self, x_protected: Tensor
    ) -> tuple[Tensor, object, Tensor, object]:
        canvas_latents, canvas_meta = self.wrapper.encode_image(x_protected)
        ref_latents, ref_meta = self.wrapper.encode_reference(x_protected.detach())
        return canvas_latents, canvas_meta, ref_latents, ref_meta

    def loss(self, x_protected: Tensor) -> tuple[Tensor, Tensor]:
        if not self._timesteps_ready or self._surrogate is None:
            raise RuntimeError("Call prepare() before loss().")
        if self.x_source is None:
            raise RuntimeError("Call set_source() before loss().")

        with torch.no_grad():
            canvas_latents, canvas_meta, ref_latents, ref_meta = (
                self._encode_canvas_and_reference(x_protected)
            )

        canvas_latents = canvas_latents.detach().requires_grad_(True)

        timestep_indices = self.wrapper.sample_timestep_indices(
            self.num_timesteps_per_step,
            self._timestep_generator,
        )

        latent_grad_accum = torch.zeros_like(canvas_latents)
        loss_acc = torch.tensor(0.0, device=self.wrapper.device)
        entropies: list[float] = []
        entropies_by_slice: dict[str, list[float]] = {
            s: [] for s in self._all_entropy_slices
        }

        for t_idx in timestep_indices:
            noise = torch.randn(
                canvas_latents.shape,
                device=canvas_latents.device,
                dtype=canvas_latents.dtype,
                generator=self.generator,
            )
            noisy = self.wrapper.noisy_canvas_latents(
                canvas_latents, t_idx, self.generator, noise=noise
            )

            loss_step, state = veto_forward_entropy_loss(
                self.wrapper,
                canvas_latents=noisy,
                canvas_meta=canvas_meta,
                reference_latents=ref_latents,
                reference_meta=ref_meta,
                surrogate=self._surrogate,
                timestep_idx=t_idx,
                guidance_scale=self.guidance_scale,
                single_stream_layer_indices=(
                    self.single_stream_layer_indices or None
                ),
                single_stream_entropy_slices=(
                    self.single_stream_entropy_slices or None
                ),
                double_stream_layer_indices=(
                    self.double_stream_layer_indices or None
                ),
                double_stream_entropy_slices=(
                    self.double_stream_entropy_slices or None
                ),
                entropy_eps=self.entropy_eps,
            )

            h_t = float(state.mean_entropy().detach().item())
            entropies.append(h_t)
            log_parts = [f"t={t_idx}", f"H_total={h_t:.4f}"]
            for slice_name in self._all_entropy_slices:
                mean_h_slice = state.mean_entropy_slice(slice_name)
                if mean_h_slice is not None:
                    h_slice = float(mean_h_slice.detach().item())
                    entropies_by_slice[slice_name].append(h_slice)
                    label = self._log_labels.get(slice_name, slice_name)
                    log_parts.append(f"{label}={h_slice:.4f}")
            log_parts.append(f"loss={float(loss_step.detach().item()):.4f}")
            print(f"[VETO] {' '.join(log_parts)}")

            loss_acc = loss_acc + loss_step
            loss_step.backward()
            with torch.no_grad():
                if canvas_latents.grad is not None:
                    latent_grad_accum = latent_grad_accum + canvas_latents.grad
                canvas_latents.grad = None

        k = len(timestep_indices)
        loss_acc = loss_acc / k
        latent_grad_accum = latent_grad_accum / k

        x_in = x_protected.detach().requires_grad_(True)
        final_latents, _ = self.wrapper.encode_image(x_in)
        final_latents.backward(latent_grad_accum)
        x_grad = x_in.grad
        if x_grad is None:
            raise RuntimeError("VETO failed to propagate DiT gradients to pixels.")

        mean_h = sum(entropies) / len(entropies) if entropies else 0.0
        self._latest_mean_entropy = float(mean_h)

        summary = f"[VETO] total_loss={float(loss_acc.item()):.4f} mean_H_total={mean_h:.4f}"
        for slice_name in self._all_entropy_slices:
            vals = entropies_by_slice.get(slice_name) or []
            if vals:
                label = self._log_labels.get(slice_name, slice_name)
                summary += f" mean_{label}={sum(vals) / len(vals):.4f}"
        print(summary)

        return loss_acc.detach(), x_grad
