import json
from typing import Any, List, Optional

import numpy as np
import torch
from diffusers import BriaFiboEditPipeline
from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from torch import Tensor

from veto.protection.wrappers.base import (
    FiboSurrogate,
    ImageMeta,
    DiTWrapper,
    ProtectionModel,
)
from veto.editing.fibo.vlm import build_edit_prompt_json
from veto.utils.images import tensor01_to_pil


class FiboEditWrapper(DiTWrapper):
    model = ProtectionModel.FIBO_EDIT

    def __init__(
        self,
        pipe: BriaFiboEditPipeline,
        device: torch.device,
        *,
        image_size: int = 512,
        max_sequence_length: int = 3000,
        fibo_base_json: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(device)
        self.pipe = pipe
        self._transformer = pipe.transformer
        self._vae = pipe.vae
        self._scheduler = pipe.scheduler
        self.image_size = int(image_size)
        self.max_sequence_length = int(max_sequence_length)
        self.fibo_base_json = fibo_base_json

    @property
    def transformer(self) -> BriaFiboTransformer2DModel:
        return self._transformer

    @property
    def vae(self):
        return self._vae

    @property
    def scheduler(self):
        return self._scheduler

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "briaai/Fibo-Edit",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        image_size: int = 512,
        max_sequence_length: int = 3000,
        fibo_base_json: dict[str, Any] | None = None,
    ) -> "FiboEditWrapper":
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        pipe = BriaFiboEditPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        return cls(
            pipe,
            device,
            image_size=image_size,
            max_sequence_length=max_sequence_length,
            fibo_base_json=fibo_base_json,
        )

    @staticmethod
    def _trim_prompt_layers(
        prompt_layers: list[Tensor],
        transformer: BriaFiboTransformer2DModel,
    ) -> tuple[Tensor, ...]:
        total = len(transformer.transformer_blocks) + len(
            transformer.single_transformer_blocks
        )
        if len(prompt_layers) >= total:
            trimmed = prompt_layers[len(prompt_layers) - total :]
        else:
            trimmed = prompt_layers + [prompt_layers[-1]] * (total - len(prompt_layers))
        return tuple(trimmed)

    @torch.no_grad()
    def encode_surrogate(
        self,
        x01: Tensor,
        prompt: str,
        *,
        guidance_scale: float = 1.0,
    ) -> FiboSurrogate:
        image = tensor01_to_pil(x01.detach().cpu())
        edit_json = build_edit_prompt_json(
            image,
            prompt,
            device=self.device,
            use_vlm=True,
            base_json=self.fibo_base_json,
        )
        prompt_str = json.dumps(edit_json)
        return self._encode_fibo_surrogate_prompt_str(
            prompt_str,
            guidance_scale=guidance_scale,
        )

    def _encode_fibo_surrogate_prompt_str(
        self,
        prompt_str: str,
        *,
        guidance_scale: float = 1.0,
    ) -> FiboSurrogate:
        (
            prompt_embeds,
            negative_prompt_embeds,
            text_ids,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_layers,
            negative_prompt_layers,
        ) = self.pipe.encode_prompt(
            prompt=prompt_str,
            negative_prompt="",
            guidance_scale=guidance_scale,
            device=self.device,
            max_sequence_length=self.max_sequence_length,
            num_images_per_prompt=1,
        )

        if guidance_scale > 1.0:
            if negative_prompt_embeds is None or negative_prompt_layers is None:
                raise RuntimeError(
                    "FIBO encode_prompt did not return negative embeddings for CFG"
                )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_layers = [
                torch.cat([negative_prompt_layers[i], prompt_layers[i]], dim=0)
                for i in range(len(prompt_layers))
            ]
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask],
                dim=0,
            )

        prompt_layers_tuple = self._trim_prompt_layers(
            list(prompt_layers),
            self.transformer,
        )
        return FiboSurrogate(
            prompt_embeds=prompt_embeds.to(
                device=self.device, dtype=self.transformer.dtype
            ),
            text_ids=text_ids.to(self.device),
            prompt_layers=tuple(
                layer.to(device=self.device, dtype=self.transformer.dtype)
                for layer in prompt_layers_tuple
            ),
            token_attention_mask=prompt_attention_mask.to(self.device),
        )

    def _latent_grid_hw(self) -> tuple[int, int]:
        h = w = self.image_size
        height = int(h) // self.pipe.vae_scale_factor
        width = int(w) // self.pipe.vae_scale_factor
        return height, width

    def _preprocess_x01(self, x01: Tensor) -> Tensor:
        """``[B,3,H,W]`` in ``[0,1]`` → Wan VAE input ``[B,3,1,H,W]`` (pipeline-faithful scale)."""
        if x01.ndim != 4 or x01.shape[1] != 3:
            raise ValueError(f"Expected x01 shape [B,3,H,W], got {tuple(x01.shape)}")
        h = w = self.image_size
        if x01.shape[-2] != h or x01.shape[-1] != w:
            x01 = torch.nn.functional.interpolate(
                x01,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        x_vae = (x01 * 2.0 - 1.0).to(dtype=self.vae.dtype)
        return x_vae.unsqueeze(2)

    def _wan_latents_mean_scaled(self, x_vae_cthw: Tensor) -> Tensor:
        device = x_vae_cthw.device
        dtype = x_vae_cthw.dtype
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(device, dtype)
        )
        encoded = self.vae.encode(x_vae_cthw).latent_dist.mean
        return (encoded - latents_mean) * latents_std

    def _pack_latents_bsd(self, latents_bchw: Tensor) -> Tensor:
        batch_size = latents_bchw.shape[0]
        num_channels = latents_bchw.shape[1]
        height, width = latents_bchw.shape[2], latents_bchw.shape[3]
        return self.pipe._pack_latents_no_patch(
            latents=latents_bchw,
            batch_size=batch_size,
            num_channels_latents=num_channels,
            height=height,
            width=width,
        )

    def _latent_image_ids(
        self,
        height: int,
        width: int,
        *,
        reference: bool,
    ) -> Tensor:
        ids = self.pipe._prepare_latent_image_ids(
            batch_size=1,
            height=height,
            width=width,
            device=self.device,
            dtype=self.transformer.dtype,
        )
        if reference:
            ids = ids.clone()
            ids[..., 0] = 1
        return ids.unsqueeze(0)

    def encode_image(self, x01: Tensor) -> tuple[Tensor, ImageMeta]:
        x_vae = self._preprocess_x01(x01)
        scaled = self._wan_latents_mean_scaled(x_vae)
        bchw = scaled[:, :, 0, :, :]
        packed = self._pack_latents_bsd(bchw)
        lh, lw = bchw.shape[2], bchw.shape[3]
        ids = self._latent_image_ids(lh, lw, reference=False)
        return packed, ImageMeta(latent_ids=ids)

    def encode_reference(self, x01: Tensor) -> tuple[Tensor, ImageMeta]:
        x_vae = self._preprocess_x01(x01)
        scaled = self._wan_latents_mean_scaled(x_vae)
        bchw = scaled[:, :, 0, :, :]
        packed = self._pack_latents_bsd(bchw)
        lh, lw = bchw.shape[2], bchw.shape[3]
        ids = self._latent_image_ids(lh, lw, reference=True)
        return packed, ImageMeta(latent_ids=ids)

    @staticmethod
    def build_joint_attention_mask(
        pipe: BriaFiboEditPipeline,
        *,
        token_attention_mask: Tensor,
        canvas_seq_len: int,
        reference_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, Tensor]:
        batch_size = int(token_attention_mask.shape[0])
        latent_attention_mask = torch.ones(
            (batch_size, canvas_seq_len),
            dtype=dtype,
            device=device,
        )
        image_latent_attention_mask = torch.ones(
            (batch_size, reference_seq_len),
            dtype=dtype,
            device=device,
        )
        segment_mask = torch.cat(
            [
                token_attention_mask.to(device=device, dtype=dtype),
                latent_attention_mask,
                image_latent_attention_mask,
            ],
            dim=1,
        )
        attention_matrix = pipe.create_attention_matrix(segment_mask)
        attention_matrix = attention_matrix.unsqueeze(dim=1).to(dtype=dtype)
        return {"attention_mask": attention_matrix}

    def set_timesteps(self, canvas_seq_len: int, num_steps: int) -> Tensor:
        print(
            f"Setting FIBO Edit timesteps (image_seq_len: {canvas_seq_len} - num_steps: {num_steps})"
        )
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        mu = calculate_shift(
            canvas_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_steps,
            self.device,
            timesteps=None,
            sigmas=sigmas,
            mu=mu,
        )
        self.pipe._num_timesteps = len(timesteps)
        return timesteps

    def sample_timestep_indices(
        self,
        k: int,
        generator: torch.Generator,
    ) -> List[int]:
        num_steps = len(self.scheduler.timesteps)
        if k > num_steps:
            raise ValueError(f"k={k} exceeds num_steps={num_steps}")
        perm = torch.randperm(num_steps, generator=generator)
        return [int(perm[i].item()) for i in range(k)]

    def noisy_canvas_latents(
        self,
        canvas_latents: Tensor,
        timestep_idx: int,
        generator: torch.Generator,
        noise: Tensor | None = None,
    ) -> Tensor:
        sigmas = self.scheduler.sigmas.to(
            device=canvas_latents.device, dtype=canvas_latents.dtype
        )
        sigma = sigmas[timestep_idx]
        if noise is None:
            noise = torch.randn(
                canvas_latents.shape,
                device=canvas_latents.device,
                dtype=canvas_latents.dtype,
                generator=generator,
            )
        elif noise.shape != canvas_latents.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} != canvas_latents shape "
                f"{tuple(canvas_latents.shape)}"
            )
        return (1.0 - sigma) * canvas_latents + sigma * noise

    def forward_truncated(
        self,
        *,
        canvas_latents: Tensor,
        canvas_meta: ImageMeta,
        reference_latents: Tensor,
        reference_meta: ImageMeta,
        surrogate: FiboSurrogate,
        timestep_idx: int,
        guidance_scale: float = 1.0,
        max_double_stream_index: Optional[int] = None,
        max_single_stream_index: Optional[int] = None,
    ) -> None:
        if max_double_stream_index is None and max_single_stream_index is None:
            raise ValueError(
                "At least one of max_double_stream_index or max_single_stream_index must be set"
            )
        latent_model_input = torch.cat([canvas_latents, reference_latents], dim=1).to(
            self.transformer.dtype
        )
        if guidance_scale > 1.0:
            latent_model_input = torch.cat(
                [latent_model_input, latent_model_input],
                dim=0,
            )

        timestep = self.scheduler.timesteps[timestep_idx].to(
            device=self.device, dtype=canvas_latents.dtype
        )
        if timestep.ndim == 0:
            timestep = timestep.expand(latent_model_input.shape[0])
        img_ids = torch.cat(
            [canvas_meta.latent_ids, reference_meta.latent_ids], dim=1
        )
        if img_ids.ndim == 3 and img_ids.shape[0] == 1:
            pass
        elif img_ids.ndim == 2:
            img_ids = img_ids.unsqueeze(0)

        canvas_seq_len = int(canvas_latents.shape[1])
        reference_seq_len = int(reference_latents.shape[1])
        joint_kwargs = self.build_joint_attention_mask(
            self.pipe,
            token_attention_mask=surrogate.token_attention_mask,
            canvas_seq_len=canvas_seq_len,
            reference_seq_len=reference_seq_len,
            dtype=self.transformer.dtype,
            device=self.device,
        )

        fibo_transformer_forward_truncated(
            self.transformer,
            hidden_states=latent_model_input,
            encoder_hidden_states=surrogate.prompt_embeds,
            text_encoder_layers=list(surrogate.prompt_layers),
            timestep=timestep,
            txt_ids=surrogate.text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=joint_kwargs,
            max_double_stream_index=max_double_stream_index,
            max_single_stream_index=max_single_stream_index,
        )


def fibo_transformer_forward_truncated(
    transformer: BriaFiboTransformer2DModel,
    *,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    text_encoder_layers: list[Tensor],
    timestep: Tensor,
    txt_ids: Tensor,
    img_ids: Tensor,
    joint_attention_kwargs: dict[str, Any],
    max_double_stream_index: Optional[int] = None,
    max_single_stream_index: Optional[int] = None,
) -> None:
    """Mirror ``BriaFiboTransformer2DModel.forward`` up to hooked layer indices."""
    n_double = len(transformer.transformer_blocks)
    n_single = len(transformer.single_transformer_blocks)
    if max_double_stream_index is not None and (
        max_double_stream_index < 0 or max_double_stream_index >= n_double
    ):
        raise IndexError(
            f"max_double_stream_index={max_double_stream_index} out of range [0, {n_double})"
        )
    if max_single_stream_index is not None and (
        max_single_stream_index < 0 or max_single_stream_index >= n_single
    ):
        raise IndexError(
            f"max_single_stream_index={max_single_stream_index} out of range [0, {n_single})"
        )

    hidden_states = transformer.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype)
    temb = transformer.time_embed(timestep, dtype=hidden_states.dtype)

    encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = transformer.pos_embed(ids)

    new_text_encoder_layers: list[Tensor] = []
    for i, text_encoder_layer in enumerate(text_encoder_layers):
        projected = transformer.caption_projection[i](text_encoder_layer)
        new_text_encoder_layers.append(projected)
    text_encoder_layers = new_text_encoder_layers

    block_id = 0
    for index_block, block in enumerate(transformer.transformer_blocks):
        if (
            max_double_stream_index is not None
            and index_block > max_double_stream_index
        ):
            break
        current_text_encoder_layer = text_encoder_layers[block_id]
        encoder_hidden_states = torch.cat(
            [
                encoder_hidden_states[:, :, : transformer.inner_dim // 2],
                current_text_encoder_layer,
            ],
            dim=-1,
        )
        block_id += 1
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    if max_single_stream_index is None:
        return

    for index_block, block in enumerate(transformer.single_transformer_blocks):
        if index_block > max_single_stream_index:
            break
        current_text_encoder_layer = text_encoder_layers[block_id]
        encoder_hidden_states = torch.cat(
            [
                encoder_hidden_states[:, :, : transformer.inner_dim // 2],
                current_text_encoder_layer,
            ],
            dim=-1,
        )
        block_id += 1
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        encoder_hidden_states = hidden_states[
            :, : encoder_hidden_states.shape[1], ...
        ]
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
