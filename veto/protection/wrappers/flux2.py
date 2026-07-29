from typing import List, Optional

import numpy as np
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2 import (
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.utils import is_torch_npu_available
from torch import Tensor

from veto.protection.wrappers.base import (
    ImageMeta,
    TextSurrogate,
    ProtectionModel,
    DiTWrapper,
)


class Flux2Wrapper(DiTWrapper):
    model = ProtectionModel.FLUX2

    def __init__(self, pipe: Flux2Pipeline, device: torch.device) -> None:
        super().__init__(device)
        self.pipe = pipe
        self._transformer = pipe.transformer
        self._vae = pipe.vae
        self._scheduler = pipe.scheduler

    @property
    def transformer(self) -> Flux2Transformer2DModel:
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
        model_id: str = "diffusers/FLUX.2-dev-bnb-4bit",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Flux2Wrapper":
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        pipe = Flux2Pipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        return cls(pipe, device)

    @torch.no_grad()
    def encode_surrogate(
        self,
        x01: Tensor,
        prompt: str,
        *,
        guidance_scale: float = 1.0,
    ) -> TextSurrogate:
        del x01, guidance_scale
        prompt_embeds, text_ids = self.encode_prompt(prompt)
        return TextSurrogate(prompt_embeds=prompt_embeds, text_ids=text_ids)

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (10, 20, 30),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        return (
            prompt_embeds.to(device=self.device, dtype=self.transformer.dtype),
            text_ids.to(self.device),
        )

    @staticmethod
    def x01_to_vae_input(x01: torch.Tensor) -> torch.Tensor:
        return x01 * 2.0 - 1.0

    def encode_image(self, x01: torch.Tensor) -> tuple[torch.Tensor, ImageMeta]:
        x_vae = self.x01_to_vae_input(x01).to(dtype=self.vae.dtype)
        image_latent_bchw = self.pipe._encode_vae_image(x_vae, generator=None)
        image_latents = self.pipe._pack_latents(image_latent_bchw)
        image_latent_ids = self.pipe._prepare_image_ids([image_latent_bchw]).to(self.device)
        return image_latents, ImageMeta(latent_ids=image_latent_ids)

    def encode_reference(self, x01: torch.Tensor) -> tuple[torch.Tensor, ImageMeta]:
        return self.encode_image(x01)

    def set_timesteps(self, canvas_seq_len: int, num_steps: int) -> torch.Tensor:
        print(
            f"Setting FLUX-2 timesteps (image_seq_len: {canvas_seq_len} - num_steps: {num_steps})"
        )
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        if (
            hasattr(self.scheduler.config, "use_flow_sigmas")
            and self.scheduler.config.use_flow_sigmas
        ):
            sigmas = None

        mu = compute_empirical_mu(image_seq_len=canvas_seq_len, num_steps=num_steps)
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_steps,
            self.device,
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
        surrogate: TextSurrogate,
        timestep_idx: int,
        guidance_scale: float = 4.0,
        max_double_stream_index: Optional[int] = None,
        max_single_stream_index: Optional[int] = None,
    ) -> None:
        if max_double_stream_index is None and max_single_stream_index is None:
            raise ValueError(
                "At least one of max_double_stream_index or max_single_stream_index "
                "must be set"
            )

        batch_size = canvas_latents.shape[0]
        timestep = self.scheduler.timesteps[timestep_idx].to(
            device=self.device, dtype=canvas_latents.dtype
        )
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)

        guidance = torch.full([1], guidance_scale, device=self.device, dtype=torch.float32)

        latent_model_input = torch.cat([canvas_latents, reference_latents], dim=1).to(
            self.transformer.dtype
        )
        latent_image_ids = torch.cat(
            [canvas_meta.latent_ids, reference_meta.latent_ids], dim=1
        )

        flux2_transformer_forward_truncated(
            self.transformer,
            hidden_states=latent_model_input,
            encoder_hidden_states=surrogate.prompt_embeds,
            timestep=timestep / 1000,
            guidance=guidance,
            img_ids=latent_image_ids,
            txt_ids=surrogate.text_ids,
            max_double_stream_index=max_double_stream_index,
            max_single_stream_index=max_single_stream_index,
        )


def flux2_transformer_forward_truncated(
    transformer: Flux2Transformer2DModel,
    *,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    timestep: Tensor,
    guidance: Tensor,
    img_ids: Tensor,
    txt_ids: Tensor,
    max_double_stream_index: Optional[int] = None,
    max_single_stream_index: Optional[int] = None,
    joint_attention_kwargs: Optional[dict] = None,
) -> None:
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
    if max_double_stream_index is None and max_single_stream_index is None:
        raise ValueError(
            "At least one of max_double_stream_index or max_single_stream_index must be set"
        )

    joint_attention_kwargs = joint_attention_kwargs or {}

    timestep = timestep.to(hidden_states.dtype) * 1000
    guidance = guidance.to(hidden_states.dtype) * 1000

    temb = transformer.time_guidance_embed(timestep, guidance)
    double_stream_mod_img = transformer.double_stream_modulation_img(temb)
    double_stream_mod_txt = transformer.double_stream_modulation_txt(temb)
    single_stream_mod = transformer.single_stream_modulation(temb)

    hidden_states = transformer.x_embedder(hidden_states)
    encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    if is_torch_npu_available():
        freqs_cos_image, freqs_sin_image = transformer.pos_embed(img_ids.cpu())
        image_rotary_emb = (freqs_cos_image.npu(), freqs_sin_image.npu())
        freqs_cos_text, freqs_sin_text = transformer.pos_embed(txt_ids.cpu())
        text_rotary_emb = (freqs_cos_text.npu(), freqs_sin_text.npu())
    else:
        image_rotary_emb = transformer.pos_embed(img_ids)
        text_rotary_emb = transformer.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    for index_block, block in enumerate(transformer.transformer_blocks):
        if (
            max_double_stream_index is not None
            and index_block > max_double_stream_index
        ):
            break
        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            encoder_hidden_states, hidden_states = transformer._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                double_stream_mod_img,
                double_stream_mod_txt,
                concat_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

    if max_single_stream_index is None:
        return

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(transformer.single_transformer_blocks):
        if index_block > max_single_stream_index:
            break
        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            hidden_states = transformer._gradient_checkpointing_func(
                block,
                hidden_states,
                None,
                single_stream_mod,
                concat_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
