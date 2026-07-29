from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch
from torch import Tensor


class ProtectionModel(str, Enum):
    """DiT backend selected by ``protection.model`` in YAML."""

    FLUX2 = "flux2"
    FIBO_EDIT = "fibo_edit"


DEFAULT_MODEL_IDS: dict[ProtectionModel, str] = {
    ProtectionModel.FLUX2: "diffusers/FLUX.2-dev-bnb-4bit",
    ProtectionModel.FIBO_EDIT: "briaai/Fibo-Edit",
}


@dataclass(frozen=True)
class ImageMeta:
    latent_ids: Tensor


@dataclass(frozen=True)
class TextSurrogate:
    prompt_embeds: Tensor
    text_ids: Tensor


@dataclass(frozen=True)
class FiboSurrogate:
    prompt_embeds: Tensor
    text_ids: Tensor
    prompt_layers: tuple[Tensor, ...]
    token_attention_mask: Tensor


class DiTWrapper(ABC):

    model: ProtectionModel

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @property
    @abstractmethod
    def transformer(self) -> torch.nn.Module:
        raise NotImplementedError

    @property
    @abstractmethod
    def vae(self) -> torch.nn.Module:
        raise NotImplementedError

    @property
    @abstractmethod
    def scheduler(self) -> torch.nn.Module:
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        return self.transformer.dtype

    @property
    def num_double_stream_layers(self) -> int:
        return len(self.transformer.transformer_blocks)

    @property
    def num_single_stream_layers(self) -> int:
        blocks = getattr(self.transformer, "single_transformer_blocks", None)
        return len(blocks) if blocks is not None else 0

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "DiTWrapper":
        raise NotImplementedError

    @abstractmethod
    def encode_surrogate(
        self,
        x01: Tensor,
        prompt: str,
        *,
        guidance_scale: float = 1.0,
    ) -> TextSurrogate | FiboSurrogate:
        raise NotImplementedError

    @abstractmethod
    def encode_image(self, x01: Tensor) -> tuple[Tensor, ImageMeta]:
        raise NotImplementedError

    @abstractmethod
    def encode_reference(self, x01: Tensor) -> tuple[Tensor, ImageMeta]:
        raise NotImplementedError

    @abstractmethod
    def set_timesteps(self, canvas_seq_len: int, num_steps: int) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_timestep_indices(
        self, k: int, generator: torch.Generator
    ) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def noisy_canvas_latents(
        self,
        canvas_latents: Tensor,
        timestep_idx: int,
        generator: torch.Generator,
        noise: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_truncated(
        self,
        *,
        canvas_latents: Tensor,
        canvas_meta: ImageMeta,
        reference_latents: Tensor,
        reference_meta: ImageMeta,
        surrogate: TextSurrogate | FiboSurrogate,
        timestep_idx: int,
        guidance_scale: float,
        max_double_stream_index: Optional[int],
        max_single_stream_index: Optional[int],
    ) -> None:
        raise NotImplementedError
