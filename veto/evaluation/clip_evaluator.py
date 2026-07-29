from typing import NamedTuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from veto.utils.images import resize_image


class CLIPScores(NamedTuple):
    clip_i_unprotected: float
    clip_i_protected: float
    clip_t_unprotected: float
    clip_t_protected: float
    clip_dir_unprotected: float
    clip_dir_protected: float


class CLIPEvaluator:
    def __init__(
        self,
        device: torch.device,
        model_id: str = "openai/clip-vit-base-patch32",
        *,
        image_size: int = 512,
    ):
        self.device = device
        self.image_size = image_size
        self.model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def _image_embed(self, pil: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=pil.convert("RGB"), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        image_features = self.model.get_image_features(
            pixel_values=pixel_values
        )
        return F.normalize(image_features, dim=-1)

    @torch.no_grad()
    def _text_embed(self, text: str) -> torch.Tensor:
        proc = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        )
        input_ids = proc["input_ids"].to(self.device)
        attention_mask = proc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            text_features = self.model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
        else:
            text_features = self.model.get_text_features(input_ids=input_ids)
        return F.normalize(text_features, dim=-1)

    @staticmethod
    def _directional_cosine(
        source_embed: torch.Tensor,
        edited_embed: torch.Tensor,
        text_orig_embed: torch.Tensor,
        text_edited_embed: torch.Tensor,
    ) -> float:
        delta_i = edited_embed - source_embed
        delta_t = text_edited_embed - text_orig_embed
        ni = float(delta_i.norm().item())
        nt = float(delta_t.norm().item())
        if ni < 1e-8 or nt < 1e-8:
            return 0.0
        delta_i_n = F.normalize(delta_i, dim=-1)
        delta_t_n = F.normalize(delta_t, dim=-1)
        return float((delta_i_n * delta_t_n).sum().item())

    @torch.no_grad()
    def evaluate(
        self,
        source_path: str,
        edited_unprotected_path: str,
        edited_protected_path: str,
        original_prompt: str,
        edited_prompt: str,
    ) -> CLIPScores:
        source = resize_image(Image.open(source_path), self.image_size)
        edited_unprotected = resize_image(
            Image.open(edited_unprotected_path), self.image_size
        )
        edited_protected = resize_image(
            Image.open(edited_protected_path), self.image_size
        )

        source_embed = self._image_embed(source)
        edited_unprotected_embed = self._image_embed(edited_unprotected)
        edited_protected_embed = self._image_embed(edited_protected)
        text_orig_embed = self._text_embed(original_prompt)
        text_edited_embed = self._text_embed(edited_prompt)

        clip_i_unprotected = float((source_embed * edited_unprotected_embed).sum().item())
        clip_i_protected = float((source_embed * edited_protected_embed).sum().item())

        clip_t_unprotected = float((text_edited_embed * edited_unprotected_embed).sum().item())
        clip_t_protected = float((text_edited_embed * edited_protected_embed).sum().item())

        clip_dir_unprotected = self._directional_cosine(source_embed, edited_unprotected_embed, text_orig_embed, text_edited_embed)
        clip_dir_protected = self._directional_cosine(source_embed, edited_protected_embed, text_orig_embed, text_edited_embed)

        return CLIPScores(
            clip_i_unprotected,
            clip_i_protected,
            clip_t_unprotected,
            clip_t_protected,
            clip_dir_unprotected,
            clip_dir_protected,
        )

    def unload(self) -> None:
        del self.model
        del self.processor
