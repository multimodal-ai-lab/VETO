from typing import Any, Optional, Union

import torch
from diffusers import BriaFiboEditPipeline
from PIL import Image

from veto.editing.backends.base import ImageEditingBackend
from veto.editing.fibo.vlm import (
    build_edit_prompt_json,
    build_fibo_edit_json,
    unload_vlm_cache,
)
from veto.utils.cuda_memory import release_cuda_memory


class FiboEditEditing(ImageEditingBackend):
    name = "fibo_edit"

    def __init__(
        self,
        model_id: str = "briaai/Fibo-Edit",
        device: Union[torch.device, str] = "cuda",
        num_inference_steps: int = 28,
        guidance_scale: float = 5,
        negative_prompt: str = "",
        image_size: int = 512,
        max_sequence_length: int = 3000,
        fibo_base_json: dict[str, Any] | None = None,
        fibo_use_vlm: bool = True,
        fibo_vlm_model_id: str = "briaai/FIBO-edit-prompt-to-JSON",
        pipeline: Optional[BriaFiboEditPipeline] = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.device = torch.device(device)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.image_size = int(image_size)
        self.max_sequence_length = max_sequence_length
        self.fibo_base_json = fibo_base_json
        self.fibo_use_vlm = fibo_use_vlm
        self.fibo_vlm_model_id = fibo_vlm_model_id
        self._pipe: Optional[BriaFiboEditPipeline] = pipeline
        self._warned_minimal_json = False

    def _load_pipe(self) -> BriaFiboEditPipeline:
        if self._pipe is not None:
            return self._pipe
        self._pipe = BriaFiboEditPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        self._pipe = self._pipe.to(self.device)
        self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    def _edit_json(self, image: Image.Image, edit_instruction: str) -> dict[str, Any]:
        if self.fibo_use_vlm:
            return build_edit_prompt_json(
                image,
                edit_instruction,
                model=self.fibo_vlm_model_id,
                device=self.device,
                use_vlm=True,
            )
        if self.fibo_base_json is not None:
            return build_fibo_edit_json(edit_instruction, base_json=self.fibo_base_json)
        if not self._warned_minimal_json:
            print(
                "[FIBO eval] fibo_use_vlm=false and no fibo_base_json; "
                "using minimal edit_instruction-only JSON."
            )
            self._warned_minimal_json = True
        return build_fibo_edit_json(edit_instruction, base_json=None)

    def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        generator: torch.Generator = None,
    ) -> Image.Image:
        pipe = self._load_pipe()
        edit_json = self._edit_json(image, prompt)
        h = w = self.image_size

        with torch.inference_mode():
            out = pipe(
                image=image,
                prompt=edit_json,
                height=h,
                width=w,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=self.negative_prompt,
                max_sequence_length=self.max_sequence_length,
                generator=generator,
            ).images[0]

        return out

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        unload_vlm_cache()
        release_cuda_memory()
