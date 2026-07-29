from typing import Optional, Union

import torch
from diffusers import Flux2Pipeline
from PIL import Image

from veto.editing.backends.base import ImageEditingBackend
from veto.utils.cuda_memory import release_cuda_memory


class Flux2Editing(ImageEditingBackend):
    name = "flux2"

    def __init__(
        self,
        model_id: str = "diffusers/FLUX.2-dev-bnb-4bit",
        device: Union[torch.device, str] = "cuda",
        num_inference_steps: int = 28,
        guidance_scale: float = 4.0,
        image_size: int = 512,
        pipeline: Optional[Flux2Pipeline] = None
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.device = torch.device(device)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self._pipe: Optional[Flux2Pipeline] = pipeline or None

    def _load_pipe(self) -> Flux2Pipeline:
        if self._pipe is not None:
            return self._pipe
        self._pipe = Flux2Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        self._pipe = self._pipe.to(self.device)
        self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    def edit_image(self, image: Image, prompt: str, generator: torch.Generator = None) -> Image:
        pipe = self._load_pipe()
        h, w = image.height, image.width

        with torch.inference_mode():
            out = pipe(
                image=image,
                prompt=prompt,
                height=h,
                width=w,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator
            ).images[0]

        return out

    def edit_images_batch(self, images, prompts: list[str], generator: torch.Generator = None) -> list[Image.Image]:
        """Run all (image, prompt) pairs in a single batched pipeline call."""
        assert len(images) == len(prompts), "images and prompts must have the same length"
        pipe = self._load_pipe()
        h = w = self.image_size
        with torch.inference_mode():
            out = pipe(
                image=images,
                prompt=prompts,
                height=h,
                width=w,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).images
        return out

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        release_cuda_memory()
