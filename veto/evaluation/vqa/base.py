from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from PIL import Image


def load_rgb_images(image_paths: Sequence[Path]) -> list[Image.Image]:
    if not image_paths:
        raise ValueError("At least one image path is required")
    return [Image.open(p).convert("RGB") for p in image_paths]


class VqaBackend(ABC):
    slug: str

    @abstractmethod
    def ask(
        self,
        image_paths: Sequence[Path],
        question: str,
        *,
        max_new_tokens: int = 24,
    ) -> str:
        """Run VQA on one or more images (order preserved) with a text question."""
        ...

    def unload(self) -> None:
        pass
