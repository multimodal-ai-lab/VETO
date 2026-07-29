from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image
import torch

from veto.data.prompts_table import PromptRow


class ImageEditingBackend(ABC):
    """
    Editing model (e.g. InstructPix2Pix, API-backed nano-banana, etc.).
    """

    name: str

    def __init__(self, **kwargs: object) -> None:
        _ = kwargs

    def edit(self, source_path: Path, row: PromptRow, out_path: Path) -> None:
        """
        Read ``source_path``, apply the edit implied by ``row.editing_instruction``, write ``out_path``.
        """

        prompt = (row.editing_instruction or "").strip()
        if not prompt:
            raise ValueError("editing_instruction is empty")

        image = Image.open(source_path).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out = self.edit_image(image, prompt)
        out.save(out_path)

    @abstractmethod
    def edit_image(self, image: Image, prompt: str, generator: torch.Generator = None) -> Image:
        ...

    def unload(self) -> None:
        """Release GPU resources held by this backend (no-op if none)."""
