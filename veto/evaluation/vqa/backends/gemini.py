import os
import time
from pathlib import Path
from typing import Sequence

from google import genai

from veto.evaluation.vqa.base import VqaBackend, load_rgb_images


class GeminiVqaBackend(VqaBackend):
    slug = "gemini"

    def __init__(self, model: str | None = None) -> None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required for VQA model 'gemini'")
        self._client = genai.Client(api_key=key)
        self._model = model or os.environ.get("VETO_VQA_GEMINI_MODEL", "gemini-3.5-flash")
        self._max_retries = 5

    def ask(
        self,
        image_paths: Sequence[Path],
        question: str,
        *,
        max_new_tokens: int = 24,
    ) -> str:
        del max_new_tokens  # Gemini API does not expose this knob here
        images = load_rgb_images(image_paths)
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=[question, *images],
                )
                return (getattr(resp, "text", None) or "").strip()
            except Exception as e:
                last_error = e
                if attempt >= self._max_retries:
                    break
                time.sleep(min(2**attempt, 16))
        assert last_error is not None
        raise last_error
