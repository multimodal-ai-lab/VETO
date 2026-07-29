import base64
import os
import time
from pathlib import Path
from typing import Sequence

from openai import OpenAI

from veto.evaluation.vqa.base import VqaBackend


def _encode_image(image_path: Path) -> str:
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }[image_path.suffix.lower()]
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


class GptVqaBackend(VqaBackend):
    slug = "gpt"

    def __init__(self, model: str | None = None) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for VQA model 'gpt'")
        self._client = OpenAI()
        self._model = model or os.environ.get("VETO_VQA_GPT_MODEL", "gpt-5.5")
        self._max_retries = 5

    def ask(
        self,
        image_paths: Sequence[Path],
        question: str,
        *,
        max_new_tokens: int = 24,
    ) -> str:
        del max_new_tokens
        content: list[dict[str, str]] = [{"type": "input_text", "text": question}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": _encode_image(Path(image_path)),
                }
            )

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.responses.create(
                    model=self._model,
                    input=[{"role": "user", "content": content}],
                )
                return (getattr(response, "output_text", None) or "").strip()
            except Exception as e:
                last_error = e
                if attempt >= self._max_retries:
                    break
                time.sleep(min(2**attempt, 16))
        assert last_error is not None
        raise last_error
