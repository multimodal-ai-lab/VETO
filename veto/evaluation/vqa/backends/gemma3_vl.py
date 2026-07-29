import os
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from veto.evaluation.vqa.base import VqaBackend, load_rgb_images

_DEFAULT_GEMMA3_MODEL = "google/gemma-3-12b-it"


class Gemma3VqaBackend(VqaBackend):
    slug = "gemma3"

    def __init__(self, model_id: str | None = None) -> None:
        mid = model_id or os.environ.get("VETO_VQA_GEMMA3_MODEL", _DEFAULT_GEMMA3_MODEL)
        self._processor = AutoProcessor.from_pretrained(mid, padding_side="left")
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            mid,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()

    def ask(
        self,
        image_paths: Sequence[Path],
        question: str,
        *,
        max_new_tokens: int = 24,
    ) -> str:
        images = load_rgb_images(image_paths)
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        dev = next(self._model.parameters()).device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                if torch.is_floating_point(v):
                    inputs[k] = v.to(dev, dtype=torch.bfloat16)
                else:
                    inputs[k] = v.to(dev)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_tokens = out[0, input_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    def unload(self) -> None:
        del self._model
        torch.cuda.empty_cache()
