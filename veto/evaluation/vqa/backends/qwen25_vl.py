import os
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from veto.evaluation.vqa.base import VqaBackend, load_rgb_images

_DEFAULT_QWEN_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


class Qwen25VlVqaBackend(VqaBackend):
    slug = "qwen"

    def __init__(self, model_id: str | None = None) -> None:
        mid = model_id or os.environ.get("VETO_VQA_QWEN_MODEL", _DEFAULT_QWEN_VL_MODEL)
        self._processor = AutoProcessor.from_pretrained(mid)
        if "72b" in mid.lower():
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                mid,
                quantization_config=quant,
                device_map="auto",
            )
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        dev = next(self._model.parameters()).device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                if torch.is_floating_point(v):
                    inputs[k] = v.to(dev, dtype=torch.bfloat16)
                else:
                    inputs[k] = v.to(dev)
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        in_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[0, in_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    def unload(self) -> None:
        del self._model
        torch.cuda.empty_cache()
