import gc
import json
import time
import warnings
from typing import Any

import torch
from diffusers.modular_pipelines import modular_pipeline
from diffusers.modular_pipelines import ModularPipelineBlocks
from PIL import Image

DEFAULT_VLM_MODEL_ID = "briaai/FIBO-edit-prompt-to-JSON"
DEFAULT_VLM_MAX_RETRIES = 3

_pipeline_cache: dict[str, object] = {}


def _patched_validate_requirements(reqs: Any) -> dict:
    """Skip diffusers strict version checks for Bria remote-code blocks."""
    if reqs is None:
        return {}
    if isinstance(reqs, dict):
        return dict(reqs)
    if isinstance(reqs, list):
        return {str(k): str(v) for k, v in reqs}
    return {}


modular_pipeline._validate_requirements = _patched_validate_requirements


def _load_vlm_pipeline(model_id: str = DEFAULT_VLM_MODEL_ID):
    if model_id in _pipeline_cache:
        return _pipeline_cache[model_id]

    blocks = ModularPipelineBlocks.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    pipeline = blocks.init_pipeline()
    _pipeline_cache[model_id] = pipeline
    return pipeline


def generate_prompt_local(
    image: Image.Image,
    instruction: str,
    *,
    model: str = DEFAULT_VLM_MODEL_ID,
    device: torch.device | str = "cuda",
    max_retries: int = DEFAULT_VLM_MAX_RETRIES,
) -> str:
    """Return structured JSON string for ``BriaFiboEditPipeline`` (Bria local VLM)."""
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    pipeline = _load_vlm_pipeline(model)
    device = torch.device(device)
    pipeline.to(device)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            output = pipeline(image=image, prompt=instruction)
            json_prompt = output.values["json_prompt"]
            if not isinstance(json_prompt, str):
                json_prompt = json.dumps(json_prompt)
            pipeline.to("cpu")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return json_prompt
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            last_err = exc
            if attempt + 1 < max_retries:
                warnings.warn(
                    f"[FIBO VLM] attempt {attempt + 1}/{max_retries} failed for "
                    f"instruction={instruction[:80]!r}: {exc}; retrying",
                    stacklevel=2,
                )
                time.sleep(0.25 * (attempt + 1))
            continue

    pipeline.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if last_err is None:
        raise RuntimeError(
            f"[FIBO VLM] failed after {max_retries} attempt(s) for "
            f"instruction={instruction[:80]!r}"
        )
    raise last_err


def build_fibo_edit_json(
    edit_instruction: str,
    base_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(base_json) if base_json else {}
    payload["edit_instruction"] = edit_instruction
    return payload


def fibo_edit_prompt_string(
    edit_instruction: str,
    base_json: dict[str, Any] | None = None,
) -> str:
    return json.dumps(build_fibo_edit_json(edit_instruction, base_json=base_json))


def build_edit_prompt_json(
    image: Image.Image,
    instruction: str,
    *,
    model: str = DEFAULT_VLM_MODEL_ID,
    device: torch.device | str = "cuda",
    use_vlm: bool = True,
    base_json: dict[str, Any] | None = None,
    max_retries: int = DEFAULT_VLM_MAX_RETRIES,
) -> dict[str, Any]:
    if use_vlm:
        raw = generate_prompt_local(
            image,
            instruction,
            model=model,
            device=device,
            max_retries=max_retries,
        )
        payload = json.loads(raw)
        payload["edit_instruction"] = instruction
        return payload
    return build_fibo_edit_json(instruction, base_json=base_json)


def unload_vlm_cache() -> None:
    _pipeline_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
