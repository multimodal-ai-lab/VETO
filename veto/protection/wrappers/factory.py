import warnings

import torch

from veto.protection.wrappers.base import (
    DEFAULT_MODEL_IDS,
    DiTWrapper,
    ProtectionModel,
)
from veto.protection.wrappers.flux2 import Flux2Wrapper
from veto.protection.wrappers.fibo_edit import FiboEditWrapper


def default_model_id(model: ProtectionModel) -> str:
    return DEFAULT_MODEL_IDS[model]


def build_wrapper(
    *,
    model: str | ProtectionModel,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    image_size: int = 1024,
    model_id: str | None = None,
    fibo_max_sequence_length: int = 3000,
    fibo_base_json: dict | None = None,
) -> DiTWrapper:
    resolved = (
        model if isinstance(model, ProtectionModel) else ProtectionModel(str(model).strip().lower())
    )
    device = torch.device(device)
    hub_id = model_id or default_model_id(resolved)

    if resolved == ProtectionModel.FLUX2:
        return Flux2Wrapper.from_pretrained(hub_id, device=device, dtype=dtype)
    if resolved == ProtectionModel.FIBO_EDIT:
        return FiboEditWrapper.from_pretrained(
            hub_id,
            device=device,
            dtype=dtype,
            image_size=image_size,
            max_sequence_length=fibo_max_sequence_length,
            fibo_base_json=fibo_base_json,
        )
    raise ValueError(f"Unknown model {model!r}; use {sorted(ProtectionModel)}")


def resolve_model(protection_cfg: dict) -> ProtectionModel:
    """Parse ``protection.model`` (default ``flux2``)."""
    if "model" in protection_cfg:
        raw = protection_cfg["model"]
    elif "model_family" in protection_cfg:
        warnings.warn(
            "protection.model_family is deprecated; use protection.model",
            DeprecationWarning,
            stacklevel=2,
        )
        raw = protection_cfg["model_family"]
    else:
        raw = ProtectionModel.FLUX2.value

    return ProtectionModel(str(raw).strip().lower())
