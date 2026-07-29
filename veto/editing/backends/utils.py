from typing import Dict, Type

from veto.editing.backends.base import ImageEditingBackend
from veto.editing.backends.fibo_edit import FiboEditEditing
from veto.editing.backends.flux2 import Flux2Editing

_REGISTRY: Dict[str, Type[ImageEditingBackend]] = {
    Flux2Editing.name: Flux2Editing,
    FiboEditEditing.name: FiboEditEditing,
}


def get_backend_class(model_key: str) -> Type[ImageEditingBackend]:
    if model_key not in _REGISTRY:
        raise KeyError(
            f"Unknown edit backend {model_key!r}. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[model_key]


def list_edit_backend_keys() -> list[str]:
    return sorted(_REGISTRY)
