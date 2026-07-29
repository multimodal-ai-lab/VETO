from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Type

from veto.data.prompts_table import PromptRow
from veto.editing.backends.base import ImageEditingBackend
from veto.editing.backends.utils import get_backend_class
from veto.utils.path_layout import (
    derive_dataset_name,
    edited_variant_dir,
    edited_unprotected_dir,
    protected_output_name,
)
from tqdm import tqdm

def is_model_fully_complete(
    prompt_rows: Sequence[PromptRow],
    out_unprotected: Path,
    out_protected: Path,
) -> bool:
    """True if every expected output file exists for both unprotected and protected branches."""
    for row in prompt_rows:
        if not (out_unprotected / row.image_name).is_file():
            return False
        if not (out_protected / row.image_name).is_file():
            return False
    return True


class ImageEditingFactory(ABC):
    """Runs the edit loop per model (save outputs, progress, resume). Subclasses only build the editor for each model name."""

    def __init__(
        self,
        output_root: Path,
        run_id: str,
        eval_variant: str,
        prompt_rows: Sequence[PromptRow],
        unprotected_source_dir: Path,
        protected_source_dir: Path,
        dataset_name: Optional[str] = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.run_id = run_id
        self.eval_variant = eval_variant
        self.prompt_rows = list(prompt_rows)
        self.unprotected_source_dir = Path(unprotected_source_dir)
        self.protected_source_dir = Path(protected_source_dir)
        self.dataset_name = dataset_name or derive_dataset_name(self.unprotected_source_dir)

    @abstractmethod
    def resolve_backend(self, model_key: str) -> ImageEditingBackend:
        """Build the editor for this model name."""
        ...

    def run(self, model_keys: Sequence[str]) -> None:
        print(f"[generation] start models={list(model_keys)}")
        for model_key in model_keys:
            backend = self.resolve_backend(model_key)
            if backend.name != model_key:
                raise ValueError(
                    f"Backend name {backend.name!r} does not match key {model_key!r}"
                )

            out_u = edited_unprotected_dir(self.output_root, self.dataset_name, model_key)
            out_p = edited_variant_dir(
                self.output_root,
                self.dataset_name,
                self.run_id,
                model_key,
                self.eval_variant,
            )
            out_u.mkdir(parents=True, exist_ok=True)
            out_p.mkdir(parents=True, exist_ok=True)

            try:
                if is_model_fully_complete(self.prompt_rows, out_u, out_p):
                    print(f"[generation] skip model={model_key} (already complete)")
                    continue

                rows_iter = tqdm(
                    self.prompt_rows,
                    total=len(self.prompt_rows),
                    desc=f"[generation:{model_key}]",
                    leave=False,
                )
                for row in rows_iter:
                    src_u = self.unprotected_source_dir / row.image_name
                    src_p = self.protected_source_dir / protected_output_name(row.image_name)
                    if not src_u.is_file():
                        raise FileNotFoundError(f"Missing unprotected source: {src_u}")
                    if not src_p.is_file():
                        raise FileNotFoundError(f"Missing protected source: {src_p}")

                    dest_u = out_u / row.image_name
                    dest_p = out_p / row.image_name

                    if not dest_u.is_file():
                        backend.edit(src_u, row, dest_u)

                    if not dest_p.is_file():
                        backend.edit(src_p, row, dest_p)
                print(f"[generation] done model={model_key}")
            finally:
                backend.unload()


class StandardImageEditingFactory(ImageEditingFactory):
    """The normal factory: looks up editors by name from the shared registry and builds them with optional per-model settings."""

    def __init__(
        self,
        output_root: Path,
        run_id: str,
        prompt_rows: Sequence[PromptRow],
        unprotected_source_dir: Path,
        protected_source_dir: Path,
        eval_variant: str = "base",
        dataset_name: Optional[str] = None,
        *,
        backend_kwargs: Optional[Mapping[str, Mapping[str, object]]] = None,
        instances: Optional[Mapping[str, ImageEditingBackend]] = None,
    ) -> None:
        super().__init__(
            output_root=output_root,
            run_id=run_id,
            eval_variant=eval_variant,
            prompt_rows=prompt_rows,
            unprotected_source_dir=unprotected_source_dir,
            protected_source_dir=protected_source_dir,
            dataset_name=dataset_name,
        )
        self._backend_kwargs: Dict[str, Dict[str, object]] = {
            k: dict(v) for k, v in (backend_kwargs or {}).items()
        }
        self._instances: Dict[str, ImageEditingBackend] = dict(instances or {})

    def resolve_backend(self, model_key: str) -> ImageEditingBackend:
        if model_key in self._instances:
            return self._instances[model_key]
        cls: Type[ImageEditingBackend] = get_backend_class(model_key)
        kwargs = self._backend_kwargs.get(model_key, {})
        return cls(**kwargs)
