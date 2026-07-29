import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from veto.utils.path_layout import resolve_dataset_images_dir

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


@dataclass(frozen=True)
class PromptRow:
    """One row of the prompts file format."""

    idx: int
    original_prompt: str
    editing_instruction: str
    edited_prompt: str
    image_name: str = ""


def load_prompts_csv(path: Path) -> List[PromptRow]:
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"idx", "original_prompt", "editing_instruction", "edited_prompt"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must contain {required}, got {reader.fieldnames!r}")
        rows: List[PromptRow] = []
        for raw in reader:
            rows.append(
                PromptRow(
                    idx=int(raw["idx"]),
                    original_prompt=raw["original_prompt"],
                    editing_instruction=raw["editing_instruction"],
                    edited_prompt=raw["edited_prompt"],
                )
            )
    return rows


def load_prompts_for_dataset(dataset_dir: Path) -> List[PromptRow]:
    """Load prompts.csv and attach each row's image basename from images/."""
    dataset_dir = Path(dataset_dir)
    rows = load_prompts_csv(dataset_dir / "prompts.csv")
    images_dir = resolve_dataset_images_dir(dataset_dir)
    names: dict[int, str] = {}
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
            names[int(path.stem)] = path.name
    out: List[PromptRow] = []
    for row in rows:
        name = names.get(row.idx)
        if name is None:
            raise FileNotFoundError(f"No image for idx {row.idx} in {images_dir}")
        out.append(
            PromptRow(
                idx=row.idx,
                original_prompt=row.original_prompt,
                editing_instruction=row.editing_instruction,
                edited_prompt=row.edited_prompt,
                image_name=name,
            )
        )
    return out
