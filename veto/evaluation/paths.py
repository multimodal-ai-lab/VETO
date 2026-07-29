from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from veto.data.prompts_table import PromptRow
from veto.utils.path_layout import (
    derive_dataset_name,
    evaluation_metrics_dir,
    evaluation_results_dir,
    protected_images_dir,
    resolve_dataset_images_dir,
)


@dataclass(frozen=True)
class EvalPaths:
    out_root: Path
    dataset_name: str
    run_id: str
    eval_variant: str
    ds_dir: Path
    u_dir: Path
    p_dir: Path
    res_dir: Path
    met_dir: Path
    prompt_rows: Sequence[PromptRow]


def assert_prompt_files_exist(ds_dir: Path) -> None:
    if not (ds_dir / "prompts.csv").is_file():
        raise FileNotFoundError(f"Dataset must contain prompts.csv: {ds_dir}")


def build_eval_paths(
    *,
    out_root: Path,
    ds_dir: Path,
    run_id: str,
    eval_variant: str,
    prompt_rows: Sequence[PromptRow],
    protected_dir: Optional[Path],
) -> EvalPaths:
    u_dir = resolve_dataset_images_dir(ds_dir)
    dataset_name = derive_dataset_name(ds_dir)
    p_dir = (
        Path(protected_dir)
        if protected_dir is not None
        else protected_images_dir(out_root, dataset_name, run_id)
    )
    res_dir = evaluation_results_dir(out_root, dataset_name, run_id, eval_variant)
    met_dir = evaluation_metrics_dir(out_root, dataset_name, run_id, eval_variant)
    return EvalPaths(
        out_root=out_root,
        dataset_name=dataset_name,
        run_id=run_id,
        eval_variant=eval_variant,
        ds_dir=ds_dir,
        u_dir=u_dir,
        p_dir=p_dir,
        res_dir=res_dir,
        met_dir=met_dir,
        prompt_rows=prompt_rows,
    )
