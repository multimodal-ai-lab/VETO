from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set

import torch
from tqdm import tqdm

from veto.evaluation.clip_evaluator import CLIPEvaluator
from veto.evaluation.paths import EvalPaths
from veto.utils.cuda_memory import release_cuda_memory
from veto.utils.io import load_csv_dicts, mean_key, save_dicts_to_csv
from veto.utils.path_layout import edited_unprotected_dir, edited_variant_dir

CLIP_ROW_FIELDS = [
    "idx",
    "edit_model",
    "clip_i_unprotected",
    "clip_i_protected",
    "clip_t_unprotected",
    "clip_t_protected",
    "clip_dir_unprotected",
    "clip_dir_protected",
]
CLIP_METRIC_FIELDS = [
    "timestamp",
    "edit_model",
    "n_images",
    "clip_i_unprotected",
    "clip_i_protected",
    "clip_t_unprotected",
    "clip_t_protected",
    "clip_dir_unprotected",
    "clip_dir_protected",
]


def parse_clip_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "idx": int(row["idx"]),
        "edit_model": str(row["edit_model"]),
        "clip_i_unprotected": float(row["clip_i_unprotected"]),
        "clip_i_protected": float(row["clip_i_protected"]),
        "clip_t_unprotected": float(row["clip_t_unprotected"]),
        "clip_t_protected": float(row["clip_t_protected"]),
        "clip_dir_unprotected": float(row["clip_dir_unprotected"]),
        "clip_dir_protected": float(row["clip_dir_protected"]),
    }


def run_clip_phase(
    paths: EvalPaths,
    *,
    edit_models: Sequence[str],
    dev: torch.device,
    image_size: int,
) -> List[Dict[str, Any]]:
    clip_rows_dir = paths.res_dir / "clip"
    clip_rows_dir.mkdir(parents=True, exist_ok=True)
    clip_eval = CLIPEvaluator(device=dev, image_size=image_size)
    try:
        all_rows: List[Dict[str, Any]] = []
        print("[evaluation] phase=clip")
        for model_key in edit_models:
            model_path = clip_rows_dir / f"{model_key}.csv"
            existing = load_csv_dicts(model_path)
            done: Set[int] = {int(r["idx"]) for r in existing if "idx" in r}
            model_rows: List[Dict[str, Any]] = [parse_clip_row(r) for r in existing]
            unprot_edits = edited_unprotected_dir(paths.out_root, paths.dataset_name, model_key)
            prot_edits = edited_variant_dir(
                paths.out_root,
                paths.dataset_name,
                paths.run_id,
                model_key,
                paths.eval_variant,
            )
            pending = [row for row in paths.prompt_rows if row.idx not in done]
            print(
                f"[evaluation] clip model={model_key} "
                f"existing={len(done)} pending={len(pending)}"
            )
            for row in tqdm(
                pending,
                total=len(pending),
                desc=f"[clip:{model_key}]",
                leave=False,
            ):
                u_edited = unprot_edits / row.image_name
                p_edited = prot_edits / row.image_name
                if not u_edited.is_file() or not p_edited.is_file():
                    raise FileNotFoundError(
                        f"Edited outputs missing for model {model_key!r}, idx {row.idx}: "
                        f"{u_edited} / {p_edited}"
                    )
                scores = clip_eval.evaluate(
                    str(paths.u_dir / row.image_name),
                    str(u_edited),
                    str(p_edited),
                    row.original_prompt,
                    row.edited_prompt,
                )
                model_rows.append(
                    {
                        "idx": row.idx,
                        "edit_model": model_key,
                        "clip_i_unprotected": scores.clip_i_unprotected,
                        "clip_i_protected": scores.clip_i_protected,
                        "clip_t_unprotected": scores.clip_t_unprotected,
                        "clip_t_protected": scores.clip_t_protected,
                        "clip_dir_unprotected": scores.clip_dir_unprotected,
                        "clip_dir_protected": scores.clip_dir_protected,
                    }
                )
            sorted_rows = sorted(model_rows, key=lambda r: int(r["idx"]))
            save_dicts_to_csv(sorted_rows, model_path, fieldnames=CLIP_ROW_FIELDS)
            all_rows.extend(sorted_rows)
        return all_rows
    finally:
        clip_eval.unload()
        release_cuda_memory()


def write_clip_summary(
    clip_rows: List[Dict[str, Any]],
    *,
    met_dir: Path,
    timestamp: str,
) -> Path:
    path = met_dir / "clip.csv"
    aggregate: List[Dict[str, Any]] = []
    for model_key in sorted({str(r["edit_model"]) for r in clip_rows}):
        mrows = [r for r in clip_rows if str(r.get("edit_model")) == model_key]
        if not mrows:
            continue
        aggregate.append(
            {
                "timestamp": timestamp,
                "edit_model": model_key,
                "n_images": len(mrows),
                "clip_i_unprotected": mean_key(mrows, "clip_i_unprotected"),
                "clip_i_protected": mean_key(mrows, "clip_i_protected"),
                "clip_t_unprotected": mean_key(mrows, "clip_t_unprotected"),
                "clip_t_protected": mean_key(mrows, "clip_t_protected"),
                "clip_dir_unprotected": mean_key(mrows, "clip_dir_unprotected"),
                "clip_dir_protected": mean_key(mrows, "clip_dir_protected"),
            }
        )
    save_dicts_to_csv(aggregate, path, fieldnames=CLIP_METRIC_FIELDS)
    return path
