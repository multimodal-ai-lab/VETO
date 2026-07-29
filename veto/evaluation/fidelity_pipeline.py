from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

import torch
from tqdm import tqdm

from veto.evaluation.fidelity_evaluator import FidelityEvaluator
from veto.evaluation.paths import EvalPaths
from veto.utils.cuda_memory import release_cuda_memory
from veto.utils.io import load_csv_dicts, mean_key, save_dicts_to_csv
from veto.utils.path_layout import protected_output_name

FIDELITY_FIELDS = ["idx", "mse", "psnr", "ssim", "lpips"]


def parse_fidelity_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "idx": int(row["idx"]),
        "mse": float(row["mse"]),
        "psnr": float(row["psnr"]),
        "ssim": float(row["ssim"]),
        "lpips": float(row["lpips"]),
    }


def run_fidelity_phase(
    paths: EvalPaths,
    *,
    dev: torch.device,
    image_size: int,
) -> List[Dict[str, Any]]:
    fidelity_rows_path = paths.res_dir / "fidelity.csv"
    fidelity_eval = FidelityEvaluator(device=dev, image_size=image_size)
    try:
        existing = load_csv_dicts(fidelity_rows_path)
        done: Set[int] = {int(r["idx"]) for r in existing if "idx" in r}
        rows: List[Dict[str, Any]] = [parse_fidelity_row(r) for r in existing]
        pending = [row for row in paths.prompt_rows if row.idx not in done]
        print(
            f"[evaluation] phase=fidelity existing={len(done)} pending={len(pending)}"
        )
        for row in tqdm(pending, total=len(pending), desc="[fidelity]"):
            orig_p = paths.u_dir / row.image_name
            prot_p = paths.p_dir / protected_output_name(row.image_name)
            f = fidelity_eval.evaluate(str(orig_p), str(prot_p))
            rows.append(
                {
                    "idx": row.idx,
                    "mse": f.mse,
                    "psnr": f.psnr,
                    "ssim": f.ssim,
                    "lpips": f.lpips,
                }
            )
        save_dicts_to_csv(
            sorted(rows, key=lambda r: int(r["idx"])),
            fidelity_rows_path,
            fieldnames=FIDELITY_FIELDS,
        )
        return rows
    finally:
        fidelity_eval.unload()
        release_cuda_memory()


def write_fidelity_summary(
    fidelity_rows: List[Dict[str, Any]],
    *,
    met_dir: Path,
    timestamp: str,
) -> Path:
    path = met_dir / "fidelity.csv"
    save_dicts_to_csv(
        [
            {
                "timestamp": timestamp,
                "n_images": len(fidelity_rows),
                "mse": mean_key(fidelity_rows, "mse"),
                "psnr": mean_key(fidelity_rows, "psnr"),
                "ssim": mean_key(fidelity_rows, "ssim"),
                "lpips": mean_key(fidelity_rows, "lpips"),
            }
        ],
        path,
        fieldnames=["timestamp", "n_images", "mse", "psnr", "ssim", "lpips"],
    )
    return path
