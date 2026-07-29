import argparse
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from veto.configs.project_config import ProjectConfig
from veto.data.prompts_table import load_prompts_for_dataset
from veto.editing.backends.utils import list_edit_backend_keys
from veto.editing.factory import StandardImageEditingFactory
from veto.evaluation.clip_pipeline import run_clip_phase, write_clip_summary
from veto.evaluation.fidelity_pipeline import run_fidelity_phase, write_fidelity_summary
from veto.evaluation.paths import EvalPaths, assert_prompt_files_exist, build_eval_paths
from veto.evaluation.vqa_evaluator import VqaEvaluator
from veto.utils.cuda_memory import release_cuda_memory


def _run_edit_generation(
    paths: EvalPaths,
    *,
    edit_models: Sequence[str],
    dev: torch.device,
    image_size: int,
    backend_kwargs: Optional[Mapping[str, Mapping[str, object]]],
) -> None:
    kw: dict[str, dict[str, object]] = {
        m: dict((backend_kwargs or {}).get(m, {})) for m in edit_models
    }
    for m in edit_models:
        kw[m].setdefault("device", dev)
        kw[m].setdefault("image_size", image_size)
    factory = StandardImageEditingFactory(
        output_root=paths.out_root,
        run_id=paths.run_id,
        eval_variant=paths.eval_variant,
        prompt_rows=list(paths.prompt_rows),
        unprotected_source_dir=paths.u_dir,
        protected_source_dir=paths.p_dir,
        dataset_name=paths.dataset_name,
        backend_kwargs=kw,
    )
    print("[evaluation] phase=generation")
    factory.run(list(edit_models))
    release_cuda_memory()


def run_vqa(
    paths: EvalPaths,
    *,
    edit_models: Sequence[str],
    vqa_models: Optional[Sequence[str]],
) -> None:
    if not vqa_models:
        return
    evaluator = VqaEvaluator(
        output_root=paths.out_root,
        dataset_name=paths.dataset_name,
        run_id=paths.run_id,
        eval_variant=paths.eval_variant,
        original_images_dir=paths.u_dir,
    )
    evaluator.evaluate(
        prompt_rows=paths.prompt_rows,
        edit_models=list(edit_models),
        vqa_models=list(vqa_models),
    )


def _print_output_paths(
    paths: EvalPaths,
    *,
    fidelity_csv: Path,
    clip_csv_dir: Path,
    fidelity_summary: Path,
    clip_summary: Path,
    image_size: int,
    vqa_models: Optional[Sequence[str]],
) -> None:
    print("[evaluation] output:")
    print(f"  dataset      {paths.dataset_name}  (image_size={image_size})")
    print(f"  results_dir  {paths.res_dir}")
    print(f"  metrics_dir  {paths.met_dir}")
    print(f"  fidelity per-row  {fidelity_csv}")
    print(f"  clip per-row dir  {clip_csv_dir}")
    print(f"  fidelity summary  {fidelity_summary}")
    print(f"  clip summary      {clip_summary}")
    if vqa_models:
        print(f"  vqa summary       {paths.met_dir / 'vqa_summary.csv'}")


def run_evaluation(
    *,
    dataset_dir: Path,
    edit_models: Sequence[str],
    run_id: str,
    output_root: Optional[Path] = None,
    protected_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    backend_kwargs: Optional[Mapping[str, Mapping[str, object]]] = None,
    image_size: int = 512,
    eval_variant: str = "base",
    vqa_models: Optional[Sequence[str]] = None,
    force: bool = False,
) -> None:
    project = ProjectConfig()
    out_root = Path(output_root) if output_root is not None else Path(project.output_folder)
    ds_dir = Path(dataset_dir)
    assert_prompt_files_exist(ds_dir)

    prompt_rows = load_prompts_for_dataset(ds_dir)
    if not prompt_rows:
        raise ValueError(f"No prompt rows in {ds_dir}")

    paths = build_eval_paths(
        out_root=out_root,
        ds_dir=ds_dir,
        run_id=run_id,
        eval_variant=eval_variant,
        prompt_rows=prompt_rows,
        protected_dir=protected_dir,
    )
    if force:
        import shutil
        from veto.utils.path_layout import edited_variant_dir
        print(f"[evaluation] force=True: cleaning up existing evaluation folders for run_id={paths.run_id}")
        shutil.rmtree(paths.res_dir, ignore_errors=True)
        shutil.rmtree(paths.met_dir, ignore_errors=True)
        for m in edit_models:
            out_p = edited_variant_dir(
                out_root,
                paths.dataset_name,
                paths.run_id,
                m,
                eval_variant,
            )
            shutil.rmtree(out_p, ignore_errors=True)

    if not paths.u_dir.is_dir():
        raise FileNotFoundError(f"Unprotected images not found: {paths.u_dir}")
    if not paths.p_dir.is_dir():
        raise FileNotFoundError(f"Protected images not found: {paths.p_dir}")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[evaluation] run_id={paths.run_id} dataset={paths.dataset_name} "
        f"n_prompts={len(prompt_rows)} edit_models={list(edit_models)} "
        f"image_size={image_size}"
    )

    _run_edit_generation(
        paths,
        edit_models=edit_models,
        dev=dev,
        image_size=image_size,
        backend_kwargs=backend_kwargs,
    )

    paths.res_dir.mkdir(parents=True, exist_ok=True)
    paths.met_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fidelity_rows = run_fidelity_phase(paths, dev=dev, image_size=image_size)
    clip_rows = run_clip_phase(
        paths, edit_models=edit_models, dev=dev, image_size=image_size
    )

    fidelity_summary = write_fidelity_summary(
        fidelity_rows, met_dir=paths.met_dir, timestamp=timestamp
    )
    clip_summary = write_clip_summary(
        clip_rows, met_dir=paths.met_dir, timestamp=timestamp
    )

    release_cuda_memory()
    run_vqa(paths, edit_models=edit_models, vqa_models=vqa_models)

    _print_output_paths(
        paths,
        fidelity_csv=paths.res_dir / "fidelity.csv",
        clip_csv_dir=paths.res_dir / "clip",
        fidelity_summary=fidelity_summary,
        clip_summary=clip_summary,
        image_size=image_size,
        vqa_models=vqa_models,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate edits then fidelity + CLIP metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Protection run identifier under outputs/{images,results,metrics}/{dataset}/.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset root containing images/ and prompts.csv.",
    )
    parser.add_argument(
        "--edit-models",
        nargs="+",
        default=list_edit_backend_keys(),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: ProjectConfig.output_folder",
    )
    parser.add_argument(
        "--protected-dir",
        type=Path,
        default=None,
        help="Protected images directory",
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--eval-variant",
        default="base",
        help="Evaluation namespace under .../evaluations/ (default: base).",
    )
    parser.add_argument(
        "--vqa-models",
        nargs="*",
        default=("qwen", "gemma3", "llava-onevision", "gemini", "gpt"),
        metavar="NAME",
        help="Optional VQA backends (e.g. gemini gpt qwen llava gemma3).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force regeneration of edits and metrics.",
    )
    args = parser.parse_args()

    dev = torch.device(args.device) if args.device else None
    run_evaluation(
        dataset_dir=args.dataset_dir,
        edit_models=args.edit_models,
        run_id=args.run_id,
        output_root=args.output_root,
        protected_dir=args.protected_dir,
        device=dev,
        image_size=args.image_size,
        eval_variant=args.eval_variant,
        vqa_models=args.vqa_models,
        force=args.force,
    )


if __name__ == "__main__":
    main()
