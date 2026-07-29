from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set

from tqdm import tqdm

from veto.data.prompts_table import PromptRow
from veto.evaluation.vqa.backends.registry import build_vqa_backend
from veto.evaluation.vqa.parser import parse_yes_no
from veto.utils.io import load_csv_dicts, save_dicts_to_csv
from veto.utils.path_layout import edited_variant_dir, evaluation_metrics_dir, evaluation_results_dir

_PER_Q_FIELDS = [
    "idx",
    "edit_model",
    "raw_answer",
    "parsed_answer",
]

_VQA_STATIC_TEMPLATE = """You are evaluating whether an image edit was successful.

You are given:
- Original image (before edit)
- Edited image (after edit)
- Edit instruction: "{editing_instruction}"
- Original scene description: "{base_prompt}"
- Intended edited description: "{edited_prompt}"

Task:
Decide if the edited image is a successful execution of the edit instruction.

Definition of SUCCESS (must satisfy ALL):
1) The requested edit is clearly present and correct.
2) Only requested changes were made; no important unrequested changes.
3) If a person/object identity is present, it must remain the same unless instruction explicitly asks to change identity.
4) No major visual defects/artifacts (blur, smearing, corruption, duplication, unnatural distortions) unless explicitly requested.

Important strict rules:
- If any condition fails, answer NO.
- Partial fulfillment is NO.
- Wrong person / wrong object instance is NO, even if the requested attribute appears.
- Unrequested blur/artifacts/corruption is NO.

Output format (exactly):
ANSWER: YES or NO
"""


class VqaEvaluator:
    def __init__(
        self,
        *,
        output_root: Path,
        dataset_name: str,
        run_id: str,
        eval_variant: str,
        original_images_dir: Path,
    ) -> None:
        self.output_root = output_root
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.eval_variant = eval_variant
        self.original_images_dir = original_images_dir

    def evaluate(
        self,
        *,
        prompt_rows: Sequence[PromptRow],
        edit_models: Sequence[str],
        vqa_models: Sequence[str],
    ) -> None:
        if not vqa_models:
            return

        res_dir = evaluation_results_dir(
            self.output_root, self.dataset_name, self.run_id, self.eval_variant
        )
        met_dir = evaluation_metrics_dir(
            self.output_root, self.dataset_name, self.run_id, self.eval_variant
        )
        res_dir.mkdir(parents=True, exist_ok=True)
        met_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        slugs_order: List[str] = []
        scores_by_slug_edit: Dict[tuple[str, str], float] = {}

        for vm in vqa_models:
            backend = build_vqa_backend(vm)
            slug = backend.slug
            slugs_order.append(slug)
            vqa_dir = res_dir / "vqa" / slug
            vqa_dir.mkdir(parents=True, exist_ok=True)
            per_q_path = vqa_dir / "per_question.csv"

            existing = load_csv_dicts(per_q_path)
            merged = [self._normalize_per_question_row(r) for r in existing]
            done = self._done_keys(merged)

            for model_key in edit_models:
                prot_root = edited_variant_dir(
                    self.output_root,
                    self.dataset_name,
                    self.run_id,
                    model_key,
                    self.eval_variant,
                )
                for row in tqdm(
                    prompt_rows,
                    desc=f"[vqa:{slug}:{model_key}]",
                    leave=False,
                ):
                    edited_image_path = prot_root / row.image_name
                    if not edited_image_path.is_file():
                        raise FileNotFoundError(
                            f"VQA missing edited protected image: {edited_image_path}"
                        )
                    original_image_path = self.original_images_dir / row.image_name
                    key = (row.idx, model_key)
                    if key in done:
                        continue
                    question = _build_static_vqa_question(row)
                    raw = backend.ask(
                        [original_image_path, edited_image_path], question
                    )
                    parsed = parse_yes_no(raw)
                    merged.append(
                        {
                            "idx": row.idx,
                            "edit_model": model_key,
                            "raw_answer": raw.replace("\n", " ").strip()[:500],
                            "parsed_answer": "yes" if parsed else "no",
                        }
                    )
                    done.add(key)

            merged.sort(key=lambda r: (str(r["edit_model"]), int(r["idx"])))
            save_dicts_to_csv(merged, per_q_path, fieldnames=_PER_Q_FIELDS)

            for mk in edit_models:
                sub = [r for r in merged if str(r.get("edit_model")) == mk]
                scores_by_slug_edit[(slug, mk)] = self._scores_from_rows(sub)
            backend.unload()

        summary_rows: List[Dict[str, Any]] = []
        for mk in edit_models:
            row: Dict[str, Any] = {
                "timestamp": ts,
                "dataset": self.dataset_name,
                "run_id": self.run_id,
                "edit_model": mk,
            }
            yes_rate_avgs: List[float] = []
            for slug in slugs_order:
                yes_rate = scores_by_slug_edit[(slug, mk)]
                row[f"vqa_{slug}_yes_rate"] = yes_rate
                yes_rate_avgs.append(yes_rate)
            row["vqa_all_models_yes_rate"] = (
                float(sum(yes_rate_avgs) / len(yes_rate_avgs)) if yes_rate_avgs else 0.0
            )
            summary_rows.append(row)

        fieldnames = (
            ["timestamp", "dataset", "run_id", "edit_model"]
            + [f"vqa_{s}_yes_rate" for s in slugs_order]
            + ["vqa_all_models_yes_rate"]
        )
        save_dicts_to_csv(summary_rows, met_dir / "vqa_summary.csv", fieldnames=fieldnames)
        print(f"[evaluation] vqa summary -> {met_dir / 'vqa_summary.csv'}")

    @staticmethod
    def _normalize_per_question_row(r: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(r)

    @staticmethod
    def _done_keys(rows: List[Mapping[str, Any]]) -> Set[tuple[int, str]]:
        out: Set[tuple[int, str]] = set()
        for r in rows:
            if "idx" in r and "edit_model" in r:
                out.add((int(r["idx"]), str(r["edit_model"])))
        return out

    @staticmethod
    def _scores_from_rows(
        rows: Sequence[Mapping[str, Any]],
    ) -> float:
        if not rows:
            raise ValueError("No VQA rows to aggregate")
        yes_values: List[float] = []
        for r in rows:
            parsed = str(r.get("parsed_answer", "")).strip().lower()
            yes_values.append(1.0 if parsed == "yes" else 0.0)
        return float(sum(yes_values) / len(yes_values)) if yes_values else 0.0


def _build_static_vqa_question(row: PromptRow) -> str:
    return _VQA_STATIC_TEMPLATE.format(
        editing_instruction=row.editing_instruction,
        base_prompt=row.original_prompt,
        edited_prompt=row.edited_prompt,
    )
