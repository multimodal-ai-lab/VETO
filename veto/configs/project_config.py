from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    project_name: str = "veto"
    local_data_folder: str = str(Path(__file__).resolve().parents[2] / "data")
    output_folder: str = "outputs"
