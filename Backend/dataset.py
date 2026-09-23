import csv
from pathlib import Path

from Backend.schemas import Contractor

DATASET_PATH = Path(__file__).resolve().parents[1] / "hackathon dataset anonymized .csv"


def load_contractors(path: Path = DATASET_PATH) -> list[Contractor]:
    """Читаем CSV стандартной библиотекой; Pydantic проверяет типы данных."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        contractors = []
        for row in csv.DictReader(file):
            for field in ("categories", "event_formats", "languages", "busy_dates"):
                row[field] = [value.strip() for value in row[field].split("|") if value.strip()]
            row["max_hours"] = row["max_hours"] or None
            contractors.append(Contractor.model_validate(row))
    if len({person.id for person in contractors}) != len(contractors):
        raise ValueError("В CSV обнаружены повторяющиеся id")
    return contractors
