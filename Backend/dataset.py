import csv
from pathlib import Path

from Backend.schemas import Contractor

# Путь относительно этого модуля, а не текущей папки запуска терминала.
DATASET_PATH = Path(__file__).resolve().parents[1] / "hackathon dataset anonymized .csv"


def load_contractors(path: Path = DATASET_PATH) -> list[Contractor]:
    """Читаем CSV стандартной библиотекой; Pydantic проверяет типы данных."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        # DictReader использует заголовки CSV как ключи каждой строки.
        contractors = []
        for row in csv.DictReader(file):
            for field in ("categories", "event_formats", "languages", "busy_dates"):
                # В CSV списки записаны через |; в Python нужны настоящие list.
                row[field] = [value.strip() for value in row[field].split("|") if value.strip()]
            # Пустая длительность — неизвестна/не применима, а не ноль часов.
            row["max_hours"] = row["max_hours"] or None
            # Модель превращает строки чисел, bool и дат в соответствующие типы.
            contractors.append(Contractor.model_validate(row))
    # Уникальность нужна для однозначного сопоставления ответа ИИ и карточек.
    if len({person.id for person in contractors}) != len(contractors):
        raise ValueError("В CSV обнаружены повторяющиеся id")
    return contractors
