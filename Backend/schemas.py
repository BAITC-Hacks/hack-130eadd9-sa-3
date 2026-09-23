from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

# Известный период датасета, а не ограничение по сегодняшней дате.
CALENDAR_START = date(2026, 9, 23)
CALENDAR_END = date(2026, 12, 31)
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class SearchRequest(BaseModel):
    # Входной JSON от клиента. Неизвестные поля запрещены, пропущенные необязательны.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    categories: list[ShortText] = Field(default_factory=list, max_length=30)
    city: ShortText | None = None
    min_price_kzt: int | None = Field(default=None, ge=0)
    max_price_kzt: int | None = Field(default=None, ge=0)
    languages: list[ShortText] = Field(default_factory=list, max_length=20)
    event_format: ShortText | None = None
    event_date: date | None = None
    duration_hours: float | None = Field(default=None, gt=0, le=24)
    wishes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def validate_filters(self) -> "SearchRequest":
        # После проверки типов проверяем взаимосвязь полей и границы календаря.
        if (self.min_price_kzt is not None and self.max_price_kzt is not None
                and self.min_price_kzt > self.max_price_kzt):
            raise ValueError("Минимальная цена не может превышать максимальную")
        if self.event_date and not CALENDAR_START <= self.event_date <= CALENDAR_END:
            raise ValueError(f"Календарь доступен только с {CALENDAR_START} по {CALENDAR_END}")
        return self


class Contractor(BaseModel):
    # Один профиль CSV. Поля imputed отмечают значения, проставленные при подготовке.
    id: str
    anon_name: str
    categories: list[str]
    city: str
    city_imputed: bool
    synthetic: bool
    price_from_kzt: int = Field(ge=0)
    price_imputed: bool
    event_formats: list[str]
    languages: list[str]
    max_hours: float | None = Field(default=None, gt=0)
    busy_dates: list[date]
    description: str


class FilterResponse(BaseModel):
    # Ответ /api/contractors/filter; маршрут /api/filter возвращает только ID.
    total_candidates: int
    matched_candidates: int
    candidates: list[Contractor]


class AIChoice(BaseModel):
    # От ИИ принимаем только ID и объяснение; имя/цену берём из собственного CSV.
    model_config = ConfigDict(extra="forbid")
    contractor_id: str
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class AISelection(BaseModel):
    # Схема ответа ИИ. Чужие ID, повторы и точное количество проверяет parse_selection.
    model_config = ConfigDict(extra="forbid")
    recommendations: list[AIChoice] = Field(min_length=1, max_length=3)


class Recommendation(BaseModel):
    # Сервер объединяет проверенный профиль с объяснением модели.
    contractor: Contractor
    reason: str


class SearchResponse(BaseModel):
    # Итоговый HTTP-ответ /api/contractors/search, включая пустые recommendations.
    total_candidates: int
    matched_candidates: int
    recommendations: list[Recommendation]
