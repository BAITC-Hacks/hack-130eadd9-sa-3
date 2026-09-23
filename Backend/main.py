# HTTP-слой: только функции с @app.get/@app.post отвечают браузеру.
# Возвращаемый ими dict или Pydantic-модель FastAPI превращает в JSON.
# return во вспомогательных функциях передаёт данные обратно Python-коду, не браузеру.
import os
from pathlib import Path

# Run в IDE может запускать этот файл напрямую, вне контекста пакета.
# Используем существующий запускатель: он задаёт корень проекта для Uvicorn.
if __name__ == "__main__" and not __package__:
    import runpy

    runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="__main__")
    raise SystemExit(0)

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from Backend.ai_service import AIServiceError, recommend_contractors
from Backend.dataset import load_contractors
from Backend.filtering import filter_contractors
from Backend.schemas import CALENDAR_END, CALENDAR_START, Contractor, FilterResponse, SearchRequest, SearchResponse

# Загружаем настройки из корня проекта; ключ не передаётся во frontend.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(title="Подбор подрядчиков", version="0.1.0")
# Разрешённые адреса отдельного frontend-сервера. Страница на / работает
# с API на том же адресе, поэтому для неё межсайтовое разрешение не требуется.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500"
    ).split(",") if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/", response_class=FileResponse, include_in_schema=False)
def frontend_page() -> FileResponse:
    """Страница и /api/filter на одном адресе: относительный fetch работает без прокси."""
    return FileResponse(
        Path(__file__).resolve().parents[1] / "frontend" / "hackathon dataset preview.html"
    )


def get_contractors() -> list[Contractor]:
    # Depends(get_contractors) вызывает эту функцию и подставляет список в people.
    # CSV перечитывается при каждом таком запросе; общего изменяемого списка нет.
    try:
        return load_contractors()
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(503, "Не удалось загрузить датасет подрядчиков.") from error


@app.get("/health")
def health() -> dict[str, str]:
    # Это только проверка HTTP-сервера, не доступности CSV или ИИ.
    return {"status": "ok"}


@app.get("/api/filters")
def filter_options(people: list[Contractor] = Depends(get_contractors)) -> dict:
    """Значения для выпадающих списков фронтенда из текущего CSV."""
    return {
        "categories": sorted({item for person in people for item in person.categories}),
        "cities": sorted({person.city for person in people}),
        "languages": sorted({item for person in people for item in person.languages}),
        "event_formats": sorted({item for person in people for item in person.event_formats}),
        "calendar_start": CALENDAR_START, "calendar_end": CALENDAR_END,
    }


@app.post("/api/contractors/filter", response_model=FilterResponse)
def preview_filter(order: SearchRequest, people: list[Contractor] = Depends(get_contractors)) -> FilterResponse:
    # FastAPI сам читает JSON запроса в order и проверяет его по SearchRequest.
    # При неверных параметрах возвращает 422, не выполняя тело этой функции.
    candidates = filter_contractors(people, order)
    # Ответ API: полные профили и счётчики, без ИИ. Удобен для /docs и отладки.
    return FilterResponse(total_candidates=len(people), matched_candidates=len(candidates), candidates=candidates)


@app.post("/api/filter")
def frontend_filter(order: SearchRequest) -> dict[str, list[str]]:
    """Принимает JSON от collectFilters() во frontend; файл не загружается."""
    # Печатаем только после проверки JSON. Ошибка обработки не должна давать
    # ложное сообщение об успехе: вторую строку выводим после фильтрации.
    print("успешно получено", flush=True)
    people = get_contractors()
    candidates = filter_contractors(people, order)
    # Сообщение означает успешную обработку сервером, даже при пустом результате.
    # Оно не подтверждает, что браузер уже прочитал ответ или обновил карточки.
    print("успешно обработано", flush=True)
    # ВОТ ОТВЕТ FRONTEND: FastAPI отправит HTTP 200 и JSON вида
    # {"contractor_ids": ["HK-44733", "HK-88430"]}. При отсутствии людей — [].
    # Текущий JS проверяет response.ok, но пока не читает этот JSON и не рисует результат.
    return {"contractor_ids": [person.id for person in candidates]}


@app.post("/api/contractors/search", response_model=SearchResponse)
def search(order: SearchRequest, people: list[Contractor] = Depends(get_contractors)) -> SearchResponse:
    # Отдельный маршрут с ИИ: текущая форма /api/filter его НЕ вызывает.
    candidates = filter_contractors(people, order)
    try:
        recommendations = recommend_contractors(order, candidates) if candidates else []
    except AIServiceError as error:
        # Превращаем внутреннюю ошибку в HTTP-ответ {"detail": "..."}.
        raise HTTPException(error.status_code, str(error)) from error
    # Ответ клиенту этого маршрута: счётчики и до 3 карточек с reason.
    # response_model=SearchResponse проверяет и сериализует возвращаемую модель.
    return SearchResponse(
        total_candidates=len(people), matched_candidates=len(candidates), recommendations=recommendations,
    )
