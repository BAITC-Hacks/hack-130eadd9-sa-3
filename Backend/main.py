import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.ai_service import AIServiceError, recommend_contractors
from backend.dataset import load_contractors
from backend.filtering import filter_contractors
from backend.schemas import CALENDAR_END, CALENDAR_START, Contractor, FilterResponse, SearchRequest, SearchResponse

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(title="Подбор подрядчиков", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500"
    ).split(",") if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def get_contractors() -> list[Contractor]:
    try:
        return load_contractors()
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(503, "Не удалось загрузить датасет подрядчиков.") from error


@app.get("/health")
def health() -> dict[str, str]:
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
    candidates = filter_contractors(people, order)
    return FilterResponse(total_candidates=len(people), matched_candidates=len(candidates), candidates=candidates)


@app.post("/api/contractors/search", response_model=SearchResponse)
def search(order: SearchRequest, people: list[Contractor] = Depends(get_contractors)) -> SearchResponse:
    candidates = filter_contractors(people, order)
    try:
        recommendations = recommend_contractors(order, candidates) if candidates else []
    except AIServiceError as error:
        raise HTTPException(error.status_code, str(error)) from error
    return SearchResponse(
        total_candidates=len(people), matched_candidates=len(candidates), recommendations=recommendations,
    )
