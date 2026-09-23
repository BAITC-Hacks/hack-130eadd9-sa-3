from backend.schemas import Contractor, SearchRequest


def normalize(value: str) -> str:
    return value.strip().casefold()


def matches_filters(person: Contractor, order: SearchRequest) -> bool:
    if order.categories and not (
        {normalize(item) for item in order.categories}
        & {normalize(item) for item in person.categories}
    ):
        return False
    if order.city and normalize(order.city) != normalize(person.city):
        return False
    if order.min_price_kzt is not None and person.price_from_kzt < order.min_price_kzt:
        return False
    if order.max_price_kzt is not None and person.price_from_kzt > order.max_price_kzt:
        return False
    if not {normalize(item) for item in order.languages}.issubset(
        {normalize(item) for item in person.languages}
    ):
        return False
    if order.event_format and normalize(order.event_format) not in {
        normalize(item) for item in person.event_formats
    }:
        return False
    if order.event_date and order.event_date in person.busy_dates:
        return False
    # Пустое max_hours не подтверждает соответствие требованию по длительности.
    if order.duration_hours is not None and (
        person.max_hours is None or person.max_hours < order.duration_hours
    ):
        return False
    return True


def filter_contractors(people: list[Contractor], order: SearchRequest) -> list[Contractor]:
    """Оставляем всех подходящих кандидатов без рейтинга и предварительного TOP-3."""
    return [person for person in people if matches_filters(person, order)]
