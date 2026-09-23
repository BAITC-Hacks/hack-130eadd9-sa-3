# Чистая Python-фильтрация: нет HTTP, файлов, ИИ или отправки в браузер.
from Backend.schemas import Contractor, SearchRequest


def normalize(value: str) -> str:
    # Игнорируем регистр и внешние пробелы, но не угадываем синонимы/опечатки.
    return value.strip().casefold()


def matches_filters(person: Contractor, order: SearchRequest) -> bool:
    # Проверяем одного человека. Первое несовпадение исключает его (return False).
    # & — пересечение множеств: достаточно одной из выбранных категорий.
    if order.categories and not (
        {normalize(item) for item in order.categories}
        & {normalize(item) for item in person.categories}
    ):
        return False
    # Разные поля связаны условием И: должны пройти все заданные ограничения.
    if order.city and normalize(order.city) != normalize(person.city):
        return False
    # Границы включаются. Сравниваем цену «от», не стоимость часа или всего заказа.
    if order.min_price_kzt is not None and person.price_from_kzt < order.min_price_kzt:
        return False
    if order.max_price_kzt is not None and person.price_from_kzt > order.max_price_kzt:
        return False
    # issubset требует ВСЕ выбранные языки. Пустой набор подходит каждому.
    if not {normalize(item) for item in order.languages}.issubset(
        {normalize(item) for item in person.languages}
    ):
        return False
    if order.event_format and normalize(order.event_format) not in {
        normalize(item) for item in person.event_formats
    }:
        return False
    # Границы известного календаря проверены в SearchRequest до этой функции.
    if order.event_date and order.event_date in person.busy_dates:
        return False
    # Пустое max_hours не подтверждает соответствие требованию по длительности.
    if order.duration_hours is not None and (
        person.max_hours is None or person.max_hours < order.duration_hours
    ):
        return False
    # Все заданные условия выполнены. wishes здесь намеренно не проверяются.
    return True


def filter_contractors(people: list[Contractor], order: SearchRequest) -> list[Contractor]:
    """Оставляем всех подходящих кандидатов без рейтинга и предварительного TOP-3."""
    # Возвращаем список обработчику в main.py, сохраняя исходный порядок CSV.
    return [person for person in people if matches_filters(person, order)]
