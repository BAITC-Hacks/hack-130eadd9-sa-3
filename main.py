"""Запуск сайта и API: python main.py из корня проекта или Run в PyCharm."""

from pathlib import Path

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "Backend.main:app",
        host="127.0.0.1",
        port=8000,
        app_dir=str(Path(__file__).resolve().parent),
    )
