#!/usr/bin/env python3
"""Запуск REST API и веб-интерфейса."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
