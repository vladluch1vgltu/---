"""
REST API системы распознавания объектов на спутниковых изображениях.

Запуск:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate.metrics import ModelEvaluator
from inference.detector import SatelliteDetector
from utils.config import load_config
from utils.security import InputValidator
from utils.weights import resolve_trained_weights

app = FastAPI(
    title="Satellite Object Detection API",
    description="Интеллектуальная система распознавания объектов на спутниковых изображениях",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = ROOT / "configs" / "default.yaml"
config = load_config(CONFIG_PATH)
validator = InputValidator(config)
detector: SatelliteDetector | None = None
evaluator = ModelEvaluator(CONFIG_PATH)

UPLOAD_DIR = ROOT / "results" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

WEB_DIR = ROOT / "web"
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")


class HealthResponse(BaseModel):
    status: str
    version: str
    classes: list[str]


class DetectResponse(BaseModel):
    success: bool
    processing_time_sec: float
    num_detections: int
    detections: list[dict]
    result_image_url: str | None = None


def get_detector() -> SatelliteDetector:
    global detector
    if detector is None:
        detector = SatelliteDetector(config_path=CONFIG_PATH)
    return detector


@app.get("/")
async def root():
    return {
        "service": config["project"]["name"],
        "version": config["project"]["version"],
        "docs": "/docs",
        "ui": "/ui/index.html",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=config["project"]["version"],
        classes=config["classes"],
    )


@app.get("/classes")
async def list_classes():
    return {"classes": config["classes"], "count": len(config["classes"])}


@app.get("/metrics")
async def get_metrics():
    """Просмотр последнего отчёта о метриках модели."""
    report_path = ROOT / "results" / "evaluation_report.json"
    if not report_path.exists():
        raise HTTPException(404, "Отчёт о метриках не найден. Запустите evaluate.py")
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)


@app.post("/detect", response_model=DetectResponse)
async def detect_image(
    file: UploadFile = File(...),
    conf: float | None = None,
    use_tiling: bool | None = None,
):
    """Загрузка изображения и запуск распознавания."""
    content = await file.read()
    ok, msg = validator.validate_upload_bytes(content, file.filename or "image.jpg")
    if not ok:
        raise HTTPException(400, msg)

    suffix = Path(file.filename or "upload.jpg").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        det = get_detector()
        if conf is not None:
            det.det_cfg["conf_threshold"] = conf

        start = time.perf_counter()
        results = det.detect(source=tmp_path, output_dir=ROOT / "results", use_tiling=use_tiling)
        elapsed = time.perf_counter() - start

        if not results:
            raise HTTPException(500, "Ошибка обработки изображения")

        r = results[0]
        stem = Path(r["image_name"]).stem
        result_img = ROOT / "results" / f"{stem}_detected.jpg"

        return DetectResponse(
            success=True,
            processing_time_sec=round(elapsed, 3),
            num_detections=r["num_detections"],
            detections=r["detections"],
            result_image_url=f"/results/image/{stem}_detected.jpg" if result_img.exists() else None,
        )
    except Exception as e:
        raise HTTPException(500, f"Ошибка детекции: {e}") from e
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.get("/results/image/{filename}")
async def get_result_file(filename: str):
    """Скачивание результата (изображение, JSON, CSV)."""
    safe_name = Path(filename).name
    path = ROOT / "results" / safe_name
    if not path.exists():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(path)


@app.post("/evaluate")
async def run_evaluation(weights: str | None = None):
    """Запуск оценки модели (требует размеченный датасет)."""
    try:
        weights_path = resolve_trained_weights(weights, config["paths"]["models"])
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    try:
        report = evaluator.evaluate(weights=weights_path)
        return JSONResponse(report)
    except Exception as e:
        raise HTTPException(500, str(e)) from e
