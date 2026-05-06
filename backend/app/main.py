"""
Entry point de la aplicación FastAPI.

- Configura CORS (orígenes desde variable de entorno).
- Sirve el frontend estático en la raíz (/).
- Expone la API REST bajo /api.
- Configura logging estructurado.
"""
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.config import settings

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mesa de Ayuda IA — Banco",
    description="Prototipo de mesa de ayuda para Desarrollo TI con agentes especializados.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ─── API Routes ───────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api")

# ─── Frontend estático ────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
FRONTEND_INDEX = os.path.join(FRONTEND_DIR, "index.html")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(FRONTEND_INDEX)

logger.info("Aplicación iniciada. Docs disponibles en /api/docs")
