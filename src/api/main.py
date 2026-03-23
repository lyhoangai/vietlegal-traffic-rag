"""FastAPI application entrypoint."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router

load_dotenv()
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="VietLegal Traffic RAG",
    description="Scoped Vietnamese traffic-law RAG app with session memory and official-source verification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the chat UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.middleware("http")
async def ensure_utf8_json(request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json") and "charset=" not in content_type.lower():
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "service": "VietLegal Traffic RAG", "version": "1.0.0"}
