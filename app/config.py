import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DATA_DIR = ROOT_DIR / "app_data"
PROJECTS_DIR = APP_DATA_DIR / "projects"
PROJECTS_FILE = APP_DATA_DIR / "projects.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"


def load_settings() -> dict:
    load_dotenv(override=True)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")

    if http_proxy:
        os.environ["http_proxy"] = http_proxy
    if https_proxy:
        os.environ["https_proxy"] = https_proxy

    return {
        "api_key": api_key,
        "model_name": model_name,
        "http_proxy": http_proxy,
        "https_proxy": https_proxy,
        "app_data_dir": APP_DATA_DIR,
    }


def ensure_app_dirs() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS

