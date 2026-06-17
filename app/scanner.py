from pathlib import Path

from app.config import SUPPORTED_EXTENSIONS, is_image, is_video


def asset_id_for_path(path: Path) -> str:
    import hashlib

    normalized = str(path.resolve()).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def file_type_for_path(path: Path) -> str | None:
    if is_image(path):
        return "image"
    if is_video(path):
        return "video"
    return None


def scan_project_files(project: dict) -> list[dict]:
    root = Path(project["root_path"])
    if not root.exists():
        raise FileNotFoundError(f"素材目录不存在: {root}")

    iterator = root.rglob("*") if project.get("recursive", True) else root.glob("*")
    files = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        stat = path.stat()
        files.append(
            {
                "id": asset_id_for_path(path),
                "path": str(path.resolve()),
                "relative_path": str(path.resolve().relative_to(root.resolve())),
                "file_type": file_type_for_path(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return sorted(files, key=lambda item: item["relative_path"].lower())


def unchanged(scanned: dict, existing: dict | None) -> bool:
    if not existing:
        return False
    return (
        existing.get("path") == scanned["path"]
        and existing.get("size") == scanned["size"]
        and existing.get("mtime") == scanned["mtime"]
        and existing.get("status") == "indexed"
        and existing.get("vector_index") is not None
    )

