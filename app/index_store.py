import json
from pathlib import Path
from typing import Callable

import numpy as np

from app.embedder import GeminiEmbedder
from app.media_preprocess import prepare_asset_for_embedding
from app.project_store import get_project_dir, update_last_indexed_at, update_project, utc_now_iso
from app.scanner import scan_project_files, unchanged


ProgressCallback = Callable[[dict], None]


def index_path(project_id: str) -> Path:
    return get_project_dir(project_id) / "index.json"


def vectors_path(project_id: str) -> Path:
    return get_project_dir(project_id) / "vectors.npy"


def load_index(project_id: str) -> list[dict]:
    path = index_path(project_id)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_index(project_id: str, records: list[dict]) -> None:
    path = index_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_vectors(project_id: str) -> np.ndarray:
    path = vectors_path(project_id)
    if not path.exists():
        return np.empty((0, 0), dtype=np.float32)
    return np.load(path).astype(np.float32)


def save_vectors(project_id: str, vectors: list[np.ndarray]) -> None:
    path = vectors_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if vectors:
        array = np.vstack(vectors).astype(np.float32)
    else:
        array = np.empty((0, 0), dtype=np.float32)
    np.save(path, array)


def reset_project_index(project_id: str) -> None:
    project_dir = get_project_dir(project_id)
    for name in ("index.json", "vectors.npy"):
        path = project_dir / name
        if path.exists():
            path.unlink()
    for folder in ("thumbnails", "previews"):
        target = project_dir / folder
        if target.exists():
            import shutil

            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    update_project(project_id, last_indexed_at=None)


def project_stats(project: dict) -> dict:
    project_id = project["project_id"]
    records = load_index(project_id)
    total = len(records)
    indexed = sum(1 for item in records if item.get("status") == "indexed")
    failed = sum(1 for item in records if item.get("status") == "failed")
    return {
        "total": total,
        "indexed": indexed,
        "failed": failed,
        "last_indexed_at": project.get("last_indexed_at"),
    }


def index_project(project: dict, progress_callback: ProgressCallback | None = None) -> dict:
    project_id = project["project_id"]
    project_dir = get_project_dir(project_id)
    (project_dir / "thumbnails").mkdir(parents=True, exist_ok=True)
    (project_dir / "previews").mkdir(parents=True, exist_ok=True)

    scanned_files = scan_project_files(project)
    existing_records = load_index(project_id)
    old_vectors = load_vectors(project_id)
    existing_by_path = {item.get("path"): item for item in existing_records}

    embedder = GeminiEmbedder()
    new_records: list[dict] = []
    new_vectors: list[np.ndarray] = []
    success = 0
    failed = 0
    skipped = 0
    total = len(scanned_files)

    for completed, scanned in enumerate(scanned_files, start=1):
        current_path = scanned["path"]
        existing = existing_by_path.get(current_path)
        now = utc_now_iso()

        if progress_callback:
            progress_callback(
                {
                    "current_file": scanned["relative_path"],
                    "completed": completed - 1,
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "skipped": skipped,
                    "status": "processing",
                }
            )

        if unchanged(scanned, existing):
            old_index = existing["vector_index"]
            if old_vectors.size and old_index < len(old_vectors):
                record = dict(existing)
                record["vector_index"] = len(new_vectors)
                new_vectors.append(old_vectors[old_index])
                new_records.append(record)
                success += 1
                skipped += 1
                continue

        record = {
            **scanned,
            "thumb_path": None,
            "preview_path": None,
            "vector_index": None,
            "status": "failed",
            "error": None,
            "created_at": existing.get("created_at") if existing else now,
            "updated_at": now,
        }

        try:
            thumb_path, preview_path, media_bytes, mime_type = prepare_asset_for_embedding(scanned, project_dir)
            vector = embedder.embed_media_bytes(media_bytes, mime_type)
            record.update(
                {
                    "thumb_path": thumb_path,
                    "preview_path": preview_path,
                    "vector_index": len(new_vectors),
                    "status": "indexed",
                    "error": None,
                    "updated_at": utc_now_iso(),
                }
            )
            new_vectors.append(vector)
            success += 1
        except Exception as exc:
            record["error"] = str(exc)
            failed += 1

        new_records.append(record)
        save_index(project_id, new_records)
        save_vectors(project_id, new_vectors)

        if progress_callback:
            progress_callback(
                {
                    "current_file": scanned["relative_path"],
                    "completed": completed,
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "skipped": skipped,
                    "status": "processing",
                }
            )

    save_index(project_id, new_records)
    save_vectors(project_id, new_vectors)
    update_last_indexed_at(project_id)

    result = {
        "total": total,
        "success": success,
        "failed": failed,
        "skipped": skipped,
    }
    if progress_callback:
        progress_callback({**result, "current_file": None, "completed": total, "status": "done"})
    return result
