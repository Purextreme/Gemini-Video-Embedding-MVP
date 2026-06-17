import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from app.config import PROJECTS_DIR, PROJECTS_FILE, ensure_app_dirs


def utc_now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_projects() -> list[dict]:
    ensure_app_dirs()
    projects = _read_json(PROJECTS_FILE, [])
    return sorted(projects, key=lambda item: item.get("created_at", ""))


def save_projects(projects: list[dict]) -> None:
    ensure_app_dirs()
    _write_json(PROJECTS_FILE, projects)


def get_project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def get_project(project_id: str) -> dict | None:
    for project in list_projects():
        if project.get("project_id") == project_id:
            return project
    return None


def create_project(name: str, root_path: str, recursive: bool) -> dict:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("项目名称不能为空")
    if not root_path.strip():
        raise ValueError("素材目录路径不能为空")

    root = Path(root_path).expanduser().resolve()
    project_id = f"project_{uuid.uuid4().hex[:10]}"
    now = utc_now_iso()
    project = {
        "project_id": project_id,
        "name": clean_name,
        "root_path": str(root),
        "recursive": bool(recursive),
        "created_at": now,
        "last_indexed_at": None,
    }

    projects = list_projects()
    projects.append(project)
    save_projects(projects)

    project_dir = get_project_dir(project_id)
    (project_dir / "thumbnails").mkdir(parents=True, exist_ok=True)
    (project_dir / "previews").mkdir(parents=True, exist_ok=True)
    _write_json(project_dir / "project.json", project)
    return project


def update_project(project_id: str, **updates) -> dict:
    projects = list_projects()
    updated_project = None
    for project in projects:
        if project.get("project_id") == project_id:
            project.update(updates)
            updated_project = project
            break
    if updated_project is None:
        raise KeyError(f"Project not found: {project_id}")

    save_projects(projects)
    _write_json(get_project_dir(project_id) / "project.json", updated_project)
    return updated_project


def update_last_indexed_at(project_id: str) -> dict:
    return update_project(project_id, last_indexed_at=utc_now_iso())


def delete_project(project_id: str) -> None:
    projects = [p for p in list_projects() if p.get("project_id") != project_id]
    save_projects(projects)

    project_dir = get_project_dir(project_id)
    if project_dir.exists():
        shutil.rmtree(project_dir)


def root_exists(project: dict) -> bool:
    return Path(project["root_path"]).exists()
