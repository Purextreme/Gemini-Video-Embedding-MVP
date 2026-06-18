from pathlib import Path

import streamlit as st

from app.explorer import reveal_in_explorer
from app.index_store import index_project, project_stats, reset_project_index
from app.project_store import create_project, delete_project, list_projects, root_exists
from app.searcher import search_assets


st.set_page_config(page_title="本地素材语义搜索", layout="wide")


def select_folder_dialog(initial_path: str = "") -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("当前环境无法打开文件夹选择对话框") from exc

    initial_dir = Path(initial_path).expanduser() if initial_path.strip() else Path.home()
    if initial_dir.is_file():
        initial_dir = initial_dir.parent

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        options = {"title": "选择素材目录"}
        if initial_dir.exists():
            options["initialdir"] = str(initial_dir)
        selected = filedialog.askdirectory(**options)
        return selected or None
    finally:
        root.destroy()


def set_project_name_from_root_path() -> None:
    root_path = st.session_state.get("create_root_path", "").strip()
    if not root_path or not st.session_state.get("create_project_name_auto", True):
        return

    folder_name = Path(root_path).name
    if folder_name:
        st.session_state["create_project_name"] = folder_name


def mark_project_name_manual() -> None:
    st.session_state["create_project_name_auto"] = False


def choose_project_root_path() -> None:
    st.session_state.pop("folder_dialog_error", None)
    try:
        selected = select_folder_dialog(st.session_state.get("create_root_path", ""))
    except Exception as exc:
        st.session_state["folder_dialog_error"] = str(exc)
        return

    if not selected:
        return

    st.session_state["create_root_path"] = selected
    set_project_name_from_root_path()


def refresh_projects() -> list[dict]:
    projects = list_projects()
    st.session_state["projects"] = projects
    return projects


def selected_project(projects: list[dict]) -> dict | None:
    if not projects:
        return None

    project_by_id = {p["project_id"]: p for p in projects}
    project_ids = list(project_by_id.keys())
    if st.session_state.get("selected_project_id") not in project_ids:
        st.session_state["selected_project_id"] = project_ids[0]

    selected_id = st.selectbox(
        "选择项目",
        project_ids,
        format_func=lambda project_id: (
            f"{project_by_id[project_id]['name']} - {project_by_id[project_id]['root_path']}"
        ),
        key="selected_project_id",
    )
    return project_by_id[selected_id]


def render_project_status(project: dict) -> None:
    stats = project_stats(project)
    exists = root_exists(project)
    cols = st.columns(3)
    cols[0].metric("素材数量", stats["total"])
    cols[1].metric("已索引", stats["indexed"])
    cols[2].metric("失败", stats["failed"])

    st.write(f"项目名称: {project['name']}")
    st.write(f"素材目录: {project['root_path']}")
    st.write(f"包含子目录: {'是' if project.get('recursive') else '否'}")
    st.write(f"最后索引时间: {stats['last_indexed_at'] or '尚未索引'}")
    if not exists:
        st.error("素材目录不存在或不可访问")


def render_create_project() -> None:
    if st.session_state.pop("clear_create_project_fields", False):
        st.session_state["create_project_name"] = ""
        st.session_state["create_root_path"] = ""
        st.session_state["create_project_name_auto"] = True

    st.session_state.setdefault("create_project_name", "")
    st.session_state.setdefault("create_root_path", "")
    st.session_state.setdefault("create_project_name_auto", True)

    name = st.text_input(
        "项目名称",
        key="create_project_name",
        help="选择素材目录后会自动使用文件夹名，可手动修改。",
        on_change=mark_project_name_manual,
    )
    root_path = st.text_input(
        "素材目录路径",
        key="create_root_path",
        on_change=set_project_name_from_root_path,
    )
    st.button("选择文件夹", use_container_width=True, on_click=choose_project_root_path)
    if st.session_state.get("folder_dialog_error"):
        st.error(st.session_state["folder_dialog_error"])

    recursive = st.checkbox("包含子目录", value=True)
    if st.button("创建项目"):
        try:
            project = create_project(name, root_path, recursive)
            st.session_state["selected_project_id"] = project["project_id"]
            st.session_state["search_results"] = []
            st.session_state["clear_create_project_fields"] = True
            refresh_projects()
            st.success("项目已创建")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def render_index_controls(project: dict) -> None:
    left, right = st.columns(2)

    if left.button("开始 / 更新索引", disabled=not root_exists(project), use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        counters = st.empty()

        def on_progress(event: dict) -> None:
            total = event.get("total", 0) or 1
            completed = event.get("completed", 0)
            progress.progress(min(completed / total, 1.0))
            current = event.get("current_file")
            if current:
                status.write(f"当前处理: {current}")
            counters.write(
                f"总数 {event.get('total', 0)} / 已完成 {completed} / "
                f"成功 {event.get('success', 0)} / 失败 {event.get('failed', 0)} / "
                f"跳过 {event.get('skipped', 0)}"
            )

        result = index_project(project, progress_callback=on_progress)
        st.success(
            f"索引完成: 总数 {result['total']}，成功 {result['success']}，"
            f"失败 {result['failed']}，跳过 {result['skipped']}"
        )
        refresh_projects()

    if right.button("清空当前项目索引", use_container_width=True):
        reset_project_index(project["project_id"])
        st.success("当前项目索引已清空，原始素材未删除")
        st.rerun()

    confirm_delete = st.checkbox("确认删除当前项目记录和缓存文件，不删除原始素材")
    if st.button("删除项目", disabled=not confirm_delete, use_container_width=True):
        delete_project(project["project_id"])
        st.session_state.pop("selected_project_id", None)
        st.session_state.pop("active_project_id", None)
        st.session_state["search_results"] = []
        refresh_projects()
        st.success("项目记录和缓存文件已删除，原始素材未删除")
        st.rerun()


def render_search(project: dict) -> None:
    with st.form("search_form"):
        query = st.text_input("搜索")
        filter_label = st.selectbox("类型", ["全部", "图片", "视频"])
        top_k = st.number_input("Top K", min_value=1, max_value=100, value=10, step=1)
        submitted = st.form_submit_button("搜索")

    if submitted:
        filter_type = {"全部": "all", "图片": "image", "视频": "video"}[filter_label]
        try:
            st.session_state["search_results"] = search_assets(
                project["project_id"],
                query,
                filter_type=filter_type,
                top_k=int(top_k),
            )
        except Exception as exc:
            st.error(str(exc))
            st.session_state["search_results"] = []

    results = st.session_state.get("search_results", [])
    for index, result in enumerate(results, start=1):
        cols = st.columns([1, 3, 1])
        thumb = result.get("thumb_path")
        if thumb and Path(thumb).exists():
            cols[0].image(thumb, use_container_width=True)
        else:
            cols[0].write("无缩略图")

        cols[1].write(f"文件名: {result['name']}")
        cols[1].write(f"类型: {result['file_type']}")
        cols[1].write(f"相似度: {result['score']:.4f}")
        cols[1].write(f"相对路径: {result.get('relative_path')}")
        cols[1].write(f"原始路径: {result['path']}")

        if cols[2].button("Reveal in Explorer", key=f"reveal_{index}_{result['path']}"):
            ok, message = reveal_in_explorer(result["path"])
            if ok:
                st.success("已打开 Explorer")
            else:
                st.error(message)


def main() -> None:
    st.title("本地素材语义搜索")
    projects = refresh_projects()

    sidebar = st.sidebar
    with sidebar:
        st.header("项目")
        render_create_project()

    project = selected_project(projects)
    if not project:
        st.info("请先创建项目")
        return
    if st.session_state.get("active_project_id") != project["project_id"]:
        st.session_state["active_project_id"] = project["project_id"]
        st.session_state["search_results"] = []

    st.header("当前项目")
    render_project_status(project)
    render_index_controls(project)

    st.header("搜索")
    render_search(project)


if __name__ == "__main__":
    main()
