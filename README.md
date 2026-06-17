# Gemini Video Embedding MVP

本项目是一个本地素材语义搜索 MVP。用户可以为不同素材目录创建独立项目，分别生成图片/视频 embedding，并在 Streamlit 前端中按语义搜索素材。

## 功能

- 多项目素材库，每个项目绑定一个本地素材目录。
- 每个项目独立保存 `index.json`、`vectors.npy`、`thumbnails/`、`previews/`。
- 支持图片和视频扫描、增量索引、缩略图生成。
- 视频会先生成 `adaptive_32frame_preview`，再提交给 Gemini embedding。
- 搜索结果按 cosine similarity 排序，可筛选图片/视频。
- Windows 下支持 Reveal in Explorer，定位原始素材文件。
- 聚类整理脚本不再接入主流程。

## 安装

建议在虚拟环境中安装依赖：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

安装 `ffmpeg` 并确保 `ffmpeg`、`ffprobe` 在 `PATH` 中可用。视频缩略图和 preview 依赖它们。

## 配置

复制 `.env.example` 为 `.env`，填入 Gemini API key：

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-2

HTTP_PROXY=
HTTPS_PROXY=
```

兼容旧变量名 `GOOGLE_API_KEY`。如果需要代理，填写 `HTTP_PROXY` 和 `HTTPS_PROXY`。

## 启动

```powershell
streamlit run streamlit_app.py
```

## 使用流程

1. 在侧边栏创建项目：输入项目名称、素材目录路径，并选择是否包含子目录。
2. 在项目下拉框中选择当前项目。
3. 点击“开始 / 更新索引”。
4. 在搜索框输入中文或英文语义描述。
5. 选择全部、图片或视频筛选。
6. 查看缩略图、相似度、相对路径和原始路径。
7. 点击 Reveal in Explorer 在 Windows Explorer 中选中原始素材。

## 数据目录

全局项目列表：

```text
app_data/projects.json
```

每个项目的独立缓存：

```text
app_data/projects/<project_id>/
  project.json
  index.json
  vectors.npy
  thumbnails/
  previews/
```

`previews/` 只保存用于 Gemini embedding 的视频 preview，不是原始素材。`thumbnails/` 只用于前端展示。清空或删除项目索引不会删除原始素材文件。

## 旧脚本

根目录下的 `generate_embeddings.py`、`search_video.py`、`convert_videos.py`、`cluster_videos.py` 是早期 CLI 实验脚本。当前主流程入口是 `streamlit_app.py`，聚类功能不再出现在主流程中。
