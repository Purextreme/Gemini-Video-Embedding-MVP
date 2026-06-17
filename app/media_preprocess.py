import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageOps

from app.config import is_image, is_video


THUMB_SIZE = (320, 180)
MAX_IMAGE_EMBED_SIDE = 1280
MAX_VIDEO_SIDE = 720


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def video_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def image_thumbnail(image_path: Path, thumb_path: Path) -> str:
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img.thumbnail(THUMB_SIZE)
        canvas = Image.new("RGB", THUMB_SIZE, "white")
        offset = ((THUMB_SIZE[0] - img.width) // 2, (THUMB_SIZE[1] - img.height) // 2)
        canvas.paste(_to_rgb(img), offset)
        canvas.save(thumb_path, "JPEG", quality=82)
    return str(thumb_path)


def compressed_image_bytes(image_path: Path) -> bytes:
    from io import BytesIO

    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img.thumbnail((MAX_IMAGE_EMBED_SIDE, MAX_IMAGE_EMBED_SIDE))
        rgb = _to_rgb(img)
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=88, optimize=True)
        return buffer.getvalue()


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        canvas = Image.new("RGB", img.size, "white")
        canvas.paste(img, mask=img.getchannel("A"))
        return canvas
    return img.convert("RGB")


def video_thumbnail(video_path: Path, thumb_path: Path) -> str:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg 或 ffprobe 不可用，无法处理视频")

    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    vf = f"thumbnail,scale='if(gt(iw,ih),{MAX_VIDEO_SIDE},-2)':'if(gt(iw,ih),-2,{MAX_VIDEO_SIDE})'"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(thumb_path),
    ]
    _run(cmd)
    return str(thumb_path)


def adaptive_32frame_preview(video_path: Path, preview_path: Path) -> str:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg 或 ffprobe 不可用，无法生成视频 preview")

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    duration = video_duration(video_path)
    scale = f"scale='if(gt(iw,ih),{MAX_VIDEO_SIDE},-2)':'if(gt(iw,ih),-2,{MAX_VIDEO_SIDE})'"

    if duration <= 32:
        vf = f"fps=1,{scale},format=yuv420p"
    else:
        vf = f"fps=32/{duration},{scale},format=yuv420p"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        vf,
        "-r",
        "1",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-movflags",
        "+faststart",
        str(preview_path),
    ]
    _run(cmd)
    return str(preview_path)


def prepare_asset_for_embedding(asset: dict, project_dir: Path) -> tuple[str, str | None, bytes, str]:
    source = Path(asset["path"])
    asset_id = asset["id"]
    thumb_path = project_dir / "thumbnails" / f"{asset_id}.jpg"

    if is_image(source):
        image_thumbnail(source, thumb_path)
        return str(thumb_path), None, compressed_image_bytes(source), "image/jpeg"

    if is_video(source):
        preview_path = project_dir / "previews" / f"{asset_id}.mp4"
        video_thumbnail(source, thumb_path)
        adaptive_32frame_preview(source, preview_path)
        return str(thumb_path), str(preview_path), preview_path.read_bytes(), "video/mp4"

    raise ValueError(f"Unsupported media type: {source}")

