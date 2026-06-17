import platform
import subprocess
from pathlib import Path


def reveal_in_explorer(path: str) -> tuple[bool, str]:
    target = Path(path)
    if not target.exists():
        return False, f"原始文件不存在: {target}"

    if platform.system() != "Windows":
        return False, "Reveal in Explorer 仅在 Windows 下可用"

    subprocess.run(["explorer", "/select,", str(target)], check=False)
    return True, str(target)

