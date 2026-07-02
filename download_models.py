from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

from image_processor import (
    REMBG_MODEL_DIR,
    REMBG_MODEL_MD5,
    REMBG_MODEL_NAME,
    REMBG_MODEL_SIZE,
    _file_md5,
)

MODEL_URL = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
    "u2net_human_seg.onnx"
)


def main() -> None:
    REMBG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["U2NET_HOME"] = str(REMBG_MODEL_DIR)
    model_path = REMBG_MODEL_DIR / f"{REMBG_MODEL_NAME}.onnx"

    if not _is_valid_model(model_path):
        model_path.unlink(missing_ok=True)
        _download_model(model_path)

    if not _is_valid_model(model_path):
        actual_md5 = _file_md5(model_path) if model_path.is_file() else "missing"
        model_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Downloaded model failed MD5 check: {actual_md5}. "
            f"Expected {REMBG_MODEL_MD5}."
        )

    print(f"Ready for offline subject detection: {model_path}")


def _is_valid_model(model_path: Path) -> bool:
    return (
        model_path.is_file()
        and model_path.stat().st_size == REMBG_MODEL_SIZE
        and _file_md5(model_path) == REMBG_MODEL_MD5
    )


def _download_model(model_path: Path) -> None:
    partial_path = model_path.with_suffix(f"{model_path.suffix}.download")
    partial_path.unlink(missing_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, partial_path)
    except Exception as urllib_error:  # noqa: BLE001
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if curl is None:
            raise SystemExit(f"Could not download model: {urllib_error}") from urllib_error

        command = [
            curl,
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "-o",
            str(partial_path),
            MODEL_URL,
        ]
        if os.name == "nt":
            command.insert(1, "--ssl-no-revoke")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            partial_path.unlink(missing_ok=True)
            raise SystemExit(f"curl failed with exit code {result.returncode}") from urllib_error
    if _file_md5(partial_path) != REMBG_MODEL_MD5:
        actual_md5 = _file_md5(partial_path)
        partial_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Downloaded model failed MD5 check: {actual_md5}. "
            f"Expected {REMBG_MODEL_MD5}."
        )
    partial_path.replace(model_path)


if __name__ == "__main__":
    main()
