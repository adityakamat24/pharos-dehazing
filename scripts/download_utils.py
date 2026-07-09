"""Shared helpers for Pharos dataset downloaders.

Only ``requests`` and ``gdown`` (both preinstalled) plus the stdlib are used. All
downloads support resume where the server allows it, and archives are extracted in
place. Nothing here imports torch or the pharos package.
"""
from __future__ import annotations

import re
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is a soft dependency here
    tqdm = None

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# (connect timeout, read timeout) — a stalled socket raises after read timeout so a
# dead/stalling link fails fast instead of blocking the whole run.
DEFAULT_TIMEOUT = (30, 180)


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def dir_size(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def count_images(path: str | Path, recursive: bool = True) -> int:
    path = Path(path)
    if not path.is_dir():
        return 0
    it = path.rglob("*") if recursive else path.iterdir()
    return sum(1 for p in it if p.suffix.lower() in IMG_EXTS)


def http_download(url: str, dest: str | Path, timeout=DEFAULT_TIMEOUT, chunk: int = 1 << 20) -> Path:
    """Stream a URL to ``dest`` with HTTP-range resume. Raises on HTTP errors."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
        if r.status_code == 416:  # requested range not satisfiable => already complete
            return dest
        mode = "wb"
        if existing and r.status_code == 206:
            mode = "ab"
        else:
            existing = 0  # server ignored range; restart cleanly
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + existing
        bar = tqdm(total=total or None, initial=existing, unit="B", unit_scale=True,
                   desc=dest.name) if tqdm else None
        with open(dest, mode) as f:
            for data in r.iter_content(chunk):
                if not data:
                    continue
                f.write(data)
                if bar:
                    bar.update(len(data))
        if bar:
            bar.close()
    return dest


def gdrive_download(file_id: str, dest: str | Path) -> Path:
    """Download a single Google Drive file by id via gdown (skips if present)."""
    import gdown

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    gdown.download(id=file_id, output=str(dest), quiet=False, resume=True)
    return dest


def gdrive_download_folder(folder_id: str, dest: str | Path) -> list[str]:
    """Download an entire Google Drive folder via gdown."""
    import gdown

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = gdown.download_folder(id=folder_id, output=str(dest), quiet=False, use_cookies=False)
    return out or []


def gdrive_list_folder(folder_id: str) -> list[tuple[str, str]]:
    """List a public Drive folder's files via gdown (no download). Returns a list of
    (file_id, relative_path). More reliable than HTML scraping."""
    import gdown

    items = gdown.download_folder(id=folder_id, skip_download=True, quiet=True, use_cookies=False)
    out: list[tuple[str, str]] = []
    for it in items or []:
        fid = getattr(it, "id", None)
        path = str(getattr(it, "path", "")).replace("\\", "/")
        if fid:
            out.append((fid, path))
    return out


def gdrive_folder_children(folder_id: str, timeout=DEFAULT_TIMEOUT) -> list[tuple[str, str, bool]]:
    """Best-effort scrape of a public Drive folder's immediate children.

    Returns a list of (name, id, is_folder). Google embeds child metadata in the
    folder page as a JS array; we regex it out. Fragile by nature — returns [] on
    any failure so callers can degrade gracefully.
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        text = requests.get(url, timeout=timeout).text
    except Exception:
        return []
    children: list[tuple[str, str, bool]] = []
    # entries look like: ["<id>",["<parent>"],"<name>",...,"application/vnd.google-apps.folder"...]
    pat = re.compile(r'\["([a-zA-Z0-9_-]{20,})",\["[^"]+"\],"((?:[^"\\]|\\.)+)"')
    for m in pat.finditer(text):
        cid, name = m.group(1), m.group(2).encode().decode("unicode_escape")
        is_folder = "application/vnd.google-apps.folder" in text[m.end():m.end() + 400]
        children.append((name, cid, is_folder))
    # de-dup preserving order
    seen: set[str] = set()
    uniq = []
    for name, cid, isf in children:
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append((name, cid, isf))
    return uniq


def extract_archive(path: str | Path, dest: str | Path, tolerant: bool = True) -> int:
    """Extract a .zip / .tar(.gz/.bz2) archive into ``dest``.

    Returns the number of members that failed to extract. When ``tolerant`` (default)
    a corrupt member is skipped instead of aborting the whole archive — real dataset
    mirrors occasionally contain one bad entry among thousands of good images.
    """
    path, dest = Path(path), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    failed = 0
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            if not tolerant:
                z.extractall(dest)
                return 0
            for info in z.infolist():
                try:
                    z.extract(info, dest)
                except Exception:  # noqa: BLE001 - skip the corrupt member, keep the rest
                    failed += 1
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            if not tolerant:
                t.extractall(dest)
                return 0
            for member in t.getmembers():
                try:
                    t.extract(member, dest)
                except Exception:  # noqa: BLE001
                    failed += 1
    else:
        raise ValueError(f"unrecognized archive format: {path}")
    return failed


def download_and_extract(url: str, raw_path: str | Path, dest: str | Path) -> Path:
    """HTTP-download an archive to ``raw_path`` then extract into ``dest``."""
    raw_path = Path(raw_path)
    http_download(url, raw_path)
    extract_archive(raw_path, dest)
    return Path(dest)


def try_urls(urls: list[str], raw_path: str | Path) -> str:
    """Try each URL in order until one downloads; return the URL that worked."""
    last_err: Exception | None = None
    for u in urls:
        try:
            http_download(u, raw_path)
            return u
        except Exception as e:  # noqa: BLE001 - record and try the next mirror
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"all urls failed; last error: {last_err}")


def cleanup_raw(raw_path: str | Path) -> None:
    """Remove a downloaded archive after successful extraction to save disk."""
    raw_path = Path(raw_path)
    try:
        if raw_path.is_file():
            raw_path.unlink()
        elif raw_path.is_dir():
            shutil.rmtree(raw_path, ignore_errors=True)
    except OSError:
        pass
