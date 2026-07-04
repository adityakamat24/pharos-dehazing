"""Download Pharos datasets into the shared, gitignored data root.

    D:/dehazing_desmoking/data/<dataset_name>/

Each dataset has its own downloader with resume support and a post-download
verification (expected file counts). Use ``--only a,b,c`` to select datasets,
``--list`` to list them, and ``--data-root`` to override the destination.

Links researched July 2026. Some sources (RTTS/URHI on UT-Austin Box, D-Fire on
OneDrive) have no clean requests/gdown path and are best-effort — a failure there
is recorded and never blocks the other datasets.

Kaggle mirrors are only attempted if ``~/.kaggle/kaggle.json`` exists; otherwise
those paths are skipped (per project policy).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_utils as du  # noqa: E402

DEFAULT_ROOT = Path("D:/dehazing_desmoking/data")


@dataclass
class Result:
    name: str
    status: str = "skipped"          # ok | partial | failed | skipped
    note: str = ""
    size: int = 0
    count: int = 0
    expected: int = 0
    extras: dict = field(default_factory=dict)


def _kaggle_available() -> bool:
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


# ---------------------------------------------------------------------------
# per-dataset downloaders
# ---------------------------------------------------------------------------
def dl_reside6k(root: Path) -> Result:
    """RESIDE-6K lives inside the shared DehazeFormer Drive folder as
    ``data/RESIDE-6K/{train,test}.zip``. List the folder, grab just those two files
    (not the multi-GB RESIDE-OUT/RS-Haze siblings), and extract them."""
    r = Result("reside6k", expected=14000)
    dest = root / "reside6k"
    parent_folder = "1Yy_GH6_bydYPU6_JJzFQwig4LTh86VI4"
    listing = du.gdrive_list_folder(parent_folder)
    targets = [(fid, p) for fid, p in listing if "RESIDE-6K/" in p and p.endswith(".zip")]
    if not targets:
        r.status = "failed"
        r.note = (f"could not list RESIDE-6K children in Drive folder {parent_folder}; "
                  "download manually (see DESIGN §6)")
        return r
    for fid, path in targets:
        fname = path.split("/")[-1]  # train.zip / test.zip
        raw = dest / "_raw" / fname
        du.gdrive_download(fid, raw)
        du.extract_archive(raw, dest)
        du.cleanup_raw(raw)
    r.count = du.count_images(dest)
    r.size = du.dir_size(dest)
    r.status = "ok" if r.count >= r.expected * 0.5 else ("partial" if r.count else "failed")
    return r


def dl_smokebench(root: Path) -> Result:
    r = Result("smokebench", expected=19950)  # 9975 pairs
    dest = root / "smokebench"
    raw = dest / "_raw" / "SmokeBench.zip"
    du.gdrive_download("1NfusIRKwB9el2TpD2xYMOB1fIxPf8PW8", raw)
    du.extract_archive(raw, dest)
    du.cleanup_raw(raw)
    r.count = du.count_images(dest)
    r.size = du.dir_size(dest)
    r.status = "ok" if r.count >= r.expected * 0.5 else ("partial" if r.count else "failed")
    return r


def dl_satehaze1k(root: Path) -> Result:
    r = Result("satehaze1k", expected=2400)  # ~1200 pairs
    dest = root / "satehaze1k"
    raw = dest / "_raw" / "Haze1k.zip"
    url = "https://www.dropbox.com/s/k2i3p7puuwl2g59/Haze1k.zip?dl=1"
    du.http_download(url, raw)
    du.extract_archive(raw, dest)
    du.cleanup_raw(raw)
    r.count = du.count_images(dest)
    r.size = du.dir_size(dest)
    r.status = "ok" if r.count >= r.expected * 0.5 else ("partial" if r.count else "failed")
    return r


def dl_rice(root: Path) -> Result:
    r = Result("rice", expected=2400)  # RICE1 ~500 pairs + RICE2 ~736 sets
    dest = root / "rice"
    raw = dest / "_raw" / "RICE.zip"
    du.gdrive_download("1CricZtIj28BGFvkD_x-W8fSexPiDtgHk", raw)
    du.extract_archive(raw, dest)
    du.cleanup_raw(raw)
    r.count = du.count_images(dest)
    r.size = du.dir_size(dest)
    r.status = "ok" if r.count else "failed"
    return r


def _eth_dataset(root: Path, name: str, urls: list[str], expected_pairs: int) -> Result:
    r = Result(name, expected=expected_pairs * 2)
    dest = root / name
    raw = dest / "_raw" / (name + ".zip")
    used = du.try_urls(urls, raw)
    du.extract_archive(raw, dest)
    du.cleanup_raw(raw)
    r.count = du.count_images(dest)
    r.size = du.dir_size(dest)
    r.note = f"via {used}"
    r.status = "ok" if r.count >= expected_pairs else ("partial" if r.count else "failed")
    return r


def dl_nhhaze(root: Path) -> Result:
    return _eth_dataset(
        root, "nhhaze",
        ["https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip"],
        expected_pairs=55,
    )


def dl_densehaze(root: Path) -> Result:
    return _eth_dataset(
        root, "densehaze",
        ["https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/files/Dense_Haze_NTIRE19.zip"],
        expected_pairs=33,
    )


def dl_ohaze(root: Path) -> Result:
    return _eth_dataset(
        root, "ohaze",
        ["https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/O-HAZE.zip",
         "http://www.vision.ee.ethz.ch/ntire18/o-haze/O-HAZE.zip"],
        expected_pairs=45,
    )


def dl_ihaze(root: Path) -> Result:
    return _eth_dataset(
        root, "ihaze",
        ["https://data.vision.ee.ethz.ch/cvl/ntire18/i-haze/I-HAZE.zip",
         "http://www.vision.ee.ethz.ch/ntire18/i-haze/I-HAZE.zip"],
        expected_pairs=35,
    )


def dl_revide(root: Path) -> Result:
    r = Result("revide", expected=0)
    dest = root / "revide"
    raw = dest / "_raw" / "REVIDE_Indoor.zip"
    du.gdrive_download("1MYaVMUtcfqXeZpnbsfoJ2JBcpZUUlXGg", raw)
    du.extract_archive(raw, dest)
    du.cleanup_raw(raw)
    r.count = du.count_images(dest)
    r.size = du.dir_size(dest)
    r.status = "ok" if r.count else "failed"
    return r


def dl_smoke5k(root: Path) -> Result:
    r = Result("smoke5k", expected=10800)  # 5400 img + 5400 mask
    dest = root / "smoke5k"
    raw = dest / "_raw" / "SMOKE5K.zip"
    du.gdrive_download("11TM8hsh9R6ZTvLAUzfD6eD051MbOufCi", raw)
    du.extract_archive(raw, dest)
    du.cleanup_raw(raw)
    r.count = du.count_images(dest)
    r.size = du.dir_size(dest)
    r.status = "ok" if r.count else "failed"
    return r


def dl_rtts(root: Path) -> Result:
    """RTTS is hosted on UT-Austin Box (bit.ly -> box). Box shared links aren't
    plain-HTTP downloadable; try the redirect, else record a manual note."""
    r = Result("rtts", expected=4322)
    dest = root / "rtts"
    raw = dest / "_raw" / "RTTS.zip"
    try:
        du.http_download("https://bit.ly/3c4gl3z", raw)
        du.extract_archive(raw, dest)
        du.cleanup_raw(raw)
        r.count = du.count_images(dest)
        r.size = du.dir_size(dest)
        r.status = "ok" if r.count else "failed"
    except Exception as e:  # noqa: BLE001
        r.status = "failed"
        r.note = (f"UT-Austin Box link not directly downloadable ({type(e).__name__}); "
                  "fetch manually from RESIDE-beta page or Baidu mirror")
    return r


def dl_urhi(root: Path) -> Result:
    r = Result("urhi", expected=4800)
    dest = root / "urhi"
    raw = dest / "_raw" / "URHI.zip"
    try:
        du.http_download("https://bit.ly/2XVx7tc", raw)
        du.extract_archive(raw, dest)
        du.cleanup_raw(raw)
        r.count = du.count_images(dest)
        r.size = du.dir_size(dest)
        r.status = "ok" if r.count else "failed"
    except Exception as e:  # noqa: BLE001
        r.status = "failed"
        r.note = f"UT-Austin Box link not directly downloadable ({type(e).__name__})"
    return r


DOWNLOADERS = {
    "reside6k": dl_reside6k,
    "smokebench": dl_smokebench,
    "satehaze1k": dl_satehaze1k,
    "rice": dl_rice,
    "nhhaze": dl_nhhaze,
    "densehaze": dl_densehaze,
    "ohaze": dl_ohaze,
    "ihaze": dl_ihaze,
    "revide": dl_revide,
    "smoke5k": dl_smoke5k,
    "rtts": dl_rtts,
    "urhi": dl_urhi,
}

# what to run by default when no --only is given (per WS-B task)
DEFAULT_RUN = ["smokebench", "reside6k", "satehaze1k", "rice", "nhhaze",
               "densehaze", "ohaze", "ihaze", "rtts"]


def print_summary(results: list[Result]) -> None:
    print("\n" + "=" * 78)
    print(f"{'dataset':<14}{'status':<10}{'images':>8}{'expect':>8}{'size':>12}   note")
    print("-" * 78)
    for r in results:
        print(f"{r.name:<14}{r.status:<10}{r.count:>8}{r.expected:>8}"
              f"{du.human_size(r.size):>12}   {r.note}")
    print("=" * 78)
    ok = sum(1 for r in results if r.status == "ok")
    print(f"{ok}/{len(results)} datasets ok\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Pharos datasets")
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT), type=str)
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    ap.add_argument("--list", action="store_true", help="list available datasets and exit")
    args = ap.parse_args()

    if args.list:
        print("available datasets:", ", ".join(DOWNLOADERS))
        print("default run:", ", ".join(DEFAULT_RUN))
        return 0

    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
    else:
        names = list(DEFAULT_RUN)

    if not _kaggle_available():
        print("[info] no ~/.kaggle/kaggle.json -> Kaggle mirrors will be skipped")

    results: list[Result] = []
    for name in names:
        fn = DOWNLOADERS.get(name)
        if fn is None:
            results.append(Result(name, status="failed", note="unknown dataset"))
            continue
        print(f"\n### {name} -> {root / name}")
        try:
            results.append(fn(root))
        except KeyboardInterrupt:
            print("interrupted")
            results.append(Result(name, status="failed", note="interrupted"))
            break
        except Exception as e:  # noqa: BLE001 - never let one dataset block the rest
            traceback.print_exc()
            results.append(Result(name, status="failed", note=f"{type(e).__name__}: {e}"))

    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
