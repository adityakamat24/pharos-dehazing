"""Interactive / headless demo for the Pharos streaming restorer.

    python -m pharos.rt.demo --source webcam --ckpt run.pt --overlay on
    python -m pharos.rt.demo --source clip.mp4 --ckpt run.pt --half --save out.mp4
    python -m pharos.rt.demo --source frames_dir --config configs/base.yaml --save out.mp4

Sources: ``webcam`` (or a device index), a video file, or a directory of images.
Keys (windowed mode): ``ESC``/``q`` quit, ``o`` toggle overlay, ``c`` confidence view,
``b`` before/after split. Headless (no display / ``--no-display``) requires ``--save``.

The real model lives in a parallel workstream and is imported lazily; if it is absent this
prints a clear message and exits non-zero rather than crashing at import time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from pharos.rt.infer import StreamingRestorer, load_model
from pharos.rt.overlay import ViewState, compose

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
_WINDOW = "Pharos"


# ---------------------------------------------------------------------------
# frame sources
# ---------------------------------------------------------------------------


class FrameSource:
    """Unified frame iterator over a webcam, a video file, or an image directory."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.is_live = False
        self.fps = 30.0
        self._kind = self._classify(source)

    @staticmethod
    def _classify(source: str) -> str:
        if source == "webcam" or source.isdigit():
            return "webcam"
        p = Path(source)
        if p.is_dir():
            return "images"
        if p.is_file():
            return "video"
        raise FileNotFoundError(f"source not found: {source}")

    def frames(self, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
        if self._kind == "images":
            yield from self._image_frames(max_frames)
        else:
            yield from self._capture_frames(max_frames)

    def _image_frames(self, max_frames: Optional[int]) -> Iterator[np.ndarray]:
        paths = sorted(p for p in Path(self.source).iterdir() if p.suffix.lower() in _IMG_EXTS)
        for i, p in enumerate(paths):
            if max_frames is not None and i >= max_frames:
                return
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame

    def _capture_frames(self, max_frames: Optional[int]) -> Iterator[np.ndarray]:
        if self._kind == "webcam":
            index = int(self.source) if self.source.isdigit() else 0
            cap = cv2.VideoCapture(index)
            self.is_live = True
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open source: {self.source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.fps = float(fps)
        try:
            i = 0
            while True:
                if max_frames is not None and i >= max_frames:
                    return
                ok, frame = cap.read()
                if not ok:
                    return
                yield frame
                i += 1
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# display helpers
# ---------------------------------------------------------------------------


def _can_display() -> bool:
    try:
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(_WINDOW)
        return True
    except cv2.error:
        return False


def _handle_key(key: int, view: ViewState) -> bool:
    """Return False to signal quit."""
    if key in (27, ord("q")):  # ESC / q
        return False
    if key == ord("o"):
        view.overlay = not view.overlay
    elif key == ord("c"):
        view.confidence_view = not view.confidence_view
    elif key == ord("b"):
        view.split = not view.split
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("pharos.rt.demo", description="Pharos streaming dehaze/desmoke demo")
    p.add_argument("--source", default="webcam", help="'webcam', a device index, a video file, or image dir")
    p.add_argument("--ckpt", default=None, help="checkpoint path (.pt) for the trained model")
    p.add_argument("--config", default=None, help="config YAML to build an (untrained) model, not --ckpt")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    p.add_argument("--half", action="store_true", help="half precision (CUDA only)")
    p.add_argument("--overlay", choices=["on", "off"], default="on", help="start with the overlay on/off")
    p.add_argument("--conf-threshold", type=float, default=0.5, help="confidence tint threshold")
    p.add_argument("--save", default=None, help="write the composed stream to this .mp4")
    p.add_argument("--out-fps", type=float, default=None, help="override output video FPS")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames (testing/headless)")
    p.add_argument("--no-display", action="store_true", help="force headless mode (requires --save)")
    p.add_argument("--auto-scene-cut", action="store_true", help="auto-reset recurrent state on scene cuts")
    p.add_argument("--image-mode", action="store_true", help="treat a dir as independent images (no state)")
    return p


def _build_restorer(args: argparse.Namespace) -> StreamingRestorer:
    if not args.ckpt and not args.config:
        raise SystemExit("error: provide --ckpt <checkpoint.pt> or --config <config.yaml>")
    try:
        if args.ckpt:
            model = load_model(args.ckpt, device=args.device if args.device != "auto" else "cpu")
        else:
            from pharos.config import load_config

            cfg = load_config(args.config)
            model = load_model(cfg, device="cpu")
    except ImportError as e:
        raise SystemExit(
            f"error: could not import the Pharos model (pharos.models): {e}\n"
            "The model workstream must be present/merged to run the demo."
        ) from e
    gate = None
    if args.config:
        try:
            from pharos.config import load_config

            gate = dict(load_config(args.config).model.gate)
        except Exception:  # noqa: BLE001
            gate = None
    return StreamingRestorer(
        model,
        device=args.device,
        half=args.half,
        gate=gate,
        auto_scene_cut=args.auto_scene_cut,
    )


def run_demo(args: argparse.Namespace) -> int:
    source = FrameSource(args.source)
    display = (not args.no_display) and _can_display()
    if not display and not args.save:
        print("error: no display available; pass --save out.mp4 to run headless", file=sys.stderr)
        return 2

    restorer = _build_restorer(args)
    view = ViewState(overlay=(args.overlay == "on"))
    stream_state = not (args.image_mode and source._kind == "images")

    writer: Optional[cv2.VideoWriter] = None
    writer_size: Optional[tuple[int, int]] = None
    out_fps = args.out_fps or source.fps
    if display:
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)

    n = 0
    try:
        for frame in source.frames(args.max_frames):
            result = restorer.restore(frame) if stream_state else restorer.restore_image(frame)
            canvas = compose(frame, result, view, threshold=args.conf_threshold,
                             fps=result["timings"]["fps_avg"])

            if args.save:
                if writer is None:
                    writer_size = (canvas.shape[1], canvas.shape[0])
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, float(out_fps), writer_size)
                    if not writer.isOpened():
                        print(f"error: could not open video writer for {args.save}", file=sys.stderr)
                        return 3
                if (canvas.shape[1], canvas.shape[0]) != writer_size:
                    canvas = cv2.resize(canvas, writer_size)
                writer.write(canvas)

            if display:
                cv2.imshow(_WINDOW, canvas)
                if not _handle_key(cv2.waitKey(1) & 0xFF, view):
                    break
            n += 1
    finally:
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    print(f"processed {n} frames" + (f"; saved {args.save}" if args.save else ""))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_demo(args)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
