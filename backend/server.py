from __future__ import annotations

import cgi
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "outputs" / "frontend_uploads"
COMPARISON_DIR = ROOT / "outputs" / "comparisons"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best_model.pt"
MAX_UPLOAD_BYTES = 1024**3
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def slugify_filename(filename: str) -> str:
    stem = Path(filename).stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-")
    return slug or "surfwatch-upload"


def public_artifact_url(handler: BaseHTTPRequestHandler, path: Path) -> str:
    relative = path.resolve().relative_to(ROOT)
    encoded = quote(relative.as_posix(), safe="/")
    host = handler.headers.get("Host", "127.0.0.1:8000")
    scheme = handler.headers.get("X-Forwarded-Proto", "http")
    return f"{scheme}://{host}/artifacts/{encoded}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pipeline_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def friendly_pipeline_error(stderr: str, stdout: str) -> str:
    combined = f"{stderr}\n{stdout}"
    if "No module named 'cv2'" in combined:
        return "OpenCV is not available to the Python environment running the comparison pipeline. Start the backend with `.venv/bin/python -m backend.server` or install project requirements into the active environment."
    if "No such file or directory" in combined and "best_model.pt" in combined:
        return "The model checkpoint was not found at checkpoints/best_model.pt."
    for line in reversed(combined.splitlines()):
        clean = line.strip()
        if clean and not clean.startswith("Traceback"):
            return clean
    return "Comparison pipeline failed."


def summary_label(metrics: dict[str, Any]) -> str:
    comparison = metrics.get("comparison", {})
    checks = [
        comparison.get("delta_mean_consecutive_iou", 0) > 0,
        comparison.get("delta_mean_consecutive_dice", 0) > 0,
        comparison.get("delta_mean_pixel_change_rate", 0) < 0,
        comparison.get("delta_std_mask_area_px", 0) < 0,
        comparison.get("delta_mean_abs_area_change_px", 0) < 0,
        comparison.get("delta_mean_small_blob_count", 0) < 0,
    ]
    improved = sum(1 for check in checks if check)

    if improved >= 4:
        return "MARSP improved temporal stability"
    if improved >= 2:
        return "MARSP changed prediction stability"
    return "Baseline remained more stable on this video"


def confidence_score(metrics: dict[str, Any]) -> float:
    marsp = metrics.get("marsp", {})
    dice = float(marsp.get("mean_consecutive_dice", 0.0))
    iou = float(marsp.get("mean_consecutive_iou", 0.0))
    return max(0.0, min(1.0, (dice + iou) / 2.0))


def run_comparison(video_name: str, input_path: Path, checkpoint: Path) -> dict[str, Any]:
    command = [
        pipeline_python(),
        "scripts/run_baseline_vs_marsp_compare.py",
        "--video-name",
        video_name,
        "--input",
        str(input_path),
        "--checkpoint",
        str(checkpoint),
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(friendly_pipeline_error(result.stderr, result.stdout))

    metrics_path = COMPARISON_DIR / f"{video_name}_baseline_vs_marsp_metrics.json"
    return read_json(metrics_path)


class SurfWatchHandler(BaseHTTPRequestHandler):
    server_version = "SurfWatchComparison/0.1"

    def end_headers(self) -> None:
        for key, value in cors_headers().items():
            self.send_header(key, value)
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/comparisons":
            self.create_comparison(parsed.query)
            return
        self.send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/artifacts/"):
            self.serve_artifact(parsed.path.removeprefix("/artifacts/"))
            return
        self.send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def create_comparison(self, query: str) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
            self.send_json({"error": "Upload is empty or exceeds the 1GB limit."}, status=HTTPStatus.BAD_REQUEST)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": str(content_length),
            },
        )

        if "file" not in form:
            self.send_json({"error": "A video file field named 'file' is required."}, status=HTTPStatus.BAD_REQUEST)
            return

        field = form["file"]
        if isinstance(field, list):
            field = field[0]

        if not field.filename:
            self.send_json({"error": "Uploaded file is missing a filename."}, status=HTTPStatus.BAD_REQUEST)
            return

        content_type = field.type or ""
        if not content_type.startswith("video/"):
            self.send_json(
                {"error": "Baseline vs MARSP comparison currently requires a video upload."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

        uploaded_at = utc_now()
        video_name = f"{slugify_filename(field.filename)}-{int(time.time())}"
        extension = Path(field.filename).suffix or ".mp4"
        input_path = UPLOAD_DIR / f"{video_name}{extension}"

        with input_path.open("wb") as handle:
            shutil.copyfileobj(field.file, handle)

        params = parse_qs(query)
        checkpoint_value = params.get("checkpoint", [None])[0]
        checkpoint = Path(checkpoint_value).expanduser() if checkpoint_value else DEFAULT_CHECKPOINT
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint

        try:
            metrics = run_comparison(video_name, input_path, checkpoint)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        artifacts = metrics.get("artifacts", {})
        side_by_side = ROOT / artifacts["side_by_side_video"]
        baseline_overlay = ROOT / artifacts["baseline_overlay"]
        marsp_overlay = ROOT / artifacts["marsp_overlay"]
        baseline_mask = ROOT / artifacts["baseline_mask"]
        marsp_mask = ROOT / artifacts["marsp_final_mask"]
        metrics_path = COMPARISON_DIR / f"{video_name}_baseline_vs_marsp_metrics.json"

        response = {
            "caseId": video_name,
            "caseName": Path(field.filename).name,
            "status": "completed",
            "createdAt": uploaded_at,
            "updatedAt": utc_now(),
            "videoUrl": public_artifact_url(self, input_path),
            "comparisonVideoUrl": public_artifact_url(self, side_by_side),
            "overlayUrl": public_artifact_url(self, side_by_side),
            "baselineOverlayUrl": public_artifact_url(self, baseline_overlay),
            "marspOverlayUrl": public_artifact_url(self, marsp_overlay),
            "predictionMaskUrl": public_artifact_url(self, marsp_mask),
            "baselineMaskUrl": public_artifact_url(self, baseline_mask),
            "metricsUrl": public_artifact_url(self, metrics_path),
            "confidenceScore": confidence_score(metrics),
            "summaryLabel": summary_label(metrics),
            "metricTable": metrics.get("metric_table", []),
            "comparison": metrics.get("comparison", {}),
            "baseline": metrics.get("baseline", {}),
            "marsp": metrics.get("marsp", {}),
        }
        self.send_json(response)

    def serve_artifact(self, raw_path: str) -> None:
        target = (ROOT / unquote(raw_path)).resolve()
        outputs_root = (ROOT / "outputs").resolve()

        if not (target == outputs_root or outputs_root in target.parents):
            self.send_error(HTTPStatus.FORBIDDEN, "Artifact path is not allowed.")
            return
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Artifact not found.")
            return

        file_size = target.stat().st_size
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")

        if range_header and range_header.startswith("bytes="):
            start_text, _, end_text = range_header.removeprefix("bytes=").partition("-")
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else file_size - 1
            end = min(end, file_size - 1)

            if start >= file_size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return

            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()

            with target.open("rb") as handle:
                handle.seek(start)
                self.wfile.write(handle.read(length))
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()

        with target.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), SurfWatchHandler)


def main() -> None:
    server = create_server()
    print("SurfWatch comparison API running at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
