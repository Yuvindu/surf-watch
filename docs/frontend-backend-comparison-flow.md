# Frontend And Backend Comparison Flow

This document explains how to run the SurfWatch web app with the local backend adapter for the baseline-vs-MARSP video comparison workflow.

## What Runs

The local demo has two servers:

- Frontend: Vite React app at `http://127.0.0.1:5173`
- Backend: Python comparison API at `http://127.0.0.1:8000`

The frontend is responsible for upload UI, progress display, history, and results rendering. The backend receives uploaded videos, runs the existing Python comparison pipeline, and serves generated videos and JSON artifacts back to the browser.

## Prerequisites

From the repository root, create and install the Python environment if it does not already exist:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

The comparison flow also expects a trained checkpoint at:

```text
checkpoints/best_model.pt
```

## Start The Backend

From the repository root:

```bash
python3 -m backend.server
```

The backend listens on:

```text
http://127.0.0.1:8000
```

The server automatically uses `.venv/bin/python` for the ML pipeline when that virtual environment exists. This matters because packages such as OpenCV (`cv2`) are usually installed in the project venv rather than the system Python.

## Start The Frontend

In a second terminal:

```bash
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

If the backend runs somewhere else, set `VITE_API_BASE_URL` before starting Vite:

```bash
cd frontend
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

## User Flow

1. The user opens the Analyse page in the React app.
2. The user uploads an `.mp4` or `.webm` video.
3. `AnalysePage` calls `runComparison` from `frontend/src/hooks/useDemoPredict.ts`.
4. `runComparison` calls `runBaselineMarspComparison` in `frontend/src/services/comparisonApi.ts`.
5. The frontend sends a `POST /api/comparisons` request with the uploaded file as `multipart/form-data`.
6. The backend saves the uploaded file under `outputs/frontend_uploads/`.
7. The backend starts a comparison job and returns a `jobId`.
8. The frontend polls `GET /api/comparisons/<jobId>` once per second.
9. The backend updates `currentStage` and `currentTask` from the actual running subprocess output.
10. The comparison script runs baseline inference, the MARSP pipeline, overlay rendering, side-by-side video generation, and metric summarisation.
11. When the job completes, the backend status response includes URLs for the generated comparison artifacts.
12. The frontend stores the completed response in `AnalysisContext` and routes to the Results page.
13. The Results page renders the side-by-side comparison video, baseline overlay, MARSP overlay, confidence summary, and metric table.

## Backend API

### `POST /api/comparisons`

Request:

- Content type: `multipart/form-data`
- File field name: `file`
- File type: video upload

Example response:

```json
{
  "jobId": "RipVIS-051-1778669621",
  "caseId": "RipVIS-051-1778669621",
  "caseName": "RipVIS-051.mp4",
  "status": "processing",
  "currentStage": "frame_extraction",
  "currentTask": "Saving uploaded video for processing",
  "createdAt": "2026-05-13T10:55:00+00:00",
  "updatedAt": "2026-05-13T10:55:00+00:00"
}
```

### `GET /api/comparisons/<jobId>`

Returns the current backend job status. The frontend polls this endpoint while the comparison is running.

Processing response:

```json
{
  "jobId": "RipVIS-051-1778669621",
  "caseId": "RipVIS-051-1778669621",
  "caseName": "RipVIS-051.mp4",
  "status": "processing",
  "currentStage": "marsp_segmentation",
  "currentTask": "Running SegFormer inference inside the MARSP pipeline",
  "createdAt": "2026-05-13T10:55:00+00:00",
  "updatedAt": "2026-05-13T10:57:00+00:00"
}
```

Completed response:

```json
{
  "jobId": "RipVIS-051-1778669621",
  "caseId": "RipVIS-051-1778669621",
  "caseName": "RipVIS-051.mp4",
  "status": "completed",
  "currentStage": "complete",
  "currentTask": "Comparison complete",
  "createdAt": "2026-05-13T10:55:00+00:00",
  "updatedAt": "2026-05-13T10:58:00+00:00",
  "result": {
    "caseId": "RipVIS-051-1778669621",
    "caseName": "RipVIS-051.mp4",
    "status": "completed",
    "createdAt": "2026-05-13T10:55:00+00:00",
    "updatedAt": "2026-05-13T10:58:00+00:00",
    "videoUrl": "http://127.0.0.1:8000/artifacts/outputs/frontend_uploads/RipVIS-051-1778669621.mp4",
    "comparisonVideoUrl": "http://127.0.0.1:8000/artifacts/outputs/comparisons/RipVIS-051-1778669621_baseline_vs_marsp.mp4",
    "baselineOverlayUrl": "http://127.0.0.1:8000/artifacts/outputs/comparisons/RipVIS-051-1778669621_baseline_overlay.mp4",
    "marspOverlayUrl": "http://127.0.0.1:8000/artifacts/outputs/comparisons/RipVIS-051-1778669621_marsp_overlay.mp4",
    "predictionMaskUrl": "http://127.0.0.1:8000/artifacts/outputs/temporal_aggregation/RipVIS-051-1778669621_stabilised_agg_mask.mp4",
    "metricsUrl": "http://127.0.0.1:8000/artifacts/outputs/comparisons/RipVIS-051-1778669621_baseline_vs_marsp_metrics.json",
    "confidenceScore": 0.82,
    "summaryLabel": "MARSP improved temporal stability",
    "metricTable": []
  }
}
```

The exact metric values depend on the uploaded video and model outputs.

### `GET /artifacts/<path>`

Serves generated files under `outputs/`.

Video playback uses HTTP range requests, so the backend supports `206 Partial Content` responses for browser video controls.

## Generated Files

Uploaded videos are stored under:

```text
outputs/frontend_uploads/
```

Comparison outputs are stored under:

```text
outputs/comparisons/
```

Typical comparison files:

```text
<video_name>_baseline_overlay.mp4
<video_name>_baseline_mask.mp4
<video_name>_baseline_prob.mp4
<video_name>_marsp_overlay.mp4
<video_name>_baseline_vs_marsp.mp4
<video_name>_baseline_vs_marsp_metrics.json
```

The MARSP pipeline also writes intermediate outputs under:

```text
outputs/motion_compensation/
outputs/video_inference/
outputs/temporal_aggregation/
outputs/marsp/
```

## Frontend Files

Important frontend integration files:

- `frontend/src/pages/AnalysePage.tsx`: upload screen, run button, progress state, and navigation to results
- `frontend/src/services/comparisonApi.ts`: `POST /api/comparisons` client
- `frontend/src/hooks/useDemoPredict.ts`: shared prediction/comparison state hook
- `frontend/src/models/analysis.ts`: backend response and analysis history types
- `frontend/src/pages/ResultsPage.tsx`: comparison video and metric table rendering
- `frontend/src/context/AnalysisContext.tsx`: in-session analysis history storage

Image uploads still use the mock prediction path. Video uploads use the real local baseline-vs-MARSP comparison backend.

## Backend Files

Important backend and pipeline files:

- `backend/server.py`: local HTTP API adapter
- `scripts/run_baseline_vs_marsp_compare.py`: baseline-vs-MARSP comparison workflow
- `scripts/run_marsp_pipeline.py`: full MARSP processing pipeline
- `scripts/run_video_segmentation.py`: frame-level SegFormer inference
- `scripts/run_motion_compensation.py`: stabilisation preprocessing
- `scripts/run_temporal_aggregation.py`: probability/mask temporal aggregation

## Troubleshooting

### `No module named 'cv2'`

The backend or pipeline is using a Python environment without OpenCV.

Check the venv:

```bash
.venv/bin/python -c "import cv2; print(cv2.__version__)"
```

If that fails, install requirements:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Frontend finishes but no results page appears

Refresh the Vite page and rerun the analysis. The frontend should navigate to `/results/<case-id>` after the polled comparison job returns `status: "completed"`.

### Port already in use

Find the process:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:5173 -sTCP:LISTEN
```

Stop the old process or start the affected server on a different port.

### Comparison takes a while

That is expected. The backend runs baseline inference, MARSP processing, overlay rendering, and metric generation in a background job. The frontend polls job status and updates the active task while the job runs.

### Uploaded file too large

The frontend currently limits uploads to 50 MB in `frontend/src/hooks/useFileHandler.ts`. The backend accepts up to 1 GB, but the browser UI will block files above the frontend limit first.
