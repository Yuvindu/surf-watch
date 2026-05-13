# SurfWatch Frontend

ML-powered rip current detection web app — React + TypeScript demo frontend for the COMP6002 SurfWatch project.

---

## Getting Started

```bash
cd frontend
npm install
npm run dev
```

App runs at `http://localhost:5173`.

---

## Connecting the Real Backend

All mock logic lives in one file: `src/services/mockApi.ts`. When the Python prediction backend is ready, only this file needs to be replaced.

### 1. Replace `simulateUpload`

**Current (mock):**
```ts
export async function simulateUpload(file: File): Promise<UploadFile> {
  await delay(1500);
  // returns a fake UploadFile
}
```

**Replace with a real API call:**
```ts
export async function simulateUpload(file: File): Promise<UploadFile> {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch('http://localhost:8000/api/upload', {
    method: 'POST',
    body: form,
  });

  const data = await res.json();
  return {
    id: data.id,
    file,
    fileType: file.type.startsWith('video') ? 'video' : 'image',
    previewUrl: URL.createObjectURL(file),
    status: 'ready',
    uploadedAt: data.uploaded_at,
  };
}
```

---

### 2. Replace `simulatePrediction`

**Current (mock):** Steps through fake pipeline stages with random delays and returns randomised bounding boxes + confidence scores.

**Replace with a real API call:**
```ts
export async function simulatePrediction(
  uploadFile: UploadFile,
  onStageChange: (stage: PipelineStage) => void,
): Promise<PredictionResult> {

  // Optionally poll a /status endpoint and call onStageChange as stages complete.
  // Minimal version — single blocking call:
  onStageChange('upload');

  const res = await fetch(`http://localhost:8000/api/predict/${uploadFile.id}`, {
    method: 'POST',
  });

  const data = await res.json();
  onStageChange('complete');

  return {
    id: data.id,
    fileId: uploadFile.id,
    fileType: uploadFile.fileType,
    frames: data.frames,           // must match FramePrediction[] shape
    averageConfidence: data.average_confidence,
    processingTimeMs: data.processing_time_ms,
    createdAt: data.created_at,
  };
}
```

---

### Expected API Response Shape

The backend must return JSON matching these TypeScript types (defined in `src/models/`):

```ts
// POST /api/predict/:fileId  →  PredictionResult
{
  id: string,
  average_confidence: number,       // 0–1
  processing_time_ms: number,
  created_at: string,               // ISO 8601
  frames: [
    {
      frame_index: number,
      timestamp: number,            // seconds
      regions: [
        {
          id: string,
          bounding_box: {
            top_left:     { x: number, y: number },  // 0–1 relative coords
            bottom_right: { x: number, y: number }
          },
          pixel_coverage: number    // percentage of frame area
        }
      ],
      confidence: {
        overall: number,
        per_region: [{ region_id: string, score: number }]
      }
    }
  ]
}
```

> Coordinates must be **relative (0–1)**, not pixels — the overlay component scales them against the rendered image dimensions.

---

### Optional: Stage Progress via Polling

If the backend exposes a `/api/status/:jobId` endpoint, you can call `onStageChange` as each stage completes:

```ts
const POLL_INTERVAL = 1000; // ms

async function pollStatus(jobId: string, onStageChange: (s: PipelineStage) => void) {
  const seen = new Set<PipelineStage>();
  while (true) {
    const res = await fetch(`http://localhost:8000/api/status/${jobId}`);
    const { stage } = await res.json();
    if (!seen.has(stage)) { seen.add(stage); onStageChange(stage); }
    if (stage === 'complete') break;
    await new Promise(r => setTimeout(r, POLL_INTERVAL));
  }
}
```

The `StatusStepper` component advances automatically as `onStageChange` fires — no other UI changes needed.

---

## Build for Production

```bash
npm run build
```

Output is in `dist/`. Point your web server (Nginx, Caddy, etc.) at that directory.

---

## Project Structure

```
src/
├── services/mockApi.ts     ← swap this file when backend is ready
├── models/                 ← TypeScript types (keep aligned with backend schema)
├── components/             ← UI components (no backend logic)
├── pages/                  ← Route-level pages
├── hooks/                  ← State logic
└── theme/theme.ts          ← Visual theme
```
