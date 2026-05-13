
**SurfWatch Frontend Demo — Full Build Prompt**

Build a complete React + TypeScript frontend application called **SurfWatch** for rip current detection in beach video/images. This is a **demo-mode** application — no real backend or ML model is connected yet. All predictions, confidence scores, and mask overlays should be simulated with realistic mock data and delays to mimic real API behaviour.

---

**Tech Stack**
- React 18+ with TypeScript (strict mode)
- Material UI (MUI) v5 for components, theming, and icons
- React Router v6 for navigation
- Vite as build tool

---

**Folder Structure (minimal, flat where possible)**

```
src/
├── assets/              # static images, logos, mock overlay PNGs
├── components/          # reusable UI components
│   ├── Layout.tsx           # app shell: navbar + sidebar + footer
│   ├── FileUploader.tsx     # drag-and-drop upload zone
│   ├── VideoPlayer.tsx      # video preview with overlay toggle
│   ├── ResultCard.tsx       # single prediction result display
│   ├── ConfidenceGauge.tsx  # circular/linear confidence indicator
│   ├── HeatmapOverlay.tsx   # simulated rip mask overlay on image/video
│   ├── StatusStepper.tsx    # pipeline progress stepper
│   ├── WaveBackground.tsx   # animated ocean wave SVG background
│   └── ThemeToggle.tsx      # light/dark mode switch
├── hooks/
│   ├── useDemoPredict.ts    # simulates prediction with delay + mock data
│   ├── useFileHandler.ts    # handles file selection, validation, preview URL
│   └── useAnalysisHistory.ts # manages local history state
├── models/
│   ├── prediction.ts        # PredictionResult, RipRegion, ConfidenceScore
│   ├── upload.ts            # UploadFile, FileType, FileStatus
│   ├── analysis.ts          # AnalysisCase, PipelineStage, AnalysisStatus
│   └── common.ts            # BoundingBox, Coordinate, Timestamp types
├── pages/
│   ├── HomePage.tsx         # landing / hero with feature overview
│   ├── AnalysePage.tsx      # main upload + prediction workflow
│   ├── ResultsPage.tsx      # view prediction output + overlay
│   ├── HistoryPage.tsx      # past analysis cases (local state)
│   └── AboutPage.tsx        # project info, how it works
├── services/
│   └── mockApi.ts           # simulated API calls with Promise delays
├── theme/
│   └── theme.ts             # MUI custom theme (Black and white)
├── utils/
│   └── helpers.ts           # file size formatter, date formatter, etc.
├── App.tsx
└── main.tsx
```

---

**Models & Data Types**

```typescript
// models/common.ts
export interface Coordinate { x: number; y: number; }
export interface BoundingBox { topLeft: Coordinate; bottomRight: Coordinate; }
export type Timestamp = string; // ISO 8601

// models/upload.ts
export type FileType = 'image' | 'video';
export type FileStatus = 'idle' | 'uploading' | 'ready' | 'error';
export interface UploadFile {
  id: string;
  file: File;
  fileType: FileType;
  previewUrl: string;
  status: FileStatus;
  uploadedAt: Timestamp;
}

// models/prediction.ts
export interface RipRegion {
  id: string;
  mask: string;           // base64 or URL to simulated overlay image
  boundingBox: BoundingBox;
  pixelCoverage: number;  // percentage of frame
}
export interface ConfidenceScore {
  overall: number;        // 0–1
  perRegion: { regionId: string; score: number }[];
}
export interface FramePrediction {
  frameIndex: number;
  timestamp: number;      // seconds into video
  regions: RipRegion[];
  confidence: ConfidenceScore;
}
export interface PredictionResult {
  id: string;
  fileId: string;
  fileType: FileType;
  frames: FramePrediction[];  // single frame for images
  averageConfidence: number;
  processingTimeMs: number;
  createdAt: Timestamp;
}

// models/analysis.ts
export type PipelineStage =
  | 'upload'
  | 'frame_extraction'
  | 'motion_compensation'
  | 'segmentation'
  | 'temporal_aggregation'
  | 'complete';
export type AnalysisStatus = 'idle' | 'processing' | 'complete' | 'error';
export interface AnalysisCase {
  id: string;
  upload: UploadFile;
  status: AnalysisStatus;
  currentStage: PipelineStage;
  prediction: PredictionResult | null;
  createdAt: Timestamp;
}
```

---

**Demo Mode Behaviour (mockApi.ts)**

- `simulateUpload(file)` → 1.5s delay → returns `UploadFile`
- `simulatePrediction(fileId)` → steps through each `PipelineStage` with 1–2s delay each, firing progress callbacks → returns `PredictionResult` with randomised but realistic mock data (confidence 0.6–0.95, 1–3 rip regions, bounding boxes within frame dimensions)
- `getAnalysisHistory()` → returns stored cases from React state (persisted in `useState` or `useReducer`, not localStorage)
- All delays use `setTimeout` wrapped in Promises
- Show the `StatusStepper` advancing through MARSP stages in real time during mock processing

---

**User Flow**  

each page should have  a same format made center focused stcuture and use good design layout also use black and white colors: 

this is not a website this is a web app so made it feel like that.  no need much website looks

1. **Home page** — ocean-themed hero section with animated wave background. Brief tagline: "ML-powered rip current detection for safer beaches." CTA button → "Start analysis"
2. **Analyse page** — drag-and-drop or click to upload an image (.jpg, .png) or video (.mp4, .webm). File validation (type + size limit 50MB). Preview shown immediately. Click "Run detection" →
3. **Processing view** (same page, or modal) — `StatusStepper` shows MARSP stages animating: Upload → Frame extraction → Motion compensation → Segmentation → Temporal aggregation → Complete. Each step shows a small description and a spinner.
4. **Results page** — displays the uploaded media with a toggleable semi-transparent rip current mask overlay (simulated coloured region). Shows: confidence gauge (circular), region count, pixel coverage %, processing time. For video: a frame scrubber/slider to step through predictions at different timestamps.
5. **History page** — card grid of past analysis cases from current session. Each card shows thumbnail, date, confidence badge, status chip. Click → opens result detail.
6. **About page** — brief explanation of rip currents, the MARSP pipeline (can reuse the workflow diagram concept), and project credits.

---

**Theme & Visual Design**

Custom MUI theme in `theme.ts`:

- **Primary colour:** Deep ocean blue `#0A4D68`
- **Secondary colour:** Seafoam/teal `#05BFDB`
- **Accent/warning:** Coral `#E85D30` (used for rip current overlay colour and danger indicators)
- **Background:** Light mode: soft off-white `#F5F9FC` / Dark mode: deep navy `#0B1929`
- **Surface cards:** Slight glass-morphism effect (subtle backdrop blur + low-opacity white/dark background)
- **Typography:** Clean sans-serif (Roboto or Inter)
- **Border radius:** 12px on cards, 8px on buttons (rounded, modern feel)

**Animated background (WaveBackground.tsx):**
- Layered animated SVG sine waves at the bottom of pages (3 layers, different opacity + speed + ocean-blue shades)
- Subtle CSS `@keyframes` horizontal translation loop for gentle wave motion
- Optional: animated floating bubble/circle particles (very subtle, low opacity) drifting upward on the home page hero section
- All animations respect `prefers-reduced-motion`

**Other visual touches:**
- Water ripple effect on the upload drop zone when hovering
- Confidence gauge uses a gradient from coral (low) → teal (medium) → green (high)
- Rip current overlay on results uses semi-transparent coral/red with pulsing opacity animation
- MUI icons throughout: `WaterDrop`, `Waves`, `Upload`, `Analytics`, `History`, `Info`, `Warning`

---

**Key Implementation Notes**

- All state lives in React hooks (`useState`, `useReducer`) — no external state library needed
- Use `React.createContext` if passing analysis state across multiple pages
- Keep every component focused on a single responsibility
- Use MUI's `sx` prop for styling (no separate CSS files)
- All mock data generation lives in `services/mockApi.ts` — when backend is ready, only this file needs to be swapped for real API calls
- Responsive design: works on desktop and tablet (minimum 768px)
- Add a visible "Demo Mode" chip/banner in the navbar so viewers know predictions are simulated

---