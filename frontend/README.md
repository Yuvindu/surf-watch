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

This repo now includes a minimal backend adapter for the baseline-vs-MARSP video comparison flow:

```bash
python3 -m backend.server
```

The frontend posts uploaded videos to `http://localhost:8000/api/comparisons` by default. To use another API host, set `VITE_API_BASE_URL` before starting Vite.

The backend wraps `scripts/run_baseline_vs_marsp_compare.py`, saves uploaded files under `outputs/frontend_uploads/`, and serves generated videos/metrics from `outputs/comparisons/`.

For the full frontend/backend startup guide and request flow, see [`../docs/frontend-backend-comparison-flow.md`](../docs/frontend-backend-comparison-flow.md).

Image uploads still use the mock prediction flow in `src/services/mockApi.ts`. Video uploads use the local comparison backend through `src/services/comparisonApi.ts`.

The current backend API is synchronous: it responds after the comparison video and metrics have been generated. The frontend advances the status stepper locally while that request is running.

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
├── services/               ← mock image flow and comparison API client
├── models/                 ← TypeScript types (keep aligned with backend schema)
├── components/             ← UI components (no backend logic)
├── pages/                  ← Route-level pages
├── hooks/                  ← State logic
└── theme/theme.ts          ← Visual theme
```
