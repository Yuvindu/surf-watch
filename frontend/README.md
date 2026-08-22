# SurfWatch Frontend

ML-powered rip current detection web app — React + TypeScript demo frontend for the COMP6002 SurfWatch project.

---

## Getting Started

Start the backend from the repository root in one terminal:

```bash
source .venv/bin/activate
python -m backend.server
```

The backend runs at `http://127.0.0.1:8000`.

Start the frontend from this directory in a second terminal:

```bash
npm install
npm run dev
```

The app runs at `http://localhost:5173`.

If the app opens to a blank page after dependency or lockfile changes, clear Vite's optimized dependency cache:

```bash
rm -rf node_modules/.vite
npm run dev -- --force
```

---

## Connecting the Real Backend

This repo now includes a minimal backend adapter for the baseline-vs-MARSP video comparison flow:

```bash
cd ..
source .venv/bin/activate
python -m backend.server
```

The frontend posts uploaded videos to `http://localhost:8000/api/comparisons` by default. To use another API host, set `VITE_API_BASE_URL` before starting Vite.

The backend wraps `scripts/run_baseline_vs_marsp_compare.py`, saves uploaded files under `outputs/frontend_uploads/`, and serves generated videos/metrics from `outputs/comparisons/`.

For the full frontend/backend startup guide and request flow, see [`../docs/frontend-backend-comparison-flow.md`](../docs/frontend-backend-comparison-flow.md).

Image uploads still use the mock prediction flow in `src/services/mockApi.ts`. Video uploads use the local comparison backend through `src/services/comparisonApi.ts`, including adjustable `window-size` and `threshold` parameters from the Analyse page.

The comparison backend runs video processing as a background job. The frontend polls job status and updates the stepper from the backend's actual current task.

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
