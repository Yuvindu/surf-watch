import type { UploadFile } from './upload';
import type { PredictionResult } from './prediction';

export type PipelineStage =
  | 'upload'
  | 'frame_extraction'
  | 'motion_compensation'
  | 'baseline_segmentation'
  | 'segmentation'
  | 'marsp_segmentation'
  | 'temporal_aggregation'
  | 'comparison_rendering'
  | 'complete';

// 'completed' matches backend; 'complete' is the frontend-internal equivalent
export type AnalysisStatus = 'idle' | 'processing' | 'complete' | 'completed' | 'error';
export type ComparisonMode = 'single' | 'multi';

export interface ArtifactIdentity {
  path: string;
  size_bytes: number;
  sha256: string;
}

export interface ComparisonProvenance {
  generated_at_utc?: string;
  git_revision?: string | null;
  segmentation_model?: string;
  checkpoint?: ArtifactIdentity;
  component_checkpoints?: Record<string, ArtifactIdentity>;
  ensemble_weights?: Record<string, number>;
  input?: ArtifactIdentity;
  parameters?: { window_size?: number; threshold?: number };
}

export interface ModelComparisonResult {
  model: string;
  status: 'success' | 'failed';
  error?: string | null;
  frameCount?: number;
  runnerRuntimeSeconds?: number;
  timingSeconds?: Record<string, number>;
  baseline?: Record<string, number>;
  marsp?: Record<string, number>;
  comparison?: Record<string, number>;
  metricTable?: ComparisonMetric[];
  provenance?: ComparisonProvenance;
  artifacts?: Record<string, string>;
  metricsUrl?: string;
}

export interface AnalysisCase {
  id: string;                      // maps to backend caseId
  caseName?: string;               // backend: caseName
  upload: UploadFile;
  status: AnalysisStatus;
  currentStage: PipelineStage;
  currentTask?: string | null;
  prediction: PredictionResult | null;
  createdAt: string;
  updatedAt?: string;              // backend: updatedAt
  jobId?: string;                  // backend: async comparison job id
  videoUrl?: string;               // backend: videoUrl (remote-hosted source)
  overlayUrl?: string;             // backend: overlayUrl (pre-rendered overlay video)
  predictionMaskUrl?: string;      // backend: predictionMaskUrl (pre-rendered mask)
  confidenceScore?: number;        // backend: confidenceScore (flat, 0–1)
  summaryLabel?: string;           // backend: summaryLabel e.g. "Rip current detected"
  model?: string;                  // backend: selected segmentation adapter id
  mode?: ComparisonMode;
  models?: string[];
  currentModel?: string;
  modelIndex?: number;
  modelCount?: number;
  modelResults?: ModelComparisonResult[];
  threeWayVideoUrl?: string;
  ensemble?: { method: string; weights: Record<string, number> };
  inputProvenance?: ArtifactIdentity;
  totalRuntimeSeconds?: number;
  summaryJsonUrl?: string;
  summaryCsvUrl?: string;
  provenance?: ComparisonProvenance;
  timingSeconds?: Record<string, number>;
  windowSize?: number;             // backend: temporal aggregation window
  threshold?: number;              // backend: mask threshold
  comparisonVideoUrl?: string;     // backend: baseline-vs-MARSP side-by-side video
  baselineOverlayUrl?: string;     // backend: baseline overlay video
  marspOverlayUrl?: string;        // backend: MARSP overlay video
  baselineMaskUrl?: string;        // backend: baseline mask video
  metricsUrl?: string;             // backend: comparison JSON artifact
  metricTable?: ComparisonMetric[];
  comparison?: Record<string, number>;
  baseline?: Record<string, number>;
  marsp?: Record<string, number>;
}

export interface ComparisonMetric {
  metric: string;
  baseline: number;
  marsp: number;
  delta: number;
  preferred_direction: 'higher' | 'lower';
}

/** Exact shape of the backend API response — do not change without coordinating with backend. */
export interface CaseResponse {
  caseId: string;
  caseName: string;
  status: 'completed' | 'processing' | 'error';
  createdAt: string;
  updatedAt: string;
  videoUrl: string;
  overlayUrl?: string;
  predictionMaskUrl?: string;
  confidenceScore?: number;
  summaryLabel: string;
  mode?: ComparisonMode;
  models?: string[];
  modelResults?: ModelComparisonResult[];
  threeWayVideoUrl?: string;
  ensemble?: { method: string; weights: Record<string, number> };
  inputProvenance?: ArtifactIdentity;
  totalRuntimeSeconds?: number;
  summaryJsonUrl?: string;
  summaryCsvUrl?: string;
  provenance?: ComparisonProvenance;
  timingSeconds?: Record<string, number>;
  model?: string;
  windowSize?: number;
  threshold?: number;
  comparisonVideoUrl?: string;
  baselineOverlayUrl?: string;
  marspOverlayUrl?: string;
  baselineMaskUrl?: string;
  metricsUrl?: string;
  metricTable?: ComparisonMetric[];
  comparison?: Record<string, number>;
  baseline?: Record<string, number>;
  marsp?: Record<string, number>;
}

/**
 * Maps a backend CaseResponse onto the frontend AnalysisCase fields.
 * Call this after receiving an API response, then merge with the local UploadFile.
 */
export function normalizeCaseResponse(
  response: CaseResponse,
  upload: UploadFile,
): AnalysisCase {
  return {
    id: response.caseId,
    caseName: response.caseName,
    upload,
    status: response.status === 'completed' ? 'complete' : response.status,
    currentStage: 'complete',
    prediction: null,
    createdAt: response.createdAt,
    updatedAt: response.updatedAt,
    videoUrl: response.videoUrl,
    overlayUrl: response.overlayUrl,
    predictionMaskUrl: response.predictionMaskUrl,
    confidenceScore: response.confidenceScore,
    summaryLabel: response.summaryLabel,
    mode: response.mode,
    models: response.models,
    modelResults: response.modelResults,
    threeWayVideoUrl: response.threeWayVideoUrl,
    ensemble: response.ensemble,
    inputProvenance: response.inputProvenance,
    totalRuntimeSeconds: response.totalRuntimeSeconds,
    summaryJsonUrl: response.summaryJsonUrl,
    summaryCsvUrl: response.summaryCsvUrl,
    provenance: response.provenance,
    timingSeconds: response.timingSeconds,
    model: response.model,
    windowSize: response.windowSize,
    threshold: response.threshold,
    comparisonVideoUrl: response.comparisonVideoUrl,
    baselineOverlayUrl: response.baselineOverlayUrl,
    marspOverlayUrl: response.marspOverlayUrl,
    baselineMaskUrl: response.baselineMaskUrl,
    metricsUrl: response.metricsUrl,
    metricTable: response.metricTable,
    comparison: response.comparison,
    baseline: response.baseline,
    marsp: response.marsp,
  };
}
