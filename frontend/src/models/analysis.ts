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

export interface AnalysisCase {
  id: string;                      // maps to backend caseId
  caseName?: string;               // backend: caseName
  upload: UploadFile;
  status: AnalysisStatus;
  currentStage: PipelineStage;
  prediction: PredictionResult | null;
  createdAt: string;
  updatedAt?: string;              // backend: updatedAt
  videoUrl?: string;               // backend: videoUrl (remote-hosted source)
  overlayUrl?: string;             // backend: overlayUrl (pre-rendered overlay video)
  predictionMaskUrl?: string;      // backend: predictionMaskUrl (pre-rendered mask)
  confidenceScore?: number;        // backend: confidenceScore (flat, 0–1)
  summaryLabel?: string;           // backend: summaryLabel e.g. "Rip current detected"
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
  overlayUrl: string;
  predictionMaskUrl: string;
  confidenceScore: number;
  summaryLabel: string;
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
