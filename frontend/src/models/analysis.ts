import type { UploadFile } from './upload';
import type { PredictionResult } from './prediction';

export type PipelineStage =
  | 'upload'
  | 'frame_extraction'
  | 'motion_compensation'
  | 'segmentation'
  | 'temporal_aggregation'
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
  };
}
