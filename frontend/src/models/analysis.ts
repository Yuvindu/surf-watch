import type { UploadFile } from './upload';
import type { PredictionResult } from './prediction';

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
  createdAt: string;
}
