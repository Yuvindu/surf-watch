import type { BoundingBox } from './common';
import type { FileType } from './upload';

export interface RipRegion {
  id: string;
  mask: string;
  boundingBox: BoundingBox;
  pixelCoverage: number;
}

export interface ConfidenceScore {
  overall: number;
  perRegion: { regionId: string; score: number }[];
}

export interface FramePrediction {
  frameIndex: number;
  timestamp: number;
  regions: RipRegion[];
  confidence: ConfidenceScore;
}

export interface PredictionResult {
  id: string;
  fileId: string;
  fileType: FileType;
  frames: FramePrediction[];
  averageConfidence: number;
  processingTimeMs: number;
  createdAt: string;
}
