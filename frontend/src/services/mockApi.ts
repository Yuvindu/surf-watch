import type { UploadFile, FileType } from '../models/upload';
import type { PredictionResult, FramePrediction, RipRegion, ConfidenceScore } from '../models/prediction';
import type { PipelineStage } from '../models/analysis';
import { generateId } from '../utils/helpers';

const delay = (ms: number) => new Promise<void>(res => setTimeout(res, ms));

function randomBetween(min: number, max: number) {
  return Math.random() * (max - min) + min;
}

function generateMockRegions(count: number): RipRegion[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `region-${i + 1}`,
    mask: '',
    boundingBox: {
      topLeft: { x: randomBetween(0.05, 0.3), y: randomBetween(0.1, 0.4) },
      bottomRight: { x: randomBetween(0.5, 0.9), y: randomBetween(0.5, 0.9) },
    },
    pixelCoverage: randomBetween(3, 25),
  }));
}

function generateMockFrame(frameIndex: number, timestamp: number): FramePrediction {
  const regionCount = Math.floor(randomBetween(1, 4));
  const regions = generateMockRegions(regionCount);
  const overall = randomBetween(0.6, 0.95);
  const confidence: ConfidenceScore = {
    overall,
    perRegion: regions.map(r => ({ regionId: r.id, score: randomBetween(0.55, 0.95) })),
  };
  return { frameIndex, timestamp, regions, confidence };
}

export async function simulateUpload(file: File): Promise<UploadFile> {
  await delay(1500);
  const fileType: FileType = file.type.startsWith('video') ? 'video' : 'image';
  return {
    id: generateId(),
    file,
    fileType,
    previewUrl: URL.createObjectURL(file),
    status: 'ready',
    uploadedAt: new Date().toISOString(),
  };
}

export const PIPELINE_STAGES: PipelineStage[] = [
  'upload',
  'frame_extraction',
  'motion_compensation',
  'segmentation',
  'temporal_aggregation',
  'complete',
];

export const COMPARISON_PIPELINE_STAGES: PipelineStage[] = [
  'upload',
  'frame_extraction',
  'baseline_segmentation',
  'motion_compensation',
  'marsp_segmentation',
  'temporal_aggregation',
  'comparison_rendering',
  'complete',
];

export const STAGE_LABELS: Record<PipelineStage, string> = {
  upload: 'Upload',
  frame_extraction: 'Frame Extraction',
  motion_compensation: 'Motion Compensation',
  baseline_segmentation: 'Baseline Segmentation',
  segmentation: 'Segmentation',
  marsp_segmentation: 'MARSP Segmentation',
  temporal_aggregation: 'Temporal Aggregation',
  comparison_rendering: 'Comparison Rendering',
  complete: 'Complete',
};

export const STAGE_DESCRIPTIONS: Record<PipelineStage, string> = {
  upload: 'Transferring file to processing server',
  frame_extraction: 'Decoding video frames for analysis',
  motion_compensation: 'Stabilising camera motion across frames',
  baseline_segmentation: 'Running baseline SegFormer inference',
  segmentation: 'Running ML model to detect rip current regions',
  marsp_segmentation: 'Running model inference in the MARSP pipeline',
  temporal_aggregation: 'Aggregating frame-level predictions over time',
  comparison_rendering: 'Rendering comparison video and computing metrics',
  complete: 'Analysis complete',
};

export async function simulatePrediction(
  uploadFile: UploadFile,
  onStageChange: (stage: PipelineStage) => void,
): Promise<PredictionResult> {
  const startTime = Date.now();

  for (const stage of PIPELINE_STAGES) {
    onStageChange(stage);
    if (stage !== 'complete') {
      await delay(randomBetween(1000, 2000));
    }
  }

  const isVideo = uploadFile.fileType === 'video';
  const frameCount = isVideo ? Math.floor(randomBetween(4, 10)) : 1;
  const frames: FramePrediction[] = Array.from({ length: frameCount }, (_, i) =>
    generateMockFrame(i, i * 2)
  );

  const averageConfidence =
    frames.reduce((sum, f) => sum + f.confidence.overall, 0) / frames.length;

  return {
    id: generateId(),
    fileId: uploadFile.id,
    fileType: uploadFile.fileType,
    frames,
    averageConfidence,
    processingTimeMs: Date.now() - startTime,
    createdAt: new Date().toISOString(),
  };
}
