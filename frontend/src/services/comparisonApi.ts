import type { UploadFile } from '../models/upload';
import type { CaseResponse, PipelineStage } from '../models/analysis';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const POLL_INTERVAL_MS = 1000;

export interface ComparisonRunOptions {
  model: string;
  windowSize: number;
  threshold: number;
}

export interface SegmentationModelOption {
  id: string;
  label: string;
  description: string;
}

interface SegmentationModelsResponse {
  defaultModel: string;
  models: SegmentationModelOption[];
}

function wait(ms: number) {
  return new Promise<void>(resolve => setTimeout(resolve, ms));
}

export interface ComparisonJobStatus {
  jobId: string;
  caseId: string;
  caseName: string;
  status: 'processing' | 'completed' | 'error';
  currentStage: PipelineStage;
  currentTask?: string;
  createdAt?: string;
  updatedAt?: string;
  model?: string;
  windowSize?: number;
  threshold?: number;
  error?: string;
  result?: CaseResponse;
}

export async function fetchSegmentationModels(): Promise<SegmentationModelsResponse> {
  const response = await fetch(`${API_BASE_URL}/api/models`);
  const data = await response.json() as SegmentationModelsResponse & { error?: string };
  if (!response.ok) {
    throw new Error(data.error ?? 'Could not load segmentation models.');
  }
  return data;
}

export async function runBaselineMarspComparison(
  uploadFile: UploadFile,
  options: ComparisonRunOptions,
  onStageChange: (stage: PipelineStage) => void,
  onTaskChange?: (task: string) => void,
  onJobStart?: (status: ComparisonJobStatus) => void,
  onJobProgress?: (status: ComparisonJobStatus) => void,
): Promise<CaseResponse> {
  const form = new FormData();
  form.append('file', uploadFile.file);
  form.append('model', options.model);
  form.append('windowSize', String(options.windowSize));
  form.append('threshold', String(options.threshold));

  onStageChange('upload');
  onTaskChange?.('Uploading video to the local comparison API');

  const startResponse = await fetch(`${API_BASE_URL}/api/comparisons`, {
    method: 'POST',
    body: form,
  });

  const startData = await startResponse.json();
  if (!startResponse.ok) {
    throw new Error(startData.error ?? 'Comparison pipeline failed.');
  }

  const startStatus = startData as ComparisonJobStatus;
  const jobId = startStatus.jobId;
  onJobStart?.(startStatus);
  onStageChange(startData.currentStage ?? 'frame_extraction');
  onTaskChange?.(startData.currentTask ?? 'Preparing comparison job');

  while (true) {
    await wait(POLL_INTERVAL_MS);

    const statusResponse = await fetch(`${API_BASE_URL}/api/comparisons/${encodeURIComponent(jobId)}`);
    const statusData = await statusResponse.json() as ComparisonJobStatus;

    if (!statusResponse.ok) {
      throw new Error(statusData.error ?? 'Could not read comparison job status.');
    }

    onStageChange(statusData.currentStage);
    if (statusData.currentTask) onTaskChange?.(statusData.currentTask);
    onJobProgress?.(statusData);

    if (statusData.status === 'error') {
      throw new Error(statusData.error ?? 'Comparison pipeline failed.');
    }

    if (statusData.status === 'completed' && statusData.result) {
      onStageChange('complete');
      onTaskChange?.('Comparison complete');
      return statusData.result;
    }
  }
}
