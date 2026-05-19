import type { UploadFile } from '../models/upload';
import type { CaseResponse, PipelineStage } from '../models/analysis';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const POLL_INTERVAL_MS = 1000;

function wait(ms: number) {
  return new Promise<void>(resolve => setTimeout(resolve, ms));
}

interface ComparisonJobStatus {
  jobId: string;
  status: 'processing' | 'completed' | 'error';
  currentStage: PipelineStage;
  currentTask?: string;
  error?: string;
  result?: CaseResponse;
}

export async function runBaselineMarspComparison(
  uploadFile: UploadFile,
  onStageChange: (stage: PipelineStage) => void,
  onTaskChange?: (task: string) => void,
): Promise<CaseResponse> {
  const form = new FormData();
  form.append('file', uploadFile.file);

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

  const jobId = startData.jobId as string;
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
