import type { UploadFile } from '../models/upload';
import type { CaseResponse, PipelineStage } from '../models/analysis';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

function wait(ms: number) {
  return new Promise<void>(resolve => setTimeout(resolve, ms));
}

async function animateStages(
  onStageChange: (stage: PipelineStage) => void,
  isComplete: () => boolean,
) {
  const stages: PipelineStage[] = [
    'upload',
    'frame_extraction',
    'motion_compensation',
    'segmentation',
    'temporal_aggregation',
  ];

  for (const stage of stages) {
    if (isComplete()) return;
    onStageChange(stage);
    await wait(stage === 'upload' ? 700 : 1800);
  }
}

export async function runBaselineMarspComparison(
  uploadFile: UploadFile,
  onStageChange: (stage: PipelineStage) => void,
): Promise<CaseResponse> {
  let complete = false;
  const stageAnimation = animateStages(onStageChange, () => complete);

  const form = new FormData();
  form.append('file', uploadFile.file);

  try {
    const response = await fetch(`${API_BASE_URL}/api/comparisons`, {
      method: 'POST',
      body: form,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error ?? 'Comparison pipeline failed.');
    }

    complete = true;
    onStageChange('complete');
    await stageAnimation;
    return data as CaseResponse;
  } catch (error) {
    complete = true;
    await stageAnimation;
    throw error;
  }
}
