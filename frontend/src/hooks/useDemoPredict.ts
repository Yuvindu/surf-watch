import { useState, useCallback } from 'react';
import type { UploadFile } from '../models/upload';
import type { PredictionResult } from '../models/prediction';
import type { PipelineStage, AnalysisStatus } from '../models/analysis';
import { simulatePrediction } from '../services/mockApi';

export function useDemoPredict() {
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [currentStage, setCurrentStage] = useState<PipelineStage>('upload');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runPrediction = useCallback(async (uploadFile: UploadFile): Promise<PredictionResult | null> => {
    setStatus('processing');
    setResult(null);
    setError(null);
    try {
      const prediction = await simulatePrediction(uploadFile, (stage) => {
        setCurrentStage(stage);
      });
      setResult(prediction);
      setStatus('complete');
      return prediction;
    } catch (err) {
      setError(String(err));
      setStatus('error');
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    setStatus('idle');
    setCurrentStage('upload');
    setResult(null);
    setError(null);
  }, []);

  return { status, currentStage, result, error, runPrediction, reset };
}
