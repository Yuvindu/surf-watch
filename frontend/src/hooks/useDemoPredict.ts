import { useState, useCallback } from 'react';
import type { UploadFile } from '../models/upload';
import type { PredictionResult } from '../models/prediction';
import type { CaseResponse, PipelineStage, AnalysisStatus, AnalysisCase } from '../models/analysis';
import { simulatePrediction } from '../services/mockApi';
import { runBaselineMarspComparison, type ComparisonRunOptions } from '../services/comparisonApi';

export function useDemoPredict() {
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [currentStage, setCurrentStage] = useState<PipelineStage>('upload');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [caseResponse, setCaseResponse] = useState<CaseResponse | null>(null);
  const [currentTask, setCurrentTask] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runPrediction = useCallback(async (uploadFile: UploadFile): Promise<PredictionResult | null> => {
    setStatus('processing');
    setResult(null);
    setCurrentTask(null);
    setError(null);
    try {
      const prediction = await simulatePrediction(uploadFile, (stage) => {
          setCurrentStage(stage);
          setCurrentTask(null);
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

  const runComparison = useCallback(async (
    uploadFile: UploadFile,
    options: ComparisonRunOptions,
    onCaseProgress?: (partial: Partial<AnalysisCase>) => void,
  ): Promise<CaseResponse | null> => {
    setStatus('processing');
    setResult(null);
    setCaseResponse(null);
    setCurrentTask(null);
    setError(null);
    try {
      const response = await runBaselineMarspComparison(
        uploadFile,
        options,
        setCurrentStage,
        setCurrentTask,
        status => onCaseProgress?.({
          jobId: status.jobId,
          currentStage: status.currentStage,
          currentTask: status.currentTask,
          updatedAt: status.updatedAt,
          windowSize: status.windowSize,
          threshold: status.threshold,
        }),
        status => onCaseProgress?.({
          status: status.status === 'completed' ? 'complete' : status.status,
          currentStage: status.currentStage,
          currentTask: status.currentTask,
          updatedAt: status.updatedAt,
          windowSize: status.windowSize,
          threshold: status.threshold,
        }),
      );
      setCaseResponse(response);
      setStatus('complete');
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus('error');
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    setStatus('idle');
    setCurrentStage('upload');
    setResult(null);
    setCaseResponse(null);
    setCurrentTask(null);
    setError(null);
  }, []);

  return { status, currentStage, currentTask, result, caseResponse, error, runPrediction, runComparison, reset };
}
