import { useContext } from 'react';
import { AnalysisContext } from '../context/AnalysisContext';

export function useAnalysisHistory() {
  return useContext(AnalysisContext);
}
