import React, { createContext, useState, useCallback } from 'react';
import type { AnalysisCase } from '../models/analysis';

interface AnalysisContextValue {
  history: AnalysisCase[];
  addCase: (c: AnalysisCase) => void;
  updateCase: (id: string, partial: Partial<AnalysisCase>) => void;
}

export const AnalysisContext = createContext<AnalysisContextValue>({
  history: [],
  addCase: () => undefined,
  updateCase: () => undefined,
});

export function AnalysisProvider({ children }: { children: React.ReactNode }) {
  const [history, setHistory] = useState<AnalysisCase[]>([]);

  const addCase = useCallback((c: AnalysisCase) => {
    setHistory(prev => [c, ...prev]);
  }, []);

  const updateCase = useCallback((id: string, partial: Partial<AnalysisCase>) => {
    setHistory(prev => prev.map(c => (c.id === id ? { ...c, ...partial } : c)));
  }, []);

  return (
    <AnalysisContext.Provider value={{ history, addCase, updateCase }}>
      {children}
    </AnalysisContext.Provider>
  );
}
