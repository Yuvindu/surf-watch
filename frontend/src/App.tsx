import { useState, useMemo } from 'react';
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { getTheme } from './theme/theme';
import { AnalysisProvider } from './context/AnalysisContext';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import AnalysePage from './pages/AnalysePage';
import ResultsPage from './pages/ResultsPage';
import HistoryPage from './pages/HistoryPage';
import AboutPage from './pages/AboutPage';

export default function App() {
  const [mode, setMode] = useState<'light' | 'dark'>('dark');
  const theme = useMemo(() => getTheme(mode), [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AnalysisProvider>
        <Layout mode={mode} onToggleTheme={() => setMode(m => m === 'light' ? 'dark' : 'light')}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/analyse" element={<AnalysePage />} />
            <Route path="/results/:id" element={<ResultsPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </Layout>
      </AnalysisProvider>
    </ThemeProvider>
  );
}
