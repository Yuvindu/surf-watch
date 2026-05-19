import { useContext, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';
import Paper from '@mui/material/Paper';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import DeleteIcon from '@mui/icons-material/Delete';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import FileUploader from '../components/FileUploader';
import StatusStepper from '../components/StatusStepper';
import { useFileHandler } from '../hooks/useFileHandler';
import { useDemoPredict } from '../hooks/useDemoPredict';
import { AnalysisContext } from '../context/AnalysisContext';
import { normalizeCaseResponse } from '../models/analysis';
import { COMPARISON_PIPELINE_STAGES, PIPELINE_STAGES } from '../services/mockApi';
import { formatFileSize, generateId } from '../utils/helpers';

export default function AnalysePage() {
  const navigate = useNavigate();
  const { uploadFile, error, handleFile, clearFile } = useFileHandler();
  const { status, currentStage, currentTask, result, caseResponse, error: analysisError, runPrediction, runComparison, reset } = useDemoPredict();
  const { addCase, updateCase } = useContext(AnalysisContext);
  const caseIdRef = useRef<string | null>(null);

  const isProcessing = status === 'processing';
  const isComplete = status === 'complete';
  const hasCompletedAnalysis = isComplete && (result || caseResponse);
  const activeStages = uploadFile?.fileType === 'video' ? COMPARISON_PIPELINE_STAGES : PIPELINE_STAGES;

  async function handleRun() {
    if (!uploadFile) return;
    const caseId = generateId();
    caseIdRef.current = caseId;
    addCase({
      id: caseId,
      upload: uploadFile,
      status: 'processing',
      currentStage: 'upload',
      prediction: null,
      createdAt: new Date().toISOString(),
    });
    if (uploadFile.fileType === 'video') {
      const response = await runComparison(uploadFile);
      if (response && caseIdRef.current) {
        const completedCaseId = caseIdRef.current;
        updateCase(completedCaseId, {
          ...normalizeCaseResponse(response, uploadFile),
          id: completedCaseId,
        });
        navigate(`/results/${completedCaseId}`);
      } else if (caseIdRef.current) {
        updateCase(caseIdRef.current, { status: 'error' });
      }
      return;
    }

    const prediction = await runPrediction(uploadFile);
    if (prediction && caseIdRef.current) {
      updateCase(caseIdRef.current, { status: 'complete', prediction });
    } else if (caseIdRef.current) {
      updateCase(caseIdRef.current, { status: 'error' });
    }
  }

  function handleViewResults() {
    if (caseIdRef.current) navigate(`/results/${caseIdRef.current}`);
  }

  function handleReset() {
    reset();
    clearFile();
    caseIdRef.current = null;
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', px: { xs: 2, sm: 3 }, py: { xs: 4, md: 6 } }}>
      <Box sx={{ width: '100%', maxWidth: 700 }}>
        <Typography variant="h4" fontWeight={800} mb={0.5} sx={{ fontSize: { xs: '1.75rem', md: '2.125rem' } }}>
          Analyse
        </Typography>
        <Typography variant="body2" color="text.secondary" mb={4}>
          Upload a beach image or video to run rip current detection.
        </Typography>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        {analysisError && <Alert severity="error" sx={{ mb: 2 }}>{analysisError}</Alert>}

        {!uploadFile && !isProcessing && !isComplete && (
          <FileUploader onFile={handleFile} disabled={false} />
        )}

        {uploadFile && !isProcessing && !isComplete && (
          <Paper sx={{ p: { xs: 2, sm: 2.5 }, mb: 3, bgcolor: 'background.paper' }}>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start', mb: 2 }}>
              {/* Thumbnail */}
              <Box sx={{ flexShrink: 0 }}>
                {uploadFile.fileType === 'image' ? (
                  <img
                    src={uploadFile.previewUrl}
                    alt="preview"
                    style={{ width: 72, height: 54, objectFit: 'cover', borderRadius: 8, display: 'block' }}
                  />
                ) : (
                  <video
                    src={uploadFile.previewUrl}
                    style={{ width: 72, height: 54, objectFit: 'cover', borderRadius: 8, display: 'block' }}
                    muted
                  />
                )}
              </Box>
              {/* File info */}
              <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                <Typography variant="subtitle2" fontWeight={600} sx={{ wordBreak: 'break-all', lineHeight: 1.3, mb: 0.5 }}>
                  {uploadFile.file.name}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  <Chip label={uploadFile.fileType} size="small" color="primary" sx={{ fontSize: '0.65rem', height: 18 }} />
                  <Chip label={formatFileSize(uploadFile.file.size)} size="small" sx={{ fontSize: '0.65rem', height: 18 }} />
                </Box>
              </Box>
              {/* Remove — icon-only on xs */}
              <Button
                variant="outlined"
                size="small"
                onClick={clearFile}
                color="inherit"
                sx={{
                  flexShrink: 0,
                  minWidth: { xs: 36, sm: 'auto' },
                  px: { xs: 1, sm: 1.5 },
                }}
              >
                <DeleteIcon fontSize="small" sx={{ display: { xs: 'block', sm: 'none' } }} />
                <Box component="span" sx={{ display: { xs: 'none', sm: 'inline' } }}>Remove</Box>
              </Button>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <Button
              variant="contained"
              fullWidth
              startIcon={<PlayArrowIcon />}
              onClick={handleRun}
            >
              {uploadFile.fileType === 'video' ? 'Run Baseline vs MARSP Comparison' : 'Run Detection'}
            </Button>
          </Paper>
        )}

        {isProcessing && (
          <Paper sx={{ p: { xs: 2.5, sm: 3 }, bgcolor: 'background.paper' }}>
            <Typography variant="subtitle1" fontWeight={700} mb={3}>
              Running MARSP pipeline…
            </Typography>
            <StatusStepper currentStage={currentStage} currentTask={currentTask} isComplete={false} stages={activeStages} />
          </Paper>
        )}

        {hasCompletedAnalysis && (
          <Paper sx={{ p: { xs: 2.5, sm: 3 }, bgcolor: 'background.paper' }}>
            <Typography variant="subtitle1" fontWeight={700} mb={3}>
              Analysis complete
            </Typography>
            <StatusStepper currentStage="complete" isComplete stages={activeStages} />
            <Box sx={{ display: 'flex', gap: 2, mt: 3, flexWrap: 'wrap' }}>
              <Button variant="contained" onClick={handleViewResults} sx={{ flexGrow: { xs: 1, sm: 0 } }}>
                View Results
              </Button>
              <Button variant="outlined" onClick={handleReset} sx={{ flexGrow: { xs: 1, sm: 0 } }}>
                New Analysis
              </Button>
            </Box>
          </Paper>
        )}
      </Box>
    </Box>
  );
}
