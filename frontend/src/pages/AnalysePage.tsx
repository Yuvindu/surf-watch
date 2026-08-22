import { useContext, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';
import Paper from '@mui/material/Paper';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import Slider from '@mui/material/Slider';
import TextField from '@mui/material/TextField';
import MenuItem from '@mui/material/MenuItem';
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
import { fetchSegmentationModels, type SegmentationModelOption } from '../services/comparisonApi';

export default function AnalysePage() {
  const navigate = useNavigate();
  const { uploadFile, error, handleFile, clearFile } = useFileHandler();
  const { status, currentStage, currentTask, result, caseResponse, error: analysisError, runPrediction, runComparison, reset } = useDemoPredict();
  const { history, addCase, updateCase } = useContext(AnalysisContext);
  const caseIdRef = useRef<string | null>(null);
  const [windowSize, setWindowSize] = useState(5);
  const [threshold, setThreshold] = useState(0.5);
  const [model, setModel] = useState('segformer');
  const [modelOptions, setModelOptions] = useState<SegmentationModelOption[]>([]);
  const [modelError, setModelError] = useState<string | null>(null);

  useEffect(() => {
    let isActive = true;
    fetchSegmentationModels()
      .then(data => {
        if (!isActive) return;
        setModelOptions(data.models);
        setModel(data.defaultModel);
        setModelError(null);
      })
      .catch(err => {
        if (!isActive) return;
        setModelError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      isActive = false;
    };
  }, []);

  const isProcessing = status === 'processing';
  const isComplete = status === 'complete';
  const hasCompletedAnalysis = isComplete && (result || caseResponse);
  const activeProcessingCase = history.find(c => c.status === 'processing');
  const isShowingPersistedProcessing = !isProcessing && !!activeProcessingCase;
  const displayedStage = isShowingPersistedProcessing ? activeProcessingCase.currentStage : currentStage;
  const displayedTask = isShowingPersistedProcessing ? activeProcessingCase.currentTask : currentTask;
  const displayedUploadType = uploadFile?.fileType ?? activeProcessingCase?.upload.fileType;
  const activeStages = displayedUploadType === 'video' ? COMPARISON_PIPELINE_STAGES : PIPELINE_STAGES;

  async function handleRun() {
    if (!uploadFile) return;
    const caseId = generateId();
    caseIdRef.current = caseId;
    addCase({
      id: caseId,
      upload: uploadFile,
      status: 'processing',
      currentStage: 'upload',
      currentTask: 'Uploading video to the local comparison API',
      prediction: null,
      createdAt: new Date().toISOString(),
      windowSize: uploadFile.fileType === 'video' ? windowSize : undefined,
      threshold: uploadFile.fileType === 'video' ? threshold : undefined,
      model: uploadFile.fileType === 'video' ? model : undefined,
    });
    if (uploadFile.fileType === 'video') {
      const response = await runComparison(uploadFile, { model, windowSize, threshold }, partial => {
        updateCase(caseId, partial);
      });
      if (response && caseIdRef.current) {
        const completedCaseId = caseIdRef.current;
        updateCase(completedCaseId, {
          ...normalizeCaseResponse(response, uploadFile),
          id: completedCaseId,
        });
        if (window.location.pathname.startsWith('/analyse')) {
          navigate(`/results/${completedCaseId}`);
        }
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
        {modelError && <Alert severity="error" sx={{ mb: 2 }}>{modelError}</Alert>}

        {!uploadFile && !isProcessing && !isComplete && !activeProcessingCase && (
          <FileUploader onFile={handleFile} disabled={false} />
        )}

        {uploadFile && !isProcessing && !isComplete && !activeProcessingCase && (
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
            {uploadFile.fileType === 'video' && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" fontWeight={700} mb={1}>
                  Comparison parameters
                </Typography>
                <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' } }}>
                  <Box sx={{ gridColumn: { sm: '1 / -1' } }}>
                    <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>
                      Segmentation model
                    </Typography>
                    <TextField
                      select
                      size="small"
                      fullWidth
                      value={model}
                      disabled={modelOptions.length === 0}
                      onChange={(event) => setModel(event.target.value)}
                    >
                      {modelOptions.map(option => (
                        <MenuItem key={option.id} value={option.id}>
                          {option.label}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>
                      Window size
                    </Typography>
                    <TextField
                      type="number"
                      size="small"
                      fullWidth
                      value={windowSize}
                      inputProps={{ min: 1, max: 31, step: 1 }}
                      onChange={(event) => {
                        const value = Number(event.target.value);
                        if (Number.isFinite(value)) {
                          setWindowSize(Math.min(31, Math.max(1, Math.round(value))));
                        }
                      }}
                    />
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>
                      Threshold: {threshold.toFixed(2)}
                    </Typography>
                    <Slider
                      value={threshold}
                      min={0}
                      max={1}
                      step={0.05}
                      valueLabelDisplay="auto"
                      onChange={(_event, value) => {
                        if (typeof value === 'number') setThreshold(value);
                      }}
                    />
                  </Box>
                </Box>
              </Box>
            )}
            <Button
              variant="contained"
              fullWidth
              startIcon={<PlayArrowIcon />}
              onClick={handleRun}
              disabled={uploadFile.fileType === 'video' && modelOptions.length === 0}
            >
              {uploadFile.fileType === 'video' ? 'Run Baseline vs MARSP Comparison' : 'Run Detection'}
            </Button>
          </Paper>
        )}

        {(isProcessing || activeProcessingCase) && (
          <Paper sx={{ p: { xs: 2.5, sm: 3 }, bgcolor: 'background.paper' }}>
            <Typography variant="subtitle1" fontWeight={700} mb={3}>
              Running MARSP pipeline…
            </Typography>
            {activeProcessingCase && !uploadFile && (
              <Typography variant="body2" color="text.secondary" mb={2} sx={{ wordBreak: 'break-all' }}>
                {activeProcessingCase.caseName ?? activeProcessingCase.upload.file.name}
              </Typography>
            )}
            <StatusStepper currentStage={displayedStage} currentTask={displayedTask} isComplete={false} stages={activeStages} />
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
