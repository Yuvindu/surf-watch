import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import Paper from '@mui/material/Paper';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Typography from '@mui/material/Typography';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import DownloadIcon from '@mui/icons-material/Download';
import type { AnalysisCase, ModelComparisonResult } from '../models/analysis';
import { formatDate } from '../utils/helpers';
import ProvenanceDetails from './ProvenanceDetails';

type MediaMode = 'side_by_side_video' | 'baseline_overlay' | 'marsp_overlay';

function modelLabel(model: string) {
  if (model === 'segformer') return 'SegFormer';
  if (model === 'unet-resnet34') return 'U-Net ResNet34';
  if (model === 'ensemble') return 'Equal-weight ensemble';
  return model;
}

function score(value: number | undefined) {
  return typeof value === 'number' ? value.toFixed(3) : '—';
}

function seconds(value: number | undefined) {
  return typeof value === 'number' ? `${value.toFixed(1)} s` : '—';
}

function MediaButtons({ result }: { result: ModelComparisonResult }) {
  const links = [
    ['Baseline mask', result.artifacts?.baseline_mask],
    ['MARSP mask', result.artifacts?.marsp_final_mask],
    ['Model metrics', result.metricsUrl],
  ] as const;
  return (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1.5 }}>
      {links.map(([label, url]) => url && (
        <Button key={label} size="small" startIcon={<DownloadIcon />} href={url} target="_blank" rel="noopener noreferrer">
          {label}
        </Button>
      ))}
    </Box>
  );
}

export default function MultiModelResults({ analysisCase }: { analysisCase: AnalysisCase }) {
  const navigate = useNavigate();
  const results = analysisCase.modelResults ?? [];
  const [activeModel, setActiveModel] = useState(
    analysisCase.threeWayVideoUrl ? 'three-way' : results.find(result => result.status === 'success')?.model ?? results[0]?.model ?? ''
  );
  const [mediaMode, setMediaMode] = useState<MediaMode>('side_by_side_video');
  const selected = results.find(result => result.model === activeModel);
  const videoUrl = activeModel === 'three-way' ? analysisCase.threeWayVideoUrl : selected?.artifacts?.[mediaMode];
  const failures = results.filter(result => result.status === 'failed');

  return (
    <Box sx={{ px: { xs: 2, sm: 3 }, py: { xs: 4, md: 6 }, maxWidth: 1160, mx: 'auto' }}>
      <Button startIcon={<ArrowBackIcon />} onClick={() => navigate(-1)} sx={{ mb: 2, color: 'text.secondary', px: 0 }}>
        Back
      </Button>
      <Typography variant="h4" fontWeight={800} sx={{ fontSize: { xs: '1.75rem', md: '2.125rem' } }}>
        Model comparison
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, mb: 2, overflowWrap: 'anywhere' }}>
        {analysisCase.caseName ?? analysisCase.upload.file.name} · {formatDate(analysisCase.createdAt)}
      </Typography>
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 3 }}>
        {analysisCase.windowSize !== undefined && <Chip size="small" label={`Window ${analysisCase.windowSize}`} />}
        {analysisCase.threshold !== undefined && <Chip size="small" label={`Threshold ${analysisCase.threshold.toFixed(2)}`} />}
        {analysisCase.totalRuntimeSeconds !== undefined && <Chip size="small" label={`Run ${seconds(analysisCase.totalRuntimeSeconds)}`} />}
      </Box>

      {failures.map(result => (
        <Alert key={result.model} severity="warning" sx={{ mb: 2 }}>
          {modelLabel(result.model)} failed: {result.error ?? 'No result was generated.'}
        </Alert>
      ))}

      <Box component="section">
        <Tabs
          value={activeModel}
          onChange={(_event, value: string) => { setActiveModel(value); setMediaMode('side_by_side_video'); }}
          variant="scrollable"
          scrollButtons="auto"
          aria-label="Model results"
          sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
        >
          {analysisCase.threeWayVideoUrl && <Tab value="three-way" label="Three-way MARSP" />}
          {results.map(result => (
            <Tab key={result.model} value={result.model} label={`${modelLabel(result.model)}${result.status === 'failed' ? ' · failed' : ''}`} />
          ))}
        </Tabs>
        {activeModel === 'three-way' && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            SegFormer, U-Net ResNet34, and the equal-weight probability ensemble on the same frames. The ensemble combines model probabilities before MARSP processing.
          </Typography>
        )}
        {selected?.status === 'success' && (
          <>
            <ToggleButtonGroup
              exclusive
              size="small"
              value={mediaMode}
              onChange={(_event, value: MediaMode | null) => { if (value) setMediaMode(value); }}
              aria-label="Video view"
              sx={{ mb: 2, maxWidth: '100%', flexWrap: 'wrap' }}
            >
              <ToggleButton value="side_by_side_video">Comparison</ToggleButton>
              <ToggleButton value="baseline_overlay">Baseline</ToggleButton>
              <ToggleButton value="marsp_overlay">MARSP</ToggleButton>
            </ToggleButtonGroup>
          </>
        )}
        {videoUrl ? (
          <Paper sx={{ overflow: 'hidden', bgcolor: '#000', borderRadius: 1 }}>
            <video key={`${activeModel}-${mediaMode}`} src={videoUrl} controls playsInline style={{ width: '100%', aspectRatio: activeModel === 'three-way' ? '16 / 3' : '16 / 9', objectFit: 'contain', display: 'block' }} />
          </Paper>
        ) : (
          <Alert severity="info">This video artifact is unavailable.</Alert>
        )}
        {selected?.status === 'success' && <MediaButtons result={selected} />}
      </Box>

      <Box component="section" sx={{ mt: 4 }}>
        <Divider sx={{ mb: 2 }} />
        <Typography variant="subtitle1" fontWeight={700} mb={1}>Across models</Typography>
        <Typography variant="body2" color="text.secondary" mb={2}>
          Temporal consistency on this uploaded video; these are not ground-truth segmentation accuracy scores.
        </Typography>
        <Box sx={{ overflowX: 'auto' }}>
          <Table size="small" sx={{ minWidth: 680 }}>
            <TableHead>
              <TableRow>
                <TableCell>Model</TableCell>
                <TableCell align="right">Frames</TableCell>
                <TableCell align="right">Baseline consecutive IoU</TableCell>
                <TableCell align="right">MARSP consecutive IoU</TableCell>
                <TableCell align="right">IoU change</TableCell>
                <TableCell align="right">MARSP time</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {results.map(result => (
                <TableRow key={result.model} selected={result.model === activeModel}>
                  <TableCell>{modelLabel(result.model)}</TableCell>
                  <TableCell align="right">{result.frameCount ?? '—'}</TableCell>
                  <TableCell align="right">{score(result.baseline?.mean_consecutive_iou)}</TableCell>
                  <TableCell align="right">{score(result.marsp?.mean_consecutive_iou)}</TableCell>
                  <TableCell align="right">{score(result.comparison?.delta_mean_consecutive_iou)}</TableCell>
                  <TableCell align="right">{seconds(result.timingSeconds?.marsp)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      </Box>

      {selected?.status === 'success' && selected.metricTable && selected.metricTable.length > 0 && (
        <Box component="section" sx={{ mt: 4 }}>
          <Typography variant="subtitle1" fontWeight={700} mb={1.5}>{modelLabel(selected.model)} metrics</Typography>
          <Box sx={{ overflowX: 'auto' }}>
            <Table size="small" sx={{ minWidth: 600 }}>
              <TableHead>
                <TableRow>
                  <TableCell>Metric</TableCell>
                  <TableCell align="right">Baseline</TableCell>
                  <TableCell align="right">MARSP</TableCell>
                  <TableCell align="right">Change</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {selected.metricTable.map(metric => (
                  <TableRow key={metric.metric}>
                    <TableCell>{metric.metric.replace(/_/g, ' ')}</TableCell>
                    <TableCell align="right">{score(metric.baseline)}</TableCell>
                    <TableCell align="right">{score(metric.marsp)}</TableCell>
                    <TableCell align="right">{score(metric.delta)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        </Box>
      )}

      <ProvenanceDetails
        provenance={selected?.provenance}
        input={analysisCase.inputProvenance}
        summaryJsonUrl={analysisCase.summaryJsonUrl}
        summaryCsvUrl={analysisCase.summaryCsvUrl}
      />
    </Box>
  );
}
