import { useContext } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';
import Divider from '@mui/material/Divider';
import Chip from '@mui/material/Chip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import WaterDropIcon from '@mui/icons-material/WaterDrop';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import LayersIcon from '@mui/icons-material/Layers';
import { AnalysisContext } from '../context/AnalysisContext';
import ConfidenceGauge from '../components/ConfidenceGauge';
import HeatmapOverlay from '../components/HeatmapOverlay';
import VideoPlayer from '../components/VideoPlayer';
import { formatDuration, formatDate } from '../utils/helpers';

export default function ResultsPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { history } = useContext(AnalysisContext);
  const analysisCase = history.find(c => c.id === id);

  if (!analysisCase) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 12, px: 2 }}>
        <Typography variant="h6" color="text.secondary" mb={2}>Result not found.</Typography>
        <Button onClick={() => navigate('/history')} startIcon={<ArrowBackIcon />}>Back to History</Button>
      </Box>
    );
  }

  const {
    upload,
    prediction,
    overlayUrl,
    confidenceScore,
    summaryLabel,
    caseName,
    comparisonVideoUrl,
    baselineOverlayUrl,
    marspOverlayUrl,
    metricTable,
    windowSize,
    threshold,
    model,
  } = analysisCase;

  // Use backend-provided confidence if available, otherwise fall back to mock prediction
  const displayConfidence = confidenceScore ?? prediction?.averageConfidence ?? 0;
  const displayName = caseName ?? upload.file.name;

  // Backend pre-renders overlay; only compute frame stats when using local mock data
  const frame0 = prediction?.frames[0] ?? null;
  const totalRegions = frame0?.regions.length ?? 0;
  const avgCoverage = frame0
    ? frame0.regions.reduce((s, r) => s + r.pixelCoverage, 0) / Math.max(frame0.regions.length, 1)
    : 0;

  if (!overlayUrl && !prediction) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 12, px: 2 }}>
        <Typography variant="h6" color="text.secondary" mb={2}>No prediction data available.</Typography>
        <Button onClick={() => navigate(-1)} startIcon={<ArrowBackIcon />}>Go back</Button>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', px: { xs: 2, sm: 3 }, py: { xs: 4, md: 6 } }}>
      <Box sx={{ width: '100%', maxWidth: 960 }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate(-1)}
          sx={{ mb: 2, color: 'text.secondary', px: 0 }}
        >
          Back
        </Button>

        <Typography
          variant="h4"
          fontWeight={800}
          mb={0.5}
          sx={{ fontSize: { xs: '1.75rem', md: '2.125rem' } }}
        >
          Results
        </Typography>
        <Typography
          variant="body2"
          color="text.secondary"
          mb={3}
          sx={{ wordBreak: 'break-all' }}
        >
          {displayName} &nbsp;·&nbsp; {formatDate(analysisCase.createdAt)}
        </Typography>

        <Grid container spacing={3}>
          {/* Media + overlay */}
          <Grid item xs={12} md={7}>
            <Paper sx={{ p: { xs: 1.5, sm: 2 }, bgcolor: 'background.paper' }}>
              {/* If the backend provided a pre-rendered comparison video, show it directly */}
              {comparisonVideoUrl || overlayUrl ? (
                <video
                  src={comparisonVideoUrl ?? overlayUrl}
                  controls
                  style={{ width: '100%', borderRadius: 4, display: 'block' }}
                />
              ) : upload.fileType === 'video' && prediction ? (
                <VideoPlayer previewUrl={upload.previewUrl} prediction={prediction} />
              ) : frame0 ? (
                <HeatmapOverlay
                  previewUrl={upload.previewUrl}
                  frame={frame0}
                  isVideo={false}
                />
              ) : (
                <video
                  src={upload.previewUrl}
                  controls
                  style={{ width: '100%', borderRadius: 4, display: 'block' }}
                />
              )}
            </Paper>

            {(baselineOverlayUrl || marspOverlayUrl) && (
              <Grid container spacing={2} sx={{ mt: 0 }}>
                {baselineOverlayUrl && (
                  <Grid item xs={12} sm={6}>
                    <Paper sx={{ p: 1.25, bgcolor: 'background.paper' }}>
                      <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                        Baseline overlay
                      </Typography>
                      <video
                        src={baselineOverlayUrl}
                        controls
                        style={{ width: '100%', borderRadius: 4, display: 'block' }}
                      />
                    </Paper>
                  </Grid>
                )}
                {marspOverlayUrl && (
                  <Grid item xs={12} sm={6}>
                    <Paper sx={{ p: 1.25, bgcolor: 'background.paper' }}>
                      <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                        MARSP overlay
                      </Typography>
                      <video
                        src={marspOverlayUrl}
                        controls
                        style={{ width: '100%', borderRadius: 4, display: 'block' }}
                      />
                    </Paper>
                  </Grid>
                )}
              </Grid>
            )}
          </Grid>

          {/* Stats panel */}
          <Grid item xs={12} md={5}>
            <Paper sx={{ p: { xs: 2.5, sm: 3 }, bgcolor: 'background.paper' }}>
              <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
                <ConfidenceGauge value={displayConfidence} size={110} />
              </Box>
              <Divider sx={{ mb: 2 }} />

              {/* summaryLabel from backend */}
              {summaryLabel && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1.5 }}>
                  <WaterDropIcon fontSize="small" sx={{ color: 'text.secondary' }} />
                  <Box>
                    <Typography variant="caption" color="text.secondary">Detection summary</Typography>
                    <Typography variant="subtitle2" fontWeight={700}>{summaryLabel}</Typography>
                  </Box>
                </Box>
              )}

              {/* Frame-level stats — only available with local mock data */}
              {(model !== undefined || windowSize !== undefined || threshold !== undefined) && (
                <>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                    Parameters
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                    {model !== undefined && (
                      <Chip label={`Model ${model}`} size="small" sx={{ fontSize: '0.65rem', height: 20 }} />
                    )}
                    {windowSize !== undefined && (
                      <Chip label={`Window ${windowSize}`} size="small" sx={{ fontSize: '0.65rem', height: 20 }} />
                    )}
                    {threshold !== undefined && (
                      <Chip label={`Threshold ${threshold.toFixed(2)}`} size="small" sx={{ fontSize: '0.65rem', height: 20 }} />
                    )}
                  </Box>
                </>
              )}

              {/* Frame-level stats — only available with local mock data */}
              {prediction && [
                {
                  icon: <WaterDropIcon fontSize="small" sx={{ color: 'text.secondary' }} />,
                  label: 'Rip regions detected',
                  value: totalRegions.toString(),
                },
                {
                  icon: <LayersIcon fontSize="small" sx={{ color: 'text.secondary' }} />,
                  label: 'Avg pixel coverage',
                  value: `${avgCoverage.toFixed(1)}%`,
                },
                {
                  icon: <AccessTimeIcon fontSize="small" sx={{ color: 'text.secondary' }} />,
                  label: 'Processing time',
                  value: formatDuration(prediction.processingTimeMs),
                },
              ].map(stat => (
                <Box key={stat.label} sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1.5 }}>
                  {stat.icon}
                  <Box>
                    <Typography variant="caption" color="text.secondary">{stat.label}</Typography>
                    <Typography variant="subtitle2" fontWeight={700}>{stat.value}</Typography>
                  </Box>
                </Box>
              ))}

              {prediction && (
                <>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                    Frames analysed
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {prediction.frames.map((_f, i) => (
                      <Chip
                        key={i}
                        label={`F${i}`}
                        size="small"
                        sx={{ fontSize: '0.6rem', height: 18 }}
                      />
                    ))}
                  </Box>
                </>
              )}
            </Paper>
          </Grid>
        </Grid>

        {metricTable && metricTable.length > 0 && (
          <Paper sx={{ mt: 3, p: { xs: 1.5, sm: 2 }, bgcolor: 'background.paper', overflowX: 'auto' }}>
            <Typography variant="subtitle1" fontWeight={700} mb={1.5}>
              Baseline vs MARSP Metrics
            </Typography>
            <Table size="small" sx={{ minWidth: 620 }}>
              <TableHead>
                <TableRow>
                  <TableCell>Metric</TableCell>
                  <TableCell align="right">Baseline</TableCell>
                  <TableCell align="right">MARSP</TableCell>
                  <TableCell align="right">Delta</TableCell>
                  <TableCell>Preferred</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {metricTable.map(metric => (
                  <TableRow key={metric.metric}>
                    <TableCell>{metric.metric.replace(/_/g, ' ')}</TableCell>
                    <TableCell align="right">{metric.baseline}</TableCell>
                    <TableCell align="right">{metric.marsp}</TableCell>
                    <TableCell align="right">{metric.delta}</TableCell>
                    <TableCell>{metric.preferred_direction}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Paper>
        )}
      </Box>
    </Box>
  );
}
