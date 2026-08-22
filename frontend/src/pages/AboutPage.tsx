import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';
import Divider from '@mui/material/Divider';

const PIPELINE_STAGES = [
  {
    step: '01',
    label: 'Upload',
    desc: 'A beach image or video is submitted through the drag-and-drop interface.',
  },
  {
    step: '02',
    label: 'Frame Extraction',
    desc: 'Videos are decoded into individual frames. Images produce a single frame.',
  },
  {
    step: '03',
    label: 'Motion Compensation',
    desc: 'Camera motion is estimated and corrected to isolate genuine water movement from shake.',
  },
  {
    step: '04',
    label: 'Segmentation',
    desc: 'A deep segmentation model runs on each frame and predicts rip current pixel regions.',
  },
  {
    step: '05',
    label: 'Temporal Aggregation',
    desc: 'Frame-level masks are aggregated over time to produce stable, noise-reduced predictions.',
  },
  {
    step: '06',
    label: 'Output',
    desc: 'Confidence scores, bounding regions, and overlay visualisations are returned to the user.',
  },
];

const STATS = [
  { value: '80%+', label: 'of rescues involve rip currents' },
  { value: '100+', label: 'km/h peak rip current speed' },
  { value: '~30m', label: 'typical rip channel width' },
];

const TECH = [
  { label: 'Frontend', value: 'React 18 + TypeScript' },
  { label: 'UI Library', value: 'Material UI v5' },
  { label: 'Build Tool', value: 'Vite' },
  { label: 'Routing', value: 'React Router v6' },
  { label: 'Mode', value: 'Demo — predictions simulated' },
];

export default function AboutPage() {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', px: { xs: 2, sm: 3 }, py: { xs: 4, md: 6 } }}>
      <Box sx={{ width: '100%', maxWidth: 900 }}>

        {/* Page header */}
        <Typography variant="overline" sx={{ letterSpacing: 4, color: 'text.secondary' }}>
          COMP6002 Research
        </Typography>
        <Typography variant="h4" fontWeight={800} mt={0.5} mb={1}>
          About SurfWatch
        </Typography>
        <Typography variant="body1" color="text.secondary" mb={5} sx={{ maxWidth: 560 }}>
          SurfWatch is a demo application exploring ML-based rip current detection in beach video and imagery.
        </Typography>

        {/* Rip current explainer */}
        <Paper sx={{ p: 4, mb: 3, borderRadius: 3 }}>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography variant="overline" sx={{ letterSpacing: 3, color: 'text.secondary' }}>
                The Hazard
              </Typography>
              <Typography variant="h5" fontWeight={700} mt={0.5} mb={2}>
                What is a rip current?
              </Typography>
              <Typography variant="body2" color="text.secondary" lineHeight={1.9}>
                A rip current is a powerful, narrow channel of fast-moving water flowing away from shore.
                They form when waves pile water against a beach and it returns seaward through the path
                of least resistance — typically a gap between sandbars or near structures.
              </Typography>
              <Typography variant="body2" color="text.secondary" lineHeight={1.9} mt={1.5}>
                Rip currents are the leading cause of beach rescues globally. They are difficult to
                spot with the naked eye, making automated detection an important safety tool.
              </Typography>
            </Grid>

            {/* Stats column */}
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {STATS.map(s => (
                  <Box
                    key={s.label}
                    sx={{
                      p: 2.5,
                      borderRadius: 2,
                      border: '1px solid',
                      borderColor: 'divider',
                      display: 'flex',
                      alignItems: 'baseline',
                      gap: 2,
                    }}
                  >
                    <Typography variant="h4" fontWeight={800} sx={{ lineHeight: 1, minWidth: 80 }}>
                      {s.value}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {s.label}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Pipeline */}
        <Paper sx={{ p: 4, mb: 3, borderRadius: 3 }}>
          <Typography variant="overline" sx={{ letterSpacing: 3, color: 'text.secondary' }}>
            Architecture
          </Typography>
          <Typography variant="h5" fontWeight={700} mt={0.5} mb={3}>
            The MARSP Pipeline
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={4} sx={{ maxWidth: 520 }}>
            MARSP — Motion-Aware Rip-current Segmentation Pipeline — is the processing chain
            SurfWatch simulates when you run a detection.
          </Typography>

          <Grid container spacing={2}>
            {PIPELINE_STAGES.map(stage => (
              <Grid item xs={12} sm={6} key={stage.step}>
                <Box
                  sx={{
                    p: 2.5,
                    borderRadius: 2,
                    border: '1px solid',
                    borderColor: 'divider',
                    height: '100%',
                  }}
                >
                  <Typography
                    variant="h5"
                    fontWeight={800}
                    sx={{ color: 'divider', lineHeight: 1, mb: 1 }}
                  >
                    {stage.step}
                  </Typography>
                  <Typography variant="subtitle2" fontWeight={700} mb={0.5}>
                    {stage.label}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {stage.desc}
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Paper>

        {/* Tech stack + credits */}
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6}>
            <Paper sx={{ p: 3, borderRadius: 3, height: '100%' }}>
              <Typography variant="overline" sx={{ letterSpacing: 3, color: 'text.secondary' }}>
                Stack
              </Typography>
              <Typography variant="h6" fontWeight={700} mt={0.5} mb={2}>
                Built with
              </Typography>
              {TECH.map((t, i) => (
                <Box key={t.label}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1 }}>
                    <Typography variant="body2" color="text.secondary">{t.label}</Typography>
                    <Typography variant="body2" fontWeight={600}>{t.value}</Typography>
                  </Box>
                  {i < TECH.length - 1 && <Divider />}
                </Box>
              ))}
            </Paper>
          </Grid>

          <Grid item xs={12} sm={6}>
            <Paper sx={{ p: 3, borderRadius: 3, height: '100%' }}>
              <Typography variant="overline" sx={{ letterSpacing: 3, color: 'text.secondary' }}>
                Project
              </Typography>
              <Typography variant="h6" fontWeight={700} mt={0.5} mb={2}>
                Credits
              </Typography>
              <Typography variant="body2" color="text.secondary" lineHeight={2}>
                COMP6002 — SurfWatch Research Group
                <br />
                Curtin University
              </Typography>
              <Divider sx={{ my: 2 }} />
              <Typography variant="caption" color="text.secondary">
                This is a demonstration application. All detections, confidence scores, and
                region masks are generated with mock data and do not reflect real model output.
              </Typography>
            </Paper>
          </Grid>
        </Grid>

      </Box>
    </Box>
  );
}
