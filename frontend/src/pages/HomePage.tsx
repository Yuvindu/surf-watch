import { useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import WaveBackground from '../components/WaveBackground';

const BUBBLE_STYLE = `
  @keyframes float-up {
    0%   { transform: translateY(0) scale(1); opacity: 0.08; }
    100% { transform: translateY(-120px) scale(1.1); opacity: 0; }
  }
  @media (prefers-reduced-motion: reduce) {
    .hero-bubble { animation: none !important; }
  }
`;

const FEATURES = [
  {
    number: '01',
    title: 'Detection',
    desc: 'ML-driven segmentation pinpoints rip current regions in beach imagery with spatial precision.',
  },
  {
    number: '02',
    title: 'Confidence Scoring',
    desc: 'Per-region and overall confidence scores give you a clear signal on detection reliability.',
  },
  {
    number: '03',
    title: 'Visual Overlay',
    desc: 'Detected rip zones are drawn directly over your image or video frame with a toggle overlay.',
  },
  {
    number: '04',
    title: 'Session History',
    desc: 'Every analysis in the current session is saved and accessible for comparison and review.',
  },
];

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <style>{BUBBLE_STYLE}</style>

      {/* Hero */}
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          minHeight: { xs: '80vh', md: '85vh' },
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
          px: { xs: 2, sm: 4 },
          pb: { xs: 8, md: 10 },
        }}
      >
        {/* floating bubbles */}
        {[...Array(8)].map((_, i) => (
          <Box
            key={i}
            className="hero-bubble"
            sx={{
              position: 'absolute',
              bottom: `${10 + (i * 11) % 30}%`,
              left: `${(i * 13 + 5) % 90}%`,
              width: 8 + (i % 4) * 6,
              height: 8 + (i % 4) * 6,
              borderRadius: '50%',
              bgcolor: 'text.primary',
              opacity: 0.07,
              animation: `float-up ${5 + i * 0.7}s ease-in-out ${i * 0.4}s infinite`,
              pointerEvents: 'none',
            }}
          />
        ))}

        <Box sx={{ textAlign: 'center', maxWidth: 640, zIndex: 1, px: 1 }}>
          <Typography
            variant="overline"
            sx={{ letterSpacing: 4, color: 'text.secondary', fontWeight: 500, display: 'block', mb: 2 }}
          >
            COMP6002 Research Project
          </Typography>
          <Typography
            variant="h2"
            fontWeight={800}
            sx={{
              letterSpacing: { xs: '-1px', md: '-1.5px' },
              fontSize: { xs: '2.2rem', sm: '3rem', md: '3.75rem' },
              mb: 2,
              color: 'text.primary',
            }}
          >
            SurfWatch
          </Typography>
          <Typography
            variant="h6"
            color="text.secondary"
            fontWeight={400}
            sx={{ mb: 5, lineHeight: 1.6, fontSize: { xs: '1rem', sm: '1.25rem' } }}
          >
            ML-powered rip current detection for safer beaches.
          </Typography>
          <Button
            variant="contained"
            size="large"
            onClick={() => navigate('/analyse')}
            sx={{ px: { xs: 4, md: 5 }, py: 1.5, fontSize: '1rem' }}
          >
            Start Analysis
          </Button>
        </Box>

        <WaveBackground height={140} />
      </Box>

      {/* Features */}
      <Box sx={{ width: '100%', maxWidth: 960, px: { xs: 2, sm: 3 }, pb: { xs: 6, md: 10 } }}>
        <Typography
          variant="overline"
          sx={{ letterSpacing: 4, color: 'text.secondary', display: 'block', mb: 1 }}
        >
          Capabilities
        </Typography>
        <Typography variant="h5" fontWeight={700} mb={4}>
          What SurfWatch does
        </Typography>
        <Grid container spacing={2.5}>
          {FEATURES.map(f => (
            <Grid item xs={12} sm={6} key={f.number}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ p: { xs: 2.5, sm: 3 } }}>
                  <Typography
                    variant="h4"
                    fontWeight={800}
                    sx={{ color: 'divider', lineHeight: 1, mb: 1.5 }}
                  >
                    {f.number}
                  </Typography>
                  <Typography variant="subtitle1" fontWeight={700} mb={0.75}>
                    {f.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {f.desc}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    </Box>
  );
}
