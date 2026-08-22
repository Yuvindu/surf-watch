import { Link } from 'react-router-dom';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import Button from '@mui/material/Button';

const NAV_LINKS = [
  { label: 'Home',    path: '/' },
  { label: 'Analyse', path: '/analyse' },
  { label: 'History', path: '/history' },
  { label: 'About',   path: '/about' },
];

const TECH = [
  'React 18 + TypeScript',
  'Material UI v5',
  'Vite',
  'React Router v6',
];

export default function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        borderTop: '1px solid',
        borderColor: 'divider',
        mt: 'auto',
      }}
    >
      <Box sx={{ maxWidth: 1100, mx: 'auto', px: { xs: 2, sm: 3 }, pt: { xs: 5, md: 6 }, pb: { xs: 4, md: 5 } }}>
        <Grid container spacing={{ xs: 4, md: 6 }}>

          {/* Brand column */}
          <Grid item xs={12} sm={6} md={4}>
            <Typography variant="h6" fontWeight={700} mb={1} letterSpacing="-0.5px">
              SurfWatch
            </Typography>
            <Typography variant="body2" color="text.secondary" lineHeight={1.8} sx={{ maxWidth: 280 }}>
              A research demo for ML-powered rip current detection in beach video and imagery.
            </Typography>
            <Box
              sx={{
                mt: 2,
                display: 'inline-block',
                px: 1.5,
                py: 0.5,
                borderRadius: 1,
                border: '1px solid',
                borderColor: 'divider',
              }}
            >
              <Typography variant="caption" color="text.secondary" fontWeight={500}>
                Video comparison mode enabled
              </Typography>
            </Box>
          </Grid>

          {/* Navigation column */}
          <Grid item xs={6} sm={3} md={2}>
            <Typography variant="overline" sx={{ letterSpacing: 3, color: 'text.secondary', display: 'block', mb: 1.5 }}>
              Navigation
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              {NAV_LINKS.map(link => (
                <Button
                  key={link.path}
                  component={Link}
                  to={link.path}
                  size="small"
                  sx={{
                    justifyContent: 'flex-start',
                    color: 'text.secondary',
                    px: 0,
                    minWidth: 0,
                    fontWeight: 400,
                    '&:hover': { color: 'text.primary', bgcolor: 'transparent' },
                  }}
                >
                  {link.label}
                </Button>
              ))}
            </Box>
          </Grid>

          {/* Tech stack column */}
          <Grid item xs={6} sm={3} md={2}>
            <Typography variant="overline" sx={{ letterSpacing: 3, color: 'text.secondary', display: 'block', mb: 1.5 }}>
              Built with
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
              {TECH.map(t => (
                <Typography key={t} variant="caption" color="text.secondary">
                  {t}
                </Typography>
              ))}
            </Box>
          </Grid>

          {/* Project column */}
          <Grid item xs={12} md={4}>
            <Typography variant="overline" sx={{ letterSpacing: 3, color: 'text.secondary', display: 'block', mb: 1.5 }}>
              Project
            </Typography>
            <Typography variant="body2" color="text.secondary" lineHeight={1.8}>
              COMP6002 — SurfWatch Research Group
              <br />
              Curtin University
            </Typography>
          </Grid>
        </Grid>

        <Divider sx={{ mt: { xs: 4, md: 5 }, mb: 2.5 }} />

        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
          <Typography variant="caption" color="text.disabled">
            © {new Date().getFullYear()} SurfWatch — COMP6002 Research Project
          </Typography>
          <Typography variant="caption" color="text.disabled">
            Image detections use demo data; video comparisons run the local pipeline
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}
