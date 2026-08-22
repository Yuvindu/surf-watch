import { useContext } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import HistoryIcon from '@mui/icons-material/History';
import { AnalysisContext } from '../context/AnalysisContext';
import ResultCard from '../components/ResultCard';

export default function HistoryPage() {
  const { history } = useContext(AnalysisContext);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', px: { xs: 2, sm: 3 }, py: { xs: 4, md: 6 } }}>
      <Box sx={{ width: '100%', maxWidth: 960 }}>
        <Typography variant="h4" fontWeight={800} mb={0.5} sx={{ fontSize: { xs: '1.75rem', md: '2.125rem' } }}>
          History
        </Typography>
        <Typography variant="body2" color="text.secondary" mb={4}>
          All analyses from this session.
        </Typography>

        {history.length === 0 ? (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 10,
              opacity: 0.4,
              gap: 1,
            }}
          >
            <HistoryIcon sx={{ fontSize: 56, color: 'text.disabled' }} />
            <Typography variant="body2" color="text.secondary">
              No analyses yet — run a detection first.
            </Typography>
          </Box>
        ) : (
          <Grid container spacing={2.5}>
            {history.map(c => (
              <Grid item xs={12} sm={6} md={4} key={c.id}>
                <ResultCard analysisCase={c} />
              </Grid>
            ))}
          </Grid>
        )}
      </Box>
    </Box>
  );
}
