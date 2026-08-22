import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import { confidenceLabel } from '../utils/helpers';

interface ConfidenceGaugeProps {
  value: number; // 0–1
  size?: number;
}

export default function ConfidenceGauge({ value, size = 100 }: ConfidenceGaugeProps) {
  const pct = Math.round(value * 100);
  const label = confidenceLabel(value);

  return (
    <Box sx={{ position: 'relative', display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
      <Box sx={{ position: 'relative', display: 'inline-flex' }}>
        {/* track */}
        <CircularProgress
          variant="determinate"
          value={100}
          size={size}
          thickness={4}
          sx={{ color: 'divider', position: 'absolute' }}
        />
        {/* fill */}
        <CircularProgress
          variant="determinate"
          value={pct}
          size={size}
          thickness={4}
          sx={{ color: 'text.primary' }}
        />
        <Box
          sx={{
            top: 0, left: 0, bottom: 0, right: 0,
            position: 'absolute',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Typography variant="h6" fontWeight={700} sx={{ color: 'text.primary', lineHeight: 1 }}>
            {pct}%
          </Typography>
        </Box>
      </Box>
      <Typography variant="caption" fontWeight={600} color="text.secondary">
        {label} Confidence
      </Typography>
    </Box>
  );
}
