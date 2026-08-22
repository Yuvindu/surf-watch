import { useState } from 'react';
import Box from '@mui/material/Box';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import Typography from '@mui/material/Typography';
import type { FramePrediction } from '../models/prediction';

const PULSE_STYLE = `
  @keyframes rip-pulse {
    0%, 100% { opacity: 0.3; }
    50%       { opacity: 0.55; }
  }
  @media (prefers-reduced-motion: reduce) {
    .rip-overlay-region { animation: none !important; opacity: 0.4 !important; }
  }
`;

interface HeatmapOverlayProps {
  previewUrl: string;
  frame: FramePrediction;
  isVideo?: boolean;
}

export default function HeatmapOverlay({ previewUrl, frame, isVideo }: HeatmapOverlayProps) {
  const [showOverlay, setShowOverlay] = useState(true);

  return (
    <>
      <style>{PULSE_STYLE}</style>
      <Box>
        <FormControlLabel
          control={
            <Switch
              checked={showOverlay}
              onChange={e => setShowOverlay(e.target.checked)}
              size="small"
              sx={{
                '& .MuiSwitch-switchBase.Mui-checked': { color: 'text.primary' },
                '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                  bgcolor: 'text.primary',
                },
              }}
            />
          }
          label={<Typography variant="caption" fontWeight={600}>Rip mask overlay</Typography>}
          sx={{ mb: 1, ml: 0 }}
        />

        <Box sx={{ position: 'relative', display: 'inline-block', width: '100%', borderRadius: '8px', overflow: 'hidden' }}>
          {isVideo ? (
            <video
              src={previewUrl}
              controls
              style={{ width: '100%', display: 'block', borderRadius: 8 }}
            />
          ) : (
            <img
              src={previewUrl}
              alt="Uploaded media"
              style={{ width: '100%', display: 'block', borderRadius: 8 }}
            />
          )}

          {showOverlay && frame.regions.map(region => {
            const { topLeft, bottomRight } = region.boundingBox;
            return (
              <Box
                key={region.id}
                className="rip-overlay-region"
                sx={{
                  position: 'absolute',
                  left:   `${topLeft.x * 100}%`,
                  top:    `${topLeft.y * 100}%`,
                  width:  `${(bottomRight.x - topLeft.x) * 100}%`,
                  height: `${(bottomRight.y - topLeft.y) * 100}%`,
                  bgcolor: 'rgba(255,255,255,0.25)',
                  border:  '2px solid rgba(255,255,255,0.75)',
                  borderRadius: '4px',
                  animation: 'rip-pulse 2s ease-in-out infinite',
                  pointerEvents: 'none',
                }}
              />
            );
          })}
        </Box>
      </Box>
    </>
  );
}
