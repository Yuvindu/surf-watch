import { useState } from 'react';
import Box from '@mui/material/Box';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import HeatmapOverlay from './HeatmapOverlay';
import type { PredictionResult } from '../models/prediction';

interface VideoPlayerProps {
  previewUrl: string;
  prediction: PredictionResult;
}

export default function VideoPlayer({ previewUrl, prediction }: VideoPlayerProps) {
  const [frameIdx, setFrameIdx] = useState(0);
  const frames = prediction.frames;
  const currentFrame = frames[frameIdx] ?? frames[0];

  return (
    <Box>
      <HeatmapOverlay
        previewUrl={previewUrl}
        frame={currentFrame}
        isVideo={prediction.fileType === 'video'}
      />

      {frames.length > 1 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption" color="text.secondary" gutterBottom>
            Frame scrubber — {frames.length} prediction frames
          </Typography>
          <Slider
            min={0}
            max={frames.length - 1}
            step={1}
            value={frameIdx}
            onChange={(_, v) => setFrameIdx(v as number)}
            valueLabelDisplay="auto"
            valueLabelFormat={v => `Frame ${v} · ${frames[v]?.timestamp.toFixed(1)}s`}
            sx={{ color: 'text.primary' }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="caption" color="text.secondary">0s</Typography>
            <Typography variant="caption" color="text.secondary">
              {frames[frames.length - 1]?.timestamp.toFixed(1)}s
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  );
}
