import { useNavigate } from 'react-router-dom';
import Card from '@mui/material/Card';
import CardActionArea from '@mui/material/CardActionArea';
import CardContent from '@mui/material/CardContent';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import ImageIcon from '@mui/icons-material/Image';
import VideocamIcon from '@mui/icons-material/Videocam';
import type { AnalysisCase } from '../models/analysis';
import { confidenceLabel, formatDate } from '../utils/helpers';

interface ResultCardProps {
  analysisCase: AnalysisCase;
}

export default function ResultCard({ analysisCase }: ResultCardProps) {
  const navigate = useNavigate();
  // Prefer backend's flat confidenceScore; fall back to nested mock value
  const conf = analysisCase.confidenceScore ?? analysisCase.prediction?.averageConfidence ?? 0;
  const displayName = analysisCase.caseName ?? analysisCase.upload.file.name;
  const hasResult = analysisCase.confidenceScore != null || analysisCase.prediction != null;

  return (
    <Card>
      <CardActionArea onClick={() => navigate(`/results/${analysisCase.id}`)}>
        {/* Thumbnail — prefer overlay preview from backend, fall back to local upload */}
        <Box sx={{ height: 140, bgcolor: 'action.hover', position: 'relative', overflow: 'hidden' }}>
          {(analysisCase.overlayUrl ?? analysisCase.upload.previewUrl) ? (
            analysisCase.upload.fileType === 'image' && !analysisCase.overlayUrl ? (
              <img
                src={analysisCase.upload.previewUrl}
                alt="thumbnail"
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
            ) : (
              <video
                src={analysisCase.overlayUrl ?? analysisCase.upload.previewUrl}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                muted
              />
            )
          ) : (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
              {analysisCase.upload.fileType === 'video'
                ? <VideocamIcon sx={{ fontSize: 40, color: 'text.disabled' }} />
                : <ImageIcon sx={{ fontSize: 40, color: 'text.disabled' }} />
              }
            </Box>
          )}

          <Chip
            label={analysisCase.upload.fileType.toUpperCase()}
            size="small"
            sx={{
              position: 'absolute', top: 8, left: 8,
              bgcolor: 'rgba(0,0,0,0.6)', color: '#fff',
              fontSize: '0.6rem', height: 18,
            }}
          />
        </Box>

        <CardContent sx={{ pb: '12px !important' }}>
          <Typography variant="body2" fontWeight={600} noWrap>
            {displayName}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {formatDate(analysisCase.createdAt)}
          </Typography>

          <Box sx={{ display: 'flex', gap: 1, mt: 1, alignItems: 'center' }}>
            <Chip
              label={analysisCase.status}
              size="small"
              variant="outlined"
              sx={{ fontSize: '0.65rem', height: 20 }}
            />
            {hasResult && (
              <Chip
                label={`${Math.round(conf * 100)}% · ${confidenceLabel(conf)}`}
                size="small"
                sx={{
                  fontSize: '0.65rem',
                  height: 20,
                  bgcolor: 'text.primary',
                  color: 'background.default',
                }}
              />
            )}
          </Box>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}
