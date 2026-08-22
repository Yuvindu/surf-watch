import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked';
import { PIPELINE_STAGES, STAGE_LABELS, STAGE_DESCRIPTIONS } from '../services/mockApi';
import type { PipelineStage } from '../models/analysis';

interface StatusStepperProps {
  currentStage: PipelineStage;
  isComplete: boolean;
  currentTask?: string | null;
  stages?: PipelineStage[];
}

export default function StatusStepper({ currentStage, isComplete, currentTask, stages = PIPELINE_STAGES }: StatusStepperProps) {
  const currentIndex = stages.indexOf(currentStage);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {stages.map((stage, i) => {
        const done = isComplete ? true : i < currentIndex;
        const active = !isComplete && i === currentIndex;
        return (
          <Box key={stage} sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 24 }}>
              {done ? (
                <CheckCircleIcon sx={{ color: 'text.primary', fontSize: 22 }} />
              ) : active ? (
                <CircularProgress size={20} thickness={5} sx={{ color: 'text.primary', mt: '1px' }} />
              ) : (
                <RadioButtonUncheckedIcon sx={{ color: 'text.disabled', fontSize: 22 }} />
              )}
              {i < stages.length - 1 && (
                <Box
                  sx={{
                    width: 2,
                    height: 28,
                    bgcolor: done ? 'text.primary' : 'divider',
                    my: '2px',
                    borderRadius: 1,
                    opacity: done ? 0.4 : 1,
                    transition: 'background-color 0.3s',
                  }}
                />
              )}
            </Box>

            <Box sx={{ pb: 1.5 }}>
              <Typography
                variant="body2"
                fontWeight={active ? 700 : done ? 500 : 400}
                color={active ? 'text.primary' : done ? 'text.secondary' : 'text.disabled'}
              >
                {STAGE_LABELS[stage]}
              </Typography>
              {active && (
                <Typography variant="caption" color="text.secondary">
                  {currentTask ?? STAGE_DESCRIPTIONS[stage]}
                </Typography>
              )}
            </Box>
          </Box>
        );
      })}
    </Box>
  );
}
