import Box from '@mui/material/Box';

const WAVE_STYLE = `
  @keyframes wave-slide {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
  }
  @media (prefers-reduced-motion: reduce) {
    .wave-path { animation: none !important; }
  }
`;

interface WaveBackgroundProps {
  height?: number;
  dark?: boolean;
}

export default function WaveBackground({ height = 120, dark = false }: WaveBackgroundProps) {
  const baseColor = dark ? '#ffffff' : '#000000';
  const waves = [
    { opacity: 0.06, duration: '14s', yOffset: 20 },
    { opacity: 0.04, duration: '10s', yOffset: 10 },
    { opacity: 0.03, duration: '7s',  yOffset: 0  },
  ];

  return (
    <>
      <style>{WAVE_STYLE}</style>
      <Box
        sx={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          width: '100%',
          height,
          overflow: 'hidden',
          pointerEvents: 'none',
          zIndex: 0,
        }}
      >
        {waves.map((wave, i) => (
          <Box
            key={i}
            className="wave-path"
            sx={{
              position: 'absolute',
              bottom: wave.yOffset,
              left: 0,
              width: '200%',
              height: '100%',
              animation: `wave-slide ${wave.duration} linear infinite`,
            }}
          >
            {[0, 1].map(j => (
              <svg
                key={j}
                viewBox="0 0 1440 120"
                preserveAspectRatio="none"
                style={{ width: '50%', height: '100%', display: 'inline-block' }}
              >
                <path
                  d="M0,60 C180,100 360,20 540,60 C720,100 900,20 1080,60 C1260,100 1440,20 1440,60 L1440,120 L0,120 Z"
                  fill={baseColor}
                  opacity={wave.opacity}
                />
              </svg>
            ))}
          </Box>
        ))}
      </Box>
    </>
  );
}
