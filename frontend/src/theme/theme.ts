import { createTheme } from '@mui/material/styles';

export const getTheme = (mode: 'light' | 'dark') =>
  createTheme({
    palette: {
      mode,
      primary:   { main: mode === 'light' ? '#111111' : '#f0f0f0' },
      secondary: { main: mode === 'light' ? '#444444' : '#aaaaaa' },
      warning:   { main: mode === 'light' ? '#333333' : '#cccccc' },
      background: {
        default: mode === 'light' ? '#ffffff' : '#0d0d0d',
        paper:   mode === 'light' ? 'rgba(255,255,255,0.9)' : 'rgba(18,18,18,0.9)',
      },
      text: {
        primary:   mode === 'light' ? '#111111' : '#f0f0f0',
        secondary: mode === 'light' ? '#555555' : '#999999',
        disabled:  mode === 'light' ? '#aaaaaa' : '#555555',
      },
      divider: mode === 'light' ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)',
    },
    typography: {
      fontFamily: '"Inter", "Roboto", sans-serif',
    },
    shape: { borderRadius: 12 },
    components: {
      MuiButton: {
        styleOverrides: {
          root: { borderRadius: 8, textTransform: 'none', fontWeight: 600 },
          containedPrimary: {
            backgroundColor: mode === 'light' ? '#111111' : '#f0f0f0',
            color:           mode === 'light' ? '#ffffff' : '#111111',
            '&:hover': {
              backgroundColor: mode === 'light' ? '#333333' : '#cccccc',
            },
          },
          outlinedPrimary: {
            borderColor: mode === 'light' ? '#111111' : '#f0f0f0',
            color:       mode === 'light' ? '#111111' : '#f0f0f0',
            '&:hover': {
              backgroundColor: mode === 'light' ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.07)',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            backdropFilter: 'blur(12px)',
            borderRadius: 12,
            border: mode === 'light'
              ? '1px solid rgba(0,0,0,0.1)'
              : '1px solid rgba(255,255,255,0.1)',
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
            border: mode === 'light'
              ? '1px solid rgba(0,0,0,0.08)'
              : '1px solid rgba(255,255,255,0.08)',
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 6,
          },
        },
      },
    },
  });
