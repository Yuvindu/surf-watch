import IconButton from '@mui/material/IconButton';
import LightModeIcon from '@mui/icons-material/LightMode';
import DarkModeIcon from '@mui/icons-material/DarkMode';

interface ThemeToggleProps {
  mode: 'light' | 'dark';
  onToggle: () => void;
}

export default function ThemeToggle({ mode, onToggle }: ThemeToggleProps) {
  const label = mode === 'light' ? 'Switch to dark mode' : 'Switch to light mode';
  return (
    <IconButton onClick={onToggle} color="inherit" size="small" title={label} aria-label={label}>
      {mode === 'light' ? <DarkModeIcon fontSize="small" /> : <LightModeIcon fontSize="small" />}
    </IconButton>
  );
}
