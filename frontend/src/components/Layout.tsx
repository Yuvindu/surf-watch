import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';
import ThemeToggle from './ThemeToggle';
import Footer from './Footer';

const NAV_LINKS = [
  { label: 'Home',    path: '/' },
  { label: 'Analyse', path: '/analyse' },
  { label: 'History', path: '/history' },
  { label: 'About',   path: '/about' },
];

interface LayoutProps {
  children: React.ReactNode;
  mode: 'light' | 'dark';
  onToggleTheme: () => void;
}

export default function Layout({ children, mode, onToggleTheme }: LayoutProps) {
  const location = useLocation();
  const [drawerOpen, setDrawerOpen] = useState(false);

  const bgColor = mode === 'light' ? 'rgba(255,255,255,0.92)' : 'rgba(13,13,13,0.92)';

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          backdropFilter: 'blur(16px)',
          backgroundColor: bgColor,
          borderBottom: '1px solid',
          borderColor: 'divider',
          color: 'text.primary',
        }}
      >
        <Toolbar sx={{ maxWidth: 1100, width: '100%', mx: 'auto', px: { xs: 2, sm: 3 }, minHeight: { xs: 56, sm: 64 } }}>
          {/* Logo */}
          <Typography
            variant="h6"
            component={Link}
            to="/"
            sx={{
              fontWeight: 700,
              letterSpacing: '-0.5px',
              textDecoration: 'none',
              color: 'text.primary',
              flexGrow: { xs: 1, md: 0 },
              mr: { md: 4 },
            }}
          >
            SurfWatch
          </Typography>

          {/* Desktop nav links */}
          <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 0.5, flexGrow: 1 }}>
            {NAV_LINKS.map(link => {
              const active = location.pathname === link.path;
              return (
                <Button
                  key={link.path}
                  component={Link}
                  to={link.path}
                  size="small"
                  sx={{
                    color: active ? 'text.primary' : 'text.secondary',
                    fontWeight: active ? 700 : 400,
                    px: 1.5,
                    borderRadius: '8px',
                    position: 'relative',
                    bgcolor: 'transparent',
                    '&:hover': {
                      bgcolor: mode === 'light' ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.06)',
                      color: 'text.primary',
                    },
                    '&::after': active ? {
                      content: '""',
                      position: 'absolute',
                      bottom: 4,
                      left: '50%',
                      transform: 'translateX(-50%)',
                      width: '20px',
                      height: '2px',
                      borderRadius: '2px',
                      bgcolor: 'text.primary',
                    } : {},
                  }}
                >
                  {link.label}
                </Button>
              );
            })}
          </Box>

          {/* Desktop theme toggle */}
          <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
            <ThemeToggle mode={mode} onToggle={onToggleTheme} />
          </Box>

          {/* Mobile: theme toggle + hamburger */}
          <Box sx={{ display: { xs: 'flex', md: 'none' }, alignItems: 'center', gap: 0.5 }}>
            <ThemeToggle mode={mode} onToggle={onToggleTheme} />
            <IconButton
              onClick={() => setDrawerOpen(true)}
              size="small"
              aria-label="Open menu"
              sx={{ color: 'text.primary' }}
            >
              <MenuIcon />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Mobile Drawer */}
      <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        PaperProps={{
          sx: {
            width: 260,
            bgcolor: mode === 'light' ? '#ffffff' : '#0d0d0d',
            borderLeft: '1px solid',
            borderColor: 'divider',
          },
        }}
      >
        {/* Drawer header */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            px: 2.5,
            py: 2,
            borderBottom: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Typography variant="subtitle1" fontWeight={700}>
            SurfWatch
          </Typography>
          <IconButton size="small" onClick={() => setDrawerOpen(false)} sx={{ color: 'text.secondary' }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>

        {/* Nav links */}
        <List disablePadding sx={{ pt: 1 }}>
          {NAV_LINKS.map(link => {
            const active = location.pathname === link.path;
            return (
              <ListItemButton
                key={link.path}
                component={Link}
                to={link.path}
                onClick={() => setDrawerOpen(false)}
                sx={{
                  mx: 1,
                  mb: 0.5,
                  borderRadius: 2,
                  bgcolor: active
                    ? (mode === 'light' ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.08)')
                    : 'transparent',
                  '&:hover': {
                    bgcolor: mode === 'light' ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.06)',
                  },
                }}
              >
                <ListItemText
                  primary={link.label}
                  primaryTypographyProps={{
                    fontWeight: active ? 700 : 400,
                    color: active ? 'text.primary' : 'text.secondary',
                    variant: 'body1',
                  }}
                />
                {active && (
                  <Box
                    sx={{
                      width: 6, height: 6,
                      borderRadius: '50%',
                      bgcolor: 'text.primary',
                      flexShrink: 0,
                    }}
                  />
                )}
              </ListItemButton>
            );
          })}
        </List>

        <Divider sx={{ mt: 'auto', mx: 2 }} />
        <Box sx={{ px: 2.5, py: 2 }}>
          <Typography variant="caption" color="text.disabled">
            COMP6002 Research Project
          </Typography>
        </Box>
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, pt: { xs: '56px', sm: '64px' } }}>
        {children}
      </Box>

      <Footer />
    </Box>
  );
}
