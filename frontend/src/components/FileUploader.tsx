import React, { useCallback, useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const RIPPLE_STYLE = `
  @keyframes ripple-out {
    0%   { transform: scale(0); opacity: 0.3; }
    100% { transform: scale(2.5); opacity: 0; }
  }
`;

interface FileUploaderProps {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function FileUploader({ onFile, disabled }: FileUploaderProps) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) onFile(file);
    },
    [onFile, disabled]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFile(file);
      e.target.value = '';
    },
    [onFile]
  );

  return (
    <>
      <style>{RIPPLE_STYLE}</style>
      <Box
        component="label"
        onDragOver={e => { e.preventDefault(); if (!disabled) setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 1.5,
          minHeight: 200,
          border: '2px dashed',
          borderColor: dragging ? 'text.primary' : 'divider',
          borderRadius: '12px',
          cursor: disabled ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease',
          position: 'relative',
          overflow: 'hidden',
          bgcolor: dragging ? 'action.hover' : 'transparent',
          '&:hover': disabled ? {} : {
            borderColor: 'text.primary',
            bgcolor: 'action.hover',
          },
          '&:hover .ripple-ring': {
            animation: 'ripple-out 1s ease-out infinite',
          },
        }}
      >
        <input
          type="file"
          accept="image/jpeg,image/png,video/mp4,video/webm"
          style={{ display: 'none' }}
          onChange={handleChange}
          disabled={disabled}
        />

        <Box
          className="ripple-ring"
          sx={{
            position: 'absolute',
            width: 80, height: 80,
            borderRadius: '50%',
            border: '2px solid',
            borderColor: 'text.primary',
            opacity: 0,
            pointerEvents: 'none',
          }}
        />

        <CloudUploadIcon sx={{ fontSize: 44, color: 'text.secondary', opacity: 0.6 }} />
        <Typography variant="subtitle1" fontWeight={600} color="text.primary">
          Drag & drop or click to upload
        </Typography>
        <Typography variant="caption" color="text.secondary">
          .jpg, .png, .mp4, .webm — max 50 MB
        </Typography>
      </Box>
    </>
  );
}
