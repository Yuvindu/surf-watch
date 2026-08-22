import { useState, useCallback } from 'react';
import type { UploadFile, FileType } from '../models/upload';
import { generateId } from '../utils/helpers';

const ACCEPTED_TYPES: Record<string, FileType> = {
  'image/jpeg': 'image',
  'image/png': 'image',
  'video/mp4': 'video',
  'video/webm': 'video',
};
const MAX_SIZE_BYTES = 50 * 1024 * 1024; // 50MB

export function useFileHandler() {
  const [uploadFile, setUploadFile] = useState<UploadFile | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback((file: File) => {
    setError(null);
    const fileType = ACCEPTED_TYPES[file.type];
    if (!fileType) {
      setError('Unsupported file type. Please upload .jpg, .png, .mp4, or .webm');
      return;
    }
    if (file.size > MAX_SIZE_BYTES) {
      setError('File exceeds the 50MB size limit.');
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    setUploadFile({
      id: generateId(),
      file,
      fileType,
      previewUrl,
      status: 'idle',
      uploadedAt: new Date().toISOString(),
    });
  }, []);

  const clearFile = useCallback(() => {
    if (uploadFile?.previewUrl) URL.revokeObjectURL(uploadFile.previewUrl);
    setUploadFile(null);
    setError(null);
  }, [uploadFile]);

  return { uploadFile, error, handleFile, clearFile };
}
