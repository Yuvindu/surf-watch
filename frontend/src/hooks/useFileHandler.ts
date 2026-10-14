import { useState, useCallback } from 'react';
import type { UploadFile, FileType } from '../models/upload';
import { generateId } from '../utils/helpers';

const ACCEPTED_TYPES: Record<string, FileType> = {
  'image/jpeg': 'image',
  'image/png': 'image',
  'video/mp4': 'video',
  'video/webm': 'video',
};
export const MAX_IMAGE_SIZE_MB = 50;
export const MAX_VIDEO_SIZE_MB = 60;

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
    const maxSizeMb = fileType === 'video' ? MAX_VIDEO_SIZE_MB : MAX_IMAGE_SIZE_MB;
    if (file.size > maxSizeMb * 1024 * 1024) {
      setError(`File exceeds the ${maxSizeMb} MB size limit.`);
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
