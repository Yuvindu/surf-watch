export type FileType = 'image' | 'video';
export type FileStatus = 'idle' | 'uploading' | 'ready' | 'error';

export interface UploadFile {
  id: string;
  file: File;
  fileType: FileType;
  previewUrl: string;
  status: FileStatus;
  uploadedAt: string;
}
