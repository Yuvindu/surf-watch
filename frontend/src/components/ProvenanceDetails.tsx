import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import Typography from '@mui/material/Typography';
import DownloadIcon from '@mui/icons-material/Download';
import type { ArtifactIdentity, ComparisonProvenance } from '../models/analysis';
import { formatFileSize } from '../utils/helpers';

interface ProvenanceDetailsProps {
  provenance?: ComparisonProvenance;
  input?: ArtifactIdentity;
  metricsUrl?: string;
  summaryJsonUrl?: string;
  summaryCsvUrl?: string;
}

function filename(path: string) {
  return path.split(/[\\/]/).pop() ?? path;
}

function Identity({ label, value }: { label: string; value?: ArtifactIdentity }) {
  if (!value) return null;
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography variant="caption" color="text.secondary" display="block">{label}</Typography>
      <Typography variant="body2" fontWeight={600} sx={{ overflowWrap: 'anywhere' }}>
        {filename(value.path)} · {formatFileSize(value.size_bytes)}
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ fontFamily: 'monospace', overflowWrap: 'anywhere' }}>
        SHA-256 {value.sha256}
      </Typography>
    </Box>
  );
}

export default function ProvenanceDetails({
  provenance, input, metricsUrl, summaryJsonUrl, summaryCsvUrl,
}: ProvenanceDetailsProps) {
  const source = input ?? provenance?.input;
  if (!source && !provenance && !metricsUrl && !summaryJsonUrl && !summaryCsvUrl) return null;

  return (
    <Box component="section" sx={{ mt: 4 }}>
      <Divider sx={{ mb: 2 }} />
      <Typography variant="subtitle1" fontWeight={700} mb={2}>Provenance</Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2.5 }}>
        <Identity label="Input video" value={source} />
        <Identity label="Model checkpoint" value={provenance?.checkpoint} />
        {Object.entries(provenance?.component_checkpoints ?? {}).map(([model, identity]) => (
          <Identity key={model} label={`${model} checkpoint`} value={identity} />
        ))}
        {provenance?.git_revision && (
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="caption" color="text.secondary" display="block">Code revision</Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', overflowWrap: 'anywhere' }}>
              {provenance.git_revision}
            </Typography>
          </Box>
        )}
        {provenance?.generated_at_utc && (
          <Box>
            <Typography variant="caption" color="text.secondary" display="block">Generated</Typography>
            <Typography variant="body2">{new Date(provenance.generated_at_utc).toLocaleString('en-AU')}</Typography>
          </Box>
        )}
      </Box>
      {(metricsUrl || summaryJsonUrl || summaryCsvUrl) && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2.5 }}>
          {metricsUrl && <Button size="small" variant="outlined" startIcon={<DownloadIcon />} href={metricsUrl} target="_blank" rel="noopener noreferrer">Model metrics</Button>}
          {summaryJsonUrl && <Button size="small" variant="outlined" startIcon={<DownloadIcon />} href={summaryJsonUrl} target="_blank" rel="noopener noreferrer">Summary JSON</Button>}
          {summaryCsvUrl && <Button size="small" variant="outlined" startIcon={<DownloadIcon />} href={summaryCsvUrl} target="_blank" rel="noopener noreferrer">Summary CSV</Button>}
        </Box>
      )}
    </Box>
  );
}
