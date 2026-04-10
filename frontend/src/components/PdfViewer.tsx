import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Source } from '../types';
import './PdfViewer.css';

const API_URL = import.meta.env.VITE_API_URL || '/api';

interface PdfViewerProps {
  source: Source;
  onClose: () => void;
}

function buildPdfUrl(source: Source): string {
  const params = new URLSearchParams();
  params.set('page', String(source.page));
  for (const hl of source.highlights) {
    params.append('hl', hl);
  }
  return `${API_URL}/documents/${encodeURIComponent(source.filename)}?${params}`;
}

function PdfViewer({ source, onClose }: PdfViewerProps) {
  const { t } = useTranslation();

  return (
    <div className="pdf-overlay" onClick={onClose}>
      <div className="pdf-modal" onClick={(e) => e.stopPropagation()}>
        <div className="pdf-header">
          <span className="pdf-title">
            {source.filename} — {t('pdf.page', { page: source.page })}
          </span>
          <button className="pdf-close" onClick={onClose} aria-label={t('pdf.close')}>
            ✕
          </button>
        </div>
        <iframe
          className="pdf-frame"
          src={buildPdfUrl(source)}
          title={source.filename}
        />
      </div>
    </div>
  );
}

interface SourceLinkProps {
  source: Source;
}

function SourceLink({ source }: SourceLinkProps) {
  const { t } = useTranslation();
  const [showPdf, setShowPdf] = useState(false);

  return (
    <>
      <button
        className="source-link"
        onClick={() => setShowPdf(true)}
      >
        <span className="source-filename">{source.filename}</span>
        <span className="source-page">{t('pdf.page', { page: source.page })}</span>
      </button>
      {showPdf && <PdfViewer source={source} onClose={() => setShowPdf(false)} />}
    </>
  );
}

export { SourceLink };
export default PdfViewer;
