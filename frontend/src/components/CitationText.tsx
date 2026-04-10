import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Source } from '../types';
import PdfViewer from './PdfViewer';
import './CitationText.css';

interface CitationTextProps {
  content: string;
  sources?: Source[];
}

const CITATION_RE = /\[([^\[\]]+#page=\d+)\]/g;

/** Parse a citation ref like `doc.pdf#page=5` into filename + page. */
function parseCitation(ref: string): { filename: string; page: number } {
  const m = ref.match(/^(.+)#page=(\d+)$/);
  return m ? { filename: m[1], page: Number(m[2]) } : { filename: ref, page: 1 };
}

function CitationText({ content, sources }: CitationTextProps) {
  const { t } = useTranslation();
  const [viewerSource, setViewerSource] = useState<Source | null>(null);

  // Assign a stable number to each unique citation ref
  const citationMap = new Map<string, number>();
  let match: RegExpExecArray | null;
  while ((match = CITATION_RE.exec(content)) !== null) {
    const ref = match[1];
    if (!citationMap.has(ref)) {
      citationMap.set(ref, citationMap.size + 1);
    }
  }

  // Split content into alternating [text, ref, text, ref, …] parts
  const parts = content.split(CITATION_RE);

  function handleClick(ref: string) {
    const { filename, page } = parseCitation(ref);
    const found = sources?.find((s) => s.filename === filename && s.page === page);
    setViewerSource(found ?? { filename, page, snippet: '' });
  }

  return (
    <>
      {parts.map((part, i) => {
        if (i % 2 === 0) {
          return <span key={i}>{part}</span>;
        }
        const num = citationMap.get(part) ?? 0;
        const { filename, page } = parseCitation(part);
        return (
          <button
            key={i}
            className="inline-citation"
            title={`${filename} — ${t('pdf.page', { page })}`}
            onClick={() => handleClick(part)}
          >
            {num}
          </button>
        );
      })}
      {viewerSource && (
        <PdfViewer source={viewerSource} onClose={() => setViewerSource(null)} />
      )}
    </>
  );
}

export default CitationText;
