import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown';
import { Source } from '../types';
import PdfViewer from './PdfViewer';
import './CitationText.css';

interface CitationTextProps {
  content: string;
  sources?: Source[];
}

export const CITATION_RE = /\[([^[\]]+#page=\d+)\]/g;

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
  CITATION_RE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = CITATION_RE.exec(content)) !== null) {
    const ref = match[1];
    if (!citationMap.has(ref)) {
      citationMap.set(ref, citationMap.size + 1);
    }
  }

  // Transform [file.pdf#page=N] into [N](citation://file.pdf#page=N) so
  // react-markdown can parse them as links while we intercept the render.
  CITATION_RE.lastIndex = 0;
  const transformed = content.replace(CITATION_RE, (_, ref) => {
    const num = citationMap.get(ref) ?? 0;
    return `[${num}](citation://${ref})`;
  });

  function handleClick(ref: string) {
    const { filename, page } = parseCitation(ref);
    const found = sources?.find((s) => s.filename === filename && s.page === page);
    setViewerSource(found ?? { filename, page, highlights: [] });
  }

  return (
    <>
      <ReactMarkdown
        urlTransform={(url) =>
          url.startsWith('citation://') ? url : defaultUrlTransform(url)
        }
        components={{
          a({ href, children }) {
            if (href?.startsWith('citation://')) {
              const ref = href.slice('citation://'.length);
              const { filename, page } = parseCitation(ref);
              return (
                <button
                  className="inline-citation"
                  title={`${filename} — ${t('pdf.page', { page })}`}
                  onClick={() => handleClick(ref)}
                >
                  {children}
                </button>
              );
            }
            return <a href={href}>{children}</a>;
          },
        }}
      >
        {transformed}
      </ReactMarkdown>
      {viewerSource && (
        <PdfViewer source={viewerSource} onClose={() => setViewerSource(null)} />
      )}
    </>
  );
}

export default CitationText;
