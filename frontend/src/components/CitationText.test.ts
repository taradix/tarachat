import { describe, it, expect } from 'vitest'
import { CITATION_RE } from './CitationText'

describe('CITATION_RE', () => {
  function findAll(text: string): string[] {
    CITATION_RE.lastIndex = 0;
    const results: string[] = [];
    let m: RegExpExecArray | null;
    while ((m = CITATION_RE.exec(text)) !== null) {
      results.push(m[1]);
    }
    return results;
  }

  it('matches a simple citation', () => {
    expect(findAll('See [report.pdf#page=3] for details')).toEqual([
      'report.pdf#page=3',
    ]);
  });

  it('matches multiple citations', () => {
    expect(
      findAll('See [a.pdf#page=1] and [b.pdf#page=2].')
    ).toEqual(['a.pdf#page=1', 'b.pdf#page=2']);
  });

  it('does not match text without brackets', () => {
    expect(findAll('no citation here')).toEqual([]);
  });

  it('does not match brackets without #page=', () => {
    expect(findAll('See [not a citation] here')).toEqual([]);
  });

  it('still matches inside nested brackets', () => {
    expect(findAll('See [[nested.pdf#page=1]] here')).toEqual([
      'nested.pdf#page=1',
    ]);
  });
});
