import type { ReactNode } from 'react';

interface DetailProps {
  /** Short label shown before the title, e.g. "derivation" or "proof". */
  kicker?: string;
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}

/**
 * A collapsible block for a full derivation. The visualisation and the
 * one-paragraph statement stay visible; the algebra is one click away, so the
 * page stays readable without leaving anything unproved.
 */
export function Detail({ kicker = 'derivation', title, children, defaultOpen }: DetailProps) {
  return (
    <details className="detail" open={defaultOpen}>
      <summary className="detail__summary">
        <span className="detail__kicker">{kicker}</span>
        <span>{title}</span>
        <span className="detail__chevron" aria-hidden="true">
          ›
        </span>
      </summary>
      <div className="detail__body">{children}</div>
    </details>
  );
}

export interface NotationEntry {
  symbol: ReactNode;
  meaning: ReactNode;
}

/** A symbol/meaning reference table. */
export function Notation({ entries }: { entries: NotationEntry[] }) {
  return (
    <table className="notation">
      <tbody>
        {entries.map((entry, i) => (
          <tr key={i}>
            <td>{entry.symbol}</td>
            <td>{entry.meaning}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** A numbered sequence of reasoning steps. */
export function Steps({ children }: { children: ReactNode }) {
  return <ol className="steps">{children}</ol>;
}
