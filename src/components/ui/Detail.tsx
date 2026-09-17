import { Fragment } from 'react';
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

interface InWordsProps {
  children: ReactNode;
  /** Defaults to "in words". Use "meaning" or "why" where that reads better. */
  tag?: string;
}

/**
 * A plain-language restatement of the equation above it. Every formula on the
 * site that a reader could stall on carries one of these, so the mathematics is
 * never the only route through an idea.
 */
export function InWords({ children, tag = 'in words' }: InWordsProps) {
  return (
    <div className="inwords">
      <span className="inwords__tag">{tag}</span>
      <div>{children}</div>
    </div>
  );
}

/** A small label/value grid for live numeric readouts inside a panel. */
export function Readout({
  rows,
}: {
  rows: Array<{ label: string; value: ReactNode; accent?: boolean }>;
}) {
  return (
    <div className="readout">
      {rows.map((row) => (
        <Fragment key={row.label}>
          <span className="readout__label">{row.label}</span>
          <span className={row.accent ? 'readout__value readout__value--accent' : 'readout__value'}>
            {row.value}
          </span>
        </Fragment>
      ))}
    </div>
  );
}
