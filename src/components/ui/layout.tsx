import type { ReactNode } from 'react';

interface PanelProps {
  title?: ReactNode;
  hint?: ReactNode;
  children: ReactNode;
  caption?: ReactNode;
  flush?: boolean;
}

export function Panel({ title, hint, children, caption, flush }: PanelProps) {
  return (
    <section className="panel">
      {title || hint ? (
        <header className="panel__head">
          <span className="panel__title">{title}</span>
          {hint ? <span className="panel__hint">{hint}</span> : null}
        </header>
      ) : null}
      <div className={flush ? 'panel__body panel__body--flush' : 'panel__body'}>{children}</div>
      {caption ? <div className="figure__caption">{caption}</div> : null}
    </section>
  );
}

export interface StatSpec {
  label: string;
  value: ReactNode;
  accent?: boolean;
}

export function Stats({ items }: { items: StatSpec[] }) {
  return (
    <div className="stats">
      {items.map((item) => (
        <div className="stat" key={item.label}>
          <div className="stat__label" title={item.label}>
            {item.label}
          </div>
          <div className={item.accent ? 'stat__value stat__value--accent' : 'stat__value'}>
            {item.value}
          </div>
        </div>
      ))}
    </div>
  );
}

export function Note({
  title,
  children,
  accent,
}: {
  title?: string;
  children: ReactNode;
  accent?: boolean;
}) {
  return (
    <div className={accent ? 'note note--accent' : 'note'}>
      {title ? <div className="note__title">{title}</div> : null}
      {children}
    </div>
  );
}

export interface LegendItem {
  color: string;
  label: string;
  /** Draws the swatch as an outline, matching hollow markers in the figures. */
  outline?: boolean;
  /** Draws the swatch as a short dashed line, matching dashed series. */
  dashed?: boolean;
}

export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <div className="legend">
      {items.map((item) => (
        <span className="legend__item" key={item.label}>
          <span
            className={item.dashed ? 'legend__swatch legend__swatch--line' : 'legend__swatch'}
            style={
              item.dashed
                ? { borderTop: `2px dashed ${item.color}` }
                : item.outline
                  ? { background: 'transparent', boxShadow: `inset 0 0 0 2px ${item.color}` }
                  : { background: item.color }
            }
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}
