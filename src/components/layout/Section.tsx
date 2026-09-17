import type { ReactNode } from 'react';

interface SectionProps {
  id: string;
  index: number;
  title: string;
  lede: ReactNode;
  children: ReactNode;
}

export function Section({ id, index, title, lede, children }: SectionProps) {
  return (
    <section className="section" id={id} aria-labelledby={`${id}-title`}>
      <div className="section__eyebrow">Section {String(index).padStart(2, '0')}</div>
      <h2 className="section__title" id={`${id}-title`}>
        {title}
      </h2>
      <div className="section__lede">{lede}</div>
      <div className="section__body">{children}</div>
    </section>
  );
}
