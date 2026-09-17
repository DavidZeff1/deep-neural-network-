import { useMemo } from 'react';
import katex from 'katex';

interface MathProps {
  children: string;
  display?: boolean;
  className?: string;
}

/** Inline LaTeX. */
export function M({ children, className }: MathProps) {
  const html = useMemo(
    () =>
      katex.renderToString(children, {
        displayMode: false,
        throwOnError: false,
        output: 'html',
      }),
    [children],
  );
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Display LaTeX, centred on its own line. */
export function MathBlock({ children, className }: MathProps) {
  const html = useMemo(
    () =>
      katex.renderToString(children, {
        displayMode: true,
        throwOnError: false,
        output: 'html',
      }),
    [children],
  );
  return <div className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

interface EquationProps {
  children: string;
  caption?: string;
  plain?: boolean;
}

/** A boxed display equation with an optional caption underneath. */
export function Equation({ children, caption, plain }: EquationProps) {
  return (
    <figure className={plain ? 'equation equation--plain' : 'equation'}>
      <MathBlock>{children}</MathBlock>
      {caption ? <figcaption className="equation__caption">{caption}</figcaption> : null}
    </figure>
  );
}
