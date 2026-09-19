import type { ReactNode } from 'react';

/**
 * Where the notebooks live. Update these two when the branch is merged and
 * every link on the site follows.
 */
const REPO = 'DavidZeff1/deep-neural-network-';
const BRANCH = 'claude/dnn-interactive-learning-site-syw69j';

interface ExerciseProps {
  /** File name inside notebooks/, e.g. "08-backprop.ipynb". */
  notebook: string;
  /** Number of exercises the notebook contains. */
  count: number;
  /** One sentence on what the notebook does with this section's material. */
  children: ReactNode;
  /** The individual exercises, shortest useful description each. */
  tasks: ReactNode[];
}

/**
 * A link to the companion Jupyter notebook for a section, with what it asks.
 * The site teaches by manipulation; the notebooks teach by implementation.
 */
export function Exercise({ notebook, count, children, tasks }: ExerciseProps) {
  const github = `https://github.com/${REPO}/blob/${BRANCH}/notebooks/${notebook}`;
  const colab = `https://colab.research.google.com/github/${REPO}/blob/${BRANCH}/notebooks/${notebook}`;
  const solution = `https://github.com/${REPO}/blob/${BRANCH}/notebooks/solutions/${notebook}`;

  return (
    <section className="exercise">
      <header className="exercise__head">
        <span className="exercise__kicker">exercises</span>
        <span className="exercise__file">notebooks/{notebook}</span>
        <span className="exercise__count">
          {count} {count === 1 ? 'exercise' : 'exercises'}
        </span>
      </header>
      <div className="exercise__body">
        <p className="exercise__intro">{children}</p>
        <ul className="exercise__list">
          {tasks.map((task, index) => (
            <li key={index}>{task}</li>
          ))}
        </ul>
      </div>
      <div className="exercise__links">
        <a className="exercise__link exercise__link--primary" href={colab} target="_blank" rel="noreferrer">
          Open in Colab
        </a>
        <a className="exercise__link" href={github} target="_blank" rel="noreferrer">
          View on GitHub
        </a>
        <a className="exercise__link" href={solution} target="_blank" rel="noreferrer">
          Solution
        </a>
      </div>
    </section>
  );
}
