import { useCallback, useEffect, useMemo, useState } from 'react';
import { useScrollSpy } from './hooks/useScrollSpy.ts';
import { useTheme } from './hooks/useTheme.ts';
import { SECTIONS } from './sections/registry.ts';
import { Hero } from './sections/Hero.tsx';

export function App() {
  const [theme, toggleTheme] = useTheme();
  const [navOpen, setNavOpen] = useState(false);
  const ids = useMemo(() => ['top', ...SECTIONS.map((section) => section.id)], []);
  const activeId = useScrollSpy(ids);

  const activeIndex = SECTIONS.findIndex((section) => section.id === activeId);
  const progress = activeIndex < 0 ? 0 : ((activeIndex + 1) / SECTIONS.length) * 100;
  const activeTitle = activeIndex < 0 ? 'Introduction' : SECTIONS[activeIndex].title;
  const activeNumber = activeIndex < 0 ? '—' : SECTIONS[activeIndex].number;

  const goTo = useCallback((id: string) => {
    setNavOpen(false);
    const element = document.getElementById(id);
    if (element) element.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setNavOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <div className="app">
      <aside className={navOpen ? 'sidebar sidebar--open' : 'sidebar'} aria-label="Course contents">
        <div className="sidebar__head">
          <span className="sidebar__title">Deep Neural Networks</span>
          <span className="sidebar__subtitle">an interactive course</span>
        </div>
        <nav className="sidebar__nav">
          <button
            type="button"
            className={activeId === 'top' ? 'navitem navitem--active' : 'navitem'}
            onClick={() => goTo('top')}
          >
            <span className="navitem__index">—</span>
            <span>Introduction</span>
          </button>
          {SECTIONS.map((section) => (
            <button
              key={section.id}
              type="button"
              className={activeId === section.id ? 'navitem navitem--active' : 'navitem'}
              onClick={() => goTo(section.id)}
              aria-current={activeId === section.id ? 'true' : undefined}
            >
              <span className="navitem__index">{section.number}</span>
              <span>{section.title}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar__foot">
          <span className="progress__label">
            {activeNumber}/12
          </span>
          <div className="progress" role="presentation">
            <div className="progress__fill" style={{ width: `${progress}%` }} />
          </div>
          <button
            type="button"
            className="iconbutton"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          >
            {theme === 'dark' ? (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
              </svg>
            ) : (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
              </svg>
            )}
          </button>
        </div>
      </aside>

      {navOpen ? (
        <button type="button" className="scrim" aria-label="Close navigation" onClick={() => setNavOpen(false)} />
      ) : null}

      <main className="main">
        <div className="topbar">
          <button
            type="button"
            className="iconbutton"
            onClick={() => setNavOpen((open) => !open)}
            aria-label="Toggle navigation"
            aria-expanded={navOpen}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M3 12h18M3 18h18" />
            </svg>
          </button>
          <span className="topbar__title">{activeTitle}</span>
          <button
            type="button"
            className="iconbutton"
            onClick={toggleTheme}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? '☀' : '☾'}
          </button>
        </div>

        <Hero onStart={() => goTo(SECTIONS[0].id)} />

        {SECTIONS.map((section) => {
          const Component = section.component;
          return <Component key={section.id} id={section.id} index={Number(section.number)} />;
        })}

        <footer className="section" style={{ paddingTop: 40, paddingBottom: 56 }}>
          <p className="faint" style={{ fontSize: 13 }}>
            Every visualisation on this page runs a real network in the browser: the forward pass,
            the gradients and the training loop are computed from the equations shown next to them.
            The implementation is in <code>src/lib/</code>, and its gradients are verified against
            finite differences in <code>tests/network.test.ts</code>.
          </p>
        </footer>
      </main>
    </div>
  );
}
