import { useSyncExternalStore } from 'react';

/**
 * Canvas drawings read theme colours once, at draw time, so they must be told
 * when the theme changes. SVG picks up CSS variables on its own and does not
 * need this.
 *
 * A single MutationObserver watches the documentElement's data-theme attribute
 * and notifies every subscribed canvas.
 */

let version = 0;
const listeners = new Set<() => void>();
let observer: MutationObserver | null = null;

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  if (!observer && typeof MutationObserver !== 'undefined') {
    observer = new MutationObserver(() => {
      version += 1;
      listeners.forEach((fn) => fn());
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });
  }
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0 && observer) {
      observer.disconnect();
      observer = null;
    }
  };
}

export function useThemeSignal(): number {
  return useSyncExternalStore(
    subscribe,
    () => version,
    () => 0,
  );
}
