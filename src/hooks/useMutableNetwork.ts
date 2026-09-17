import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MLP } from '../lib/network.ts';
import type { NetworkConfig } from '../lib/network.ts';

export interface MutableNetwork {
  network: MLP;
  /** Increments whenever the parameters change; use it as a redraw key. */
  version: number;
  /** Applies an in-place edit and triggers a re-render. */
  mutate: (fn: (network: MLP) => void) => void;
  /** Re-initialises the weights, optionally with a new seed. */
  reset: (seed?: number) => void;
}

/**
 * Keeps one MLP instance alive across renders and rebuilds it when the
 * architecture changes. The network is mutated in place — copying every weight
 * on each training step would dominate the frame budget.
 */
export function useMutableNetwork(config: NetworkConfig): MutableNetwork {
  const key = useMemo(() => JSON.stringify(config), [config]);
  const ref = useRef<MLP>(new MLP(config));
  const keyRef = useRef(key);
  const [version, setVersion] = useState(0);

  useEffect(() => {
    if (keyRef.current === key) return;
    keyRef.current = key;
    ref.current = new MLP(config);
    setVersion((v) => v + 1);
  }, [key, config]);

  const mutate = useCallback((fn: (network: MLP) => void) => {
    fn(ref.current);
    setVersion((v) => v + 1);
  }, []);

  const reset = useCallback((seed?: number) => {
    ref.current.initialise(seed);
    setVersion((v) => v + 1);
  }, []);

  // `version` is read here so consumers re-render when it changes.
  return { network: ref.current, version, mutate, reset };
}
