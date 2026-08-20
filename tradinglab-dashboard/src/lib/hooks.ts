import { useEffect, useRef, useState } from 'react';
import type { Loadable } from '../models/truth';

// Runs a repository read and tracks it as a Loadable. `read` must be a stable
// callback (wrap in useCallback) — it is the effect's sole dependency.
export function useLoadable<T>(read: () => Promise<Loadable<T>>): Loadable<T> {
  const [loadable, setLoadable] = useState<Loadable<T>>({ state: 'loading' });
  const first = useRef(true);

  useEffect(() => {
    let alive = true;
    // Reset to loading on subsequent reads only; the initial state is already
    // 'loading', so we avoid a synchronous setState on first mount.
    if (!first.current) setLoadable({ state: 'loading' });
    first.current = false;

    read()
      .then((r) => {
        if (alive) setLoadable(r);
      })
      .catch((e) => {
        if (alive)
          setLoadable({ state: 'error', error: String(e?.message ?? e) });
      });
    return () => {
      alive = false;
    };
  }, [read]);

  return loadable;
}
