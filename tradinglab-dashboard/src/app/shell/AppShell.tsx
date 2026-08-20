import { useEffect, useRef, type ReactNode } from 'react';
import { useLocation } from 'react-router-dom';
import { LeftNav } from './LeftNav';
import { SystemStatusBar } from './SystemStatusBar';
import './shell.css';

export function AppShell({ children }: { children: ReactNode }) {
  const mainRef = useRef<HTMLElement>(null);
  const { pathname } = useLocation();

  // On route change, move focus to the main region so keyboard and
  // screen-reader users land at the new page content, not the old position.
  useEffect(() => {
    mainRef.current?.focus();
  }, [pathname]);

  return (
    <div className="tl-shell">
      <a className="tl-skiplink" href="#tl-main">
        Skip to content
      </a>
      <div className="tl-brand">
        <span className="tl-brand__mark" aria-hidden>
          T
        </span>
        <span className="tl-brand__name">
          Trading<span>Lab</span>
        </span>
      </div>
      <SystemStatusBar />
      <LeftNav />
      <main
        ref={mainRef}
        className="tl-content"
        id="tl-main"
        tabIndex={-1}
        aria-label="Main content"
      >
        <div className="tl-content__inner">{children}</div>
      </main>
    </div>
  );
}
