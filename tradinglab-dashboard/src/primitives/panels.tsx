import type { ReactNode } from 'react';

export function AppPanel({
  children,
  raised = false,
  flush = false,
  className = '',
  style,
}: {
  children: ReactNode;
  raised?: boolean;
  flush?: boolean;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <section
      className={`tl-panel ${raised ? 'tl-panel--raised' : ''} ${className}`}
      style={style}
    >
      {flush ? (
        children
      ) : (
        <div className="tl-panel__body">{children}</div>
      )}
    </section>
  );
}

export function PanelHeader({
  title,
  subtitle,
  meta,
}: {
  title: string;
  subtitle?: string;
  meta?: ReactNode;
}) {
  return (
    <div className="tl-panelhdr">
      <div className="tl-panelhdr__titles">
        <span className="tl-panelhdr__title">{title}</span>
        {subtitle && <span className="tl-panelhdr__sub">{subtitle}</span>}
      </div>
      {meta && <div className="tl-panelhdr__meta">{meta}</div>}
    </div>
  );
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return <h3 className="tl-sectiontitle">{children}</h3>;
}

export function WarningBanner({
  tone = 'warning',
  children,
}: {
  tone?: 'warning' | 'critical' | 'info';
  children: ReactNode;
}) {
  const glyph = tone === 'critical' ? '✕' : tone === 'info' ? 'i' : '▲';
  return (
    <div className={`tl-banner tl-banner--${tone}`} role="alert">
      <span aria-hidden>{glyph}</span>
      <span className="tl-banner__body">{children}</span>
    </div>
  );
}
