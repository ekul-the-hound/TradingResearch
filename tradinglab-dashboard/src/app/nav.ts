export interface NavItem {
  to: string;
  label: string;
  glyph: string; // simple text glyph, no icon lib dependency
  offline?: boolean; // Execution shows a static OFFLINE marker
}

export const NAV_ITEMS: NavItem[] = [
  { to: '/research-command', label: 'Research Command', glyph: '◧' },
  { to: '/strategy-lab', label: 'Strategy Lab', glyph: '▤' },
  { to: '/portfolio-builder', label: 'Portfolio Builder', glyph: '◑' },
  { to: '/challenge-readiness', label: 'Challenge Readiness', glyph: '◈' },
  { to: '/discovery-inbox', label: 'Discovery Inbox', glyph: '☰' },
  { to: '/data-integrity', label: 'Data & Integrity', glyph: '⚿' },
  { to: '/execution', label: 'Execution', glyph: '⧉', offline: true },
];
