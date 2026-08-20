// The ONE canonical place metrics are formatted (CLAUDE.md §7, ARCHITECTURE §6).
// Components must not re-implement number/percent/date formatting inline.

export const DASH = '—';

export function pct(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return DASH;
  return `${v.toFixed(digits)}%`;
}

export function ratio(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return DASH;
  return v.toFixed(digits);
}

export function int(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return DASH;
  return Math.round(v).toLocaleString('en-US');
}

export function prob(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return DASH;
  return `${(v * 100).toFixed(digits)}%`;
}

// Actual timestamp only. Never fabricate "just now" / relative guesses.
export function ts(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return DASH;
  return d.toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
}

export function shortFingerprint(fp: string | null | undefined): string {
  if (!fp) return DASH;
  return fp.length <= 12 ? fp : `${fp.slice(0, 8)}…${fp.slice(-4)}`;
}
