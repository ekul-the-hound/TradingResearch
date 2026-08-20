import { useId } from 'react';

export interface Series {
  label: string;
  color: string;
  points: { x: number; y: number }[]; // x is index/time-ordinal, y is value
}

// A small, dependency-free line chart. Renders inside a ChartFrame's plot area.
// Draws a zero baseline when the y-range crosses zero (no omitted baselines).
export function LineChart({
  series,
  height = 180,
  yUnit = '',
}: {
  series: Series[];
  height?: number;
  yUnit?: string;
}) {
  const clip = useId();
  const all = series.flatMap((s) => s.points);
  if (all.length === 0) return null;

  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = Math.min(...ys, 0);
  let yMax = Math.max(...ys, 0);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }

  const W = 600;
  const H = height;
  const padL = 44;
  const padR = 8;
  const padT = 8;
  const padB = 20;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const sx = (x: number) =>
    padL + ((x - xMin) / (xMax - xMin || 1)) * plotW;
  const sy = (y: number) =>
    padT + (1 - (y - yMin) / (yMax - yMin || 1)) * plotH;

  const zeroY = sy(0);
  const yTicks = [yMin, (yMin + yMax) / 2, yMax];

  const labelText = series.map((s) => s.label).join(' and ') || 'series';

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      height={H}
      role="img"
      aria-label={`Line chart of ${labelText}`}
      style={{ display: 'block' }}
    >
      <clipPath id={clip}>
        <rect x={padL} y={padT} width={plotW} height={plotH} />
      </clipPath>

      {/* y grid + labels */}
      {yTicks.map((t, i) => (
        <g key={i}>
          <line
            x1={padL}
            x2={W - padR}
            y1={sy(t)}
            y2={sy(t)}
            stroke="var(--c-border)"
            strokeWidth={0.5}
          />
          <text
            x={padL - 6}
            y={sy(t) + 3}
            textAnchor="end"
            fontSize={9}
            fill="var(--c-text-muted)"
          >
            {t.toFixed(1)}
            {yUnit}
          </text>
        </g>
      ))}

      {/* zero baseline emphasized when in range */}
      {yMin < 0 && yMax > 0 && (
        <line
          x1={padL}
          x2={W - padR}
          y1={zeroY}
          y2={zeroY}
          stroke="var(--c-text-muted)"
          strokeWidth={1}
          strokeDasharray="2 2"
        />
      )}

      {series.map((s) => (
        <polyline
          key={s.label}
          clipPath={`url(#${clip})`}
          fill="none"
          stroke={s.color}
          strokeWidth={1.5}
          points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')}
        />
      ))}
    </svg>
  );
}
