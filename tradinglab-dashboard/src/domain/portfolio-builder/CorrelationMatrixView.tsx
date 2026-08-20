import type { CorrelationMatrix } from '../../models/portfolio';

// Diverging color for correlation in [-1, 1]. Low correlation (good for
// diversification) reads cool; high reads warm.
function cellColor(v: number): string {
  if (v >= 0.7) return 'rgba(242,109,109,0.35)';
  if (v >= 0.4) return 'rgba(246,184,75,0.30)';
  if (v >= 0.2) return 'rgba(145,162,181,0.20)';
  return 'rgba(58,203,143,0.22)';
}

export function CorrelationMatrixView({ m }: { m: CorrelationMatrix }) {
  return (
    <div className="tl-corr-wrap">
      <table className="tl-corr">
        <thead>
          <tr>
            <th />
            {m.labels.map((l, i) => (
              <th key={m.ids[i]} title={l}>
                {l.length > 10 ? `${l.slice(0, 10)}…` : l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {m.values.map((row, i) => (
            <tr key={m.ids[i]}>
              <th title={m.labels[i]}>
                {m.labels[i].length > 12 ? `${m.labels[i].slice(0, 12)}…` : m.labels[i]}
              </th>
              {row.map((v, j) => (
                <td
                  key={m.ids[j]}
                  style={{ background: i === j ? 'var(--c-raised)' : cellColor(v) }}
                  title={`${m.labels[i]} × ${m.labels[j]}: ${v.toFixed(2)}`}
                >
                  {v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="tl-corr-legend">
        Lower correlation improves diversification. Diagonal is self-correlation
        (1.00).
      </p>
    </div>
  );
}
