import { useMemo, useState, type ReactNode } from 'react';

export interface Column<Row> {
  key: string;
  header: ReactNode;
  render: (row: Row) => ReactNode;
  sortValue?: (row: Row) => number | string;
  numeric?: boolean;
  title?: string; // header tooltip (metric definition)
}

export function DenseDataTable<Row>({
  columns,
  rows,
  getRowId,
  onRowActivate,
  initialSortKey,
}: {
  columns: Column<Row>[];
  rows: Row[];
  getRowId: (row: Row) => string;
  onRowActivate?: (row: Row) => void;
  initialSortKey?: string;
}) {
  const [sortKey, setSortKey] = useState<string | null>(initialSortKey ?? null);
  const [dir, setDir] = useState<'asc' | 'desc'>('desc');

  const sorted = useMemo(() => {
    if (!sortKey) return rows;
    const col = columns.find((c) => c.key === sortKey);
    if (!col?.sortValue) return rows;
    const s = [...rows].sort((a, b) => {
      const av = col.sortValue!(a);
      const bv = col.sortValue!(b);
      if (av < bv) return dir === 'asc' ? -1 : 1;
      if (av > bv) return dir === 'asc' ? 1 : -1;
      return 0;
    });
    return s;
  }, [rows, columns, sortKey, dir]);

  function toggleSort(col: Column<Row>) {
    if (!col.sortValue) return;
    if (sortKey === col.key) {
      setDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(col.key);
      setDir('desc');
    }
  }

  return (
    <div className="tl-table-wrap">
      <table className="tl-table">
        <thead>
          <tr>
            {columns.map((col) => {
              const active = sortKey === col.key;
              return (
                <th
                  key={col.key}
                  title={col.title}
                  aria-sort={
                    col.sortValue
                      ? active
                        ? dir === 'asc'
                          ? 'ascending'
                          : 'descending'
                        : 'none'
                      : undefined
                  }
                  onClick={() => toggleSort(col)}
                  onKeyDown={(e) => {
                    if (col.sortValue && (e.key === 'Enter' || e.key === ' ')) {
                      e.preventDefault();
                      toggleSort(col);
                    }
                  }}
                  tabIndex={col.sortValue ? 0 : undefined}
                >
                  {col.header}
                  {active && (
                    <span className="tl-table__sortarrow" aria-hidden>
                      {dir === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr
              key={getRowId(row)}
              data-clickable={onRowActivate ? 'true' : undefined}
              tabIndex={onRowActivate ? 0 : undefined}
              onClick={onRowActivate ? () => onRowActivate(row) : undefined}
              onKeyDown={
                onRowActivate
                  ? (e) => {
                      if (e.key === 'Enter') onRowActivate(row);
                    }
                  : undefined
              }
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={col.numeric ? 'tl-td--num' : undefined}
                >
                  {col.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
