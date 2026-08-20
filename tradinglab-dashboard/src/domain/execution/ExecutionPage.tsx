import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useRepository } from '../../data/useRepository';
import { useLoadable } from '../../lib/hooks';
import {
  AppPanel,
  LoadableView,
  PanelHeader,
  SectionTitle,
  StatusChip,
  TruthLabel,
  WarningBanner,
} from '../../primitives';
import type {
  ExecutionComponent,
  ExecutionStatus,
} from '../../models/execution';
import './execution.css';

const COMPONENT_CHIP: Record<
  ExecutionComponent['state'],
  { s: 'PASS' | 'FAIL' | 'WARNING' | 'INFO' | 'UNKNOWN' | 'OFFLINE'; label: string }
> = {
  CONNECTED: { s: 'PASS', label: 'CONNECTED' },
  SCAFFOLD: { s: 'INFO', label: 'SCAFFOLD' },
  ABSENT: { s: 'OFFLINE', label: 'ABSENT' },
  NOT_CHECKED: { s: 'UNKNOWN', label: 'NOT CHECKED' },
};

export function ExecutionPage() {
  const repo = useRepository();
  const navigate = useNavigate();
  const loadable = useLoadable(
    useCallback(() => repo.getExecutionStatus(), [repo]),
  );

  return (
    <>
      <h1 className="tl-page-title">Execution</h1>
      <p className="tl-page-sub">
        Live / paper session and broker readiness. This surface is read-only.
      </p>

      <LoadableView loadable={loadable} emptyTitle="No execution data">
        {(d: ExecutionStatus) => (
          <>
            {/* The defining state: offline */}
            <div className="tl-exec-hero">
              <div className="tl-exec-hero__badge">
                <StatusChip status="OFFLINE" label="EXECUTION OFFLINE" />
              </div>
              <p className="tl-exec-hero__msg">{d.message}</p>
              <div className="tl-exec-hero__truth">
                <TruthLabel label="Mode" value={d.mode} tone="warn" />
                <TruthLabel
                  label="Broker"
                  value={d.broker === 'NOT_CONFIGURED' ? 'NOT CONFIGURED' : d.broker}
                  tone="warn"
                />
                <TruthLabel
                  label="Session"
                  value={d.session.active ? 'ACTIVE' : 'NONE'}
                  tone={d.session.active ? 'pass' : 'neutral'}
                />
                <TruthLabel label="Live P&L" value="NOT AVAILABLE" />
              </div>
            </div>

            <WarningBanner tone="info">
              No live P&L, tickers, or order controls are shown because no broker
              is connected. Nothing on this page can place, modify, or cancel
              orders.
            </WarningBanner>

            <div style={{ height: 'var(--s-4)' }} />

            <div className="tl-grid">
              {/* Preconditions */}
              <div style={{ gridColumn: 'span 6' }}>
                <AppPanel flush>
                  <PanelHeader
                    title="Readiness preconditions"
                    subtitle="All must be met before any paper/live session"
                  />
                  <ul className="tl-precond">
                    {d.preconditions.map((p) => (
                      <li key={p.label} className="tl-precond__item">
                        <StatusChip
                          status={p.met ? 'PASS' : 'INCOMPLETE'}
                          label={p.met ? 'MET' : 'PENDING'}
                        />
                        <div className="tl-precond__body">
                          <span className="tl-precond__label">{p.label}</span>
                          <span className="tl-precond__detail">{p.detail}</span>
                          <span className="tl-precond__comp mono">
                            {p.blockingComponent}
                          </span>
                        </div>
                      </li>
                    ))}
                  </ul>
                </AppPanel>
              </div>

              {/* Components */}
              <div style={{ gridColumn: 'span 6' }}>
                <AppPanel flush>
                  <PanelHeader
                    title="Execution components"
                    subtitle="Present-in-repo vs. connected"
                  />
                  <div className="tl-exec-comps">
                    {d.components.map((c) => {
                      const chip = COMPONENT_CHIP[c.state];
                      return (
                        <div className="tl-exec-comp" key={c.name}>
                          <div className="tl-exec-comp__top">
                            <span className="tl-exec-comp__name">{c.name}</span>
                            <StatusChip status={chip.s} label={chip.label} />
                          </div>
                          <span className="tl-exec-comp__detail">{c.detail}</span>
                        </div>
                      );
                    })}
                  </div>
                </AppPanel>
              </div>
            </div>

            <div style={{ height: 'var(--s-4)' }} />

            <AppPanel>
              <SectionTitle>What unblocks execution</SectionTitle>
              <p className="tl-di-note">
                Execution activates only after a strategy clears validation and
                portfolio selection, the consistency rule is configured, the MT5
                EA bridge is installed, and the live governor/kill switch are
                armed against a configured broker account. Until then this page
                stays offline by design.
              </p>
              <div style={{ marginTop: 'var(--s-3)', display: 'flex', gap: 'var(--s-2)' }}>
                <button className="tl-linklike" onClick={() => navigate('/challenge-readiness')}>
                  → Configure challenge rules
                </button>
                <button className="tl-linklike" onClick={() => navigate('/strategy-lab')}>
                  → Review strategies
                </button>
              </div>
            </AppPanel>
          </>
        )}
      </LoadableView>
    </>
  );
}
