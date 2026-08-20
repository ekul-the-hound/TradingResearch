import { useCallback } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { useRepository } from '../../data/useRepository';
import { useLoadable } from '../../lib/hooks';
import { LoadableView, StatusChip } from '../../primitives';
import { DetailHeader } from './DetailHeader';
import { PerformanceTab } from './PerformanceTab';
import { ValidationTab } from './ValidationTab';
import { RiskTab } from './RiskTab';
import { ChallengeFitTab, ProvenanceTab } from './ChallengeProvenanceTabs';
import './strategy-detail.css';

const TABS = [
  { id: 'performance', label: 'Performance' },
  { id: 'validation', label: 'Validation' },
  { id: 'risk', label: 'Risk' },
  { id: 'challenge', label: 'Challenge Fit' },
  { id: 'provenance', label: 'Provenance' },
] as const;

type TabId = (typeof TABS)[number]['id'];

export function StrategyDetailPage() {
  const { strategyId = '' } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const repo = useRepository();

  const activeTab: TabId =
    (TABS.find((t) => `#${t.id}` === location.hash)?.id as TabId) ??
    'performance';

  const detail = useLoadable(
    useCallback(() => repo.getStrategyDetail(strategyId), [repo, strategyId]),
  );
  const ftmo = useLoadable(
    useCallback(() => repo.getFTMOCompliance(strategyId), [repo, strategyId]),
  );
  const sim = useLoadable(
    useCallback(() => repo.getChallengeSimulation(strategyId), [repo, strategyId]),
  );

  return (
    <div>
      <button className="tl-backlink" onClick={() => navigate('/strategy-lab')}>
        ← Strategy Lab
      </button>

      <LoadableView
        loadable={detail}
        emptyTitle="Strategy not found"
        emptyMessage={`No strategy with id "${strategyId}".`}
      >
        {(d, isFixture) => (
          <>
            {isFixture && (
              <div style={{ margin: '6px 0' }}>
                <StatusChip status="INFO" label="DEV FIXTURE" />
              </div>
            )}
            <DetailHeader detail={d} />

            <nav className="tl-tabs" role="tablist" aria-label="Strategy evidence">
              {TABS.map((t, i) => (
                <button
                  key={t.id}
                  role="tab"
                  id={`tab-${t.id}`}
                  aria-controls="tl-tabpanel"
                  aria-selected={activeTab === t.id}
                  tabIndex={activeTab === t.id ? 0 : -1}
                  className={`tl-tab ${activeTab === t.id ? 'is-active' : ''}`}
                  onClick={() =>
                    navigate(`/strategies/${strategyId}#${t.id}`, { replace: true })
                  }
                  onKeyDown={(e) => {
                    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
                      e.preventDefault();
                      const dir = e.key === 'ArrowRight' ? 1 : -1;
                      const next = TABS[(i + dir + TABS.length) % TABS.length];
                      navigate(`/strategies/${strategyId}#${next.id}`, {
                        replace: true,
                      });
                    }
                  }}
                >
                  {t.label}
                </button>
              ))}
            </nav>

            <div
              className="tl-tabpanel"
              role="tabpanel"
              id="tl-tabpanel"
              aria-labelledby={`tab-${activeTab}`}
            >
              {activeTab === 'performance' && <PerformanceTab perf={d.performance} />}
              {activeTab === 'validation' && <ValidationTab v={d.validation} />}
              {activeTab === 'risk' && <RiskTab risk={d.risk} />}
              {activeTab === 'challenge' && (
                <ChallengeFitTab ftmo={ftmo} sim={sim} />
              )}
              {activeTab === 'provenance' && <ProvenanceTab detail={d} />}
            </div>
          </>
        )}
      </LoadableView>
    </div>
  );
}
