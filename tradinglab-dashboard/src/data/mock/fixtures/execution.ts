import type { ExecutionStatus } from '../../../models/execution';

// DEVELOPMENT FIXTURE. The execution surface is intentionally OFFLINE: no broker
// bridge, no live session, no live P&L. The MT5 transport is a file-IPC contract
// only; the EA bridge is not present. This page must never fabricate live data
// (CLAUDE.md §2). Preconditions/components reflect real repo modules.

export const FIXTURE_EXECUTION: ExecutionStatus = {
  broker: 'NOT_CONFIGURED',
  mode: 'OFFLINE',
  session: {
    active: false,
    sessionId: null,
    startedAt: null,
  },
  livePnL: null,
  overallStatus: 'OFFLINE',
  message:
    'Execution is offline. No broker bridge is connected and no live or paper session exists. This surface is read-only and will not place orders.',
  preconditions: [
    {
      label: 'A validated, promoted strategy exists',
      met: false,
      detail: 'No strategy has passed validation and portfolio selection yet.',
      blockingComponent: 'validation_framework',
    },
    {
      label: 'Consistency rule configured',
      met: false,
      detail: 'FTMO consistency threshold is not set.',
      blockingComponent: 'consistency_rule',
    },
    {
      label: 'MT5 transport bridge present',
      met: false,
      detail: 'File-IPC contract exists; the EA bridge is not installed.',
      blockingComponent: 'mt5_transport',
    },
    {
      label: 'Live governor + kill switch armed',
      met: false,
      detail: 'Governor and kill switch are scaffolded but not connected to a broker.',
      blockingComponent: 'live_governor',
    },
    {
      label: 'Broker account configured',
      met: false,
      detail: 'No broker credentials or account are configured.',
      blockingComponent: 'broker_adapter',
    },
  ],
  components: [
    {
      name: 'MT5 transport',
      state: 'SCAFFOLD',
      detail: 'mt5_transport.py — file-IPC contract only, EA bridge absent.',
    },
    {
      name: 'Broker adapter',
      state: 'SCAFFOLD',
      detail: 'broker_adapter.py — interface present, not connected.',
    },
    {
      name: 'Live governor',
      state: 'SCAFFOLD',
      detail: 'live_governor.py — risk governor scaffolded.',
    },
    {
      name: 'Kill switch',
      state: 'SCAFFOLD',
      detail: 'kill_switch.py — present, not armed against a live account.',
    },
    {
      name: 'Live session',
      state: 'ABSENT',
      detail: 'No live_session has been started.',
    },
    {
      name: 'Broker connection',
      state: 'NOT_CHECKED',
      detail: 'No broker endpoint is configured to check.',
    },
  ],
};
