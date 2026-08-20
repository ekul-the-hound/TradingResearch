import type { BrokerState, Status } from './truth';

export interface ExecutionPrecondition {
  label: string;
  met: boolean;
  detail: string;
  blockingComponent: string; // e.g. "mt5_transport", "live_governor"
}

export interface ExecutionComponent {
  name: string;
  // present-in-repo but not connected == SCAFFOLD; missing == ABSENT
  state: 'CONNECTED' | 'SCAFFOLD' | 'ABSENT' | 'NOT_CHECKED';
  detail: string;
}

export interface ExecutionStatus {
  // The whole page hinges on this: default is OFFLINE / NOT_CONFIGURED.
  broker: BrokerState;
  mode: 'OFFLINE' | 'PAPER' | 'DEMO' | 'LIVE';
  // no live session exists
  session: {
    active: boolean;
    sessionId: string | null;
    startedAt: string | null;
  };
  // there is NO live P&L — this is always null/absent, never fabricated
  livePnL: null;
  preconditions: ExecutionPrecondition[];
  components: ExecutionComponent[];
  overallStatus: Status; // OFFLINE
  message: string;
}
