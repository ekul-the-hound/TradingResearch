import type { ReactNode } from 'react';
import type { Loadable } from '../models/truth';
import {
  EmptyState,
  ErrorState,
  LoadingState,
  UnavailableState,
} from '../primitives';

// Renders the non-ready states uniformly; delegates ready to the child render.
export function LoadableView<T>({
  loadable,
  children,
  emptyTitle,
  emptyMessage,
  loadingRows,
}: {
  loadable: Loadable<T>;
  children: (data: T, isFixture: boolean) => ReactNode;
  emptyTitle?: string;
  emptyMessage?: string;
  loadingRows?: number;
}) {
  switch (loadable.state) {
    case 'loading':
      return <LoadingState rows={loadingRows} />;
    case 'error':
      return <ErrorState message={loadable.error} />;
    case 'unavailable':
      return <UnavailableState reason={loadable.reason} />;
    case 'empty':
      return <EmptyState title={emptyTitle} message={emptyMessage} />;
    case 'ready':
      return <>{children(loadable.data, loadable.isFixture)}</>;
  }
}
