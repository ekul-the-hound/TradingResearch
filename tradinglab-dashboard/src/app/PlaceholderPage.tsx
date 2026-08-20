import { AppPanel, EmptyState } from '../primitives';

export function PlaceholderPage({ title }: { title: string }) {
  return (
    <>
      <h1 className="tl-page-title">{title}</h1>
      <p className="tl-page-sub">This route is scheduled in a later build prompt.</p>
      <AppPanel flush>
        <EmptyState
          title="Not yet built"
          message="The design system and shell are in place. This page's content arrives in its dedicated prompt."
        />
      </AppPanel>
    </>
  );
}
