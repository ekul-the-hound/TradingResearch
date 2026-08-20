import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { AppShell } from '../src/app/shell/AppShell';
import { RepositoryProvider } from '../src/data/RepositoryProvider';
import { MockResearchRepository } from '../src/data/mock/MockResearchRepository';

function renderShell(children: React.ReactNode) {
  return render(
    <RepositoryProvider repository={new MockResearchRepository()}>
      <MemoryRouter>
        <AppShell>{children}</AppShell>
      </MemoryRouter>
    </RepositoryProvider>,
  );
}

describe('shell accessibility', () => {
  it('exposes a skip link to main content', () => {
    renderShell(<div>content</div>);
    const skip = screen.getByRole('link', { name: /skip to content/i });
    expect(skip).toHaveAttribute('href', '#tl-main');
  });

  it('has a labeled primary navigation with all routes', () => {
    renderShell(<div>content</div>);
    const nav = screen.getByRole('navigation', { name: /primary/i });
    const links = within(nav).getAllByRole('link');
    const labels = links.map((l) => l.textContent);
    expect(labels.join(' ')).toMatch(/Research Command/);
    expect(labels.join(' ')).toMatch(/Strategy Lab/);
    expect(labels.join(' ')).toMatch(/Execution/);
  });

  it('has a labeled main landmark', () => {
    renderShell(<div>content</div>);
    expect(screen.getByRole('main', { name: /main content/i })).toBeTruthy();
  });

  it('the status bar exposes a status role', () => {
    renderShell(<div>content</div>);
    // System status region
    expect(
      screen.getAllByRole('status').some((el) =>
        /system status|reading system status/i.test(el.getAttribute('aria-label') ?? el.textContent ?? ''),
      ),
    ).toBe(true);
  });
});
