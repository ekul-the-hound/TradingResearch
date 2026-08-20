import { AppShell } from './shell/AppShell';
import { AppRoutes } from './routes';

export default function App() {
  return (
    <AppShell>
      <AppRoutes />
    </AppShell>
  );
}
