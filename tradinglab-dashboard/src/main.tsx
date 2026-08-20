import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import './design/globals.css';
import App from './app/App';
import { RepositoryProvider } from './data/RepositoryProvider';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RepositoryProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </RepositoryProvider>
  </StrictMode>,
);
