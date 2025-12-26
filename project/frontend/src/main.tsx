import { createRoot } from 'react-dom/client'
import { ErrorBoundary } from "react-error-boundary";
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider } from "next-themes";

import App from './App.tsx'
import { ErrorFallback } from './ErrorFallback.tsx'
import { ApiKeyProvider } from './contexts/ApiKeyContext.tsx'
import { AuthProvider } from './contexts/AuthContext.tsx'
import { initializeTelemetry, getMeter } from './lib/telemetry'
import { initializeMetrics } from './lib/metrics'

import "./main.css"
import "./styles/theme.css"
import "./index.css"

// Initialize OpenTelemetry
try {
  initializeTelemetry();
  const meter = getMeter();
  initializeMetrics(meter);
} catch (error) {
  console.error('Failed to initialize telemetry:', error);
}

createRoot(document.getElementById('root')!).render(
  <ErrorBoundary FallbackComponent={ErrorFallback}>
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      enableColorScheme={false}
    >
      <BrowserRouter>
        <AuthProvider>
          <ApiKeyProvider>
            <App />
          </ApiKeyProvider>
        </AuthProvider>
      </BrowserRouter>
    </ThemeProvider>
   </ErrorBoundary>
)
