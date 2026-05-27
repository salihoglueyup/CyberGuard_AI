import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
import { lazy, Suspense, useEffect } from 'react';
import ErrorBoundary from './components/ErrorBoundary';
import ProtectedRoute from './components/ProtectedRoute';
import Layout from './components/Layout';
import Login from './pages/auth/Login';
import Register from './pages/auth/Register';
import PerformanceOverlay from './components/shared/PerformanceOverlay';

// Lazy loaded pages — Core
const Dashboard = lazy(() => import('./pages/core/Dashboard'));
const Settings = lazy(() => import('./pages/core/Settings'));
const Logs = lazy(() => import('./pages/core/Logs'));
const Reports = lazy(() => import('./pages/core/Reports'));
const Analytics = lazy(() => import('./pages/core/Analytics'));
const Database = lazy(() => import('./pages/core/Database'));
const Help = lazy(() => import('./pages/core/Help'));
const NotificationCenter = lazy(() => import('./pages/core/NotificationCenter'));

// Lazy loaded pages — AI & ML
const AIAssistant = lazy(() => import('./pages/ai/AIAssistant'));
const AIHub = lazy(() => import('./pages/ai/AIHub'));
const AIMLHub = lazy(() => import('./pages/ai/AIMLHub'));
const MLModels = lazy(() => import('./pages/ai/MLModels'));
const AdvancedModels = lazy(() => import('./pages/ai/AdvancedModels'));
const AdvancedML = lazy(() => import('./pages/ai/AdvancedML'));
const AttackTraining = lazy(() => import('./pages/ai/AttackTraining'));
const Predictions = lazy(() => import('./pages/ai/Predictions'));
const XAIExplainer = lazy(() => import('./pages/ai/XAIExplainer'));
const AutoMLPipeline = lazy(() => import('./pages/ai/AutoMLPipeline'));

// Lazy loaded pages — Security
const MalwareScanner = lazy(() => import('./pages/security/MalwareScanner'));
const VulnScanner = lazy(() => import('./pages/security/VulnScanner'));
const SecurityHub = lazy(() => import('./pages/security/SecurityHub'));
const SandboxPage = lazy(() => import('./pages/security/SandboxPage'));
const ContainerSecurity = lazy(() => import('./pages/security/ContainerSecurity'));
const BlockchainAudit = lazy(() => import('./pages/security/BlockchainAudit'));
const ThreatHunting = lazy(() => import('./pages/security/ThreatHunting'));
const DarkWebMonitor = lazy(() => import('./pages/security/DarkWebMonitor'));
const ComplianceDashboard = lazy(() => import('./pages/security/ComplianceDashboard'));
const ForensicsLab = lazy(() => import('./pages/security/ForensicsLab'));
const HoneypotManager = lazy(() => import('./pages/security/HoneypotManager'));
const ApiSecurity = lazy(() => import('./pages/security/ApiSecurity'));
const UserBehavior = lazy(() => import('./pages/security/UserBehavior'));
const PentestDashboard = lazy(() => import('./pages/security/PentestDashboard'));

// Lazy loaded pages — Monitoring
const NetworkMonitor = lazy(() => import('./pages/monitoring/NetworkMonitor'));
const Network3D = lazy(() => import('./pages/monitoring/Network3D'));
const AttackMap = lazy(() => import('./pages/monitoring/AttackMap'));
const IncidentTimeline = lazy(() => import('./pages/monitoring/IncidentTimeline'));
const ThreatIntel = lazy(() => import('./pages/monitoring/ThreatIntel'));
const SIEMIntegration = lazy(() => import('./pages/monitoring/SIEMIntegration'));
const GlobeView = lazy(() => import('./pages/monitoring/GlobeView'));
const TopologyMap = lazy(() => import('./pages/monitoring/TopologyMap'));

// 404 → redirect to home
const NotFoundRedirect = () => {
  const nav = useNavigate();
  useEffect(() => { nav('/', { replace: true }); }, [nav]);
  return null;
};

const PageLoader = () => (
  <div className="flex flex-col items-center justify-center h-64 gap-4">
    <div className="relative">
      <div className="w-12 h-12 rounded-full border-2 border-[var(--hud-border)] border-t-[var(--hud-cyan)] animate-spin" />
      <div className="absolute inset-0 w-12 h-12 rounded-full border-2 border-transparent border-b-[var(--hud-purple)] animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }} />
    </div>
    <p className="text-[10px] font-mono text-[var(--hud-text-dim)] tracking-wide animate-pulse">Modül Yükleniyor</p>
  </div>
);

function App() {
  return (
    <ErrorBoundary>
    <BrowserRouter>
      <Routes>
        {/* Auth Pages (no layout) */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Main App (with layout + auth guard) */}
        <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
          <Route index element={<Suspense fallback={<PageLoader />}><Dashboard /></Suspense>} />
          <Route path="network" element={<Suspense fallback={<PageLoader />}><NetworkMonitor /></Suspense>} />
          <Route path="scanner" element={<Suspense fallback={<PageLoader />}><MalwareScanner /></Suspense>} />
          <Route path="assistant" element={<Suspense fallback={<PageLoader />}><AIAssistant /></Suspense>} />
          <Route path="models" element={<Suspense fallback={<PageLoader />}><MLModels /></Suspense>} />
          <Route path="advanced-models" element={<Suspense fallback={<PageLoader />}><AdvancedModels /></Suspense>} />
          <Route path="attack-training" element={<Suspense fallback={<PageLoader />}><AttackTraining /></Suspense>} />

          <Route path="advanced-ml" element={<Suspense fallback={<PageLoader />}><AdvancedML /></Suspense>} />
          <Route path="predictions" element={<Suspense fallback={<PageLoader />}><Predictions /></Suspense>} />
          <Route path="ai-hub" element={<Suspense fallback={<PageLoader />}><AIHub /></Suspense>} />
          <Route path="aiml-hub" element={<Suspense fallback={<PageLoader />}><AIMLHub /></Suspense>} />

          <Route path="threat-intel" element={<Suspense fallback={<PageLoader />}><ThreatIntel /></Suspense>} />
          <Route path="analytics" element={<Suspense fallback={<PageLoader />}><Analytics /></Suspense>} />
          <Route path="logs" element={<Suspense fallback={<PageLoader />}><Logs /></Suspense>} />
          <Route path="reports" element={<Suspense fallback={<PageLoader />}><Reports /></Suspense>} />
          <Route path="database" element={<Suspense fallback={<PageLoader />}><Database /></Suspense>} />

          <Route path="settings" element={<Suspense fallback={<PageLoader />}><Settings /></Suspense>} />
          <Route path="help" element={<Suspense fallback={<PageLoader />}><Help /></Suspense>} />

          <Route path="xai" element={<Suspense fallback={<PageLoader />}><XAIExplainer /></Suspense>} />
          <Route path="security-hub" element={<Suspense fallback={<PageLoader />}><SecurityHub /></Suspense>} />
          <Route path="automl" element={<Suspense fallback={<PageLoader />}><AutoMLPipeline /></Suspense>} />
          <Route path="vuln-scanner" element={<Suspense fallback={<PageLoader />}><VulnScanner /></Suspense>} />
          <Route path="incidents" element={<Suspense fallback={<PageLoader />}><IncidentTimeline /></Suspense>} />

          <Route path="attack-map" element={<Suspense fallback={<PageLoader />}><AttackMap /></Suspense>} />
          <Route path="notifications" element={<Suspense fallback={<PageLoader />}><NotificationCenter /></Suspense>} />
          <Route path="darkweb" element={<Suspense fallback={<PageLoader />}><DarkWebMonitor /></Suspense>} />
          <Route path="sandbox" element={<Suspense fallback={<PageLoader />}><SandboxPage /></Suspense>} />
          <Route path="threat-hunting" element={<Suspense fallback={<PageLoader />}><ThreatHunting /></Suspense>} />
          <Route path="siem" element={<Suspense fallback={<PageLoader />}><SIEMIntegration /></Suspense>} />
          <Route path="blockchain" element={<Suspense fallback={<PageLoader />}><BlockchainAudit /></Suspense>} />
          <Route path="container" element={<Suspense fallback={<PageLoader />}><ContainerSecurity /></Suspense>} />
          <Route path="network3d" element={<Suspense fallback={<PageLoader />}><Network3D /></Suspense>} />
          <Route path="globe" element={<Suspense fallback={<PageLoader />}><GlobeView /></Suspense>} />
          <Route path="topology" element={<Suspense fallback={<PageLoader />}><TopologyMap /></Suspense>} />
          <Route path="compliance" element={<Suspense fallback={<PageLoader />}><ComplianceDashboard /></Suspense>} />
          <Route path="forensics" element={<Suspense fallback={<PageLoader />}><ForensicsLab /></Suspense>} />
          <Route path="honeypot" element={<Suspense fallback={<PageLoader />}><HoneypotManager /></Suspense>} />
          <Route path="api-security" element={<Suspense fallback={<PageLoader />}><ApiSecurity /></Suspense>} />
          <Route path="user-behavior" element={<Suspense fallback={<PageLoader />}><UserBehavior /></Suspense>} />
          <Route path="pentest" element={<Suspense fallback={<PageLoader />}><PentestDashboard /></Suspense>} />
          <Route path="*" element={<NotFoundRedirect />} />
        </Route>
      </Routes>
    </BrowserRouter>
    <PerformanceOverlay />
    </ErrorBoundary>
  );
}

export default App;


