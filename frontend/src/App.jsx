import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import NetworkMonitor from './pages/NetworkMonitor';
import MalwareScanner from './pages/MalwareScanner';
import AIAssistant from './pages/AIAssistant';
import MLModels from './pages/MLModels';
import AdvancedModels from './pages/AdvancedModels';
import AttackTraining from './pages/AttackTraining';
import Settings from './pages/Settings';
import Logs from './pages/Logs';
import Reports from './pages/Reports';
import Analytics from './pages/Analytics';
import Database from './pages/Database';
import Help from './pages/Help';
import ThreatIntel from './pages/ThreatIntel';
import Login from './pages/Login';
import Register from './pages/Register';
import AdvancedML from './pages/AdvancedML';
import Predictions from './pages/Predictions';
import AIHub from './pages/AIHub';
import AIMLHub from './pages/AIMLHub';
// New pages
import XAIExplainer from './pages/XAIExplainer';
import SecurityHub from './pages/SecurityHub';
import AutoMLPipeline from './pages/AutoMLPipeline';
import VulnScanner from './pages/VulnScanner';
import IncidentTimeline from './pages/IncidentTimeline';
// Mega Update Pages
import AttackMap from './pages/AttackMap';
import NotificationCenter from './pages/NotificationCenter';
import DarkWebMonitor from './pages/DarkWebMonitor';
import SandboxPage from './pages/SandboxPage';
import ThreatHunting from './pages/ThreatHunting';
import SIEMIntegration from './pages/SIEMIntegration';
import BlockchainAudit from './pages/BlockchainAudit';
import Network3D from './pages/Network3D';
import ContainerSecurity from './pages/ContainerSecurity';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Auth Pages (no layout) */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Main App (with layout) */}
        <Route path="/" element={<Layout />}>
          {/* Main Pages */}
          <Route index element={<Dashboard />} />
          <Route path="network" element={<NetworkMonitor />} />
          <Route path="scanner" element={<MalwareScanner />} />
          <Route path="assistant" element={<AIAssistant />} />
          <Route path="models" element={<MLModels />} />
          <Route path="advanced-models" element={<AdvancedModels />} />
          <Route path="attack-training" element={<AttackTraining />} />

          {/* Advanced ML & Predictions */}
          <Route path="advanced-ml" element={<AdvancedML />} />
          <Route path="predictions" element={<Predictions />} />
          <Route path="ai-hub" element={<AIHub />} />
          <Route path="aiml-hub" element={<AIMLHub />} />

          {/* Tools Pages */}
          <Route path="threat-intel" element={<ThreatIntel />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="logs" element={<Logs />} />
          <Route path="reports" element={<Reports />} />
          <Route path="database" element={<Database />} />

          {/* Settings & Help */}
          <Route path="settings" element={<Settings />} />
          <Route path="help" element={<Help />} />

          {/* New Pages */}
          <Route path="xai" element={<XAIExplainer />} />
          <Route path="security-hub" element={<SecurityHub />} />
          <Route path="automl" element={<AutoMLPipeline />} />
          <Route path="vuln-scanner" element={<VulnScanner />} />
          <Route path="incidents" element={<IncidentTimeline />} />

          {/* Mega Update Pages */}
          <Route path="attack-map" element={<AttackMap />} />
          <Route path="notifications" element={<NotificationCenter />} />
          <Route path="darkweb" element={<DarkWebMonitor />} />
          <Route path="sandbox" element={<SandboxPage />} />
          <Route path="threat-hunting" element={<ThreatHunting />} />
          <Route path="siem" element={<SIEMIntegration />} />
          <Route path="blockchain" element={<BlockchainAudit />} />
          <Route path="container" element={<ContainerSecurity />} />
          <Route path="network3d" element={<Network3D />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;


