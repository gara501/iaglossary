import { Routes, Route, Navigate } from 'react-router-dom'
import GlossaryPage from './pages/GlossaryPage'
import LearningPage from './pages/LearningPage'
import LoginPage from './pages/LoginPage'
import AdminLayout from './pages/admin/AdminLayout'
import GlossaryAdmin from './pages/admin/GlossaryAdmin'
import LearningAdmin from './pages/admin/LearningAdmin'
import SimulationsLayout from './pages/SimulationsLayout'
import TokenizationSim from './pages/TokenizationSim'
import RagPipelineSim from './pages/RagPipelineSim'
import TransformerAttentionSim from './pages/TransformerAttentionSim'
import HowToLayout from './pages/HowToLayout'
import HowChatGPTWorks from './pages/HowChatGPTWorks'
import HowClaudeCodeWorks from './pages/HowClaudeCodeWorks'
import LanguageSelector from './components/LanguageSelector'
import ProtectedRoute from './components/ProtectedRoute'

function App() {
    return (
        <>
            <Routes>
                {/* Public routes */}
                <Route path="/" element={<><LanguageSelector /><GlossaryPage /></>} />
                <Route path="/learning" element={<><LanguageSelector /><LearningPage /></>} />

                {/* Simulations */}
                <Route path="/simulations" element={<><LanguageSelector /><SimulationsLayout /></>}>
                    <Route index element={<Navigate to="/simulations/tokenization" replace />} />
                    <Route path="tokenization" element={<TokenizationSim />} />
                    <Route path="rag" element={<RagPipelineSim />} />
                    <Route path="attention" element={<TransformerAttentionSim />} />
                </Route>

                {/* How To */}
                <Route path="/howto" element={<><LanguageSelector /><HowToLayout /></>}>
                    <Route index element={<Navigate to="/howto/chatgpt" replace />} />
                    <Route path="chatgpt" element={<HowChatGPTWorks />} />
                    <Route path="claudecode" element={<HowClaudeCodeWorks />} />
                </Route>

                {/* Auth */}
                <Route path="/login" element={<LoginPage />} />

                {/* Protected admin routes */}
                <Route path="/admin" element={
                    <ProtectedRoute>
                        <AdminLayout />
                    </ProtectedRoute>
                }>
                    <Route index element={<Navigate to="/admin/glossary" replace />} />
                    <Route path="glossary" element={<GlossaryAdmin />} />
                    <Route path="learning" element={<LearningAdmin />} />
                </Route>
            </Routes>
        </>
    )
}

export default App
