import { useState } from 'react'
import GlossaryPage from './pages/GlossaryPage'
import LearningPage from './pages/LearningPage'
import LanguageSelector from './components/LanguageSelector'

function App() {
    const [currentPage, setCurrentPage] = useState<'glossary' | 'learning'>('glossary')

    return (
        <>
            <LanguageSelector currentPage={currentPage} onPageChange={setCurrentPage} />
            {currentPage === 'glossary' ? <GlossaryPage /> : <LearningPage onReturn={() => setCurrentPage('glossary')} />}
        </>
    )
}

export default App
