import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ChakraProvider } from '@chakra-ui/react'
import { BrowserRouter } from 'react-router-dom'
import { LanguageProvider } from './context/LanguageContext'
import { ThemeProvider } from './context/ThemeContext'
import { AuthProvider } from './context/AuthContext'
import App from './App'
import './index.css'

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <ChakraProvider>
            <BrowserRouter basename="/iaglossary">
                <LanguageProvider>
                    <ThemeProvider>
                        <AuthProvider>
                            <App />
                        </AuthProvider>
                    </ThemeProvider>
                </LanguageProvider>
            </BrowserRouter>
        </ChakraProvider>
    </StrictMode>,
)
