import React from 'react'
import ReactDOM from 'react-dom/client'
import { ChakraProvider } from '@chakra-ui/react'
import App from './App'
import theme from './theme'
import { LanguageProvider } from './context/LanguageContext'
import { ThemeProvider } from './context/ThemeContext'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <ThemeProvider>
            <LanguageProvider>
                <ChakraProvider theme={theme}>
                    <App />
                </ChakraProvider>
            </LanguageProvider>
        </ThemeProvider>
    </React.StrictMode>,
)

