import { createContext, useContext, useState, ReactNode } from 'react'

type ColorMode = 'dark' | 'light'

interface ThemeContextType {
    colorMode: ColorMode
    toggleColorMode: () => void
}

const ThemeContext = createContext<ThemeContextType>({
    colorMode: 'dark',
    toggleColorMode: () => { },
})

export function ThemeProvider({ children }: { children: ReactNode }) {
    const [colorMode, setColorMode] = useState<ColorMode>('dark')
    const toggleColorMode = () => setColorMode((m) => (m === 'dark' ? 'light' : 'dark'))
    return (
        <ThemeContext.Provider value={{ colorMode, toggleColorMode }}>
            {children}
        </ThemeContext.Provider>
    )
}

export function useColorMode() {
    return useContext(ThemeContext)
}
