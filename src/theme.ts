import { extendTheme } from '@chakra-ui/react'

const theme = extendTheme({
    config: {
        initialColorMode: 'dark',
        useSystemColorMode: false,
    },
    styles: {
        global: {
            body: {
                bg: '#060d1f',
                color: 'white',
            },
        },
    },
    fonts: {
        heading: "'Space Grotesk', sans-serif",
        body: "'Inter', sans-serif",
    },
    colors: {
        glass: {
            50: 'rgba(255,255,255,0.03)',
            100: 'rgba(255,255,255,0.06)',
            200: 'rgba(255,255,255,0.09)',
            300: 'rgba(255,255,255,0.13)',
            400: 'rgba(255,255,255,0.18)',
            border: 'rgba(255,255,255,0.10)',
            borderHover: 'rgba(255,255,255,0.20)',
        },
        accent: {
            300: '#a5b4fc',
            400: '#818cf8',
            500: '#6366f1',
            600: '#4f46e5',
            700: '#3730a3',
        },
        blue: {
            300: '#93c5fd',
            400: '#60a5fa',
            500: '#3b82f6',
        },
    },
    components: {
        Modal: {
            baseStyle: {
                dialog: {
                    bg: 'rgba(10, 15, 40, 0.85)',
                    backdropFilter: 'blur(24px)',
                    border: '1px solid rgba(255,255,255,0.10)',
                    borderRadius: '24px',
                },
                overlay: {
                    bg: 'rgba(0,0,0,0.65)',
                    backdropFilter: 'blur(8px)',
                },
            },
        },
        Badge: {
            baseStyle: {
                borderRadius: 'full',
                fontWeight: '600',
                letterSpacing: '0.06em',
                fontSize: '10px',
            },
        },
    },
})

export default theme
