import { extendTheme } from '@chakra-ui/react'

const theme = extendTheme({
    config: {
        initialColorMode: 'dark',
        useSystemColorMode: false,
    },
    styles: {
        global: {
            body: {
                bg: '#0a0608',
                color: '#f1e8ea',
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
            100: 'rgba(255,255,255,0.05)',
            200: 'rgba(255,255,255,0.08)',
            300: 'rgba(255,255,255,0.12)',
            400: 'rgba(255,255,255,0.17)',
            border: 'rgba(255,255,255,0.08)',
            borderHover: 'rgba(225,29,72,0.25)',
        },
        accent: {
            300: '#fda4af',  // rose-300
            400: '#fb7185',  // rose-400
            500: '#f43f5e',  // rose-500
            600: '#e11d48',  // rose-600
            700: '#be123c',  // rose-700
        },
    },
    components: {
        Modal: {
            baseStyle: {
                dialog: {
                    bg: 'rgba(12, 5, 8, 0.88)',
                    backdropFilter: 'blur(24px)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: '24px',
                },
                overlay: {
                    bg: 'rgba(0,0,0,0.70)',
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
