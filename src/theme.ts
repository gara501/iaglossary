import { extendTheme, type ThemeConfig } from '@chakra-ui/react'

const config: ThemeConfig = {
    initialColorMode: 'dark',
    useSystemColorMode: false,
}

const theme = extendTheme({
    config,
    fonts: {
        heading: `'Space Grotesk', sans-serif`,
        body: `'Inter', sans-serif`,
    },
    colors: {
        brand: {
            50: '#f0e6ff',
            100: '#d4b3ff',
            200: '#b980ff',
            300: '#9d4dff',
            400: '#8226ff',
            500: '#6600e6',
            600: '#5000b3',
            700: '#3a0080',
            800: '#25004d',
            900: '#10001a',
        },
        accent: {
            400: '#00d4ff',
            500: '#00b8e6',
        },
    },
    styles: {
        global: {
            body: {
                bg: '#0a0a1a',
                color: 'white',
            },
        },
    },
    components: {
        Button: {
            variants: {
                letter: {
                    bg: 'whiteAlpha.100',
                    color: 'white',
                    borderRadius: 'md',
                    fontWeight: '600',
                    fontSize: 'sm',
                    _hover: {
                        bg: 'brand.500',
                        transform: 'translateY(-2px)',
                        boxShadow: '0 4px 15px rgba(130, 38, 255, 0.4)',
                    },
                    _active: {
                        bg: 'brand.400',
                    },
                    transition: 'all 0.2s',
                },
                letterActive: {
                    bg: 'brand.500',
                    color: 'white',
                    borderRadius: 'md',
                    fontWeight: '700',
                    fontSize: 'sm',
                    boxShadow: '0 4px 15px rgba(130, 38, 255, 0.5)',
                    transform: 'translateY(-2px)',
                    _hover: {
                        bg: 'brand.400',
                    },
                },
            },
        },
    },
})

export default theme
