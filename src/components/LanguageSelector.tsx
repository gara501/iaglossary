import { Box, HStack, Text, Icon } from '@chakra-ui/react'
import { FiSun, FiMoon } from 'react-icons/fi'
import { useLanguage } from '../context/LanguageContext'
import { useColorMode } from '../context/ThemeContext'

export default function LanguageSelector({ currentPage, onPageChange }: { currentPage: 'glossary' | 'learning', onPageChange: (page: 'glossary' | 'learning') => void }) {
    const { language, setLanguage } = useLanguage()
    const { colorMode, toggleColorMode } = useColorMode()
    const dark = colorMode === 'dark'

    const pillBg = dark ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.72)'
    const pillBorder = dark ? 'rgba(255,255,255,0.08)' : 'rgba(191,201,209,0.60)'
    const activeBg = dark ? 'rgba(255,155,81,0.20)' : '#FF9B51'
    const activeBdr = dark ? 'rgba(255,155,81,0.45)' : '#e07e38'
    const activeText = dark ? '#FF9B51' : '#ffffff'
    const inactiveText = dark ? 'rgba(234,239,239,0.38)' : 'rgba(37,52,63,0.42)'

    return (
        <Box position="fixed" top={4} right={4} zIndex={1000}>
            <HStack spacing={2}>
                {/* Page Toggle */}
                <HStack
                    spacing={0}
                    bg={pillBg}
                    backdropFilter="blur(16px)"
                    border={`1px solid ${pillBorder}`}
                    borderRadius="12px" p="3px"
                    boxShadow={dark ? '0 4px 24px rgba(0,0,0,0.30)' : '0 2px 12px rgba(37,52,63,0.08)'}
                >
                    {(['glossary', 'learning'] as const).map((page) => {
                        const isActive = currentPage === page
                        return (
                            <Box
                                key={page} as="button"
                                onClick={() => onPageChange(page)}
                                px={3} py={1.5} borderRadius="9px"
                                bg={isActive ? activeBg : 'transparent'}
                                border={isActive ? `1px solid ${activeBdr}` : '1px solid transparent'}
                                transition="all 0.18s"
                                _hover={{ bg: isActive ? undefined : 'rgba(255,155,81,0.08)' }}
                                cursor="pointer"
                            >
                                <Text fontSize="11px" fontWeight="700"
                                    color={isActive ? activeText : inactiveText}
                                    letterSpacing="0.06em">
                                    {page === 'glossary' ? (language === 'es' ? 'GLOSARIO' : 'GLOSSARY') : (language === 'es' ? 'APRENDIZAJE' : 'LEARNING')}
                                </Text>
                            </Box>
                        )
                    })}
                </HStack>

                {/* Dark/Light toggle */}
                <Box
                    as="button"
                    onClick={toggleColorMode}
                    w="38px" h="38px" borderRadius="11px"
                    bg={pillBg}
                    border={`1px solid ${pillBorder}`}
                    backdropFilter="blur(16px)"
                    display="flex" alignItems="center" justifyContent="center"
                    cursor="pointer" transition="all 0.18s"
                    _hover={{ bg: dark ? 'rgba(255,155,81,0.12)' : 'rgba(255,155,81,0.12)', borderColor: 'rgba(255,155,81,0.40)' }}
                    boxShadow={dark ? '0 4px 20px rgba(0,0,0,0.30)' : '0 2px 12px rgba(37,52,63,0.08)'}
                    title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                    <Icon as={dark ? FiSun : FiMoon} boxSize={4} color="#FF9B51" />
                </Box>

                {/* Language switcher */}
                <HStack
                    spacing={0}
                    bg={pillBg}
                    backdropFilter="blur(16px)"
                    border={`1px solid ${pillBorder}`}
                    borderRadius="12px" p="3px"
                    boxShadow={dark ? '0 4px 24px rgba(0,0,0,0.30)' : '0 2px 12px rgba(37,52,63,0.08)'}
                >
                    {(['en', 'es'] as const).map((lang) => {
                        const isActive = language === lang
                        return (
                            <Box
                                key={lang} as="button"
                                onClick={() => setLanguage(lang)}
                                px={3} py={1.5} borderRadius="9px"
                                bg={isActive ? activeBg : 'transparent'}
                                border={isActive ? `1px solid ${activeBdr}` : '1px solid transparent'}
                                transition="all 0.18s"
                                _hover={{ bg: isActive ? undefined : 'rgba(255,155,81,0.08)' }}
                                cursor="pointer"
                            >
                                <HStack spacing={1.5}>
                                    <Text fontSize="13px">{lang === 'en' ? '🇺🇸' : '🇪🇸'}</Text>
                                    <Text fontSize="11px" fontWeight="700"
                                        color={isActive ? activeText : inactiveText}
                                        letterSpacing="0.06em">
                                        {lang.toUpperCase()}
                                    </Text>
                                </HStack>
                            </Box>
                        )
                    })}
                </HStack>
            </HStack>
        </Box>
    )
}
