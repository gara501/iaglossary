import { Box, HStack, Text, Icon } from '@chakra-ui/react'
import { FiSun, FiMoon } from 'react-icons/fi'
import { useLanguage } from '../context/LanguageContext'
import { useColorMode } from '../context/ThemeContext'

export default function LanguageSelector() {
    const { language, setLanguage } = useLanguage()
    const { colorMode, toggleColorMode } = useColorMode()
    const dark = colorMode === 'dark'

    const pillBg = dark ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.75)'
    const pillBorder = dark ? 'rgba(255,255,255,0.08)' : 'rgba(197,0,60,0.14)'
    const activeBg = dark ? 'rgba(197,0,60,0.22)' : '#c5003c'
    const activeBdr = dark ? 'rgba(197,0,60,0.45)' : '#880425'
    const activeText = dark ? '#f3e600' : '#f3e600'
    const inactiveText = dark ? 'rgba(245,234,236,0.35)' : 'rgba(26,10,13,0.40)'

    return (
        <Box position="fixed" top={4} right={4} zIndex={1000}>
            <HStack spacing={2}>
                {/* Dark/Light toggle */}
                <Box
                    as="button"
                    onClick={toggleColorMode}
                    w="38px" h="38px"
                    borderRadius="11px"
                    bg={pillBg}
                    border={`1px solid ${pillBorder}`}
                    backdropFilter="blur(16px)"
                    display="flex" alignItems="center" justifyContent="center"
                    cursor="pointer"
                    transition="all 0.18s"
                    _hover={{ bg: dark ? 'rgba(197,0,60,0.14)' : 'rgba(197,0,60,0.10)', borderColor: 'rgba(197,0,60,0.30)' }}
                    boxShadow={dark ? '0 4px 20px rgba(0,0,0,0.30)' : '0 2px 12px rgba(0,0,0,0.08)'}
                    title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                    <Icon
                        as={dark ? FiSun : FiMoon}
                        boxSize={4}
                        color={dark ? '#f3e600' : '#c5003c'}
                    />
                </Box>

                {/* Language switcher */}
                <HStack
                    spacing={0}
                    bg={pillBg}
                    backdropFilter="blur(16px)"
                    border={`1px solid ${pillBorder}`}
                    borderRadius="12px"
                    p="3px"
                    boxShadow={dark ? '0 4px 24px rgba(0,0,0,0.35)' : '0 2px 12px rgba(0,0,0,0.08)'}
                >
                    {(['en', 'es'] as const).map((lang) => {
                        const isActive = language === lang
                        return (
                            <Box
                                key={lang}
                                as="button"
                                onClick={() => setLanguage(lang)}
                                px={3} py={1.5}
                                borderRadius="9px"
                                bg={isActive ? activeBg : 'transparent'}
                                border={isActive ? `1px solid ${activeBdr}` : '1px solid transparent'}
                                transition="all 0.18s"
                                _hover={{ bg: isActive ? undefined : (dark ? 'rgba(197,0,60,0.08)' : 'rgba(197,0,60,0.06)') }}
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
