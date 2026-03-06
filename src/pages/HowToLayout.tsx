import { useEffect } from 'react'
import { Box, Flex, Text, VStack, Icon } from '@chakra-ui/react'
import { Outlet, NavLink, Navigate, useLocation } from 'react-router-dom'
import { FiCpu, FiCode } from 'react-icons/fi'
import { useLanguage } from '../context/LanguageContext'
import { useColorMode } from '../context/ThemeContext'
import { useStrings } from '../i18n/strings'

interface HowToPage {
    key: string
    path: string
    icon: React.ElementType
    titleKey: 'howChatGPTTitle' | 'howClaudeCodeTitle'
    subtitleKey: 'howChatGPTSubtitle' | 'howClaudeCodeSubtitle'
}

const HOW_TO_PAGES: HowToPage[] = [
    {
        key: 'chatgpt',
        path: '/howto/chatgpt',
        icon: FiCpu,
        titleKey: 'howChatGPTTitle',
        subtitleKey: 'howChatGPTSubtitle',
    },
    {
        key: 'claudecode',
        path: '/howto/claudecode',
        icon: FiCode,
        titleKey: 'howClaudeCodeTitle',
        subtitleKey: 'howClaudeCodeSubtitle',
    },
]

export default function HowToLayout() {
    const { language } = useLanguage()
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'
    const s = useStrings(language)
    const location = useLocation()

    useEffect(() => {
        document.body.classList.remove('dark', 'light')
        document.body.classList.add(colorMode)
    }, [colorMode])

    useEffect(() => {
        document.body.classList.add('dark')
    }, [])

    if (location.pathname === '/howto' || location.pathname === '/howto/') {
        return <Navigate to="/howto/chatgpt" replace />
    }

    const sidebarBg = dark ? 'rgba(13, 21, 32, 0.85)' : 'rgba(255, 255, 255, 0.80)'
    const sidebarBorder = dark ? 'rgba(255, 255, 255, 0.07)' : 'rgba(191, 201, 209, 0.55)'
    const titleColor = dark ? 'rgba(234,239,239,0.40)' : 'rgba(37,52,63,0.40)'

    return (
        <Box minH="100vh" position="relative">
            <div className="glass-bg" />

            <Box position="fixed" top="-15%" left="-5%" w="700px" h="700px" borderRadius="full"
                bg={dark ? 'rgba(0,229,255,0.06)' : 'rgba(0,229,255,0.04)'}
                filter="blur(120px)" pointerEvents="none" zIndex={0}
                style={{ animation: 'float-blob 18s ease-in-out infinite' }} />
            <Box position="fixed" bottom="-20%" right="-10%" w="600px" h="600px" borderRadius="full"
                bg={dark ? 'rgba(37,52,63,0.80)' : 'rgba(191,201,209,0.50)'}
                filter="blur(100px)" pointerEvents="none" zIndex={0}
                style={{ animation: 'float-blob 22s ease-in-out infinite reverse' }} />

            <Flex position="relative" zIndex={1} minH="100vh" pt={{ base: '70px', md: '72px' }}>
                {/* ── Sidebar ── */}
                <Box
                    as="aside"
                    w={{ base: '72px', md: '240px' }}
                    flexShrink={0}
                    bg={sidebarBg}
                    backdropFilter="blur(20px)"
                    borderRight={`1px solid ${sidebarBorder}`}
                    position="sticky"
                    top="72px"
                    h="calc(100vh - 72px)"
                    overflowY="auto"
                    py={6}
                    px={{ base: 2, md: 4 }}
                    transition="width 0.2s"
                    boxShadow={dark ? '4px 0 24px rgba(0,0,0,0.25)' : '4px 0 16px rgba(37,52,63,0.06)'}
                >
                    <Text
                        display={{ base: 'none', md: 'block' }}
                        fontSize="10px"
                        fontWeight="700"
                        letterSpacing="0.12em"
                        textTransform="uppercase"
                        color={titleColor}
                        mb={4}
                        px={2}
                    >
                        {s.howToSidebarTitle}
                    </Text>

                    <VStack spacing={1} align="stretch">
                        {HOW_TO_PAGES.map((page) => {
                            const isActive = location.pathname.startsWith(page.path)
                            return (
                                <NavLink key={page.key} to={page.path} style={{ textDecoration: 'none' }}>
                                    <Flex
                                        align="center"
                                        gap={3}
                                        px={{ base: 2, md: 3 }}
                                        py={{ base: 3, md: 2.5 }}
                                        borderRadius="12px"
                                        cursor="pointer"
                                        transition="all 0.18s"
                                        bg={isActive
                                            ? dark ? 'rgba(0,229,255,0.10)' : 'rgba(0,229,255,0.09)'
                                            : 'transparent'}
                                        border={`1px solid ${isActive
                                            ? dark ? 'rgba(0,229,255,0.30)' : 'rgba(0,190,220,0.35)'
                                            : 'transparent'}`}
                                        boxShadow={isActive
                                            ? dark ? '0 0 16px rgba(0,229,255,0.12)' : '0 0 12px rgba(0,180,210,0.10)'
                                            : 'none'}
                                        _hover={{
                                            bg: isActive ? undefined : dark ? 'rgba(255,255,255,0.04)' : 'rgba(37,52,63,0.05)',
                                            borderColor: isActive ? undefined : dark ? 'rgba(255,255,255,0.08)' : 'rgba(37,52,63,0.12)',
                                        }}
                                    >
                                        <Icon
                                            as={page.icon}
                                            boxSize={4}
                                            color={isActive
                                                ? dark ? '#00e5ff' : '#0099bb'
                                                : dark ? 'rgba(234,239,239,0.35)' : 'rgba(37,52,63,0.35)'}
                                            flexShrink={0}
                                        />
                                        <Box display={{ base: 'none', md: 'block' }} minW={0}>
                                            <Text
                                                fontSize="13px"
                                                fontWeight="600"
                                                color={isActive
                                                    ? dark ? '#00e5ff' : '#0099bb'
                                                    : dark ? 'rgba(234,239,239,0.70)' : 'rgba(37,52,63,0.70)'}
                                                lineHeight={1.3}
                                                noOfLines={1}
                                            >
                                                {s[page.titleKey]}
                                            </Text>
                                            <Text
                                                fontSize="10px"
                                                color={dark ? 'rgba(234,239,239,0.30)' : 'rgba(37,52,63,0.38)'}
                                                fontWeight="400"
                                                lineHeight={1.4}
                                                mt={0.5}
                                                noOfLines={2}
                                            >
                                                {s[page.subtitleKey]}
                                            </Text>
                                        </Box>
                                    </Flex>
                                </NavLink>
                            )
                        })}
                    </VStack>
                </Box>

                {/* ── Main content ── */}
                <Box flex={1} overflowX="hidden">
                    <Outlet />
                </Box>
            </Flex>
        </Box>
    )
}
