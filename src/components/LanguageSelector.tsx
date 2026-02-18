import { Box, HStack, Text } from '@chakra-ui/react'
import { useLanguage } from '../context/LanguageContext'

export default function LanguageSelector() {
    const { language, setLanguage } = useLanguage()

    return (
        <Box
            position="fixed"
            top={4}
            right={4}
            zIndex={1000}
        >
            <HStack
                spacing={0}
                bg="rgba(255,255,255,0.06)"
                backdropFilter="blur(16px)"
                border="1px solid rgba(255,255,255,0.10)"
                borderRadius="12px"
                p="3px"
                boxShadow="0 4px 24px rgba(0,0,0,0.3)"
            >
                {(['en', 'es'] as const).map((lang) => {
                    const isActive = language === lang
                    return (
                        <Box
                            key={lang}
                            as="button"
                            onClick={() => setLanguage(lang)}
                            px={3}
                            py={1.5}
                            borderRadius="9px"
                            bg={isActive ? 'rgba(100,130,255,0.25)' : 'transparent'}
                            border={isActive ? '1px solid rgba(120,150,255,0.35)' : '1px solid transparent'}
                            transition="all 0.18s"
                            _hover={{ bg: isActive ? undefined : 'rgba(255,255,255,0.06)' }}
                            cursor="pointer"
                        >
                            <HStack spacing={1.5}>
                                <Text fontSize="13px">{lang === 'en' ? '🇺🇸' : '🇪🇸'}</Text>
                                <Text
                                    fontSize="11px"
                                    fontWeight="700"
                                    color={isActive ? 'rgba(200,215,255,0.95)' : 'rgba(160,185,255,0.45)'}
                                    letterSpacing="0.06em"
                                >
                                    {lang.toUpperCase()}
                                </Text>
                            </HStack>
                        </Box>
                    )
                })}
            </HStack>
        </Box>
    )
}
