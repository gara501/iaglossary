import { Box, Button, HStack, Text } from '@chakra-ui/react'
import { motion } from 'framer-motion'
import { useLanguage, Language } from '../context/LanguageContext'

const MotionBox = motion(Box)

export default function LanguageSelector() {
    const { language, setLanguage } = useLanguage()

    const options: { code: Language; label: string; flag: string }[] = [
        { code: 'en', label: 'EN', flag: '🇺🇸' },
        { code: 'es', label: 'ES', flag: '🇪🇸' },
    ]

    return (
        <MotionBox
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            position="fixed"
            top={4}
            right={4}
            zIndex={100}
        >
            <HStack
                spacing={1}
                bg="rgba(15, 15, 40, 0.85)"
                backdropFilter="blur(12px)"
                border="1px solid"
                borderColor="whiteAlpha.100"
                borderRadius="xl"
                p={1}
                boxShadow="0 4px 20px rgba(0,0,0,0.4)"
            >
                {options.map(({ code, label, flag }) => {
                    const isActive = language === code
                    return (
                        <Button
                            key={code}
                            onClick={() => setLanguage(code)}
                            size="sm"
                            h="34px"
                            px={3}
                            borderRadius="lg"
                            bg={isActive ? 'brand.500' : 'transparent'}
                            color={isActive ? 'white' : 'whiteAlpha.600'}
                            fontWeight={isActive ? '700' : '500'}
                            fontSize="xs"
                            letterSpacing="wider"
                            boxShadow={isActive ? '0 0 12px rgba(130,38,255,0.4)' : 'none'}
                            _hover={{
                                bg: isActive ? 'brand.400' : 'whiteAlpha.100',
                                color: 'white',
                            }}
                            transition="all 0.2s"
                        >
                            <Text mr={1} fontSize="sm">{flag}</Text>
                            {label}
                        </Button>
                    )
                })}
            </HStack>
        </MotionBox>
    )
}
