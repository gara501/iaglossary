import { Box, Text, HStack, Icon } from '@chakra-ui/react'
import { motion } from 'framer-motion'
import { FaBrain } from 'react-icons/fa'
import { useColorMode } from '../context/ThemeContext'

const MotionBox = motion(Box)

interface LogoProps {
    subtitle?: string
}

export default function Logo({ subtitle = 'AI & Generative AI Terms' }: LogoProps) {
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'

    return (
        <MotionBox
            initial={{ opacity: 0, y: -16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            display="flex"
            justifyContent="center"
            mb={3}
        >
            <HStack spacing={4} align="center">
                <Box
                    w="58px"
                    h="58px"
                    borderRadius="18px"
                    bg={dark ? 'rgba(197, 0, 60, 0.16)' : 'rgba(197, 0, 60, 0.10)'}
                    border={dark ? '1px solid rgba(197, 0, 60, 0.28)' : '1px solid rgba(197, 0, 60, 0.22)'}
                    backdropFilter="blur(12px)"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    boxShadow={dark
                        ? '0 8px 32px rgba(197, 0, 60, 0.22), inset 0 1px 0 rgba(255,255,255,0.08)'
                        : '0 4px 20px rgba(197, 0, 60, 0.14)'}
                >
                    <Icon as={FaBrain} color={dark ? '#f3e600' : '#c5003c'} boxSize={6} />
                </Box>

                <Box>
                    <Text
                        fontFamily="Space Grotesk, sans-serif"
                        fontWeight="800"
                        fontSize={{ base: '2xl', md: '3xl' }}
                        color={dark ? 'white' : '#1a0a0d'}
                        lineHeight={1}
                        className="glow-text"
                        letterSpacing="-0.02em"
                    >
                        IA Glossary
                    </Text>
                    <Text
                        fontSize="10px"
                        color={dark ? 'rgba(243, 230, 0, 0.55)' : 'rgba(136, 4, 37, 0.60)'}
                        fontWeight="700"
                        letterSpacing="0.16em"
                        textTransform="uppercase"
                        mt="3px"
                    >
                        {subtitle}
                    </Text>
                </Box>
            </HStack>
        </MotionBox>
    )
}
