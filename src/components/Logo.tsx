import { Box, Text, HStack, Icon } from '@chakra-ui/react'
import { motion } from 'framer-motion'
import { FaBrain } from 'react-icons/fa'

const MotionBox = motion(Box)

interface LogoProps {
    subtitle?: string
}

export default function Logo({ subtitle = 'AI & Generative AI Terms' }: LogoProps) {
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
                {/* Icon bubble */}
                <Box
                    w="56px"
                    h="56px"
                    borderRadius="18px"
                    bg="rgba(99, 120, 255, 0.18)"
                    border="1px solid rgba(120, 150, 255, 0.25)"
                    backdropFilter="blur(12px)"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    boxShadow="0 8px 32px rgba(80, 100, 255, 0.25), inset 0 1px 0 rgba(255,255,255,0.12)"
                >
                    <Icon as={FaBrain} color="rgba(180,200,255,0.9)" boxSize={6} />
                </Box>

                <Box>
                    <Text
                        fontFamily="Space Grotesk, sans-serif"
                        fontWeight="800"
                        fontSize={{ base: '2xl', md: '3xl' }}
                        color="white"
                        lineHeight={1}
                        className="glow-text"
                        letterSpacing="-0.02em"
                    >
                        IA Glossary
                    </Text>
                    <Text
                        fontSize="10px"
                        color="rgba(160, 185, 255, 0.65)"
                        fontWeight="600"
                        letterSpacing="0.14em"
                        textTransform="uppercase"
                        mt="2px"
                    >
                        {subtitle}
                    </Text>
                </Box>
            </HStack>
        </MotionBox>
    )
}
