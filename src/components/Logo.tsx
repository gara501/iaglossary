import { Box, Text, HStack, Icon } from '@chakra-ui/react'
import { motion } from 'framer-motion'
import { FaBrain } from 'react-icons/fa'

const MotionBox = motion(Box)

export default function Logo() {
    return (
        <MotionBox
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            display="flex"
            justifyContent="center"
            mb={4}
        >
            <HStack spacing={3} align="center">
                <Box
                    position="relative"
                    w="60px"
                    h="60px"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                >
                    {/* Outer glow ring */}
                    <Box
                        position="absolute"
                        inset={0}
                        borderRadius="full"
                        bg="transparent"
                        border="2px solid"
                        borderColor="brand.400"
                        opacity={0.6}
                        sx={{
                            animation: 'pulse 2s infinite',
                            '@keyframes pulse': {
                                '0%, 100%': { transform: 'scale(1)', opacity: 0.6 },
                                '50%': { transform: 'scale(1.15)', opacity: 0.2 },
                            },
                        }}
                    />
                    {/* Inner circle */}
                    <Box
                        w="50px"
                        h="50px"
                        borderRadius="full"
                        bgGradient="linear(135deg, brand.500, accent.400)"
                        display="flex"
                        alignItems="center"
                        justifyContent="center"
                        boxShadow="0 0 30px rgba(130, 38, 255, 0.6), 0 0 60px rgba(0, 212, 255, 0.2)"
                    >
                        <Icon as={FaBrain} color="white" boxSize={6} />
                    </Box>
                </Box>
                <Box>
                    <Text
                        fontFamily="Space Grotesk, sans-serif"
                        fontWeight="800"
                        fontSize={{ base: '2xl', md: '3xl' }}
                        bgGradient="linear(to-r, brand.300, accent.400)"
                        bgClip="text"
                        lineHeight={1}
                        className="glow-text"
                    >
                        IA Glossary
                    </Text>
                    <Text
                        fontSize="xs"
                        color="whiteAlpha.600"
                        fontWeight="500"
                        letterSpacing="widest"
                        textTransform="uppercase"
                    >
                        AI & Generative AI Terms
                    </Text>
                </Box>
            </HStack>
        </MotionBox>
    )
}
