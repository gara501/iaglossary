import { Box, Flex, Button, Text } from '@chakra-ui/react'
import { motion } from 'framer-motion'

const MotionBox = motion(Box)
const MotionButton = motion(Button)

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')

interface AlphabetNavProps {
    activeLetter: string | null
    availableLetters: Set<string>
    onLetterClick: (letter: string | null) => void
    allLabel?: string
    clearLabel?: string
}

export default function AlphabetNav({
    activeLetter,
    availableLetters,
    onLetterClick,
    allLabel = 'ALL',
    clearLabel = '✕ Clear filter',
}: AlphabetNavProps) {
    return (
        <MotionBox
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            w="100%"
        >
            <Flex
                wrap="wrap"
                gap={1.5}
                justify="center"
                align="center"
                mb={2}
            >
                {/* ALL button */}
                <MotionButton
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    size="sm"
                    onClick={() => onLetterClick(null)}
                    variant={activeLetter === null ? 'letterActive' : 'letter'}
                    minW="42px"
                    h="36px"
                    fontSize="xs"
                    fontWeight="700"
                    letterSpacing="wider"
                >
                    {allLabel}
                </MotionButton>

                {ALPHABET.map((letter, i) => {
                    const isAvailable = availableLetters.has(letter)
                    const isActive = activeLetter === letter
                    return (
                        <MotionButton
                            key={letter}
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: 0.4 + i * 0.015 }}
                            whileHover={isAvailable ? { scale: 1.1 } : {}}
                            whileTap={isAvailable ? { scale: 0.9 } : {}}
                            size="sm"
                            onClick={() => isAvailable && onLetterClick(letter)}
                            variant={isActive ? 'letterActive' : 'letter'}
                            minW="36px"
                            h="36px"
                            opacity={isAvailable ? 1 : 0.25}
                            cursor={isAvailable ? 'pointer' : 'not-allowed'}
                            position="relative"
                        >
                            {letter}
                            {isAvailable && (
                                <Box
                                    position="absolute"
                                    bottom="2px"
                                    left="50%"
                                    transform="translateX(-50%)"
                                    w="3px"
                                    h="3px"
                                    borderRadius="full"
                                    bg={isActive ? 'white' : 'brand.300'}
                                />
                            )}
                        </MotionButton>
                    )
                })}
            </Flex>
            {activeLetter && (
                <Flex justify="center">
                    <Text
                        fontSize="xs"
                        color="whiteAlpha.500"
                        cursor="pointer"
                        onClick={() => onLetterClick(null)}
                        _hover={{ color: 'brand.300' }}
                        transition="color 0.2s"
                    >
                        {clearLabel}
                    </Text>
                </Flex>
            )}
        </MotionBox>
    )
}
