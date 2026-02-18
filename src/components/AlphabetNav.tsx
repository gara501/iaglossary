import { Box, Flex, Text } from '@chakra-ui/react'
import { motion } from 'framer-motion'

const MotionBox = motion(Box)

const EN_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
const ES_ALPHABET = 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ'.split('')

interface AlphabetNavProps {
    activeLetter: string | null
    availableLetters: Set<string>
    onLetterClick: (letter: string | null) => void
    allLabel?: string
    clearLabel?: string
    language?: string
}

export default function AlphabetNav({
    activeLetter,
    availableLetters,
    onLetterClick,
    allLabel = 'ALL',
    clearLabel = '✕ Clear filter',
    language = 'en',
}: AlphabetNavProps) {
    const ALPHABET = language === 'es' ? ES_ALPHABET : EN_ALPHABET
    return (
        <MotionBox
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            w="100%"
        >
            <Flex wrap="wrap" gap={1.5} justify="center" align="center" mb={2}>
                {/* ALL button */}
                <button
                    className={`letter-btn${activeLetter === null ? ' active' : ''}`}
                    onClick={() => onLetterClick(null)}
                    style={{ width: 'auto', padding: '0 12px', fontSize: '11px', letterSpacing: '0.08em' }}
                >
                    {allLabel}
                </button>

                {ALPHABET.map((letter) => {
                    const isAvailable = availableLetters.has(letter)
                    const isActive = activeLetter === letter
                    return (
                        <button
                            key={letter}
                            className={`letter-btn${isActive ? ' active' : ''}`}
                            onClick={() => isAvailable && onLetterClick(letter)}
                            disabled={!isAvailable}
                        >
                            {letter}
                            {isAvailable && (
                                <span
                                    style={{
                                        position: 'absolute',
                                        bottom: '3px',
                                        left: '50%',
                                        transform: 'translateX(-50%)',
                                        width: '3px',
                                        height: '3px',
                                        borderRadius: '50%',
                                        background: isActive ? 'white' : 'rgba(160,185,255,0.6)',
                                    }}
                                />
                            )}
                        </button>
                    )
                })}
            </Flex>

            {activeLetter && (
                <Flex justify="center">
                    <Text
                        fontSize="11px"
                        color="rgba(160,185,255,0.45)"
                        cursor="pointer"
                        onClick={() => onLetterClick(null)}
                        _hover={{ color: 'rgba(160,185,255,0.8)' }}
                        transition="color 0.2s"
                        fontWeight="500"
                    >
                        {clearLabel}
                    </Text>
                </Flex>
            )}
        </MotionBox>
    )
}
