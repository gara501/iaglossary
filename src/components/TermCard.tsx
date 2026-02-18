import { Box, Text, Flex, Icon } from '@chakra-ui/react'
import { motion } from 'framer-motion'
import { FiArrowRight } from 'react-icons/fi'
import { GlossaryTerm } from '../types/glossary'

const MotionBox = motion(Box)

interface TermCardProps {
    term: GlossaryTerm
    index: number
    onClick: () => void
    readMoreLabel?: string
}

export default function TermCard({ term, index, onClick, readMoreLabel = 'Read more' }: TermCardProps) {
    return (
        <MotionBox
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: index * 0.035 }}
            onClick={onClick}
            cursor="pointer"
            className="glass-card"
            p={5}
            h="100%"
            display="flex"
            flexDirection="column"
            role="button"
            tabIndex={0}
            onKeyDown={(e: React.KeyboardEvent) => e.key === 'Enter' && onClick()}
            _focus={{ outline: '2px solid rgba(225,29,72,0.45)', outlineOffset: '2px' }}
        >
            {/* Top row: category badge + letter */}
            <Flex justify="space-between" align="center" mb={3}>
                <span className="glass-badge">{term.category}</span>
                <Box
                    w="28px"
                    h="28px"
                    borderRadius="9px"
                    bg="rgba(225, 29, 72, 0.12)"
                    border="1px solid rgba(225, 29, 72, 0.20)"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    flexShrink={0}
                >
                    <Text
                        fontSize="11px"
                        fontWeight="800"
                        color="rgba(251, 113, 133, 0.9)"
                        fontFamily="Space Grotesk, sans-serif"
                    >
                        {term.letter}
                    </Text>
                </Box>
            </Flex>

            {/* Term name */}
            <Text
                fontFamily="Space Grotesk, sans-serif"
                fontWeight="700"
                fontSize="md"
                color="rgba(255, 240, 243, 0.95)"
                mb={2}
                lineHeight={1.35}
                letterSpacing="-0.01em"
            >
                {term.term}
            </Text>

            {/* Summary */}
            <Text
                fontSize="13px"
                color="rgba(255, 200, 210, 0.45)"
                lineHeight={1.65}
                flex={1}
                noOfLines={3}
                fontWeight="400"
            >
                {term.summary}
            </Text>

            {/* Read more */}
            <Flex align="center" mt={4} gap={1.5}>
                <Text fontSize="12px" fontWeight="600" color="rgba(225, 29, 72, 0.65)">
                    {readMoreLabel}
                </Text>
                <Icon as={FiArrowRight} boxSize="11px" color="rgba(225, 29, 72, 0.65)" />
            </Flex>
        </MotionBox>
    )
}
