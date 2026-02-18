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
            _focus={{ outline: '2px solid rgba(120,150,255,0.5)', outlineOffset: '2px' }}
        >
            {/* Top row: category badge + letter */}
            <Flex justify="space-between" align="center" mb={3}>
                <span className="glass-badge">{term.category}</span>
                <Box
                    w="28px"
                    h="28px"
                    borderRadius="9px"
                    bg="rgba(100,130,255,0.15)"
                    border="1px solid rgba(120,150,255,0.2)"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    flexShrink={0}
                >
                    <Text
                        fontSize="11px"
                        fontWeight="800"
                        color="rgba(160,185,255,0.9)"
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
                color="rgba(235,240,255,0.95)"
                mb={2}
                lineHeight={1.35}
                letterSpacing="-0.01em"
            >
                {term.term}
            </Text>

            {/* Summary */}
            <Text
                fontSize="13px"
                color="rgba(180,200,255,0.55)"
                lineHeight={1.65}
                flex={1}
                noOfLines={3}
                fontWeight="400"
            >
                {term.summary}
            </Text>

            {/* Read more */}
            <Flex align="center" mt={4} gap={1.5}>
                <Text fontSize="12px" fontWeight="600" color="rgba(140,165,255,0.7)">
                    {readMoreLabel}
                </Text>
                <Icon as={FiArrowRight} boxSize="11px" color="rgba(140,165,255,0.7)" />
            </Flex>
        </MotionBox>
    )
}
