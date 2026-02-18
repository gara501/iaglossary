import { Box, Text, Flex, Icon } from '@chakra-ui/react'
import { motion } from 'framer-motion'
import { FiArrowRight } from 'react-icons/fi'
import { GlossaryTerm } from '../types/glossary'
import { useColorMode } from '../context/ThemeContext'

const MotionBox = motion(Box)

interface TermCardProps {
    term: GlossaryTerm
    index: number
    onClick: () => void
    readMoreLabel?: string
}

export default function TermCard({ term, index, onClick, readMoreLabel = 'Read more' }: TermCardProps) {
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'

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
            _focus={{ outline: '2px solid rgba(197,0,60,0.50)', outlineOffset: '2px' }}
        >
            {/* Top row: category badge + letter */}
            <Flex justify="space-between" align="center" mb={3}>
                <span className="glass-badge">{term.category}</span>
                <Box
                    w="30px"
                    h="30px"
                    borderRadius="9px"
                    bg={dark ? 'rgba(197, 0, 60, 0.14)' : 'rgba(197, 0, 60, 0.08)'}
                    border={dark ? '1px solid rgba(197, 0, 60, 0.24)' : '1px solid rgba(197, 0, 60, 0.18)'}
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    flexShrink={0}
                >
                    <Text
                        fontSize="12px"
                        fontWeight="800"
                        color={dark ? '#f3e600' : '#c5003c'}
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
                fontSize="17px"
                color={dark ? '#ffffff' : '#1a0a0d'}
                mb={2}
                lineHeight={1.3}
                letterSpacing="-0.01em"
            >
                {term.term}
            </Text>

            {/* Summary — improved contrast */}
            <Text
                fontSize="14px"
                color={dark ? 'rgba(245, 234, 236, 0.68)' : 'rgba(26, 10, 13, 0.65)'}
                lineHeight={1.7}
                flex={1}
                noOfLines={3}
                fontWeight="400"
            >
                {term.summary}
            </Text>

            {/* Read more */}
            <Flex align="center" mt={4} gap={1.5}>
                <Text fontSize="13px" fontWeight="700" color={dark ? 'rgba(197,0,60,0.80)' : '#c5003c'}>
                    {readMoreLabel}
                </Text>
                <Icon as={FiArrowRight} boxSize="12px" color={dark ? 'rgba(197,0,60,0.80)' : '#c5003c'} />
            </Flex>
        </MotionBox>
    )
}
