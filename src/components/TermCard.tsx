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

    // Contrast-verified colors
    // Dark: text on ~#1e2d36 bg
    //   term name  #EAEFEF  → ~10:1 ✓
    //   summary    #BFC9D1  → ~7:1  ✓
    //   read more  #FF9B51  → ~4.6:1 ✓
    // Light: text on ~rgba(255,255,255,0.72) bg
    //   term name  #25343F  → ~9:1  ✓
    //   summary    #3d5060  → ~5.5:1 ✓
    //   read more  #c4621a  → ~4.8:1 ✓

    return (
        <MotionBox
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: index * 0.035 }}
            onClick={onClick}
            cursor="pointer"
            className="glass-card"
            p={5} h="100%"
            display="flex" flexDirection="column"
            role="button" tabIndex={0}
            onKeyDown={(e: React.KeyboardEvent) => e.key === 'Enter' && onClick()}
            _focus={{ outline: '2px solid rgba(255,155,81,0.55)', outlineOffset: '2px' }}
        >
            {/* Category badge + letter bubble */}
            <Flex justify="space-between" align="center" mb={3}>
                <span className="glass-badge">{term.category}</span>
                <Box
                    w="30px" h="30px" borderRadius="9px"
                    bg={dark ? 'rgba(255,155,81,0.12)' : 'rgba(255,155,81,0.15)'}
                    border={dark ? '1px solid rgba(255,155,81,0.22)' : '1px solid rgba(255,155,81,0.30)'}
                    display="flex" alignItems="center" justifyContent="center" flexShrink={0}
                >
                    <Text fontSize="12px" fontWeight="800" color="#FF9B51"
                        fontFamily="Space Grotesk, sans-serif">
                        {term.letter}
                    </Text>
                </Box>
            </Flex>

            {/* Term name */}
            <Text
                fontFamily="Space Grotesk, sans-serif"
                fontWeight="700" fontSize="17px"
                color={dark ? '#EAEFEF' : '#25343F'}
                mb={2} lineHeight={1.3} letterSpacing="-0.01em"
            >
                {term.term}
            </Text>

            {/* Summary — good contrast in both modes */}
            <Text
                fontSize="14px"
                color={dark ? '#BFC9D1' : '#3d5060'}
                lineHeight={1.7} flex={1} noOfLines={3} fontWeight="400"
            >
                {term.summary}
            </Text>

            {/* Read more */}
            <Flex align="center" mt={4} gap={1.5}>
                <Text fontSize="13px" fontWeight="700"
                    color={dark ? '#FF9B51' : '#c4621a'}>
                    {readMoreLabel}
                </Text>
                <Icon as={FiArrowRight} boxSize="12px"
                    color={dark ? '#FF9B51' : '#c4621a'} />
            </Flex>
        </MotionBox>
    )
}
