import {
    Modal, ModalOverlay, ModalContent, ModalHeader, ModalBody, ModalCloseButton,
    Text, Box, Flex, Divider, Wrap, WrapItem, Tag, Icon,
} from '@chakra-ui/react'
import { motion, AnimatePresence } from 'framer-motion'
import { FiBookOpen, FiTag, FiLink } from 'react-icons/fi'
import { GlossaryTerm } from '../types/glossary'
import { useColorMode } from '../context/ThemeContext'

const MotionModalContent = motion(ModalContent)

interface TermModalProps {
    term: GlossaryTerm | null
    isOpen: boolean
    onClose: () => void
    summaryLabel?: string
    definitionLabel?: string
    relatedTermsLabel?: string
}

export default function TermModal({
    term, isOpen, onClose,
    summaryLabel = 'Summary',
    definitionLabel = 'Full Definition',
    relatedTermsLabel = 'Related Terms',
}: TermModalProps) {
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'

    if (!term) return null

    const dialogBg = dark ? 'rgba(8, 2, 4, 0.92)' : 'rgba(255, 248, 250, 0.96)'
    const borderColor = dark ? 'rgba(255,255,255,0.07)' : 'rgba(197,0,60,0.14)'
    const labelColor = dark ? 'rgba(243,230,0,0.65)' : 'rgba(136,4,37,0.70)'
    const textColor = dark ? 'rgba(245,234,236,0.88)' : 'rgba(26,10,13,0.85)'
    const summaryBg = dark ? 'rgba(197,0,60,0.07)' : 'rgba(197,0,60,0.05)'
    const summaryBorder = dark ? 'rgba(197,0,60,0.14)' : 'rgba(197,0,60,0.14)'
    const summaryText = dark ? 'rgba(245,234,236,0.75)' : 'rgba(26,10,13,0.72)'
    const tagBg = dark ? 'rgba(255,255,255,0.05)' : 'rgba(197,0,60,0.06)'
    const tagColor = dark ? 'rgba(245,234,236,0.72)' : 'rgba(136,4,37,0.80)'
    const tagBorder = dark ? 'rgba(255,255,255,0.08)' : 'rgba(197,0,60,0.14)'

    return (
        <AnimatePresence>
            {isOpen && (
                <Modal isOpen={isOpen} onClose={onClose} size="xl" isCentered scrollBehavior="inside">
                    <ModalOverlay bg="rgba(0,0,0,0.65)" backdropFilter="blur(10px)" />
                    <MotionModalContent
                        initial={{ opacity: 0, scale: 0.93, y: 16 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.93, y: 16 }}
                        transition={{ duration: 0.22, ease: 'easeOut' }}
                        bg={dialogBg}
                        backdropFilter="blur(28px)"
                        border={`1px solid ${borderColor}`}
                        borderRadius="24px"
                        overflow="hidden"
                        boxShadow={dark
                            ? '0 32px 80px rgba(0,0,0,0.60), 0 0 0 1px rgba(197,0,60,0.08)'
                            : '0 20px 60px rgba(197,0,60,0.10), 0 0 0 1px rgba(197,0,60,0.10)'}
                        mx={4}
                    >
                        {/* Accent line */}
                        <Box h="2px" bgGradient="linear(to-r, #c5003c, #f3e600, rgba(197,0,60,0.2))" />

                        <ModalHeader pt={6} pb={3}>
                            <Flex align="flex-start" gap={3} pr={8}>
                                <Box
                                    minW="46px" h="46px" borderRadius="14px"
                                    bg={dark ? 'rgba(197,0,60,0.15)' : 'rgba(197,0,60,0.08)'}
                                    border={dark ? '1px solid rgba(197,0,60,0.26)' : '1px solid rgba(197,0,60,0.20)'}
                                    display="flex" alignItems="center" justifyContent="center"
                                    flexShrink={0} mt={0.5}
                                >
                                    <Text fontFamily="Space Grotesk, sans-serif" fontWeight="800" fontSize="lg"
                                        color={dark ? '#f3e600' : '#c5003c'}>
                                        {term.letter}
                                    </Text>
                                </Box>
                                <Box>
                                    <span className="glass-badge" style={{ marginBottom: '6px', display: 'inline-block' }}>
                                        {term.category}
                                    </span>
                                    <Text fontFamily="Space Grotesk, sans-serif" fontWeight="800"
                                        fontSize={{ base: 'xl', md: '2xl' }}
                                        color={dark ? '#ffffff' : '#1a0a0d'}
                                        lineHeight={1.2} letterSpacing="-0.02em">
                                        {term.term}
                                    </Text>
                                </Box>
                            </Flex>
                        </ModalHeader>

                        <ModalCloseButton
                            color={dark ? 'rgba(245,234,236,0.40)' : 'rgba(26,10,13,0.40)'}
                            _hover={{ color: dark ? 'white' : '#1a0a0d', bg: 'rgba(197,0,60,0.10)' }}
                            borderRadius="10px" top={5} right={5}
                        />

                        <ModalBody pb={8}>
                            {/* Summary */}
                            <Box bg={summaryBg} border={`1px solid ${summaryBorder}`} borderRadius="16px" p={4} mb={5}>
                                <Flex align="center" gap={2} mb={2}>
                                    <Icon as={FiBookOpen} color={labelColor} boxSize={3.5} />
                                    <Text fontSize="10px" fontWeight="700" color={labelColor} letterSpacing="0.10em" textTransform="uppercase">
                                        {summaryLabel}
                                    </Text>
                                </Flex>
                                <Text fontSize="14px" color={summaryText} fontStyle="italic" lineHeight={1.75}>
                                    {term.summary}
                                </Text>
                            </Box>

                            <Divider borderColor={dark ? 'rgba(255,255,255,0.05)' : 'rgba(197,0,60,0.08)'} mb={5} />

                            {/* Definition */}
                            <Box mb={6}>
                                <Flex align="center" gap={2} mb={3}>
                                    <Icon as={FiTag} color={labelColor} boxSize={3.5} />
                                    <Text fontSize="10px" fontWeight="700" color={labelColor} letterSpacing="0.10em" textTransform="uppercase">
                                        {definitionLabel}
                                    </Text>
                                </Flex>
                                <Text fontSize="15px" color={textColor} lineHeight={1.85} fontWeight="400">
                                    {term.definition}
                                </Text>
                            </Box>

                            {/* Related terms */}
                            {(term.relatedTerms ?? []).length > 0 && (
                                <Box>
                                    <Flex align="center" gap={2} mb={3}>
                                        <Icon as={FiLink} color={labelColor} boxSize={3.5} />
                                        <Text fontSize="10px" fontWeight="700" color={labelColor} letterSpacing="0.10em" textTransform="uppercase">
                                            {relatedTermsLabel}
                                        </Text>
                                    </Flex>
                                    <Wrap spacing={2}>
                                        {(term.relatedTerms ?? []).map((related) => (
                                            <WrapItem key={related}>
                                                <Tag size="sm" bg={tagBg} color={tagColor} border={`1px solid ${tagBorder}`}
                                                    borderRadius="full" fontWeight="600" fontSize="13px"
                                                    _hover={{ bg: 'rgba(197,0,60,0.12)', borderColor: 'rgba(197,0,60,0.30)', color: dark ? '#f3e600' : '#880425' }}
                                                    transition="all 0.18s" cursor="default">
                                                    {related}
                                                </Tag>
                                            </WrapItem>
                                        ))}
                                    </Wrap>
                                </Box>
                            )}
                        </ModalBody>
                    </MotionModalContent>
                </Modal>
            )}
        </AnimatePresence>
    )
}
