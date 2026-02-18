import {
    Modal,
    ModalOverlay,
    ModalContent,
    ModalHeader,
    ModalBody,
    ModalCloseButton,
    Text,
    Box,
    Flex,
    Divider,
    Wrap,
    WrapItem,
    Tag,
    Icon,
} from '@chakra-ui/react'
import { motion, AnimatePresence } from 'framer-motion'
import { FiBookOpen, FiTag, FiLink } from 'react-icons/fi'
import { GlossaryTerm } from '../types/glossary'

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
    term,
    isOpen,
    onClose,
    summaryLabel = 'Summary',
    definitionLabel = 'Full Definition',
    relatedTermsLabel = 'Related Terms',
}: TermModalProps) {
    if (!term) return null

    return (
        <AnimatePresence>
            {isOpen && (
                <Modal isOpen={isOpen} onClose={onClose} size="xl" isCentered scrollBehavior="inside">
                    <ModalOverlay
                        bg="rgba(0,0,0,0.65)"
                        backdropFilter="blur(10px)"
                    />
                    <MotionModalContent
                        initial={{ opacity: 0, scale: 0.93, y: 16 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.93, y: 16 }}
                        transition={{ duration: 0.22, ease: 'easeOut' }}
                        bg="rgba(12, 5, 8, 0.90)"
                        backdropFilter="blur(28px)"
                        border="1px solid rgba(255,255,255,0.08)"
                        borderRadius="24px"
                        overflow="hidden"
                        boxShadow="0 32px 80px rgba(0,0,0,0.55), 0 0 0 1px rgba(225,29,72,0.08)"
                        mx={4}
                    >
                        {/* Top accent line — neon crimson */}
                        <Box
                            h="2px"
                            bgGradient="linear(to-r, rgba(225,29,72,0.9), rgba(251,113,133,0.6), rgba(225,29,72,0.2))"
                        />

                        <ModalHeader pt={6} pb={3}>
                            <Flex align="flex-start" gap={3} pr={8}>
                                {/* Letter bubble */}
                                <Box
                                    minW="44px"
                                    h="44px"
                                    borderRadius="14px"
                                    bg="rgba(225, 29, 72, 0.14)"
                                    border="1px solid rgba(225, 29, 72, 0.24)"
                                    display="flex"
                                    alignItems="center"
                                    justifyContent="center"
                                    flexShrink={0}
                                    mt={0.5}
                                >
                                    <Text
                                        fontFamily="Space Grotesk, sans-serif"
                                        fontWeight="800"
                                        fontSize="lg"
                                        color="rgba(251, 113, 133, 0.9)"
                                    >
                                        {term.letter}
                                    </Text>
                                </Box>

                                <Box>
                                    <span className="glass-badge" style={{ marginBottom: '6px', display: 'inline-block' }}>
                                        {term.category}
                                    </span>
                                    <Text
                                        fontFamily="Space Grotesk, sans-serif"
                                        fontWeight="800"
                                        fontSize={{ base: 'xl', md: '2xl' }}
                                        color="rgba(255, 240, 243, 0.97)"
                                        lineHeight={1.2}
                                        letterSpacing="-0.02em"
                                    >
                                        {term.term}
                                    </Text>
                                </Box>
                            </Flex>
                        </ModalHeader>

                        <ModalCloseButton
                            color="rgba(255,200,210,0.4)"
                            _hover={{ color: 'white', bg: 'rgba(225,29,72,0.12)' }}
                            borderRadius="10px"
                            top={5}
                            right={5}
                        />

                        <ModalBody pb={8}>
                            {/* Summary box */}
                            <Box
                                bg="rgba(225, 29, 72, 0.06)"
                                border="1px solid rgba(225, 29, 72, 0.12)"
                                borderRadius="16px"
                                p={4}
                                mb={5}
                            >
                                <Flex align="center" gap={2} mb={2}>
                                    <Icon as={FiBookOpen} color="rgba(251,113,133,0.65)" boxSize={3.5} />
                                    <Text fontSize="10px" fontWeight="700" color="rgba(251,113,133,0.65)" letterSpacing="0.1em" textTransform="uppercase">
                                        {summaryLabel}
                                    </Text>
                                </Flex>
                                <Text fontSize="sm" color="rgba(255, 210, 220, 0.72)" fontStyle="italic" lineHeight={1.7}>
                                    {term.summary}
                                </Text>
                            </Box>

                            <Divider borderColor="rgba(255,255,255,0.05)" mb={5} />

                            {/* Full definition */}
                            <Box mb={6}>
                                <Flex align="center" gap={2} mb={3}>
                                    <Icon as={FiTag} color="rgba(251,113,133,0.55)" boxSize={3.5} />
                                    <Text fontSize="10px" fontWeight="700" color="rgba(251,113,133,0.55)" letterSpacing="0.1em" textTransform="uppercase">
                                        {definitionLabel}
                                    </Text>
                                </Flex>
                                <Text fontSize="sm" color="rgba(255, 230, 235, 0.85)" lineHeight={1.8} fontWeight="400">
                                    {term.definition}
                                </Text>
                            </Box>

                            {/* Related terms */}
                            {term.relatedTerms && term.relatedTerms.length > 0 && (
                                <Box>
                                    <Flex align="center" gap={2} mb={3}>
                                        <Icon as={FiLink} color="rgba(251,113,133,0.55)" boxSize={3.5} />
                                        <Text fontSize="10px" fontWeight="700" color="rgba(251,113,133,0.55)" letterSpacing="0.1em" textTransform="uppercase">
                                            {relatedTermsLabel}
                                        </Text>
                                    </Flex>
                                    <Wrap spacing={2}>
                                        {term.relatedTerms.map((related) => (
                                            <WrapItem key={related}>
                                                <Tag
                                                    size="sm"
                                                    bg="rgba(255,255,255,0.05)"
                                                    color="rgba(255, 200, 210, 0.70)"
                                                    border="1px solid rgba(255,255,255,0.08)"
                                                    borderRadius="full"
                                                    fontWeight="500"
                                                    fontSize="12px"
                                                    _hover={{ bg: 'rgba(225,29,72,0.14)', borderColor: 'rgba(225,29,72,0.30)', color: 'white' }}
                                                    transition="all 0.18s"
                                                    cursor="default"
                                                >
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
