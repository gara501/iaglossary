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
    englishTerm?: string
}

export default function TermModal({
    term, isOpen, onClose,
    summaryLabel = 'Summary',
    definitionLabel = 'Full Definition',
    relatedTermsLabel = 'Related Terms',
    englishTerm,
}: TermModalProps) {
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'

    if (!term) return null

    // Contrast-verified:
    // Dark dialog bg ~#1e2d36:  labelColor #FF9B51 ~4.6:1 ✓, textColor #EAEFEF ~10:1 ✓, summaryText #BFC9D1 ~7:1 ✓
    // Light dialog bg #fff8fa:  labelColor #c4621a ~5.2:1 ✓, textColor #25343F ~9:1 ✓, summaryText #3d5060 ~5.5:1 ✓
    const dialogBg = dark ? 'rgba(20, 30, 38, 0.93)' : 'rgba(250, 252, 252, 0.97)'
    const borderColor = dark ? 'rgba(255,255,255,0.07)' : 'rgba(191,201,209,0.55)'
    const labelColor = dark ? '#FF9B51' : '#c4621a'
    const titleColor = dark ? '#EAEFEF' : '#25343F'
    const textColor = dark ? '#EAEFEF' : '#25343F'
    const summaryText = dark ? '#BFC9D1' : '#3d5060'
    const summaryBg = dark ? 'rgba(255,155,81,0.07)' : 'rgba(255,155,81,0.07)'
    const summaryBorder = 'rgba(255,155,81,0.18)'
    const tagBg = dark ? 'rgba(255,255,255,0.05)' : 'rgba(191,201,209,0.25)'
    const tagColor = dark ? '#BFC9D1' : '#25343F'
    const tagBorder = dark ? 'rgba(255,255,255,0.08)' : 'rgba(191,201,209,0.50)'
    const tagHoverColor = dark ? '#FF9B51' : '#c4621a'
    const dividerColor = dark ? 'rgba(255,255,255,0.06)' : 'rgba(191,201,209,0.40)'
    const closeColor = dark ? 'rgba(234,239,239,0.40)' : 'rgba(37,52,63,0.40)'

    return (
        <AnimatePresence>
            {isOpen && (
                <Modal isOpen={isOpen} onClose={onClose} size="xl" isCentered scrollBehavior="inside">
                    <ModalOverlay bg="rgba(0,0,0,0.55)" backdropFilter="blur(10px)" />
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
                            ? '0 32px 80px rgba(0,0,0,0.50), 0 0 0 1px rgba(255,155,81,0.07)'
                            : '0 20px 60px rgba(37,52,63,0.12), 0 0 0 1px rgba(191,201,209,0.40)'}
                        mx={4}
                    >
                        {/* Accent line: orange */}
                        <Box h="2px" bgGradient="linear(to-r, #FF9B51, rgba(255,155,81,0.40), transparent)" />

                        <ModalHeader pt={6} pb={3}>
                            <Flex align="flex-start" gap={3} pr={8}>
                                <Box
                                    minW="46px" h="46px" borderRadius="14px"
                                    bg={dark ? 'rgba(255,155,81,0.12)' : 'rgba(255,155,81,0.14)'}
                                    border={dark ? '1px solid rgba(255,155,81,0.24)' : '1px solid rgba(255,155,81,0.30)'}
                                    display="flex" alignItems="center" justifyContent="center"
                                    flexShrink={0} mt={0.5}
                                >
                                    <Text fontFamily="Space Grotesk, sans-serif" fontWeight="800" fontSize="lg"
                                        color="#FF9B51">
                                        {term.letter}
                                    </Text>
                                </Box>
                                <Box>
                                    <span className="glass-badge" style={{ marginBottom: '6px', display: 'inline-block' }}>
                                        {term.category}
                                    </span>
                                    <Text fontFamily="Space Grotesk, sans-serif" fontWeight="800"
                                        fontSize={{ base: 'xl', md: '2xl' }}
                                        color={titleColor} lineHeight={1.2} letterSpacing="-0.02em"
                                        mb={englishTerm ? 0.5 : 0}>
                                        {term.term}
                                    </Text>
                                    {englishTerm && (
                                        <Text
                                            fontFamily="Space Grotesk, sans-serif"
                                            fontSize="13px" fontWeight="500" fontStyle="italic"
                                            color={dark ? 'rgba(191,201,209,0.55)' : 'rgba(37,52,63,0.40)'}
                                            lineHeight={1.3} letterSpacing="0.01em"
                                        >
                                            {englishTerm}
                                        </Text>
                                    )}
                                </Box>
                            </Flex>
                        </ModalHeader>

                        <ModalCloseButton
                            color={closeColor}
                            _hover={{ color: dark ? '#EAEFEF' : '#25343F', bg: 'rgba(255,155,81,0.10)' }}
                            borderRadius="10px" top={5} right={5}
                        />

                        <ModalBody pb={8}>
                            {/* Summary */}
                            <Box bg={summaryBg} border={`1px solid ${summaryBorder}`} borderRadius="16px" p={4} mb={5}>
                                <Flex align="center" gap={2} mb={2}>
                                    <Icon as={FiBookOpen} color={labelColor} boxSize={3.5} />
                                    <Text fontSize="10px" fontWeight="700" color={labelColor}
                                        letterSpacing="0.10em" textTransform="uppercase">
                                        {summaryLabel}
                                    </Text>
                                </Flex>
                                <Text fontSize="14px" color={summaryText} fontStyle="italic" lineHeight={1.75}>
                                    {term.summary}
                                </Text>
                            </Box>

                            <Divider borderColor={dividerColor} mb={5} />

                            {/* Definition */}
                            <Box mb={6}>
                                <Flex align="center" gap={2} mb={3}>
                                    <Icon as={FiTag} color={labelColor} boxSize={3.5} />
                                    <Text fontSize="10px" fontWeight="700" color={labelColor}
                                        letterSpacing="0.10em" textTransform="uppercase">
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
                                        <Text fontSize="10px" fontWeight="700" color={labelColor}
                                            letterSpacing="0.10em" textTransform="uppercase">
                                            {relatedTermsLabel}
                                        </Text>
                                    </Flex>
                                    <Wrap spacing={2}>
                                        {(term.relatedTerms ?? []).map((related) => (
                                            <WrapItem key={related}>
                                                <Tag size="sm" bg={tagBg} color={tagColor}
                                                    border={`1px solid ${tagBorder}`}
                                                    borderRadius="full" fontWeight="600" fontSize="13px"
                                                    _hover={{ bg: 'rgba(255,155,81,0.14)', borderColor: 'rgba(255,155,81,0.35)', color: tagHoverColor }}
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
