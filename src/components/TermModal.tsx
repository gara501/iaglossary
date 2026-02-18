import {
    Modal,
    ModalOverlay,
    ModalContent,
    ModalHeader,
    ModalBody,
    ModalCloseButton,
    Text,
    Badge,
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

const categoryColors: Record<string, string> = {
    Architecture: 'purple',
    'Model Type': 'blue',
    Model: 'cyan',
    Training: 'green',
    'Generative AI': 'pink',
    Prompting: 'orange',
    Representation: 'teal',
    Deployment: 'yellow',
    Infrastructure: 'red',
    Fundamentals: 'gray',
    'Learning Paradigm': 'messenger',
    'Safety & Alignment': 'whatsapp',
}

interface TermModalProps {
    term: GlossaryTerm | null
    isOpen: boolean
    onClose: () => void
}

export default function TermModal({ term, isOpen, onClose }: TermModalProps) {
    if (!term) return null

    const badgeColor = categoryColors[term.category] || 'purple'

    return (
        <AnimatePresence>
            {isOpen && (
                <Modal isOpen={isOpen} onClose={onClose} size="xl" isCentered scrollBehavior="inside">
                    <ModalOverlay
                        bg="blackAlpha.800"
                        backdropFilter="blur(12px)"
                    />
                    <MotionModalContent
                        initial={{ opacity: 0, scale: 0.9, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 20 }}
                        transition={{ duration: 0.25, ease: 'easeOut' }}
                        bg="rgba(15, 15, 40, 0.95)"
                        border="1px solid"
                        borderColor="brand.700"
                        borderRadius="2xl"
                        overflow="hidden"
                        boxShadow="0 25px 80px rgba(130, 38, 255, 0.3), 0 0 0 1px rgba(130, 38, 255, 0.2)"
                        mx={4}
                    >
                        {/* Top gradient bar */}
                        <Box
                            h="3px"
                            bgGradient="linear(to-r, brand.500, accent.400, brand.300)"
                        />

                        <ModalHeader pt={6} pb={2}>
                            <Flex align="flex-start" gap={3} pr={8}>
                                {/* Letter circle */}
                                <Box
                                    minW="48px"
                                    h="48px"
                                    borderRadius="xl"
                                    bgGradient="linear(135deg, brand.600, accent.400)"
                                    display="flex"
                                    alignItems="center"
                                    justifyContent="center"
                                    boxShadow="0 0 20px rgba(130, 38, 255, 0.4)"
                                    mt={0.5}
                                >
                                    <Text
                                        fontFamily="Space Grotesk, sans-serif"
                                        fontWeight="800"
                                        fontSize="xl"
                                        color="white"
                                    >
                                        {term.letter}
                                    </Text>
                                </Box>

                                <Box>
                                    <Badge
                                        colorScheme={badgeColor}
                                        variant="subtle"
                                        borderRadius="full"
                                        px={2}
                                        py={0.5}
                                        fontSize="2xs"
                                        fontWeight="600"
                                        letterSpacing="wider"
                                        textTransform="uppercase"
                                        mb={1}
                                    >
                                        {term.category}
                                    </Badge>
                                    <Text
                                        fontFamily="Space Grotesk, sans-serif"
                                        fontWeight="800"
                                        fontSize={{ base: 'xl', md: '2xl' }}
                                        color="white"
                                        lineHeight={1.2}
                                    >
                                        {term.term}
                                    </Text>
                                </Box>
                            </Flex>
                        </ModalHeader>

                        <ModalCloseButton
                            color="whiteAlpha.600"
                            _hover={{ color: 'white', bg: 'whiteAlpha.100' }}
                            borderRadius="lg"
                            top={5}
                            right={5}
                        />

                        <ModalBody pb={8}>
                            {/* Summary */}
                            <Box
                                bg="whiteAlpha.50"
                                border="1px solid"
                                borderColor="brand.800"
                                borderRadius="xl"
                                p={4}
                                mb={5}
                            >
                                <Flex align="center" gap={2} mb={2}>
                                    <Icon as={FiBookOpen} color="brand.300" boxSize={4} />
                                    <Text fontSize="xs" fontWeight="700" color="brand.300" letterSpacing="wider" textTransform="uppercase">
                                        Summary
                                    </Text>
                                </Flex>
                                <Text fontSize="sm" color="whiteAlpha.800" fontStyle="italic" lineHeight={1.7}>
                                    {term.summary}
                                </Text>
                            </Box>

                            <Divider borderColor="whiteAlpha.100" mb={5} />

                            {/* Full definition */}
                            <Box mb={6}>
                                <Flex align="center" gap={2} mb={3}>
                                    <Icon as={FiTag} color="accent.400" boxSize={4} />
                                    <Text fontSize="xs" fontWeight="700" color="accent.400" letterSpacing="wider" textTransform="uppercase">
                                        Full Definition
                                    </Text>
                                </Flex>
                                <Text
                                    fontSize="sm"
                                    color="whiteAlpha.900"
                                    lineHeight={1.8}
                                >
                                    {term.definition}
                                </Text>
                            </Box>

                            {/* Related terms */}
                            {term.relatedTerms && term.relatedTerms.length > 0 && (
                                <Box>
                                    <Flex align="center" gap={2} mb={3}>
                                        <Icon as={FiLink} color="brand.300" boxSize={4} />
                                        <Text fontSize="xs" fontWeight="700" color="brand.300" letterSpacing="wider" textTransform="uppercase">
                                            Related Terms
                                        </Text>
                                    </Flex>
                                    <Wrap spacing={2}>
                                        {term.relatedTerms.map((related) => (
                                            <WrapItem key={related}>
                                                <Tag
                                                    size="sm"
                                                    bg="whiteAlpha.100"
                                                    color="whiteAlpha.800"
                                                    border="1px solid"
                                                    borderColor="whiteAlpha.200"
                                                    borderRadius="full"
                                                    fontWeight="500"
                                                    _hover={{ bg: 'brand.800', borderColor: 'brand.500', color: 'white' }}
                                                    transition="all 0.2s"
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
