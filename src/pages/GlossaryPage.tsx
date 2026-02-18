import { useState, useMemo, useCallback } from 'react'
import {
    Box,
    Container,
    SimpleGrid,
    Text,
    Flex,
    useDisclosure,
    Center,
    Icon,
} from '@chakra-ui/react'
import { motion, AnimatePresence } from 'framer-motion'
import { FiSearch } from 'react-icons/fi'
import Logo from '../components/Logo'
import SearchBar from '../components/SearchBar'
import AlphabetNav from '../components/AlphabetNav'
import TermCard from '../components/TermCard'
import TermModal from '../components/TermModal'
import glossaryData from '../data/glossaryData'
import { GlossaryTerm } from '../types/glossary'

const MotionBox = motion(Box)

export default function GlossaryPage() {
    const [search, setSearch] = useState('')
    const [activeLetter, setActiveLetter] = useState<string | null>(null)
    const [selectedTerm, setSelectedTerm] = useState<GlossaryTerm | null>(null)
    const { isOpen, onOpen, onClose } = useDisclosure()

    // Compute available letters
    const availableLetters = useMemo(() => {
        return new Set(glossaryData.map((t) => t.letter))
    }, [])

    // Filter terms
    const filteredTerms = useMemo(() => {
        let terms = glossaryData

        if (activeLetter) {
            terms = terms.filter((t) => t.letter === activeLetter)
        }

        if (search.trim()) {
            const q = search.toLowerCase()
            terms = terms.filter(
                (t) =>
                    t.term.toLowerCase().includes(q) ||
                    t.summary.toLowerCase().includes(q) ||
                    t.definition.toLowerCase().includes(q) ||
                    t.category.toLowerCase().includes(q)
            )
        }

        return terms.sort((a, b) => a.term.localeCompare(b.term))
    }, [activeLetter, search])

    const handleCardClick = useCallback((term: GlossaryTerm) => {
        setSelectedTerm(term)
        onOpen()
    }, [onOpen])

    const handleLetterClick = useCallback((letter: string | null) => {
        setActiveLetter(letter)
        setSearch('')
    }, [])

    return (
        <Box
            minH="100vh"
            className="bg-grid"
            position="relative"
        >
            {/* Background blobs */}
            <Box
                position="fixed"
                top="-20%"
                left="-10%"
                w="600px"
                h="600px"
                borderRadius="full"
                bg="brand.600"
                opacity={0.06}
                filter="blur(100px)"
                pointerEvents="none"
                zIndex={0}
            />
            <Box
                position="fixed"
                bottom="-20%"
                right="-10%"
                w="500px"
                h="500px"
                borderRadius="full"
                bg="accent.400"
                opacity={0.04}
                filter="blur(100px)"
                pointerEvents="none"
                zIndex={0}
            />

            <Container maxW="1400px" px={{ base: 4, md: 8 }} py={10} position="relative" zIndex={1}>
                {/* Header */}
                <MotionBox
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    textAlign="center"
                    mb={10}
                >
                    <Logo />

                    <Text
                        fontSize={{ base: 'sm', md: 'md' }}
                        color="whiteAlpha.600"
                        maxW="500px"
                        mx="auto"
                        mt={2}
                        mb={8}
                        lineHeight={1.7}
                    >
                        Explore {glossaryData.length} essential terms in Artificial Intelligence
                        and Generative AI — from fundamentals to cutting-edge concepts.
                    </Text>

                    {/* Search */}
                    <SearchBar value={search} onChange={setSearch} />
                </MotionBox>

                {/* A-Z Nav */}
                <Box mb={10}>
                    <AlphabetNav
                        activeLetter={activeLetter}
                        availableLetters={availableLetters}
                        onLetterClick={handleLetterClick}
                    />
                </Box>

                {/* Results count */}
                <Flex align="center" justify="space-between" mb={6} px={1}>
                    <Text fontSize="sm" color="whiteAlpha.500">
                        {filteredTerms.length === glossaryData.length
                            ? `${glossaryData.length} terms`
                            : `${filteredTerms.length} of ${glossaryData.length} terms`}
                        {activeLetter && ` · Letter "${activeLetter}"`}
                        {search && ` · "${search}"`}
                    </Text>
                </Flex>

                {/* Grid */}
                <AnimatePresence mode="wait">
                    {filteredTerms.length > 0 ? (
                        <SimpleGrid
                            key="grid"
                            columns={{ base: 1, sm: 2, lg: 3, xl: 4 }}
                            spacing={5}
                        >
                            {filteredTerms.map((term, i) => (
                                <TermCard
                                    key={term.id}
                                    term={term}
                                    index={i}
                                    onClick={() => handleCardClick(term)}
                                />
                            ))}
                        </SimpleGrid>
                    ) : (
                        <MotionBox
                            key="empty"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.3 }}
                        >
                            <Center flexDirection="column" py={20} gap={4}>
                                <Box
                                    w="80px"
                                    h="80px"
                                    borderRadius="2xl"
                                    bg="whiteAlpha.50"
                                    display="flex"
                                    alignItems="center"
                                    justifyContent="center"
                                    border="1px solid"
                                    borderColor="whiteAlpha.100"
                                >
                                    <Icon as={FiSearch} boxSize={8} color="whiteAlpha.300" />
                                </Box>
                                <Text color="whiteAlpha.500" fontSize="lg" fontWeight="600">
                                    No terms found
                                </Text>
                                <Text color="whiteAlpha.400" fontSize="sm">
                                    Try a different search or letter filter
                                </Text>
                            </Center>
                        </MotionBox>
                    )}
                </AnimatePresence>

                {/* Footer */}
                <Box mt={16} textAlign="center">
                    <Text fontSize="xs" color="whiteAlpha.300">
                        IA Glossary · AI & Generative AI Reference · {new Date().getFullYear()}
                    </Text>
                </Box>
            </Container>

            {/* Modal */}
            <TermModal term={selectedTerm} isOpen={isOpen} onClose={onClose} />
        </Box>
    )
}
