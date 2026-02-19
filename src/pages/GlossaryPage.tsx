import { useState, useMemo, useCallback, useEffect } from 'react'
import {
    Box, Container, SimpleGrid, Text, Flex, useDisclosure, Center, Icon,
} from '@chakra-ui/react'
import { motion, AnimatePresence } from 'framer-motion'
import { FiSearch } from 'react-icons/fi'
import Logo from '../components/Logo'
import SearchBar from '../components/SearchBar'
import AlphabetNav from '../components/AlphabetNav'
import Pagination from '../components/Pagination'
import TermCard from '../components/TermCard'
import TermModal from '../components/TermModal'
import glossaryDataEn from '../data/glossaryData'
import glossaryDataEs from '../data/glossaryDataEs'
import { GlossaryTerm } from '../types/glossary'
import { useLanguage } from '../context/LanguageContext'
import { useStrings } from '../i18n/strings'
import { useColorMode } from '../context/ThemeContext'

const MotionBox = motion(Box)

export default function GlossaryPage() {
    const [search, setSearch] = useState('')
    const [activeLetter, setActiveLetter] = useState<string | null>(null)
    const [selectedTerm, setSelectedTerm] = useState<GlossaryTerm | null>(null)
    const { isOpen, onOpen, onClose } = useDisclosure()

    const [currentPage, setCurrentPage] = useState(1)
    const ITEMS_PER_PAGE = 10

    const { language } = useLanguage()
    const s = useStrings(language)
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'

    // Apply body class for CSS dark/light mode
    useEffect(() => {
        document.body.classList.remove('dark', 'light')
        document.body.classList.add(colorMode)
    }, [colorMode])

    // Apply dark class on mount
    useEffect(() => {
        document.body.classList.add('dark')
    }, [])

    const glossaryData = language === 'es' ? glossaryDataEs : glossaryDataEn

    // Build English term lookup for Spanish mode
    const enTermById = useMemo(() => {
        if (language !== 'es') return null
        const map: Record<string, string> = {}
        for (const t of glossaryDataEn) map[t.id] = t.term
        return map
    }, [language])

    const availableLetters = useMemo(() => {
        return new Set(glossaryData.map((t) => t.letter))
    }, [glossaryData])

    useMemo(() => {
        setActiveLetter(null)
        setSearch('')
        setCurrentPage(1)
    }, [language])

    useEffect(() => {
        setCurrentPage(1)
    }, [activeLetter, search])

    const filteredTerms = useMemo(() => {
        let terms = glossaryData
        if (activeLetter) terms = terms.filter((t) => t.letter === activeLetter)
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
    }, [activeLetter, search, glossaryData])

    const totalPages = Math.ceil(filteredTerms.length / ITEMS_PER_PAGE)
    const paginatedTerms = useMemo(() => {
        const start = (currentPage - 1) * ITEMS_PER_PAGE
        return filteredTerms.slice(start, start + ITEMS_PER_PAGE)
    }, [filteredTerms, currentPage])

    const handleCardClick = useCallback((term: GlossaryTerm) => {
        setSelectedTerm(term)
        onOpen()
    }, [onOpen])

    const handleLetterClick = useCallback((letter: string | null) => {
        setActiveLetter(letter)
        setSearch('')
        setCurrentPage(1)
    }, [])

    const handleSearchChange = useCallback((value: string) => {
        setSearch(value)
        setCurrentPage(1)
    }, [])

    const subtitle = s.subtitle.replace('{count}', String(glossaryData.length))

    const subtitleColor = dark ? 'rgba(234,239,239,0.55)' : 'rgba(37,52,63,0.60)'
    const countColor = dark ? 'rgba(191,201,209,0.65)' : 'rgba(37,52,63,0.50)'
    const footerColor = dark ? 'rgba(234,239,239,0.22)' : 'rgba(37,52,63,0.30)'
    const emptyIconColor = dark ? 'rgba(255,155,81,0.35)' : 'rgba(255,155,81,0.45)'
    const emptyTextColor = dark ? '#BFC9D1' : '#25343F'
    const emptyHintColor = dark ? 'rgba(191,201,209,0.50)' : 'rgba(37,52,63,0.45)'

    // Blob colors — orange tint in dark, muted in light
    const blob1 = dark ? 'rgba(255, 155, 81, 0.12)' : 'rgba(255, 155, 81, 0.10)'
    const blob2 = dark ? 'rgba(37,  52,  63, 0.80)' : 'rgba(191, 201, 209, 0.50)'
    const blob3 = dark ? 'rgba(191, 201, 209, 0.05)' : 'rgba(191, 201, 209, 0.30)'

    return (
        <Box minH="100vh" position="relative">
            <div className="glass-bg" />

            {/* Animated blobs */}
            <Box position="fixed" top="-15%" left="-5%" w="700px" h="700px" borderRadius="full"
                bg={blob1} filter="blur(120px)" pointerEvents="none" zIndex={0}
                style={{ animation: 'float-blob 18s ease-in-out infinite' }} />
            <Box position="fixed" bottom="-20%" right="-10%" w="600px" h="600px" borderRadius="full"
                bg={blob2} filter="blur(100px)" pointerEvents="none" zIndex={0}
                style={{ animation: 'float-blob 22s ease-in-out infinite reverse' }} />
            <Box position="fixed" top="40%" left="60%" w="400px" h="400px" borderRadius="full"
                bg={blob3} filter="blur(80px)" pointerEvents="none" zIndex={0}
                style={{ animation: 'float-blob 26s ease-in-out infinite 4s' }} />

            <Container maxW="1400px" px={{ base: 4, md: 8 }} py={12} position="relative" zIndex={1}>
                {/* Header */}
                <MotionBox
                    initial={{ opacity: 0, y: -16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    textAlign="center"
                    mb={10}
                >
                    <Logo subtitle={s.logoSubtitle} />

                    <Text fontSize="15px" color={subtitleColor} maxW="500px" mx="auto"
                        mt={3} mb={8} lineHeight={1.75} fontWeight="400">
                        {subtitle}
                    </Text>

                    <SearchBar value={search} onChange={handleSearchChange} placeholder={s.searchPlaceholder} />
                </MotionBox>

                {/* A-Z Nav */}
                <Box mb={10}>
                    <AlphabetNav
                        activeLetter={activeLetter}
                        availableLetters={availableLetters}
                        onLetterClick={handleLetterClick}
                        allLabel={s.allButton}
                        clearLabel={s.clearFilter}
                        language={language}
                    />
                </Box>

                {/* Results count */}
                <Flex align="center" mb={5} px={1}>
                    <Text fontSize="13px" color={countColor} fontWeight="500">
                        {s.termsCount(filteredTerms.length, glossaryData.length)}
                        {activeLetter && s.letterFilter(activeLetter)}
                        {search && s.searchFilter(search)}
                    </Text>
                </Flex>

                {/* Grid */}
                <AnimatePresence mode="wait">
                    {paginatedTerms.length > 0 ? (
                        <>
                            <SimpleGrid key="grid" columns={{ base: 1, sm: 2, lg: 3, xl: 4 }} spacing={4}>
                                {paginatedTerms.map((term, i) => (
                                    <TermCard key={term.id} term={term} index={i}
                                        onClick={() => handleCardClick(term)} readMoreLabel={s.readMore}
                                        englishTerm={enTermById ? enTermById[term.id] : undefined} />
                                ))}
                            </SimpleGrid>

                            <Pagination
                                currentPage={currentPage}
                                totalPages={totalPages}
                                onPageChange={setCurrentPage}
                                prevLabel={language === 'es' ? 'Anterior' : 'Prev'}
                                nextLabel={language === 'es' ? 'Siguiente' : 'Next'}
                            />
                        </>
                    ) : (
                        <MotionBox key="empty"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.25 }}>
                            <Center flexDirection="column" py={20} gap={4}>
                                <Box w="72px" h="72px" borderRadius="20px" className="glass-card"
                                    display="flex" alignItems="center" justifyContent="center">
                                    <Icon as={FiSearch} boxSize={7} color={emptyIconColor} />
                                </Box>
                                <Text color={emptyTextColor} fontSize="16px" fontWeight="600">{s.noTermsFound}</Text>
                                <Text color={emptyHintColor} fontSize="14px">{s.noTermsHint}</Text>
                            </Center>
                        </MotionBox>
                    )}
                </AnimatePresence>

                {/* Footer */}
                <Box mt={16} textAlign="center">
                    <Text fontSize="12px" color={footerColor} letterSpacing="0.04em">
                        {s.footerText} · {new Date().getFullYear()}
                    </Text>
                </Box>
            </Container>

            <TermModal
                term={selectedTerm} isOpen={isOpen} onClose={onClose}
                summaryLabel={s.summaryLabel}
                definitionLabel={s.definitionLabel}
                relatedTermsLabel={s.relatedTermsLabel}
                englishTerm={enTermById && selectedTerm ? enTermById[selectedTerm.id] : undefined}
            />
        </Box>
    )
}
