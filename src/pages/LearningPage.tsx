import { useEffect, useState, useMemo } from 'react'
import {
    Box, Container, SimpleGrid, Text, Flex, Heading, Link, Badge, Icon, Button,
} from '@chakra-ui/react'
import { motion } from 'framer-motion'
import { FiArrowLeft, FiExternalLink, FiBookOpen } from 'react-icons/fi'
import Logo from '../components/Logo'
import learningDataEn from '../data/learningData'
import learningDataEs from '../data/learningDataES'
import { useLanguage } from '../context/LanguageContext'
import { useStrings } from '../i18n/strings'
import { useColorMode } from '../context/ThemeContext'
import Pagination from '../components/Pagination'

const MotionBox = motion(Box)

interface LearningPageProps {
    onReturn: () => void;
}

export default function LearningPage({ onReturn }: LearningPageProps) {
    const { language } = useLanguage()
    const s = useStrings(language)
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'

    useEffect(() => {
        document.body.classList.remove('dark', 'light')
        document.body.classList.add(colorMode)
    }, [colorMode])

    const [currentPage, setCurrentPage] = useState(1)
    const ITEMS_PER_PAGE = 6

    const learningData = language === 'es' ? learningDataEs : learningDataEn

    const totalPages = Math.ceil(learningData.length / ITEMS_PER_PAGE)
    const paginatedData = useMemo(() => {
        const start = (currentPage - 1) * ITEMS_PER_PAGE
        return learningData.slice(start, start + ITEMS_PER_PAGE)
    }, [learningData, currentPage])

    useEffect(() => {
        setCurrentPage(1)
    }, [language])

    const subtitleColor = dark ? 'rgba(234,239,239,0.55)' : 'rgba(37,52,63,0.60)'
    const footerColor = dark ? 'rgba(234,239,239,0.22)' : 'rgba(37,52,63,0.30)'

    const blob1 = dark ? 'rgba(255, 155, 81, 0.12)' : 'rgba(255, 155, 81, 0.10)'
    const blob2 = dark ? 'rgba(37,  52,  63, 0.80)' : 'rgba(191, 201, 209, 0.50)'
    const blob3 = dark ? 'rgba(191, 201, 209, 0.05)' : 'rgba(191, 201, 209, 0.30)'

    const cardBg = dark ? 'rgba(255, 255, 255, 0.03)' : 'rgba(255, 255, 255, 0.65)'
    const cardBorder = dark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(191, 201, 209, 0.50)'
    const titleColor = dark ? '#EAEFEF' : '#25343F'
    const textColor = dark ? '#BFC9D1' : 'rgba(37, 52, 63, 0.85)'
    const creatorColor = dark ? '#FF9B51' : '#e07e38'

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

            <Container maxW="1400px" px={{ base: 4, md: 8 }} pt={{ base: 28, md: 12 }} pb={12} position="relative" zIndex={1}>
                {/* Header */}
                <MotionBox
                    initial={{ opacity: 0, y: -16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    textAlign="center"
                    mb={12}
                >
                    <Logo subtitle={s.logoSubtitle} />

                    <Heading size="lg" mt={6} color={titleColor} letterSpacing="-0.02em">
                        {language === 'es' ? 'Recursos de Aprendizaje' : 'Learning Resources'}
                    </Heading>

                    <Text fontSize="15px" color={subtitleColor} maxW="600px" mx="auto"
                        mt={3} mb={8} lineHeight={1.75} fontWeight="400">
                        {language === 'es'
                            ? 'Amplía tus conocimientos con estos cursos y recursos seleccionados sobre IA y Agentes.'
                            : 'Expand your knowledge with these curated courses and resources on AI and Agents.'}
                    </Text>

                    <Button
                        leftIcon={<FiArrowLeft />}
                        variant="ghost"
                        onClick={onReturn}
                        color={creatorColor}
                        _hover={{ bg: dark ? 'rgba(255,155,81,0.12)' : 'rgba(255,155,81,0.08)' }}
                    >
                        {language === 'es' ? 'Volver al Glosario' : 'Back to Glossary'}
                    </Button>
                </MotionBox>

                {/* Grid */}
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6} maxW="1000px" mx="auto">
                    {paginatedData.map((item, i) => (
                        <MotionBox
                            key={item.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: i * 0.1 }}
                            p={8}
                            borderRadius="24px"
                            bg={cardBg}
                            border="1px solid"
                            borderColor={cardBorder}
                            backdropFilter="blur(20px)"
                            boxShadow={dark ? '0 8px 32px rgba(0,0,0,0.2)' : '0 8px 32px rgba(37,52,63,0.05)'}
                            position="relative"
                            overflow="hidden"
                            role="group"
                            display="flex"
                            flexDirection="column"
                            justifyContent="space-between"
                        >
                            <Box>
                                <Flex justify="space-between" align="start" mb={4}>
                                    <Badge
                                        px={2} py={0.5} borderRadius="6px"
                                        variant="subtle" colorScheme="orange" fontSize="10px"
                                        bg={dark ? 'rgba(255,155,81,0.15)' : 'rgba(255,155,81,0.10)'}
                                    >
                                        {item.category || 'COURSE'}
                                    </Badge>
                                    <Icon as={FiBookOpen} color={creatorColor} opacity={0.5} boxSize={5} />
                                </Flex>

                                <Heading size="md" mb={2} color={titleColor} fontWeight="700">
                                    {item.title}
                                </Heading>

                                <Text fontSize="13px" color={creatorColor} fontWeight="600" mb={4} textTransform="uppercase" letterSpacing="0.05em">
                                    {item.creator}
                                </Text>

                                <Text fontSize="15px" color={textColor} lineHeight="1.6" mb={6}>
                                    {item.summary}
                                </Text>
                            </Box>

                            <Link
                                href={item.link}
                                isExternal
                                display="inline-flex"
                                alignItems="center"
                                color={creatorColor}
                                fontWeight="700"
                                fontSize="14px"
                                _hover={{ textDecoration: 'none', opacity: 0.8 }}
                            >
                                {language === 'es' ? 'Ir al recurso' : 'Visit resource'}
                                <Icon as={FiExternalLink} ml={2} />
                            </Link>
                        </MotionBox>
                    ))}
                </SimpleGrid>

                <Pagination
                    currentPage={currentPage}
                    totalPages={totalPages}
                    onPageChange={setCurrentPage}
                    prevLabel={language === 'es' ? 'Anterior' : 'Prev'}
                    nextLabel={language === 'es' ? 'Siguiente' : 'Next'}
                />

                {/* Footer */}
                <Box mt={16} textAlign="center">
                    <Text fontSize="12px" color={footerColor} letterSpacing="0.04em">
                        {s.footerText} · {new Date().getFullYear()}
                    </Text>
                </Box>
            </Container>
        </Box>
    )
}
