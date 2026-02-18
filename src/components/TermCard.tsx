import {
    Box,
    Text,
    Badge,
    Flex,
    Icon,
} from '@chakra-ui/react'
import { motion } from 'framer-motion'
import { FiArrowRight } from 'react-icons/fi'
import { GlossaryTerm } from '../types/glossary'

const MotionBox = motion(Box)

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
    // Spanish categories
    Arquitectura: 'purple',
    'Tipo de Modelo': 'blue',
    Modelo: 'cyan',
    Entrenamiento: 'green',
    'IA Generativa': 'pink',
    Representación: 'teal',
    Despliegue: 'yellow',
    Infraestructura: 'red',
    Fundamentos: 'gray',
    'Paradigma de Aprendizaje': 'messenger',
    'Seguridad y Alineación': 'whatsapp',
}

interface TermCardProps {
    term: GlossaryTerm
    index: number
    onClick: () => void
    readMoreLabel?: string
}

export default function TermCard({ term, index, onClick, readMoreLabel = 'Read more' }: TermCardProps) {
    const badgeColor = categoryColors[term.category] || 'purple'

    return (
        <MotionBox
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: index * 0.04 }}
            onClick={onClick}
            cursor="pointer"
            className="card-hover"
            position="relative"
            overflow="hidden"
            borderRadius="2xl"
            border="1px solid"
            borderColor="whiteAlpha.100"
            bg="rgba(18, 18, 42, 0.8)"
            backdropFilter="blur(10px)"
            p={5}
            h="100%"
            display="flex"
            flexDirection="column"
            role="button"
            tabIndex={0}
            onKeyDown={(e: React.KeyboardEvent) => e.key === 'Enter' && onClick()}
            _focus={{ outline: '2px solid', outlineColor: 'brand.400', outlineOffset: '2px' }}
        >
            {/* Top gradient accent */}
            <Box
                position="absolute"
                top={0}
                left={0}
                right={0}
                h="2px"
                bgGradient="linear(to-r, brand.500, accent.400)"
                opacity={0}
                transition="opacity 0.3s"
                sx={{
                    '.card-hover:hover &': { opacity: 1 },
                }}
            />

            {/* Letter badge */}
            <Box
                position="absolute"
                top={4}
                right={4}
                w="32px"
                h="32px"
                borderRadius="lg"
                bg="whiteAlpha.50"
                border="1px solid"
                borderColor="whiteAlpha.100"
                display="flex"
                alignItems="center"
                justifyContent="center"
            >
                <Text
                    fontSize="sm"
                    fontWeight="800"
                    bgGradient="linear(to-br, brand.300, accent.400)"
                    bgClip="text"
                    fontFamily="Space Grotesk, sans-serif"
                >
                    {term.letter}
                </Text>
            </Box>

            {/* Category badge */}
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
                mb={3}
                w="fit-content"
            >
                {term.category}
            </Badge>

            {/* Term name */}
            <Text
                fontFamily="Space Grotesk, sans-serif"
                fontWeight="700"
                fontSize="lg"
                color="white"
                mb={2}
                lineHeight={1.3}
                pr={8}
            >
                {term.term}
            </Text>

            {/* Summary */}
            <Text
                fontSize="sm"
                color="whiteAlpha.700"
                lineHeight={1.6}
                flex={1}
                noOfLines={3}
            >
                {term.summary}
            </Text>

            {/* Read more */}
            <Flex align="center" mt={4} color="brand.300" gap={1}>
                <Text fontSize="xs" fontWeight="600">
                    {readMoreLabel}
                </Text>
                <Icon as={FiArrowRight} boxSize={3} />
            </Flex>
        </MotionBox>
    )
}
