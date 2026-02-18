import { Box, InputGroup, InputLeftElement, Input, Icon } from '@chakra-ui/react'
import { FiSearch } from 'react-icons/fi'
import { motion } from 'framer-motion'
import { useColorMode } from '../context/ThemeContext'

const MotionBox = motion(Box)

interface SearchBarProps {
    value: string
    onChange: (value: string) => void
    placeholder?: string
}

export default function SearchBar({ value, onChange, placeholder = 'Search AI terms...' }: SearchBarProps) {
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'

    return (
        <MotionBox
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            w="100%"
            maxW="560px"
            mx="auto"
        >
            <InputGroup size="lg">
                <InputLeftElement pointerEvents="none" h="full" pl={1}>
                    <Icon as={FiSearch} color={dark ? 'rgba(243,230,0,0.40)' : 'rgba(197,0,60,0.45)'} boxSize={4} />
                </InputLeftElement>
                <Input
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    placeholder={placeholder}
                    className="glass-input"
                    h="52px"
                    pl="44px"
                    fontSize="15px"
                    fontWeight="500"
                    _placeholder={{ color: dark ? 'rgba(245,234,236,0.30)' : 'rgba(26,10,13,0.35)', fontWeight: '400' }}
                />
            </InputGroup>
        </MotionBox>
    )
}
