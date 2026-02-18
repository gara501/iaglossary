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
            w="100%" maxW="560px" mx="auto"
        >
            <InputGroup size="lg">
                <InputLeftElement pointerEvents="none" h="full" pl={1}>
                    <Icon as={FiSearch}
                        color={dark ? 'rgba(255,155,81,0.50)' : 'rgba(196,98,26,0.50)'}
                        boxSize={4} />
                </InputLeftElement>
                <Input
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    placeholder={placeholder}
                    className="glass-input"
                    h="52px" pl="44px" fontSize="15px" fontWeight="500"
                    _placeholder={{
                        color: dark ? 'rgba(234,239,239,0.35)' : 'rgba(37,52,63,0.38)',
                        fontWeight: '400'
                    }}
                />
            </InputGroup>
        </MotionBox>
    )
}
