import { Box, InputGroup, InputLeftElement, Input, Icon } from '@chakra-ui/react'
import { FiSearch } from 'react-icons/fi'
import { motion } from 'framer-motion'

const MotionBox = motion(Box)

interface SearchBarProps {
    value: string
    onChange: (value: string) => void
    placeholder?: string
}

export default function SearchBar({ value, onChange, placeholder = 'Search AI terms...' }: SearchBarProps) {
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
                    <Icon as={FiSearch} color="rgba(160,185,255,0.5)" boxSize={4} />
                </InputLeftElement>
                <Input
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    placeholder={placeholder}
                    className="glass-input"
                    h="50px"
                    pl="44px"
                    fontSize="sm"
                    fontWeight="500"
                    _placeholder={{ color: 'rgba(160,185,255,0.4)', fontWeight: '400' }}
                />
            </InputGroup>
        </MotionBox>
    )
}
