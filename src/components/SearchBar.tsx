import {
    InputGroup,
    InputLeftElement,
    Input,
    Icon,
    Box,
} from '@chakra-ui/react'
import { FiSearch } from 'react-icons/fi'
import { motion } from 'framer-motion'

const MotionBox = motion(Box)

interface SearchBarProps {
    value: string
    onChange: (value: string) => void
}

export default function SearchBar({ value, onChange }: SearchBarProps) {
    return (
        <MotionBox
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            w="100%"
            maxW="600px"
            mx="auto"
        >
            <InputGroup size="lg">
                <InputLeftElement pointerEvents="none" h="full">
                    <Icon as={FiSearch} color="brand.300" boxSize={5} />
                </InputLeftElement>
                <Input
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    placeholder="Search AI terms..."
                    bg="whiteAlpha.50"
                    border="1px solid"
                    borderColor="brand.700"
                    borderRadius="xl"
                    color="white"
                    fontSize="md"
                    h="54px"
                    pl="48px"
                    _placeholder={{ color: 'whiteAlpha.400' }}
                    _hover={{
                        borderColor: 'brand.500',
                        bg: 'whiteAlpha.100',
                    }}
                    _focus={{
                        borderColor: 'brand.400',
                        bg: 'whiteAlpha.100',
                        boxShadow: '0 0 0 1px rgba(130, 38, 255, 0.5), 0 0 20px rgba(130, 38, 255, 0.15)',
                        outline: 'none',
                    }}
                    transition="all 0.2s"
                />
            </InputGroup>
        </MotionBox>
    )
}
