import { Box, HStack, Text, Icon } from '@chakra-ui/react'
import { FiChevronLeft, FiChevronRight } from 'react-icons/fi'
import { useColorMode } from '../context/ThemeContext'

interface PaginationProps {
    currentPage: number;
    totalPages: number;
    onPageChange: (page: number) => void;
    prevLabel?: string;
    nextLabel?: string;
}

export default function Pagination({
    currentPage,
    totalPages,
    onPageChange,
    prevLabel = "Prev",
    nextLabel = "Next"
}: PaginationProps) {
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'

    if (totalPages <= 1) return null

    const pillBg = dark ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.72)'
    const pillBorder = dark ? 'rgba(255,255,255,0.08)' : 'rgba(191,201,209,0.60)'
    const activeBg = dark ? 'rgba(255,155,81,0.20)' : '#FF9B51'
    const activeBdr = dark ? 'rgba(255,155,81,0.45)' : '#e07e38'
    const activeText = dark ? '#FF9B51' : '#ffffff'
    const inactiveText = dark ? 'rgba(234,239,239,0.45)' : 'rgba(37,52,63,0.50)'

    const renderPageButton = (page: number) => {
        const isActive = currentPage === page
        return (
            <Box
                key={page}
                as="button"
                onClick={() => onPageChange(page)}
                w="36px" h="36px" borderRadius="10px"
                bg={isActive ? activeBg : 'transparent'}
                border={isActive ? `1px solid ${activeBdr}` : '1px solid transparent'}
                display="flex" alignItems="center" justifyContent="center"
                transition="all 0.18s"
                _hover={{ bg: isActive ? undefined : 'rgba(255,155,81,0.08)' }}
                cursor="pointer"
            >
                <Text
                    fontSize="13px" fontWeight="700"
                    color={isActive ? activeText : inactiveText}
                >
                    {page}
                </Text>
            </Box>
        )
    }

    const pages = []
    const maxVisible = 5
    let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2))
    let endPage = Math.min(totalPages, startPage + maxVisible - 1)

    if (endPage - startPage + 1 < maxVisible) {
        startPage = Math.max(1, endPage - maxVisible + 1)
    }

    for (let i = startPage; i <= endPage; i++) {
        pages.push(renderPageButton(i))
    }

    return (
        <HStack spacing={4} justify="center" mt={8} mb={4}>
            <HStack
                spacing={1}
                bg={pillBg}
                backdropFilter="blur(16px)"
                border={`1px solid ${pillBorder}`}
                borderRadius="14px" p="4px"
                boxShadow={dark ? '0 4px 24px rgba(0,0,0,0.20)' : '0 2px 12px rgba(37,52,63,0.05)'}
            >
                <Box
                    as="button"
                    disabled={currentPage === 1}
                    onClick={() => onPageChange(currentPage - 1)}
                    px={3} py={1.5} borderRadius="10px"
                    display="flex" alignItems="center" gap={1.5}
                    transition="all 0.18s"
                    opacity={currentPage === 1 ? 0.3 : 1}
                    cursor={currentPage === 1 ? 'not-allowed' : 'pointer'}
                    _hover={{ bg: currentPage === 1 ? undefined : 'rgba(255,155,81,0.08)' }}
                >
                    <Icon as={FiChevronLeft} boxSize={3.5} color="#FF9B51" />
                    <Text fontSize="11px" fontWeight="700" color={inactiveText} letterSpacing="0.06em">
                        {prevLabel.toUpperCase()}
                    </Text>
                </Box>

                <HStack spacing={1}>
                    {pages}
                </HStack>

                <Box
                    as="button"
                    disabled={currentPage === totalPages}
                    onClick={() => onPageChange(currentPage + 1)}
                    px={3} py={1.5} borderRadius="10px"
                    display="flex" alignItems="center" gap={1.5}
                    transition="all 0.18s"
                    opacity={currentPage === totalPages ? 0.3 : 1}
                    cursor={currentPage === totalPages ? 'not-allowed' : 'pointer'}
                    _hover={{ bg: currentPage === totalPages ? undefined : 'rgba(255,155,81,0.08)' }}
                >
                    <Text fontSize="11px" fontWeight="700" color={inactiveText} letterSpacing="0.06em">
                        {nextLabel.toUpperCase()}
                    </Text>
                    <Icon as={FiChevronRight} boxSize={3.5} color="#FF9B51" />
                </Box>
            </HStack>
        </HStack>
    )
}
