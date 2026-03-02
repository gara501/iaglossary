import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { Box, Flex, Heading, Button, Icon, HStack } from '@chakra-ui/react';
import { FiBookOpen, FiList, FiLogOut, FiArrowLeft } from 'react-icons/fi';
import { useAuth } from '../../context/AuthContext';
import { useColorMode } from '../../context/ThemeContext';

export default function AdminLayout() {
    const { signOut, user } = useAuth();
    const navigate = useNavigate();
    const { colorMode } = useColorMode();
    const dark = colorMode === 'dark';

    const handleLogout = async () => {
        await signOut();
        navigate('/login');
    };

    const headerBg = dark ? 'rgba(25, 30, 35, 0.90)' : 'rgba(255, 255, 255, 0.85)';
    const headerBorder = dark ? 'rgba(255,255,255,0.08)' : 'rgba(191,201,209,0.4)';
    const titleColor = dark ? '#EAEFEF' : '#25343F';
    const userColor = dark ? '#BFC9D1' : 'rgba(37,52,63,0.6)';

    const linkStyle = (isActive: boolean) => ({
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        padding: '8px 16px',
        borderRadius: '10px',
        fontSize: '14px',
        fontWeight: 600,
        color: isActive ? '#FF9B51' : (dark ? '#BFC9D1' : '#25343F'),
        background: isActive
            ? (dark ? 'rgba(255,155,81,0.15)' : 'rgba(255,155,81,0.10)')
            : 'transparent',
        textDecoration: 'none',
        transition: 'all 0.2s',
    });

    return (
        <Box minH="100vh" bg={dark ? '#0f1318' : '#f5f7f9'}>
            {/* Header */}
            <Box
                position="sticky"
                top={0}
                zIndex={10}
                bg={headerBg}
                borderBottom="1px solid"
                borderColor={headerBorder}
                backdropFilter="blur(24px)"
                px={6}
                py={3}
            >
                <Flex align="center" justify="space-between" maxW="1200px" mx="auto">
                    <HStack spacing={4}>
                        <Heading size="md" color={titleColor} letterSpacing="-0.02em">
                            🛠 Admin
                        </Heading>

                        <HStack spacing={1} ml={4}>
                            <NavLink to="/admin/glossary" style={({ isActive }) => linkStyle(isActive)}>
                                <Icon as={FiList} />
                                Glossary
                            </NavLink>
                            <NavLink to="/admin/learning" style={({ isActive }) => linkStyle(isActive)}>
                                <Icon as={FiBookOpen} />
                                Learning
                            </NavLink>
                        </HStack>
                    </HStack>

                    <HStack spacing={3}>
                        <Box fontSize="12px" color={userColor}>{user?.email}</Box>
                        <Button
                            size="sm"
                            variant="ghost"
                            leftIcon={<FiArrowLeft />}
                            color={userColor}
                            onClick={() => navigate('/')}
                            _hover={{ color: 'orange.400' }}
                        >
                            Site
                        </Button>
                        <Button
                            size="sm"
                            variant="ghost"
                            leftIcon={<FiLogOut />}
                            color={userColor}
                            onClick={handleLogout}
                            _hover={{ color: 'red.400' }}
                        >
                            Logout
                        </Button>
                    </HStack>
                </Flex>
            </Box>

            {/* Content */}
            <Box maxW="1200px" mx="auto" px={6} py={8}>
                <Outlet />
            </Box>
        </Box>
    );
}
