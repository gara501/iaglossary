import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Box, Container, Heading, Input, Button, Text, VStack, FormControl, FormLabel,
    InputGroup, InputLeftElement, Icon, useToast,
} from '@chakra-ui/react';
import { FiMail, FiLock, FiLogIn } from 'react-icons/fi';
import { useAuth } from '../context/AuthContext';
import { useColorMode } from '../context/ThemeContext';

export default function LoginPage() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const { signIn } = useAuth();
    const navigate = useNavigate();
    const toast = useToast();
    const { colorMode } = useColorMode();
    const dark = colorMode === 'dark';

    // Apply body class for CSS dark/light mode
    useEffect(() => {
        document.body.classList.remove('dark', 'light');
        document.body.classList.add(colorMode);
    }, [colorMode]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        const { error } = await signIn(email, password);
        setLoading(false);

        if (error) {
            toast({
                title: 'Login failed',
                description: error.message,
                status: 'error',
                duration: 4000,
                isClosable: true,
            });
        } else {
            navigate('/admin');
        }
    };

    const cardBg = dark ? 'rgba(255, 255, 255, 0.04)' : 'rgba(240, 244, 248, 0.95)';
    const cardBorder = dark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(37, 52, 63, 0.15)';
    const titleColor = dark ? '#EAEFEF' : '#1a2a35';
    const textColor = dark ? '#BFC9D1' : '#3d5568';
    const inputBg = dark ? 'rgba(255,255,255,0.06)' : '#ffffff';
    const inputBorder = dark ? 'rgba(255,255,255,0.12)' : 'rgba(37,52,63,0.20)';
    const labelColor = dark ? '#BFC9D1' : '#2c4256';

    const blob1 = dark ? 'rgba(255, 155, 81, 0.12)' : 'rgba(255, 155, 81, 0.15)';
    const blob2 = dark ? 'rgba(37,  52,  63, 0.80)' : 'rgba(37, 52, 63, 0.12)';

    return (
        <Box minH="100vh" position="relative" display="flex" alignItems="center" justifyContent="center">
            <div className="glass-bg" />
            <Box position="fixed" top="-15%" left="-5%" w="700px" h="700px" borderRadius="full"
                bg={blob1} filter="blur(120px)" pointerEvents="none" zIndex={0}
                style={{ animation: 'float-blob 18s ease-in-out infinite' }} />
            <Box position="fixed" bottom="-20%" right="-10%" w="600px" h="600px" borderRadius="full"
                bg={blob2} filter="blur(100px)" pointerEvents="none" zIndex={0}
                style={{ animation: 'float-blob 22s ease-in-out infinite reverse' }} />

            <Container maxW="420px" position="relative" zIndex={1}>
                <Box
                    as="form"
                    onSubmit={handleSubmit}
                    p={10}
                    borderRadius="24px"
                    bg={cardBg}
                    border="1px solid"
                    borderColor={cardBorder}
                    backdropFilter="blur(24px)"
                    boxShadow={dark ? '0 16px 48px rgba(0,0,0,0.3)' : '0 16px 48px rgba(37,52,63,0.15)'}
                >
                    <VStack spacing={6} align="stretch">
                        <Box textAlign="center">
                            <Heading size="lg" color={titleColor} letterSpacing="-0.02em">
                                Admin Login
                            </Heading>
                            <Text fontSize="14px" color={textColor} mt={2}>
                                Sign in to manage glossary & learning content
                            </Text>
                        </Box>

                        <FormControl>
                            <FormLabel fontSize="13px" color={labelColor} fontWeight="600">Email</FormLabel>
                            <InputGroup>
                                <InputLeftElement pointerEvents="none">
                                    <Icon as={FiMail} color="orange.400" />
                                </InputLeftElement>
                                <Input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="admin@example.com"
                                    bg={inputBg}
                                    border="1px solid"
                                    borderColor={inputBorder}
                                    borderRadius="12px"
                                    color={dark ? titleColor : '#1a2a35'}
                                    _placeholder={{ color: textColor, opacity: 0.5 }}
                                    _focus={{ borderColor: 'orange.400', boxShadow: '0 0 0 1px var(--chakra-colors-orange-400)' }}
                                    required
                                />
                            </InputGroup>
                        </FormControl>

                        <FormControl>
                            <FormLabel fontSize="13px" color={labelColor} fontWeight="600">Password</FormLabel>
                            <InputGroup>
                                <InputLeftElement pointerEvents="none">
                                    <Icon as={FiLock} color="orange.400" />
                                </InputLeftElement>
                                <Input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    bg={inputBg}
                                    border="1px solid"
                                    borderColor={inputBorder}
                                    borderRadius="12px"
                                    color={dark ? titleColor : '#1a2a35'}
                                    _placeholder={{ color: textColor, opacity: 0.5 }}
                                    _focus={{ borderColor: 'orange.400', boxShadow: '0 0 0 1px var(--chakra-colors-orange-400)' }}
                                    required
                                />
                            </InputGroup>
                        </FormControl>

                        <Button
                            type="submit"
                            isLoading={loading}
                            loadingText="Signing in..."
                            leftIcon={<FiLogIn />}
                            bg="linear-gradient(135deg, #FF9B51, #e07e38)"
                            color="white"
                            borderRadius="12px"
                            size="lg"
                            fontWeight="700"
                            _hover={{ opacity: 0.9, transform: 'translateY(-1px)' }}
                            _active={{ transform: 'translateY(0)' }}
                            transition="all 0.2s"
                        >
                            Sign In
                        </Button>

                        <Button
                            variant="ghost"
                            size="sm"
                            color={textColor}
                            onClick={() => navigate('/')}
                            _hover={{ color: 'orange.400' }}
                        >
                            ← Back to Glossary
                        </Button>
                    </VStack>
                </Box>
            </Container>
        </Box>
    );
}
