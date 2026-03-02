import { useState, useEffect, useCallback } from 'react';
import {
    Box, Heading, Button, Table, Thead, Tbody, Tr, Th, Td, IconButton, HStack, Select,
    Modal, ModalOverlay, ModalContent, ModalHeader, ModalBody, ModalFooter, ModalCloseButton,
    FormControl, FormLabel, Input, Textarea, useDisclosure, useToast, Badge, Text,
    AlertDialog, AlertDialogOverlay, AlertDialogContent, AlertDialogHeader, AlertDialogBody,
    AlertDialogFooter, Spinner, Center, Flex,
} from '@chakra-ui/react';
import { FiPlus, FiEdit2, FiTrash2 } from 'react-icons/fi';
import { useRef } from 'react';
import { GlossaryTerm } from '../../types/glossary';
import {
    fetchGlossaryTerms, createGlossaryTerm, updateGlossaryTerm, deleteGlossaryTerm
} from '../../services/glossaryService';
import { useColorMode } from '../../context/ThemeContext';

const emptyForm: GlossaryTerm = {
    id: '', term: '', letter: '', summary: '', definition: '', category: '', relatedTerms: [],
};

export default function GlossaryAdmin() {
    const [terms, setTerms] = useState<GlossaryTerm[]>([]);
    const [loading, setLoading] = useState(true);
    const [lang, setLang] = useState('en');
    const [form, setForm] = useState<GlossaryTerm>(emptyForm);
    const [editingSlug, setEditingSlug] = useState<string | null>(null);
    const [deleteSlug, setDeleteSlug] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);

    const { isOpen, onOpen, onClose } = useDisclosure();
    const { isOpen: isDeleteOpen, onOpen: onDeleteOpen, onClose: onDeleteClose } = useDisclosure();
    const cancelRef = useRef<HTMLButtonElement>(null);
    const toast = useToast();
    const { colorMode } = useColorMode();
    const dark = colorMode === 'dark';

    const loadTerms = useCallback(async () => {
        setLoading(true);
        try {
            const data = await fetchGlossaryTerms(lang);
            setTerms(data);
        } catch (err) {
            toast({ title: 'Error loading terms', description: String(err), status: 'error' });
        } finally {
            setLoading(false);
        }
    }, [lang, toast]);

    useEffect(() => { loadTerms(); }, [loadTerms]);

    const handleNew = () => {
        setForm(emptyForm);
        setEditingSlug(null);
        onOpen();
    };

    const handleEdit = (term: GlossaryTerm) => {
        setForm({ ...term });
        setEditingSlug(term.id);
        onOpen();
    };

    const handleDelete = (slug: string) => {
        setDeleteSlug(slug);
        onDeleteOpen();
    };

    const confirmDelete = async () => {
        if (!deleteSlug) return;
        try {
            await deleteGlossaryTerm(deleteSlug, lang);
            toast({ title: 'Term deleted', status: 'success', duration: 2000 });
            loadTerms();
        } catch (err) {
            toast({ title: 'Delete failed', description: String(err), status: 'error' });
        }
        onDeleteClose();
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            if (editingSlug) {
                await updateGlossaryTerm(editingSlug, lang, form);
                toast({ title: 'Term updated', status: 'success', duration: 2000 });
            } else {
                await createGlossaryTerm(form, lang);
                toast({ title: 'Term created', status: 'success', duration: 2000 });
            }
            onClose();
            loadTerms();
        } catch (err) {
            toast({ title: 'Save failed', description: String(err), status: 'error' });
        } finally {
            setSaving(false);
        }
    };

    const updateForm = (field: keyof GlossaryTerm, value: string | string[]) => {
        setForm(prev => ({ ...prev, [field]: value }));
    };

    const cardBg = dark ? 'rgba(255,255,255,0.03)' : 'white';
    const tableBorder = dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    const titleColor = dark ? '#EAEFEF' : '#25343F';
    const textColor = dark ? '#BFC9D1' : '#25343F';

    return (
        <Box>
            <Flex justify="space-between" align="center" mb={6}>
                <Box>
                    <Heading size="lg" color={titleColor}>Glossary Terms</Heading>
                    <Text fontSize="14px" color={textColor} mt={1}>{terms.length} terms</Text>
                </Box>
                <HStack spacing={3}>
                    <Select
                        value={lang} onChange={e => setLang(e.target.value)}
                        w="100px" size="sm" borderRadius="8px"
                        bg={cardBg} color={textColor} borderColor={tableBorder}
                    >
                        <option value="en">EN</option>
                        <option value="es">ES</option>
                    </Select>
                    <Button leftIcon={<FiPlus />} colorScheme="orange" size="sm" borderRadius="10px" onClick={handleNew}>
                        New Term
                    </Button>
                </HStack>
            </Flex>

            {loading ? (
                <Center py={20}><Spinner size="xl" color="orange.400" /></Center>
            ) : (
                <Box overflowX="auto" borderRadius="16px" border="1px solid" borderColor={tableBorder} bg={cardBg}>
                    <Table size="sm">
                        <Thead>
                            <Tr>
                                <Th color={textColor}>Term</Th>
                                <Th color={textColor}>Letter</Th>
                                <Th color={textColor}>Category</Th>
                                <Th color={textColor} w="60px">Actions</Th>
                            </Tr>
                        </Thead>
                        <Tbody>
                            {terms.map(t => (
                                <Tr key={t.id} _hover={{ bg: dark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)' }}>
                                    <Td color={titleColor} fontWeight="500" fontSize="13px">{t.term}</Td>
                                    <Td><Badge colorScheme="orange" fontSize="10px">{t.letter}</Badge></Td>
                                    <Td color={textColor} fontSize="12px">{t.category}</Td>
                                    <Td>
                                        <HStack spacing={1}>
                                            <IconButton
                                                aria-label="Edit" icon={<FiEdit2 />} size="xs"
                                                variant="ghost" colorScheme="blue"
                                                onClick={() => handleEdit(t)}
                                            />
                                            <IconButton
                                                aria-label="Delete" icon={<FiTrash2 />} size="xs"
                                                variant="ghost" colorScheme="red"
                                                onClick={() => handleDelete(t.id)}
                                            />
                                        </HStack>
                                    </Td>
                                </Tr>
                            ))}
                        </Tbody>
                    </Table>
                </Box>
            )}

            {/* Create/Edit Modal */}
            <Modal isOpen={isOpen} onClose={onClose} size="xl" scrollBehavior="inside">
                <ModalOverlay backdropFilter="blur(4px)" />
                <ModalContent bg={dark ? '#1a1f25' : 'white'} borderRadius="16px">
                    <ModalHeader color={titleColor}>
                        {editingSlug ? 'Edit Term' : 'New Term'}
                    </ModalHeader>
                    <ModalCloseButton />
                    <ModalBody pb={6}>
                        <FormControl mb={4}>
                            <FormLabel fontSize="13px" color={textColor}>Term</FormLabel>
                            <Input value={form.term} onChange={e => updateForm('term', e.target.value)}
                                borderRadius="10px" color={titleColor} />
                        </FormControl>
                        <HStack spacing={4} mb={4}>
                            <FormControl>
                                <FormLabel fontSize="13px" color={textColor}>Letter</FormLabel>
                                <Input value={form.letter} onChange={e => updateForm('letter', e.target.value.toUpperCase())}
                                    maxLength={1} borderRadius="10px" color={titleColor} />
                            </FormControl>
                            <FormControl>
                                <FormLabel fontSize="13px" color={textColor}>Category</FormLabel>
                                <Input value={form.category} onChange={e => updateForm('category', e.target.value)}
                                    borderRadius="10px" color={titleColor} />
                            </FormControl>
                        </HStack>
                        <FormControl mb={4}>
                            <FormLabel fontSize="13px" color={textColor}>Summary</FormLabel>
                            <Textarea value={form.summary} onChange={e => updateForm('summary', e.target.value)}
                                rows={3} borderRadius="10px" color={titleColor} />
                        </FormControl>
                        <FormControl mb={4}>
                            <FormLabel fontSize="13px" color={textColor}>Definition</FormLabel>
                            <Textarea value={form.definition} onChange={e => updateForm('definition', e.target.value)}
                                rows={6} borderRadius="10px" color={titleColor} />
                        </FormControl>
                        <FormControl>
                            <FormLabel fontSize="13px" color={textColor}>Related Terms (comma-separated)</FormLabel>
                            <Input
                                value={(form.relatedTerms || []).join(', ')}
                                onChange={e => updateForm('relatedTerms', e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
                                borderRadius="10px" color={titleColor}
                            />
                        </FormControl>
                    </ModalBody>
                    <ModalFooter>
                        <Button variant="ghost" mr={3} onClick={onClose} color={textColor}>Cancel</Button>
                        <Button colorScheme="orange" onClick={handleSave} isLoading={saving} borderRadius="10px">
                            {editingSlug ? 'Update' : 'Create'}
                        </Button>
                    </ModalFooter>
                </ModalContent>
            </Modal>

            {/* Delete Confirmation */}
            <AlertDialog isOpen={isDeleteOpen} leastDestructiveRef={cancelRef} onClose={onDeleteClose}>
                <AlertDialogOverlay>
                    <AlertDialogContent bg={dark ? '#1a1f25' : 'white'} borderRadius="16px">
                        <AlertDialogHeader color={titleColor}>Delete Term</AlertDialogHeader>
                        <AlertDialogBody color={textColor}>
                            Are you sure? This action cannot be undone.
                        </AlertDialogBody>
                        <AlertDialogFooter>
                            <Button ref={cancelRef} onClick={onDeleteClose}>Cancel</Button>
                            <Button colorScheme="red" onClick={confirmDelete} ml={3}>Delete</Button>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialogOverlay>
            </AlertDialog>
        </Box>
    );
}
