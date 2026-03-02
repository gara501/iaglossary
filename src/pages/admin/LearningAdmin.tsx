import { useState, useEffect, useCallback, useRef } from 'react';
import {
    Box, Heading, Button, Table, Thead, Tbody, Tr, Th, Td, IconButton, HStack, Select,
    Modal, ModalOverlay, ModalContent, ModalHeader, ModalBody, ModalFooter, ModalCloseButton,
    FormControl, FormLabel, Input, Textarea, useDisclosure, useToast, Badge, Text,
    AlertDialog, AlertDialogOverlay, AlertDialogContent, AlertDialogHeader, AlertDialogBody,
    AlertDialogFooter, Spinner, Center, Flex, Link, Icon,
} from '@chakra-ui/react';
import { FiPlus, FiEdit2, FiTrash2, FiExternalLink } from 'react-icons/fi';
import { LearningItem } from '../../types/learning';
import {
    fetchLearningItems, createLearningItem, updateLearningItem, deleteLearningItem
} from '../../services/learningService';
import { useColorMode } from '../../context/ThemeContext';

const emptyForm: LearningItem = {
    id: '', title: '', creator: '', summary: '', link: '', category: '',
};

export default function LearningAdmin() {
    const [items, setItems] = useState<LearningItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [lang, setLang] = useState('en');
    const [form, setForm] = useState<LearningItem>(emptyForm);
    const [editingSlug, setEditingSlug] = useState<string | null>(null);
    const [deleteSlug, setDeleteSlug] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);

    const { isOpen, onOpen, onClose } = useDisclosure();
    const { isOpen: isDeleteOpen, onOpen: onDeleteOpen, onClose: onDeleteClose } = useDisclosure();
    const cancelRef = useRef<HTMLButtonElement>(null);
    const toast = useToast();
    const { colorMode } = useColorMode();
    const dark = colorMode === 'dark';

    const loadItems = useCallback(async () => {
        setLoading(true);
        try {
            const data = await fetchLearningItems(lang);
            setItems(data);
        } catch (err) {
            toast({ title: 'Error loading items', description: String(err), status: 'error' });
        } finally {
            setLoading(false);
        }
    }, [lang, toast]);

    useEffect(() => { loadItems(); }, [loadItems]);

    const handleNew = () => {
        setForm(emptyForm);
        setEditingSlug(null);
        onOpen();
    };

    const handleEdit = (item: LearningItem) => {
        setForm({ ...item });
        setEditingSlug(item.id);
        onOpen();
    };

    const handleDelete = (slug: string) => {
        setDeleteSlug(slug);
        onDeleteOpen();
    };

    const confirmDelete = async () => {
        if (!deleteSlug) return;
        try {
            await deleteLearningItem(deleteSlug, lang);
            toast({ title: 'Item deleted', status: 'success', duration: 2000 });
            loadItems();
        } catch (err) {
            toast({ title: 'Delete failed', description: String(err), status: 'error' });
        }
        onDeleteClose();
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            if (editingSlug) {
                await updateLearningItem(editingSlug, lang, form);
                toast({ title: 'Item updated', status: 'success', duration: 2000 });
            } else {
                await createLearningItem(form, lang);
                toast({ title: 'Item created', status: 'success', duration: 2000 });
            }
            onClose();
            loadItems();
        } catch (err) {
            toast({ title: 'Save failed', description: String(err), status: 'error' });
        } finally {
            setSaving(false);
        }
    };

    const updateForm = (field: keyof LearningItem, value: string) => {
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
                    <Heading size="lg" color={titleColor}>Learning Items</Heading>
                    <Text fontSize="14px" color={textColor} mt={1}>{items.length} items</Text>
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
                        New Item
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
                                <Th color={textColor}>Title</Th>
                                <Th color={textColor}>Creator</Th>
                                <Th color={textColor}>Category</Th>
                                <Th color={textColor}>Link</Th>
                                <Th color={textColor} w="60px">Actions</Th>
                            </Tr>
                        </Thead>
                        <Tbody>
                            {items.map(item => (
                                <Tr key={item.id} _hover={{ bg: dark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)' }}>
                                    <Td color={titleColor} fontWeight="500" fontSize="13px" maxW="250px" isTruncated>
                                        {item.title}
                                    </Td>
                                    <Td color={textColor} fontSize="12px">{item.creator}</Td>
                                    <Td><Badge colorScheme="orange" fontSize="10px">{item.category || '—'}</Badge></Td>
                                    <Td>
                                        <Link href={item.link} isExternal color="orange.400" fontSize="12px">
                                            <Icon as={FiExternalLink} />
                                        </Link>
                                    </Td>
                                    <Td>
                                        <HStack spacing={1}>
                                            <IconButton
                                                aria-label="Edit" icon={<FiEdit2 />} size="xs"
                                                variant="ghost" colorScheme="blue"
                                                onClick={() => handleEdit(item)}
                                            />
                                            <IconButton
                                                aria-label="Delete" icon={<FiTrash2 />} size="xs"
                                                variant="ghost" colorScheme="red"
                                                onClick={() => handleDelete(item.id)}
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
                        {editingSlug ? 'Edit Item' : 'New Item'}
                    </ModalHeader>
                    <ModalCloseButton />
                    <ModalBody pb={6}>
                        <FormControl mb={4}>
                            <FormLabel fontSize="13px" color={textColor}>Title</FormLabel>
                            <Input value={form.title} onChange={e => updateForm('title', e.target.value)}
                                borderRadius="10px" color={titleColor} />
                        </FormControl>
                        <HStack spacing={4} mb={4}>
                            <FormControl>
                                <FormLabel fontSize="13px" color={textColor}>Creator</FormLabel>
                                <Input value={form.creator} onChange={e => updateForm('creator', e.target.value)}
                                    borderRadius="10px" color={titleColor} />
                            </FormControl>
                            <FormControl>
                                <FormLabel fontSize="13px" color={textColor}>Category</FormLabel>
                                <Input value={form.category || ''} onChange={e => updateForm('category', e.target.value)}
                                    borderRadius="10px" color={titleColor} placeholder="Course, Article, Study..." />
                            </FormControl>
                        </HStack>
                        <FormControl mb={4}>
                            <FormLabel fontSize="13px" color={textColor}>Summary</FormLabel>
                            <Textarea value={form.summary} onChange={e => updateForm('summary', e.target.value)}
                                rows={4} borderRadius="10px" color={titleColor} />
                        </FormControl>
                        <FormControl>
                            <FormLabel fontSize="13px" color={textColor}>Link (URL)</FormLabel>
                            <Input value={form.link} onChange={e => updateForm('link', e.target.value)}
                                borderRadius="10px" color={titleColor} type="url" placeholder="https://..." />
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
                        <AlertDialogHeader color={titleColor}>Delete Item</AlertDialogHeader>
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
