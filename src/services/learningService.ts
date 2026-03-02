import supabase from '../utils/supabase';
import { LearningItem } from '../types/learning';

export interface LearningItemRow {
    id: string;
    slug: string;
    title: string;
    creator: string;
    summary: string;
    link: string;
    category: string;
    lang: string;
    created_at: string;
    updated_at: string;
}

function rowToLearningItem(row: LearningItemRow): LearningItem {
    return {
        id: row.slug,
        title: row.title,
        creator: row.creator,
        summary: row.summary,
        link: row.link,
        category: row.category,
    };
}

export async function fetchLearningItems(lang: string): Promise<LearningItem[]> {
    const { data, error } = await supabase
        .from('learning_items')
        .select('*')
        .eq('lang', lang)
        .order('title', { ascending: true });

    if (error) throw error;
    return (data || []).map(rowToLearningItem);
}

export async function createLearningItem(
    item: Omit<LearningItem, 'id'> & { id?: string },
    lang: string
): Promise<LearningItemRow> {
    const slug = item.id || item.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    const { data, error } = await supabase
        .from('learning_items')
        .insert({
            slug,
            title: item.title,
            creator: item.creator,
            summary: item.summary,
            link: item.link,
            category: item.category || '',
            lang,
        })
        .select()
        .single();

    if (error) throw error;
    return data;
}

export async function updateLearningItem(
    slug: string,
    lang: string,
    updates: Partial<LearningItem>
): Promise<LearningItemRow> {
    const updateData: Record<string, unknown> = {};
    if (updates.title !== undefined) updateData.title = updates.title;
    if (updates.creator !== undefined) updateData.creator = updates.creator;
    if (updates.summary !== undefined) updateData.summary = updates.summary;
    if (updates.link !== undefined) updateData.link = updates.link;
    if (updates.category !== undefined) updateData.category = updates.category;

    const { data, error } = await supabase
        .from('learning_items')
        .update(updateData)
        .eq('slug', slug)
        .eq('lang', lang)
        .select()
        .single();

    if (error) throw error;
    return data;
}

export async function deleteLearningItem(slug: string, lang: string): Promise<void> {
    const { error } = await supabase
        .from('learning_items')
        .delete()
        .eq('slug', slug)
        .eq('lang', lang);

    if (error) throw error;
}
