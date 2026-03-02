import supabase from '../utils/supabase';
import { GlossaryTerm } from '../types/glossary';

export interface GlossaryTermRow {
    id: string;
    slug: string;
    term: string;
    letter: string;
    summary: string;
    definition: string;
    category: string;
    related_terms: string[];
    lang: string;
    created_at: string;
    updated_at: string;
}

function rowToGlossaryTerm(row: GlossaryTermRow): GlossaryTerm {
    return {
        id: row.slug,
        term: row.term,
        letter: row.letter,
        summary: row.summary,
        definition: row.definition,
        category: row.category,
        relatedTerms: row.related_terms || [],
    };
}

export async function fetchGlossaryTerms(lang: string): Promise<GlossaryTerm[]> {
    const { data, error } = await supabase
        .from('glossary_terms')
        .select('*')
        .eq('lang', lang)
        .order('term', { ascending: true });

    if (error) throw error;
    return (data || []).map(rowToGlossaryTerm);
}

export async function createGlossaryTerm(
    term: Omit<GlossaryTerm, 'id'> & { id?: string },
    lang: string
): Promise<GlossaryTermRow> {
    const slug = term.id || term.term.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    const { data, error } = await supabase
        .from('glossary_terms')
        .insert({
            slug,
            term: term.term,
            letter: term.letter || term.term.charAt(0).toUpperCase(),
            summary: term.summary,
            definition: term.definition,
            category: term.category,
            related_terms: term.relatedTerms || [],
            lang,
        })
        .select()
        .single();

    if (error) throw error;
    return data;
}

export async function updateGlossaryTerm(
    slug: string,
    lang: string,
    updates: Partial<GlossaryTerm>
): Promise<GlossaryTermRow> {
    const updateData: Record<string, unknown> = {};
    if (updates.term !== undefined) updateData.term = updates.term;
    if (updates.letter !== undefined) updateData.letter = updates.letter;
    if (updates.summary !== undefined) updateData.summary = updates.summary;
    if (updates.definition !== undefined) updateData.definition = updates.definition;
    if (updates.category !== undefined) updateData.category = updates.category;
    if (updates.relatedTerms !== undefined) updateData.related_terms = updates.relatedTerms;

    const { data, error } = await supabase
        .from('glossary_terms')
        .update(updateData)
        .eq('slug', slug)
        .eq('lang', lang)
        .select()
        .single();

    if (error) throw error;
    return data;
}

export async function deleteGlossaryTerm(slug: string, lang: string): Promise<void> {
    const { error } = await supabase
        .from('glossary_terms')
        .delete()
        .eq('slug', slug)
        .eq('lang', lang);

    if (error) throw error;
}
