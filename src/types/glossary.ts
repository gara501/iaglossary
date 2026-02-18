export interface GlossaryTerm {
    id: string;
    term: string;
    letter: string;
    summary: string;
    definition: string;
    category: string;
    relatedTerms?: string[];
}
