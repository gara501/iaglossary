import { Language } from '../context/LanguageContext'

export interface UIStrings {
    subtitle: string
    searchPlaceholder: string
    allButton: string
    clearFilter: string
    termsCount: (count: number, total: number) => string
    letterFilter: (letter: string) => string
    searchFilter: (query: string) => string
    noTermsFound: string
    noTermsHint: string
    readMore: string
    summaryLabel: string
    definitionLabel: string
    relatedTermsLabel: string
    footerText: string
    logoSubtitle: string
}

const strings: Record<Language, UIStrings> = {
    en: {
        subtitle: 'Explore {count} essential terms in Artificial Intelligence and Generative AI — from fundamentals to cutting-edge concepts.',
        searchPlaceholder: 'Search AI terms...',
        allButton: 'ALL',
        clearFilter: '✕ Clear filter',
        termsCount: (count, total) =>
            count === total ? `${total} terms` : `${count} of ${total} terms`,
        letterFilter: (letter) => ` · Letter "${letter}"`,
        searchFilter: (query) => ` · "${query}"`,
        noTermsFound: 'No terms found',
        noTermsHint: 'Try a different search or letter filter',
        readMore: 'Read more',
        summaryLabel: 'Summary',
        definitionLabel: 'Full Definition',
        relatedTermsLabel: 'Related Terms',
        footerText: 'IA Glossary · AI & Generative AI Reference - Aram',
        logoSubtitle: 'AI & Generative AI Terms',
    },
    es: {
        subtitle: 'Explora {count} términos esenciales de Inteligencia Artificial e IA Generativa — desde fundamentos hasta conceptos de vanguardia.',
        searchPlaceholder: 'Buscar términos de IA...',
        allButton: 'TODOS',
        clearFilter: '✕ Quitar filtro',
        termsCount: (count, total) =>
            count === total ? `${total} términos` : `${count} de ${total} términos`,
        letterFilter: (letter) => ` · Letra "${letter}"`,
        searchFilter: (query) => ` · "${query}"`,
        noTermsFound: 'No se encontraron términos',
        noTermsHint: 'Intenta con otra búsqueda o filtro de letra',
        readMore: 'Leer más',
        summaryLabel: 'Resumen',
        definitionLabel: 'Definición Completa',
        relatedTermsLabel: 'Términos Relacionados',
        footerText: 'IA Glossary · Referencia de IA e IA Generativa - Aram',
        logoSubtitle: 'Términos de IA e IA Generativa',
    },
}

export function useStrings(language: Language): UIStrings {
    return strings[language]
}
