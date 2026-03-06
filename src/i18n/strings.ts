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
    // Simulations
    simulationsNav: string
    simulationsSidebarTitle: string
    tokenizationSimTitle: string
    tokenizationSimSubtitle: string
    ragSimTitle: string
    ragSimSubtitle: string
    attentionSimTitle: string
    attentionSimSubtitle: string
    // How To
    howToNav: string
    howToSidebarTitle: string
    howChatGPTTitle: string
    howChatGPTSubtitle: string
    howClaudeCodeTitle: string
    howClaudeCodeSubtitle: string
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
        // Simulations
        simulationsNav: 'SIMULATIONS',
        simulationsSidebarTitle: 'Simulations',
        tokenizationSimTitle: 'Tokenization',
        tokenizationSimSubtitle: 'How LLMs split text into tokens',
        ragSimTitle: 'RAG Pipeline',
        ragSimSubtitle: 'Retrieval-Augmented Generation flow',
        attentionSimTitle: 'Transformer Attention',
        attentionSimSubtitle: 'How self-attention and multi-head attention work',
        // How To
        howToNav: 'HOW TO',
        howToSidebarTitle: 'How To',
        howChatGPTTitle: 'How ChatGPT Works',
        howChatGPTSubtitle: 'Architecture flow from prompt to response',
        howClaudeCodeTitle: 'How Claude Code Works',
        howClaudeCodeSubtitle: 'Agentic CLI + cloud model loop',
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
        // Simulations
        simulationsNav: 'SIMULACIONES',
        simulationsSidebarTitle: 'Simulaciones',
        tokenizationSimTitle: 'Tokenización',
        tokenizationSimSubtitle: 'Cómo los LLMs dividen el texto en tokens',
        ragSimTitle: 'Pipeline RAG',
        ragSimSubtitle: 'Flujo de Retrieval-Augmented Generation',
        attentionSimTitle: 'Atención Transformer',
        attentionSimSubtitle: 'Cómo funciona la auto-atención y la atención multi-cabeza',
        // How To
        howToNav: 'CÓMO FUNCIONA',
        howToSidebarTitle: 'Cómo Funciona',
        howChatGPTTitle: 'Cómo funciona ChatGPT',
        howChatGPTSubtitle: 'Arquitectura desde el prompt hasta la respuesta',
        howClaudeCodeTitle: 'Cómo funciona Claude Code',
        howClaudeCodeSubtitle: 'Bucle agéntico CLI + modelo en la nube',
    },
}

export function useStrings(language: Language): UIStrings {
    return strings[language]
}
