import { LearningItem } from '../types/learning';

const learningDataES: LearningItem[] = [
    {
        id: "agents-course",
        title: "Curso de Agentes",
        creator: "Hugging Face",
        summary: "Este curso gratuito te llevará en un viaje, desde principiante hasta experto, para comprender, usar y construir agentes de IA.",
        link: "https://huggingface.co/learn/agents-course/unit1/what-are-llms",
        category: "Curso"
    },
    {
        id: "llm-course",
        title: "Curso de LLM",
        creator: "Hugging Face",
        summary: "Este curso te enseñará sobre modelos de lenguaje grandes (LLM) y procesamiento de lenguaje natural (NLP) utilizando bibliotecas del ecosistema Hugging Face — 🤗 Transformers, 🤗 Datasets, 🤗 Tokenizers y 🤗 Accelerate — así como el Hugging Face Hub.",
        link: "https://huggingface.co/learn/llm-course/chapter1/1",
        category: "Curso"
    },
    {
        id: "claude-code-action",
        title: "Claude Code en Acción",
        creator: "Anthropic",
        summary: "Este curso proporciona capacitación integral sobre el uso de Claude Code para tareas de desarrollo de software, cubriendo la arquitectura subyacente de los asistentes de codificación de IA, técnicas de implementación práctica y estrategias de integración avanzada. Aprenderás sobre los enfoques de gestión de contexto de Claude Code y cómo extender la funcionalidad a través de servidores MCP e integración con GitHub.",
        link: "https://anthropic.skilljar.com/claude-code-in-action",
        category: "Curso"
    },
    {
        id: "contextual-retrieval",
        title: "Recuperación Contextual",
        creator: "Anthropic",
        summary: "Este artículo explora cómo Claude puede recuperar información de documentos externos para mejorar la precisión y relevancia de sus respuestas.",
        link: "https://www.anthropic.com/engineering/contextual-retrieval",
        category: "Artículo"
    },
    {
        id: "qlora",
        title: "QLORA: Efficient Finetuning of Quantized LLMs",
        creator: "Tim Dettmers",
        summary: "Articulo donde se explica a fondo que es QLORA y como funciona el finetuning de LLMs.",
        link: "https://arxiv.org/pdf/2305.14314",
        category: "Estudio"
    },
    {
        id: "transformer-math",
        title: "Transformer Math",
        creator: "Quentin Anthony, Stella Biderman, Hailey Schoelkopf",
        summary: "Artículo donde se explican las matemáticas básicas relacionadas con la computación y el uso de memoria para transformers.",
        link: "https://blog.eleuther.ai/transformer-math/",
        category: "Artículo"
    },

];

export default learningDataES;
