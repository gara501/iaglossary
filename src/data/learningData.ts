import { LearningItem } from '../types/learning';

const learningData: LearningItem[] = [
    {
        id: "agents-course",
        title: "Agents Course",
        creator: "Hugging Face",
        summary: "This free course will take you on a journey, from beginner to expert, in understanding, using and building AI agents.",
        link: "https://huggingface.co/learn/agents-course/unit1/what-are-llms",
        category: "Course"
    },
    {
        id: "llm-course",
        title: "LLM Course",
        creator: "Hugging Face",
        summary: "This course will teach you about large language models (LLMs) and natural language processing (NLP) using libraries from the Hugging Face ecosystem — 🤗 Transformers, 🤗 Datasets, 🤗 Tokenizers, and 🤗 Accelerate — as well as the Hugging Face Hub.",
        link: "https://huggingface.co/learn/llm-course/chapter1/1",
        category: "Course"
    },
    {
        id: "claude-code-action",
        title: "Claude Code in Action",
        creator: "Anthropic",
        summary: "This course provides comprehensive training on using Claude Code for software development tasks, covering the underlying architecture of AI coding assistants, practical implementation techniques, and advanced integration strategies. You'll learn about Claude Code's context management approaches, and how to extend functionality through MCP servers and GitHub integration.",
        link: "https://anthropic.skilljar.com/claude-code-in-action",
        category: "Course"
    },
    {
        id: "contextual-retrieval",
        title: "Contextual Retrieval",
        creator: "Anthropic",
        summary: "This article explores how Claude can retrieve information from external documents to improve the accuracy and relevance of its responses.",
        link: "https://www.anthropic.com/engineering/contextual-retrieval",
        category: "Article"
    },
    {
        id: "qlora",
        title: "QLORA: Efficient Finetuning of Quantized LLMs",
        creator: "Tim Dettmers",
        summary: "QLORA finetuning of quantized LLMs.",
        link: "https://arxiv.org/pdf/2305.14314",
        category: "Study"
    },
    {
        id: "transformer-math",
        title: "Transformer Math",
        creator: "Quentin Anthony, Stella Biderman, Hailey Schoelkopf",
        summary: "We present basic math related to computation and memory usage for transformers",
        link: "https://blog.eleuther.ai/transformer-math/",
        category: "Article"
    },
    {
        id: "agent-skills",
        title: "Agent Skills with Anthropic",
        creator: "DeepLearning.AI",
        summary: "Learn how to build agents that can use tools, access information, and perform complex tasks",
        link: "https://learn.deeplearning.ai/courses/agent-skills-with-anthropic",
        category: "Course"
    },
    {
        id: "generative-ai-with-llms",
        title: "Generative AI with LLMs",
        creator: "DeepLearning.AI",
        summary: "Understand the generative AI lifecycle. Describe transformer architecture powering LLMs. Apply training/tuning/inference methods. Hear from researchers on generative AI challenges/opportunities.",
        link: "https://learn.deeplearning.ai/courses/generative-ai-with-llms/lesson/rs5m7/course-introduction",
        category: "Course"
    },

];

export default learningData;
