-- ============================================
-- Migration: Create glossary_terms and learning_items
-- Generated from local TypeScript data files
-- ============================================

-- 1. Create glossary_terms table
CREATE TABLE IF NOT EXISTS glossary_terms (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    slug TEXT NOT NULL,
    term TEXT NOT NULL,
    letter TEXT NOT NULL,
    summary TEXT NOT NULL,
    definition TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    related_terms TEXT[] DEFAULT '{}',
    lang TEXT NOT NULL CHECK (lang IN ('en', 'es')),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(slug, lang)
);

-- 2. Create learning_items table
CREATE TABLE IF NOT EXISTS learning_items (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    creator TEXT NOT NULL,
    summary TEXT NOT NULL,
    link TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    lang TEXT NOT NULL CHECK (lang IN ('en', 'es')),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(slug, lang)
);

-- 3. Enable Row Level Security
ALTER TABLE glossary_terms ENABLE ROW LEVEL SECURITY;
ALTER TABLE learning_items ENABLE ROW LEVEL SECURITY;

-- 4. RLS Policies: public read, authenticated write
CREATE POLICY "Public read glossary" ON glossary_terms
    FOR SELECT USING (true);

CREATE POLICY "Authenticated insert glossary" ON glossary_terms
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Authenticated update glossary" ON glossary_terms
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Authenticated delete glossary" ON glossary_terms
    FOR DELETE TO authenticated USING (true);

CREATE POLICY "Public read learning" ON learning_items
    FOR SELECT USING (true);

CREATE POLICY "Authenticated insert learning" ON learning_items
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Authenticated update learning" ON learning_items
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Authenticated delete learning" ON learning_items
    FOR DELETE TO authenticated USING (true);

-- 5. Auto-update updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER glossary_terms_updated_at
    BEFORE UPDATE ON glossary_terms
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER learning_items_updated_at
    BEFORE UPDATE ON learning_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================
-- SEED DATA
-- ============================================

-- Glossary terms (en)
INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'attention-mechanism',
    'Attention Mechanism',
    'A',
    'A technique that allows models to focus on relevant parts of the input when producing output.',
    'The attention mechanism is a fundamental component in modern neural networks, particularly in Transformer architectures. It allows a model to dynamically focus on different parts of the input sequence when generating each element of the output. Instead of compressing all input information into a fixed-size vector, attention computes a weighted sum of input representations, where the weights reflect the relevance of each input element to the current output step. This enables models to handle long-range dependencies and capture complex relationships in data.',
    'Architecture',
    '{"Transformer","Self-Attention","Multi-Head Attention"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'autoregressive-model',
    'Autoregressive Model',
    'A',
    'A model that generates output sequentially, each token conditioned on previous tokens.',
    'An autoregressive model generates sequences by predicting one element at a time, conditioning each prediction on all previously generated elements. In language models like GPT, this means generating text token by token from left to right. The model learns the joint probability of a sequence by decomposing it into a product of conditional probabilities. This approach is powerful for generation tasks but can be slow at inference time since tokens must be produced sequentially.',
    'Model Type',
    '{"GPT","Language Model","Token"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'bert',
    'BERT',
    'B',
    'Bidirectional Encoder Representations from Transformers — a pre-trained language model by Google.',
    'BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model developed by Google in 2018. Unlike GPT, BERT uses a bidirectional approach, meaning it considers both left and right context simultaneously when processing text. It is pre-trained using two tasks: Masked Language Modeling (MLM), where random tokens are masked and the model predicts them, and Next Sentence Prediction (NSP). BERT set new state-of-the-art results on numerous NLP benchmarks and popularized the fine-tuning paradigm for NLP tasks.',
    'Model',
    '{"Transformer","Fine-tuning","Pre-training","GPT"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'chain-of-thought',
    'Chain-of-Thought Prompting',
    'C',
    'A prompting technique that encourages LLMs to reason step-by-step before answering.',
    'Chain-of-Thought (CoT) prompting is a technique where the model is guided to produce intermediate reasoning steps before arriving at a final answer. By including examples that show step-by-step reasoning in the prompt, or by simply instructing the model to ''think step by step'', CoT significantly improves performance on complex reasoning tasks such as math problems, logical puzzles, and multi-step question answering. It was introduced by Google researchers and has become a standard technique for eliciting better reasoning from large language models.',
    'Prompting',
    '{"Prompt Engineering","Few-Shot Learning","Reasoning"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'context-window',
    'Context Window',
    'C',
    'The maximum amount of text (tokens) a language model can process at once.',
    'The context window refers to the maximum number of tokens that a language model can consider at one time during inference. Tokens within the context window can attend to each other via the attention mechanism. Early models like GPT-2 had context windows of 1,024 tokens, while modern models like GPT-4 Turbo support up to 128,000 tokens. A larger context window allows the model to process longer documents, maintain longer conversations, and perform tasks requiring extensive context, but it also increases computational cost.',
    'Architecture',
    '{"Token","Attention Mechanism","Transformer"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'diffusion-model',
    'Diffusion Model',
    'D',
    'A generative model that learns to reverse a noise-adding process to generate data.',
    'Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process. During training, data (e.g., images) is progressively corrupted with Gaussian noise over many steps. The model learns to reverse this process, starting from pure noise and iteratively denoising to produce realistic samples. Stable Diffusion, DALL-E 2, and Midjourney are prominent examples. Diffusion models have achieved state-of-the-art results in image generation, audio synthesis, and video generation.',
    'Generative AI',
    '{"Stable Diffusion","DALL-E","Generative Model","Latent Space"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'embedding',
    'Embedding',
    'E',
    'A dense vector representation of data (words, sentences, images) in a continuous space.',
    'An embedding is a learned representation of data as a dense vector in a continuous, high-dimensional space. Words, sentences, images, or other entities are mapped to vectors such that semantically similar items are close together in the embedding space. Word embeddings like Word2Vec and GloVe were early examples; modern models produce contextual embeddings where the same word has different representations depending on context. Embeddings are foundational to most modern AI systems, enabling efficient similarity search, clustering, and downstream task performance.',
    'Representation',
    '{"Vector Database","Semantic Search","Word2Vec","Transformer"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'fine-tuning',
    'Fine-Tuning',
    'F',
    'Adapting a pre-trained model to a specific task by training on task-specific data.',
    'Fine-tuning is the process of taking a pre-trained model and continuing to train it on a smaller, task-specific dataset. This allows the model to adapt its general knowledge to a particular domain or task while retaining the broad capabilities learned during pre-training. Fine-tuning can be full (updating all parameters) or parameter-efficient (e.g., LoRA, adapters). It is a cornerstone of modern NLP and computer vision, enabling high performance on specialized tasks without training from scratch.',
    'Training',
    '{"Pre-training","Transfer Learning","LoRA","RLHF"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'foundation-model',
    'Foundation Model',
    'F',
    'A large model trained on broad data that can be adapted to many downstream tasks.',
    'A foundation model is a large AI model trained on vast amounts of diverse data using self-supervised learning. The term was coined by Stanford researchers in 2021. Foundation models serve as a base that can be fine-tuned or prompted for a wide range of downstream tasks. Examples include GPT-4, PaLM, LLaMA, CLIP, and Stable Diffusion. Their scale and generality make them powerful starting points, but also raise concerns about bias, safety, and the concentration of AI capabilities.',
    'Model Type',
    '{"Pre-training","Fine-tuning","Large Language Model","Transfer Learning"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'gpt',
    'GPT',
    'G',
    'Generative Pre-trained Transformer — OpenAI''s family of large autoregressive language models.',
    'GPT (Generative Pre-trained Transformer) is a family of large language models developed by OpenAI. GPT models are trained using unsupervised pre-training on massive text corpora, followed by fine-tuning for specific tasks. They use a decoder-only Transformer architecture and generate text autoregressively. GPT-3 (175B parameters) demonstrated remarkable few-shot learning capabilities. GPT-4 is a multimodal model capable of processing both text and images. The GPT series has been foundational in demonstrating the power of scaling language models.',
    'Model',
    '{"Transformer","Autoregressive Model","OpenAI","Large Language Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'generative-adversarial-network',
    'Generative Adversarial Network',
    'G',
    'A framework where two neural networks compete: a generator and a discriminator.',
    'A Generative Adversarial Network (GAN) consists of two neural networks trained simultaneously in a competitive framework. The generator network creates synthetic data samples, while the discriminator network tries to distinguish real data from generated data. Through this adversarial process, the generator learns to produce increasingly realistic outputs. GANs were introduced by Ian Goodfellow in 2014 and have been used for image synthesis, style transfer, data augmentation, and more. Variants include DCGAN, StyleGAN, and CycleGAN.',
    'Generative AI',
    '{"Diffusion Model","Generative Model","Latent Space"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'hallucination',
    'Hallucination',
    'H',
    'When an AI model generates plausible-sounding but factually incorrect or fabricated information.',
    'Hallucination in AI refers to the phenomenon where a language model generates content that is confidently stated but factually incorrect, nonsensical, or entirely fabricated. This occurs because LLMs are trained to produce statistically likely text rather than verified facts. Hallucinations can range from subtle errors (wrong dates, names) to completely invented citations or events. Mitigating hallucinations is a major research challenge, with approaches including Retrieval-Augmented Generation (RAG), better training data, and improved alignment techniques.',
    'Safety & Alignment',
    '{"Retrieval-Augmented Generation","Alignment","Large Language Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'inference',
    'Inference',
    'I',
    'The process of using a trained model to generate predictions or outputs on new data.',
    'Inference is the phase where a trained AI model is used to make predictions or generate outputs on new, unseen data. Unlike training, inference does not update model weights. It involves a forward pass through the network. For large language models, inference can be computationally expensive due to the size of the models and the autoregressive nature of text generation. Techniques like quantization, batching, speculative decoding, and hardware acceleration (GPUs, TPUs) are used to make inference faster and more cost-effective.',
    'Deployment',
    '{"Training","Quantization","Autoregressive Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'instruction-tuning',
    'Instruction Tuning',
    'I',
    'Fine-tuning a language model on instruction-response pairs to improve instruction-following.',
    'Instruction tuning is a fine-tuning technique where a pre-trained language model is trained on a dataset of (instruction, response) pairs. This teaches the model to follow natural language instructions more reliably. Models like InstructGPT, FLAN, and Alpaca use instruction tuning. It significantly improves the model''s ability to generalize to new tasks described in natural language, without requiring task-specific examples. Instruction tuning is often combined with RLHF to produce helpful, harmless, and honest AI assistants.',
    'Training',
    '{"Fine-tuning","RLHF","Prompt Engineering","Alignment"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'latent-space',
    'Latent Space',
    'L',
    'A compressed, abstract representation space learned by a model to encode data.',
    'Latent space is the multi-dimensional space in which a model represents compressed, abstract features of data. In autoencoders, the encoder maps input data to a point in latent space, and the decoder reconstructs the data from that point. In diffusion models and VAEs, the latent space captures the underlying structure of the data distribution. Navigating and manipulating latent space allows for controlled generation, interpolation between data points, and style transfer. Understanding latent space is key to understanding how generative models work.',
    'Architecture',
    '{"Embedding","Diffusion Model","Variational Autoencoder","Generative Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'large-language-model',
    'Large Language Model',
    'L',
    'A neural network with billions of parameters trained on massive text corpora.',
    'A Large Language Model (LLM) is a type of neural network with billions to trillions of parameters, trained on vast amounts of text data. LLMs learn statistical patterns in language and can perform a wide range of tasks including text generation, translation, summarization, question answering, and code generation. Notable LLMs include GPT-4, Claude, Gemini, LLaMA, and Mistral. The ''large'' refers to both the number of parameters and the scale of training data. LLMs exhibit emergent capabilities that were not explicitly trained for.',
    'Model Type',
    '{"GPT","Transformer","Foundation Model","Emergent Capabilities"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'lora',
    'LoRA',
    'L',
    'Low-Rank Adaptation — an efficient fine-tuning method that trains only small adapter matrices.',
    'LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable rank-decomposition matrices into each layer of the Transformer architecture. Instead of updating all model parameters, LoRA trains a much smaller number of parameters (often <1% of the original), making fine-tuning feasible on consumer hardware. LoRA has become extremely popular for customizing large language models and image generation models like Stable Diffusion, enabling domain-specific adaptation without full fine-tuning costs.',
    'Training',
    '{"Fine-tuning","Parameter-Efficient Fine-Tuning","Transformer","Stable Diffusion"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'multimodal',
    'Multimodal AI',
    'M',
    'AI systems that can process and generate multiple types of data (text, images, audio, video).',
    'Multimodal AI refers to systems capable of understanding and generating multiple modalities of data, such as text, images, audio, and video. Unlike unimodal models that handle only one data type, multimodal models can reason across modalities. Examples include GPT-4V (text + images), Gemini (text, images, audio, video), CLIP (text + images), and Flamingo. Multimodal capabilities enable applications like visual question answering, image captioning, audio transcription, and video understanding. This is considered a key step toward more general AI systems.',
    'Model Type',
    '{"GPT","CLIP","Vision-Language Model","Foundation Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'neural-network',
    'Neural Network',
    'N',
    'A computational model inspired by the brain, composed of interconnected layers of nodes.',
    'A neural network is a machine learning model inspired by the structure of biological neural networks in the brain. It consists of layers of interconnected nodes (neurons), where each connection has a learnable weight. Data flows through the network (forward pass), and the model learns by adjusting weights to minimize a loss function via backpropagation. Deep neural networks with many layers are the foundation of modern AI, enabling breakthroughs in image recognition, natural language processing, speech recognition, and generative AI.',
    'Fundamentals',
    '{"Deep Learning","Backpropagation","Transformer","Convolutional Neural Network"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'prompt-engineering',
    'Prompt Engineering',
    'P',
    'The practice of crafting inputs to guide AI models toward desired outputs.',
    'Prompt engineering is the discipline of designing and optimizing input prompts to elicit desired behaviors from AI language models. Since LLMs are sensitive to how instructions are phrased, prompt engineering can significantly affect output quality. Techniques include zero-shot prompting, few-shot prompting, chain-of-thought prompting, role prompting, and structured output prompting. Prompt engineering has emerged as a critical skill for effectively using LLMs in applications, and has spawned research into automatic prompt optimization and prompt injection attacks.',
    'Prompting',
    '{"Chain-of-Thought Prompting","Few-Shot Learning","Large Language Model","Zero-Shot Learning"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'pre-training',
    'Pre-training',
    'P',
    'Training a model on large-scale data before fine-tuning on specific tasks.',
    'Pre-training is the initial phase of training a foundation model on a large, diverse dataset using self-supervised objectives. For language models, this typically involves predicting masked tokens (BERT) or next tokens (GPT). For vision models, it may involve contrastive learning or masked image modeling. Pre-training allows the model to learn general representations of language, vision, or other modalities. The pre-trained model is then adapted for specific tasks via fine-tuning or prompting, dramatically reducing the data and compute needed for downstream tasks.',
    'Training',
    '{"Fine-tuning","Foundation Model","Self-Supervised Learning","Transfer Learning"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'quantization',
    'Quantization',
    'Q',
    'Reducing model size by representing weights with lower-precision numbers.',
    'Quantization is a model compression technique that reduces the numerical precision of model weights and activations from high-precision formats (e.g., float32) to lower-precision formats (e.g., int8, int4). This reduces memory requirements and speeds up inference with minimal loss in model quality. Techniques include post-training quantization (PTQ) and quantization-aware training (QAT). Quantization has been crucial for deploying large language models on consumer hardware, with tools like GPTQ, AWQ, and llama.cpp enabling running LLMs on laptops.',
    'Deployment',
    '{"Inference","Model Compression","LoRA","Large Language Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'rag',
    'Retrieval-Augmented Generation',
    'R',
    'Combining a retrieval system with a generative model to ground responses in external knowledge.',
    'Retrieval-Augmented Generation (RAG) is an architecture that enhances language model responses by retrieving relevant documents from an external knowledge base and incorporating them into the prompt. The process involves: (1) encoding the query as an embedding, (2) searching a vector database for relevant documents, (3) including retrieved documents in the prompt context, and (4) generating a response grounded in the retrieved information. RAG reduces hallucinations, enables access to up-to-date information, and allows LLMs to reason over private or domain-specific knowledge.',
    'Architecture',
    '{"Vector Database","Embedding","Hallucination","Large Language Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'rlhf',
    'RLHF',
    'R',
    'Reinforcement Learning from Human Feedback — aligning AI models using human preference data.',
    'Reinforcement Learning from Human Feedback (RLHF) is a training technique used to align language models with human values and preferences. The process involves: (1) supervised fine-tuning on demonstration data, (2) training a reward model on human preference comparisons, and (3) optimizing the language model using the reward model via reinforcement learning (typically PPO). RLHF was used to train InstructGPT and ChatGPT, dramatically improving their helpfulness and safety. It is a key technique in AI alignment research.',
    'Training',
    '{"Alignment","Fine-tuning","Instruction Tuning","Reward Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'self-attention',
    'Self-Attention',
    'S',
    'A mechanism where each element in a sequence attends to all other elements in the same sequence.',
    'Self-attention (also called intra-attention) is a mechanism where each position in a sequence computes attention weights over all other positions in the same sequence. For each position, queries, keys, and values are computed from the input, and the output is a weighted sum of values where weights are determined by query-key compatibility. Self-attention enables the model to capture long-range dependencies and relationships between any two positions regardless of distance. It is the core operation in Transformer architectures and scales quadratically with sequence length.',
    'Architecture',
    '{"Attention Mechanism","Transformer","Multi-Head Attention","Context Window"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'stable-diffusion',
    'Stable Diffusion',
    'S',
    'An open-source latent diffusion model for high-quality text-to-image generation.',
    'Stable Diffusion is an open-source latent diffusion model developed by Stability AI, released in 2022. It generates high-quality images from text descriptions by performing the diffusion process in a compressed latent space rather than pixel space, making it computationally efficient. It uses a CLIP text encoder to condition generation on text prompts and a U-Net denoising network. Stable Diffusion can run on consumer GPUs and has spawned a large ecosystem of fine-tuned models, LoRA adaptations, and tools like AUTOMATIC1111 and ComfyUI.',
    'Generative AI',
    '{"Diffusion Model","Latent Space","LoRA","DALL-E"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'token',
    'Token',
    'T',
    'The basic unit of text that language models process — roughly a word or word fragment.',
    'A token is the fundamental unit of text that language models process. Tokenization splits text into tokens using algorithms like Byte-Pair Encoding (BPE) or WordPiece. A token is typically a word, subword, or character, depending on the tokenizer. For example, ''tokenization'' might be split into [''token'', ''ization'']. The number of tokens in a text affects processing cost and context window usage. On average, 1 token ≈ 4 characters or 0.75 words in English. API pricing for LLMs is typically based on token count.',
    'Fundamentals',
    '{"Context Window","Tokenization","Large Language Model","Embedding"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'transformer',
    'Transformer',
    'T',
    'The dominant neural network architecture for NLP, based entirely on attention mechanisms.',
    'The Transformer is a neural network architecture introduced in the 2017 paper ''Attention Is All You Need'' by Vaswani et al. at Google. It replaced recurrent networks (RNNs, LSTMs) with self-attention mechanisms, enabling parallel processing of sequences and better capture of long-range dependencies. The architecture consists of encoder and decoder stacks, each containing multi-head self-attention and feed-forward layers with residual connections and layer normalization. Transformers are the foundation of virtually all modern large language models, vision models, and multimodal models.',
    'Architecture',
    '{"Self-Attention","Attention Mechanism","BERT","GPT","Large Language Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'transfer-learning',
    'Transfer Learning',
    'T',
    'Applying knowledge learned from one task or domain to improve performance on another.',
    'Transfer learning is a machine learning paradigm where a model trained on one task is adapted for a different but related task. In deep learning, this typically involves using pre-trained model weights as initialization for a new task. The pre-trained model has learned useful feature representations that transfer across tasks. Transfer learning dramatically reduces the data and compute needed for new tasks. It is the foundation of the modern AI development workflow: pre-train on large data, fine-tune on task-specific data.',
    'Training',
    '{"Pre-training","Fine-tuning","Foundation Model","Domain Adaptation"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'vector-database',
    'Vector Database',
    'V',
    'A database optimized for storing and searching high-dimensional embedding vectors.',
    'A vector database is a specialized database designed to store, index, and efficiently search high-dimensional embedding vectors. Unlike traditional databases that search by exact match, vector databases use approximate nearest neighbor (ANN) algorithms (e.g., HNSW, IVF) to find vectors most similar to a query vector. They are essential infrastructure for RAG systems, semantic search, recommendation engines, and similarity-based applications. Popular vector databases include Pinecone, Weaviate, Qdrant, Chroma, and pgvector (PostgreSQL extension).',
    'Infrastructure',
    '{"Embedding","Retrieval-Augmented Generation","Semantic Search"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'zero-shot-learning',
    'Zero-Shot Learning',
    'Z',
    'A model''s ability to perform tasks it has never explicitly been trained on.',
    'Zero-shot learning refers to a model''s ability to perform a task without having seen any examples of that task during training or in the prompt. Large language models exhibit zero-shot capabilities because their pre-training on diverse data gives them broad knowledge and reasoning abilities. For example, GPT-4 can translate text to a language it was never explicitly trained to translate, or solve novel logic puzzles. Zero-shot performance is a key measure of a model''s generalization ability and is contrasted with few-shot learning, where a small number of examples are provided.',
    'Learning Paradigm',
    '{"Few-Shot Learning","Prompt Engineering","Large Language Model","Generalization"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'agent',
    'Agent',
    'A',
    'An AI system that perceives its environment and takes actions to achieve goals.',
    'In AI, an agent is any software or program that interacts with the world (or a simulation) by receiving inputs and producing outputs or actions. Agents operate within an environment, perceive its state through sensors or data inputs, and act upon it through actuators or API calls. They range from simple rule-based systems (like a thermostat) to complex autonomous agents powered by large language models that can plan, use tools, browse the web, write code, and execute multi-step tasks. Agentic AI systems are increasingly used in automation, robotics, and AI assistants.',
    'Architecture',
    '{"Reinforcement Learning","Large Language Model","Tool Use","Autonomous AI"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'generative-ai',
    'Generative AI',
    'G',
    'AI that creates new content such as text, images, audio, or video.',
    'Generative AI refers to a class of AI models designed to produce original data that resembles the examples they were trained on. These systems learn statistical patterns from large datasets and use those patterns to generate new content — including natural-sounding text, realistic images, music, video, code, and 3D models. Key architectures include Transformers (for text), Diffusion Models (for images), and GANs. Prominent examples are GPT-4 (text), DALL-E and Stable Diffusion (images), Sora (video), and MusicLM (audio). Generative AI has transformed creative industries, software development, and scientific research.',
    'Generative AI',
    '{"Large Language Model","Diffusion Model","Generative Adversarial Network","Foundation Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'context',
    'Context',
    'C',
    'Information surrounding an input that helps a model interpret meaning accurately.',
    'In AI, context refers to any relevant background information that helps a model understand or respond appropriately. For language models, context includes the conversation history, system instructions, user intent, topic, tone, and any documents provided in the prompt. The amount of context a model can use is bounded by its context window. Effective use of context is critical for accurate, relevant responses — a model with insufficient context may misinterpret ambiguous queries. Context engineering — the practice of structuring context inputs optimally — has become a key skill in building AI applications.',
    'Fundamentals',
    '{"Context Window","Prompt Engineering","Context Engineering","Retrieval-Augmented Generation"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'claude',
    'Claude',
    'C',
    'A family of AI assistant models developed by Anthropic, known for safety and helpfulness.',
    'Claude is a family of large language models developed by Anthropic, an AI safety company. Claude models are designed with a strong emphasis on being helpful, harmless, and honest (the ''HHH'' framework). They are trained using Constitutional AI (CAI) and RLHF techniques to align model behavior with human values. Claude excels at tasks including summarization, reasoning, coding assistance, analysis, and creative writing. The Claude model family includes multiple tiers (Haiku, Sonnet, Opus) optimized for different speed/capability trade-offs. Claude is widely used via API and in Anthropic''s consumer products.',
    'Model',
    '{"Large Language Model","RLHF","Alignment","OpenAI","GPT"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'openai',
    'OpenAI',
    'O',
    'An AI research organization that develops advanced AI systems including the GPT series.',
    'OpenAI is an AI research laboratory and technology company founded in 2015, with a mission to ensure that artificial general intelligence (AGI) benefits all of humanity. It is responsible for developing some of the most influential AI systems, including the GPT series of language models, DALL-E image generation models, Codex (code generation), Whisper (speech recognition), and the Sora video generation model. OpenAI also created ChatGPT, one of the most widely used AI applications in history. The organization operates as a capped-profit company with a non-profit parent, balancing commercial operations with safety research.',
    'Organization',
    '{"GPT","ChatGPT","DALL-E","Large Language Model","AGI"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'chatgpt',
    'ChatGPT',
    'C',
    'A conversational AI assistant by OpenAI, built on GPT large language models.',
    'ChatGPT is a conversational AI application developed by OpenAI, launched in November 2022. It is built on top of the GPT series of large language models (initially GPT-3.5, later GPT-4) and fine-tuned with RLHF to be a helpful, conversational assistant. ChatGPT can engage in multi-turn dialogue, answer questions, write and debug code, draft documents, summarize text, translate languages, and perform many other language tasks. It became one of the fastest-growing consumer applications in history, reaching 100 million users in two months. It supports plugins, image input (GPT-4V), and custom GPTs.',
    'Model',
    '{"GPT","OpenAI","RLHF","Large Language Model","Instruction Tuning"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'deep-learning',
    'Deep Learning',
    'D',
    'A subset of machine learning using neural networks with many layers to learn from data.',
    'Deep learning is a subfield of machine learning that uses artificial neural networks with many layers (hence ''deep'') to automatically learn hierarchical representations from raw data. Each layer learns increasingly abstract features — for example, in image recognition, early layers detect edges, middle layers detect shapes, and later layers detect objects. Deep learning has driven breakthroughs in computer vision (CNNs), natural language processing (Transformers), speech recognition (RNNs, attention), and generative AI (GANs, diffusion models). It requires large datasets and significant compute, typically using GPUs or TPUs.',
    'Fundamentals',
    '{"Neural Network","Transformer","Convolutional Neural Network","Backpropagation"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'reinforcement-learning',
    'Reinforcement Learning',
    'R',
    'A learning paradigm where an agent learns by receiving rewards or penalties for its actions.',
    'Reinforcement learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent observes the current state, takes an action, receives a reward signal (positive or negative), and updates its policy to maximize cumulative reward over time. Key algorithms include Q-learning, SARSA, and policy gradient methods like PPO and A3C. RL has achieved superhuman performance in games (AlphaGo, Atari), robotics, and autonomous driving. In the context of LLMs, RLHF uses RL to align model outputs with human preferences.',
    'Learning Paradigm',
    '{"RLHF","Agent","Deep Learning","Policy Gradient"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'computer-vision',
    'Computer Vision',
    'C',
    'AI that enables machines to interpret and understand visual information from images and videos.',
    'Computer vision is a field of AI focused on enabling machines to extract meaningful information from visual inputs such as images and videos. Core tasks include image classification (identifying what is in an image), object detection (locating objects), semantic segmentation (labeling each pixel), and image generation. Modern computer vision relies heavily on convolutional neural networks (CNNs) and Vision Transformers (ViT). Applications span autonomous vehicles, medical imaging, facial recognition, augmented reality, quality control in manufacturing, and satellite imagery analysis.',
    'Field',
    '{"Neural Network","Convolutional Neural Network","Deep Learning","Multimodal AI"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'nlp',
    'NLP',
    'N',
    'Natural Language Processing — AI that enables computers to understand and generate human language.',
    'Natural Language Processing (NLP) is a branch of AI that combines linguistics, computer science, and machine learning to enable computers to process, understand, and generate human language. NLP encompasses a wide range of tasks: text classification, sentiment analysis, machine translation, question answering, summarization, named entity recognition, and dialogue systems. Modern NLP is dominated by Transformer-based models like BERT and GPT. NLP powers applications including search engines, virtual assistants, chatbots, grammar checkers, and content moderation systems.',
    'Field',
    '{"Natural Language Understanding","Natural Language Generation","Transformer","BERT","Large Language Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'supervised-learning',
    'Supervised Learning',
    'S',
    'Machine learning where models are trained on labeled input-output pairs.',
    'Supervised learning is a machine learning paradigm where a model is trained on a dataset of labeled examples — pairs of inputs and their correct outputs. The model learns a mapping from inputs to outputs by minimizing the difference between its predictions and the true labels. Common tasks include classification (predicting a category) and regression (predicting a continuous value). Examples include spam detection, image classification, and price prediction. Supervised learning requires labeled data, which can be expensive to obtain, but it is the most widely used ML paradigm in production systems.',
    'Learning Paradigm',
    '{"Unsupervised Learning","Fine-tuning","Neural Network","Transfer Learning"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'unsupervised-learning',
    'Unsupervised Learning',
    'U',
    'Machine learning on unlabeled data to discover hidden patterns or structure.',
    'Unsupervised learning is a machine learning paradigm where models learn patterns and structure from data without labeled outputs. The model must discover the underlying organization of the data on its own. Common techniques include clustering (grouping similar data points, e.g., K-means), dimensionality reduction (e.g., PCA, t-SNE, autoencoders), and density estimation. Unsupervised learning is valuable when labeled data is scarce or expensive. Self-supervised learning — where models generate their own labels from unlabeled data — is a powerful variant used to pre-train large language models.',
    'Learning Paradigm',
    '{"Self-Supervised Learning","Supervised Learning","Clustering","Pre-training"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'data-mining',
    'Data Mining',
    'D',
    'The process of discovering useful patterns and insights from large datasets.',
    'Data mining is the process of applying statistical, mathematical, and computational techniques to extract meaningful patterns, correlations, and insights from large collections of data. It sits at the intersection of statistics, machine learning, and database systems. Common data mining tasks include classification, clustering, association rule learning (e.g., market basket analysis), anomaly detection, and regression. Data mining is foundational to business intelligence, fraud detection, scientific discovery, and recommendation systems. Modern data mining increasingly leverages machine learning and AI techniques.',
    'Data Science',
    '{"Machine Learning","Supervised Learning","Unsupervised Learning","Pattern Recognition"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'entity-annotation',
    'Entity Annotation',
    'E',
    'Labeling meaningful entities (names, places, dates) in text or data for AI training.',
    'Entity annotation is the process of marking up entities — such as person names, organizations, locations, dates, and product names — in text datasets so that AI models can learn to recognize them. It is a critical step in creating training data for Named Entity Recognition (NER) systems and other NLP tasks. Annotation can be done manually by human annotators or semi-automatically using pre-trained models. High-quality entity annotation is essential for training accurate information extraction systems used in search, knowledge graphs, and document processing pipelines.',
    'NLP',
    '{"Entity Extraction","Named Entity Recognition","NLP","Supervised Learning"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'entity-extraction',
    'Entity Extraction',
    'E',
    'Automatically identifying and categorizing key entities from unstructured text.',
    'Entity extraction, also known as Named Entity Recognition (NER), is an NLP task where an AI model automatically identifies and classifies named entities in unstructured text into predefined categories such as people, organizations, locations, dates, monetary values, and more. For example, in the sentence ''Apple was founded by Steve Jobs in Cupertino in 1976,'' a NER model would extract Apple (organization), Steve Jobs (person), Cupertino (location), and 1976 (date). Entity extraction is fundamental to information retrieval, knowledge graph construction, and document intelligence.',
    'NLP',
    '{"Entity Annotation","NLP","Natural Language Understanding","Information Extraction"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'intent',
    'Intent',
    'I',
    'The goal or purpose behind a user''s input in a conversational AI system.',
    'In conversational AI and NLP, intent refers to the underlying goal or purpose that a user aims to achieve with their input. For example, the query ''What''s the weather like tomorrow?'' has the intent ''get weather forecast.'' Intent recognition (or intent classification) is the task of automatically identifying the user''s intent from their utterance. It is a core component of dialogue systems, virtual assistants, and chatbots. Modern systems use machine learning classifiers or large language models to detect intent, enabling appropriate routing and response generation.',
    'NLP',
    '{"Natural Language Understanding","NLP","Dialogue System","Entity Extraction"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'model',
    'Model',
    'M',
    'A mathematical system trained on data to make predictions, classifications, or generate outputs.',
    'In AI and machine learning, a model is a computational system that has learned patterns from training data and can apply that knowledge to new inputs. Models are defined by their architecture (the structure of the computation) and their parameters (the learned weights). After training, a model can make predictions (regression, classification), generate content (language, images), or take actions (agents). The term encompasses everything from simple linear regression models to billion-parameter neural networks. Model selection, training, evaluation, and deployment are the core stages of the machine learning lifecycle.',
    'Fundamentals',
    '{"Neural Network","Training","Inference","Foundation Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'nlu',
    'Natural Language Understanding',
    'N',
    'AI that interprets the meaning, intent, and context of human language.',
    'Natural Language Understanding (NLU) is a subfield of NLP focused on enabling machines to comprehend the meaning, intent, sentiment, and context of human language — going beyond surface-level text processing. NLU tasks include intent recognition, sentiment analysis, semantic role labeling, coreference resolution, and reading comprehension. NLU is the ''understanding'' component of conversational AI systems, enabling them to correctly interpret what users mean rather than just what they say. Modern NLU is powered by large pre-trained language models like BERT and its variants.',
    'NLP',
    '{"NLP","Natural Language Generation","Intent","BERT","Sentiment Analysis"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'nlg',
    'Natural Language Generation',
    'N',
    'AI that produces coherent, human-like text or speech from data or other inputs.',
    'Natural Language Generation (NLG) is a subfield of NLP focused on automatically producing coherent, fluent, and contextually appropriate text or speech from structured data, knowledge, or other inputs. NLG tasks include text summarization, report generation, dialogue response generation, machine translation, and creative writing. Modern NLG is dominated by autoregressive language models like GPT-4, which generate text token by token. NLG is the ''generation'' component of conversational AI and is used in chatbots, automated journalism, data-to-text systems, and virtual assistants.',
    'NLP',
    '{"NLP","Natural Language Understanding","Large Language Model","Autoregressive Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'overfitting',
    'Overfitting',
    'O',
    'When a model memorizes training data too closely and fails to generalize to new data.',
    'Overfitting occurs when a machine learning model learns the training data too precisely — including its noise and random fluctuations — rather than the underlying general patterns. An overfitted model performs very well on training data but poorly on unseen test data, because it has essentially memorized the training examples rather than learning transferable patterns. Overfitting is more likely with complex models and small datasets. Common mitigation strategies include regularization (L1/L2), dropout, early stopping, data augmentation, and cross-validation.',
    'Fundamentals',
    '{"Supervised Learning","Regularization","Generalization","Neural Network"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'pattern-recognition',
    'Pattern Recognition',
    'P',
    'The ability of algorithms to identify recurring structures or regularities in data.',
    'Pattern recognition is the ability of algorithms and AI systems to detect, classify, and respond to recurring structures, regularities, or relationships in data. It is one of the foundational tasks of AI and machine learning, underlying applications such as image recognition (detecting faces or objects), speech recognition (identifying phonemes and words), handwriting recognition, anomaly detection, and biometric identification. Modern pattern recognition is largely achieved through deep learning, where neural networks automatically learn hierarchical feature representations from raw data.',
    'Fundamentals',
    '{"Deep Learning","Computer Vision","Neural Network","Classification"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'context-engineering',
    'Context Engineering',
    'C',
    'Structuring and optimizing the information provided to AI models to improve output quality.',
    'Context engineering is the practice of deliberately designing and structuring the information provided to an AI model — including system prompts, conversation history, retrieved documents, examples, and environmental data — to maximize the quality and relevance of its outputs. It goes beyond basic prompt engineering to encompass the full information architecture around a model call: what to include, how to format it, what to retrieve, and how to prioritize. As AI systems become more capable, context engineering has emerged as a critical discipline for building reliable, accurate AI applications.',
    'Prompting',
    '{"Prompt Engineering","Retrieval-Augmented Generation","Context Window","Context"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'turing-test',
    'Turing Test',
    'T',
    'A test proposed by Alan Turing to assess whether a machine''s behavior is indistinguishable from a human''s.',
    'The Turing Test, proposed by mathematician Alan Turing in his 1950 paper ''Computing Machinery and Intelligence,'' is a test of a machine''s ability to exhibit intelligent behavior indistinguishable from that of a human. In the original formulation (the Imitation Game), a human evaluator converses via text with both a human and a machine without knowing which is which; if the evaluator cannot reliably distinguish the machine from the human, the machine is said to have passed the test. While influential as a philosophical benchmark, the Turing Test is now considered insufficient as a measure of true AI intelligence, as modern LLMs can pass it without possessing genuine understanding.',
    'Fundamentals',
    '{"Large Language Model","AGI","Narrow AI","ChatGPT"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'narrow-ai',
    'Narrow AI',
    'N',
    'AI designed to perform a specific, limited set of tasks — also called Weak AI.',
    'Narrow AI (also called Weak AI or Artificial Narrow Intelligence, ANI) refers to AI systems designed and trained to perform a specific task or a limited set of related tasks. Unlike hypothetical Artificial General Intelligence (AGI), narrow AI cannot transfer its knowledge to domains outside its training. Examples include image classifiers, spam filters, recommendation engines, chess-playing programs, and speech recognition systems. Despite the ''narrow'' label, modern narrow AI systems like GPT-4 can perform impressively across many language tasks, blurring the line with more general capabilities.',
    'Fundamentals',
    '{"AGI","Turing Test","Large Language Model","Foundation Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'spec-driven-development',
    'Spec-Driven Development',
    'S',
    'A development methodology where detailed specifications guide design, implementation, and testing.',
    'Spec-driven development (SDD) is a software engineering methodology in which clear, structured specifications are written before implementation begins. These specs define expected behavior, inputs, outputs, edge cases, and acceptance criteria. In the context of AI systems, SDD is increasingly important for defining how AI components should behave, what outputs are acceptable, and how to evaluate correctness. It aligns closely with test-driven development (TDD) and behavior-driven development (BDD), and is gaining traction as a way to build more reliable, auditable AI-powered applications.',
    'Engineering',
    '{"Context Engineering","Prompt Engineering","Alignment","Evaluation"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'nlu-nlg-nlp',
    'NLU / NLG / NLP',
    'N',
    'The three pillars of language AI: Understanding, Generation, and Processing.',
    'NLU (Natural Language Understanding), NLG (Natural Language Generation), and NLP (Natural Language Processing) are three closely related but distinct subfields of language AI. NLP is the broadest term, covering all computational techniques for processing human language. NLU focuses specifically on comprehension — extracting meaning, intent, and structure from text. NLG focuses on production — generating coherent, contextually appropriate language from data or knowledge. Modern large language models like GPT-4 integrate all three capabilities: they process input (NLP), understand it (NLU), and generate responses (NLG) in a unified architecture.',
    'NLP',
    '{"NLP","Natural Language Understanding","Natural Language Generation","Large Language Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'tokenization',
    'Tokenization',
    'T',
    'The process of converting raw text into tokens that an LLM can process.',
    'Tokenization is the preprocessing step in which raw text is split into tokens — the atomic units an LLM understands. Tokenizers like Byte-Pair Encoding (BPE) or SentencePiece split text into subword units, balancing vocabulary size with coverage. Different models use different tokenizers with different vocabularies, which is why the same text can produce different token counts across models. Understanding tokenization is important for managing context window usage and API costs.',
    'Foundations',
    '{"Token","Context Window","EOS Token"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'eos-token',
    'EOS Token (End of Sequence)',
    'E',
    'A special token that signals the end of a model''s generated output.',
    'The End of Sequence (EOS) token is a special token used to indicate that an LLM has completed its generation. Each model family uses its own EOS token — for example, GPT-4 uses <|endoftext|>, Llama 3 uses <|eot_id|>, and SmolLM2 uses <|im_end|>. The model stops generating once it predicts this token. EOS tokens are part of a broader set of special tokens that structure the model''s inputs and outputs.',
    'Foundations',
    '{"Token","Autoregressive Model","Special Tokens"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'special-tokens',
    'Special Tokens',
    'S',
    'Reserved tokens used to structure LLM inputs and outputs, such as marking the start or end of messages.',
    'Special tokens are reserved tokens in an LLM''s vocabulary that carry structural meaning rather than linguistic content. They are used to demarcate the beginning or end of a sequence, separate system instructions from user messages, or signal tool use and function calls. Examples include EOS tokens, BOS (Beginning of Sequence) tokens, and chat-specific tokens. Different models use different sets of special tokens, making prompt migration between models non-trivial.',
    'Foundations',
    '{"EOS Token","Token","System Prompt","Chat Template"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'encoder',
    'Encoder',
    'E',
    'A type of Transformer that converts input text into dense vector representations (embeddings).',
    'An encoder is a Transformer variant that processes an input sequence and produces a dense vector representation (embedding) of that input. Encoder-based models like BERT are trained to understand and represent text, making them well-suited for tasks like text classification, semantic search, and Named Entity Recognition (NER). Unlike decoders, encoders do not generate new text token by token; instead, they produce fixed-size representations of the entire input.',
    'Foundations',
    '{"Decoder","Transformer","Embedding","Semantic Search"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'decoder',
    'Decoder',
    'D',
    'A type of Transformer designed to generate new tokens, one at a time, for tasks like text generation.',
    'A decoder is a Transformer variant that generates new text by predicting one token at a time, conditioned on all previous tokens. Decoder-only models like GPT-4, Llama, and Mistral are the most common architecture for modern LLMs. They are used for text generation, chatbots, code generation, and reasoning. Their unidirectional attention means they can only look at previous tokens, making them ideal for generation tasks.',
    'Foundations',
    '{"Encoder","Transformer","Autoregressive Model","Large Language Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'prompt',
    'Prompt',
    'P',
    'The input text provided to an LLM to guide its response.',
    'A prompt is the input text passed to an LLM that instructs or guides its generation. Prompts can include instructions, examples, context, conversation history, retrieved documents, and system-level directives. The wording, structure, and content of a prompt significantly affect the quality and relevance of the model''s output. Prompt design is one of the most accessible and powerful ways to improve LLM performance without retraining.',
    'Prompting',
    '{"Prompt Engineering","System Prompt","Chain-of-Thought","Context Engineering"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'system-prompt',
    'System Prompt',
    'S',
    'A special prompt that sets the behavior, persona, and constraints of an LLM before the conversation starts.',
    'A system prompt is an instruction block that is passed to an LLM before any user message, used to configure the model''s behavior, persona, tone, and constraints. It typically contains role descriptions, output format requirements, safety rules, and task-specific instructions. System prompts are a primary tool for customizing LLM behavior in production applications and are part of the broader prompt engineering toolkit.',
    'Prompting',
    '{"Prompt Engineering","Prompt","Chat Template","Context Engineering"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'few-shot-prompting',
    'Few-Shot Prompting',
    'F',
    'A prompting technique where a few input-output examples are included in the prompt to guide the model''s behavior.',
    'Few-shot prompting (also called n-shot prompting) is a technique where a small number of input-output examples are embedded directly into the prompt to demonstrate the desired task format and output style. This leverages the LLM''s in-context learning ability — the model infers the pattern from the examples and applies it to new inputs. As a rule of thumb, providing at least 5 examples helps the model generalize, and examples should be representative of the real production distribution.',
    'Prompting',
    '{"In-Context Learning","Zero-Shot Prompting","Prompt Engineering","Chain-of-Thought"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'in-context-learning',
    'In-Context Learning',
    'I',
    'The ability of an LLM to learn a new task from examples provided directly in the prompt, without weight updates.',
    'In-context learning is the capability of LLMs to adapt to new tasks or behaviors based solely on examples and instructions provided within the prompt, without any changes to the model''s underlying weights. This is in contrast to fine-tuning, which requires updating the model parameters. In-context learning is enabled by the model''s ability to detect and generalize patterns from the few examples it sees in the context window.',
    'Prompting',
    '{"Few-Shot Prompting","Zero-Shot Prompting","Fine-Tuning","Prompt Engineering"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'zero-shot-prompting',
    'Zero-Shot Prompting',
    'Z',
    'Asking an LLM to perform a task with instructions only, providing no examples.',
    'Zero-shot prompting is a technique where an LLM is asked to perform a task based only on a natural language instruction, with no input-output examples provided. The model relies entirely on its pre-trained knowledge and instruction-following ability. While simple and flexible, zero-shot performance can be less reliable than few-shot prompting for complex or specialized tasks.',
    'Prompting',
    '{"Few-Shot Prompting","In-Context Learning","Prompt Engineering"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'semantic-search',
    'Semantic Search',
    'S',
    'A search technique that finds documents based on meaning and intent rather than exact keyword matches.',
    'Semantic search is a retrieval approach that finds relevant documents by comparing the meaning of a query to the meaning of documents, using embedding vectors and similarity metrics. Unlike keyword search, semantic search can surface relevant results even when the exact words in the query don''t appear in the document. It is a key component of RAG pipelines. For best results, semantic search is often combined with keyword search in a hybrid approach.',
    'Retrieval',
    '{"Embedding","Vector Database","Hybrid Search","Retrieval-Augmented Generation"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'hybrid-search',
    'Hybrid Search',
    'H',
    'A retrieval method that combines semantic (vector) search with traditional keyword (BM25) search for better results.',
    'Hybrid search combines semantic search (using embedding similarity) with keyword-based search (such as BM25 or TF-IDF) to retrieve documents. The two rankings are typically merged using techniques like Reciprocal Rank Fusion (RRF). Hybrid search often outperforms either method alone, especially for precise factual queries where keyword matching is important, or for domain-specific terminology that embedding models may not handle well. It is considered a best practice in production RAG systems.',
    'Retrieval',
    '{"Semantic Search","Retrieval-Augmented Generation","Vector Database"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'grounding',
    'Grounding',
    'G',
    'Anchoring LLM outputs to verifiable external sources to improve factual accuracy.',
    'Grounding is the practice of connecting LLM-generated outputs to reliable external sources such as retrieved documents, databases, or real-time data. A grounded response is traceable back to a source, reducing hallucination risk. RAG is the most common grounding technique. Grounding is especially important in high-stakes applications like medical, legal, or financial AI systems where factual accuracy is critical.',
    'Evaluation',
    '{"Hallucination","Retrieval-Augmented Generation","Guardrails"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'tool-use',
    'Tool Use',
    'T',
    'The ability of an LLM or AI agent to call external functions or APIs to extend its capabilities.',
    'Tool use (also called function calling) is the ability of an LLM to invoke external tools or APIs as part of its reasoning process. Tools can include web search, code interpreters, database queries, calculators, and custom APIs. The model receives a description of available tools, decides which to call and with what arguments, and processes the tool''s output to inform its next action. Tool use is a fundamental capability of AI agents.',
    'Agents',
    '{"AI Agent","Function Calling","Agentic Loop","ReAct","MCP"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'react',
    'ReAct (Reason + Act)',
    'R',
    'A framework where an LLM interleaves reasoning steps with actions to solve tasks iteratively.',
    'ReAct is an agent framework that interleaves reasoning (thinking through what to do next) with acting (calling tools or taking actions) in an iterative loop. The model outputs a Thought describing its reasoning, then an Action specifying a tool call, then an Observation reporting the result, and repeats until the task is complete. ReAct improves on pure reasoning approaches by grounding the agent''s thoughts in real observations from the environment.',
    'Agents',
    '{"AI Agent","Tool Use","Agentic Loop","Chain-of-Thought","Thought-Action-Observation"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'agentic-loop',
    'Agentic Loop',
    'A',
    'The iterative cycle of reasoning, acting, and observing that drives an AI agent''s behavior.',
    'The agentic loop is the core operational cycle of an AI agent: the agent receives a goal or observation, reasons about the best next action, executes that action (e.g., calls a tool), receives an observation of the result, updates its understanding, and repeats the cycle until the goal is achieved or it determines it cannot proceed. This loop enables agents to tackle complex, multi-step tasks that cannot be solved in a single LLM call.',
    'Agents',
    '{"AI Agent","ReAct","Tool Use","Thought-Action-Observation","Orchestration"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'multi-agent-system',
    'Multi-Agent System',
    'M',
    'A system composed of multiple AI agents collaborating or competing to accomplish complex tasks.',
    'A multi-agent system is an architecture where multiple AI agents — each with their own role, tools, and capabilities — work together (or in a structured hierarchy) to accomplish tasks that are too complex for a single agent. Common patterns include a coordinator/orchestrator agent that delegates subtasks to specialized worker agents. Multi-agent systems can improve parallelism, specialization, and robustness but introduce coordination complexity and the risk of error propagation.',
    'Agents',
    '{"AI Agent","Orchestration","Agentic Loop","Tool Use"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'orchestration',
    'Orchestration',
    'O',
    'The coordination of LLM calls, tool invocations, and data flows to build complex AI pipelines.',
    'Orchestration refers to the design and management of sequences of LLM calls, tool invocations, memory retrievals, and data transformations that collectively implement a complex AI workflow or agent. Orchestration frameworks like LangChain, LlamaIndex, and smolagents provide abstractions for building these pipelines. Good orchestration design prioritizes determinism, observability, and graceful error handling, and determines when to use deterministic flows versus LLM-driven decision-making.',
    'Agents',
    '{"AI Agent","Multi-Agent System","LLMOps","Tool Use","Agentic Loop"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'memory',
    'Memory',
    'M',
    'Mechanisms that allow AI agents to store and retrieve information across multiple steps or sessions.',
    'Memory in AI agent systems refers to mechanisms that allow information to persist and be retrieved beyond the immediate context window. Common memory types include: short-term memory (the current context window), episodic memory (logs of past interactions), semantic memory (retrieved knowledge from vector stores), and procedural memory (learned skills or plans). Effective memory management is essential for agents handling long tasks, personalization, and multi-session continuity.',
    'Agents',
    '{"AI Agent","Context Window","Retrieval-Augmented Generation","Agentic Loop"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'mcp',
    'Model Context Protocol (MCP)',
    'M',
    'An open standard protocol for connecting AI models with external tools and data sources.',
    'The Model Context Protocol (MCP) is an open protocol that standardizes how AI models and agents connect to external tools, APIs, and data sources. Developed by Anthropic and supported by frameworks like Kiro, MCP enables AI systems to access context from diverse external sources — databases, file systems, APIs — through a consistent interface. MCP servers expose capabilities that agents can discover and use, making it easier to build composable, tool-augmented AI applications.',
    'Agents',
    '{"Tool Use","AI Agent","Function Calling","Orchestration"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'function-calling',
    'Function Calling',
    'F',
    'A model capability to generate structured calls to predefined functions or APIs as part of its output.',
    'Function calling (also known as tool use) is a capability in which an LLM generates structured JSON or code to invoke a predefined external function or API, rather than generating plain text. The model is given a schema describing available functions and their parameters, decides which function to call and with what arguments, and returns a structured call that an application can execute. Function calling is essential for building reliable tool-using agents.',
    'Agents',
    '{"Tool Use","AI Agent","MCP","Structured Output"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'structured-output',
    'Structured Output',
    'S',
    'Constraining LLM output to a defined schema (e.g., JSON) to facilitate downstream processing.',
    'Structured output is the practice of constraining an LLM''s generation to follow a specific format — such as JSON, XML, or a typed schema — rather than producing free-form text. This simplifies downstream parsing, integration with other systems, and automated validation. Techniques include prompt-based formatting instructions, output parsers, and model-native features like JSON mode or grammar-constrained sampling. Structured output is especially important in agent pipelines and API integrations.',
    'Prompting',
    '{"Function Calling","Tool Use","Prompt Engineering","Guardrails"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'guardrails',
    'Guardrails',
    'G',
    'Rules and mechanisms that constrain or validate LLM inputs and outputs to ensure safe and appropriate behavior.',
    'Guardrails are safety and quality mechanisms applied to LLM inputs or outputs to enforce constraints such as content policy compliance, output format validity, factual accuracy, and instruction adherence. They can be implemented as prompt-based instructions, output classifiers, rule-based filters, or separate validation models. Guardrails are often interchangeable with evaluation mechanisms and are a core component of responsible AI production systems.',
    'Evaluation',
    '{"Hallucination","LLM-as-Judge","Structured Output","Grounding","Safety"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'llm-as-judge',
    'LLM-as-Judge',
    'L',
    'Using an LLM to evaluate the quality or correctness of another LLM''s outputs.',
    'LLM-as-Judge is an evaluation technique where a powerful LLM (often a different model or a larger one) is used to assess the quality, relevance, safety, or correctness of outputs generated by another model. It enables scalable automated evaluation without requiring human annotators for every sample. While useful, LLM-as-Judge has known limitations: it can exhibit bias toward longer or more confident-sounding responses, and it may not catch subtle factual errors. It works best for relative comparisons rather than absolute scoring.',
    'Evaluation',
    '{"Evaluation","Guardrails","Hallucination","Evals"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'evals',
    'Evals (Evaluations)',
    'E',
    'Tests and benchmarks used to measure LLM or AI system performance on specific tasks.',
    'Evals (short for evaluations) are systematic tests designed to measure the performance, accuracy, safety, and reliability of LLMs or AI systems. They range from unit tests on specific input-output pairs to large benchmarks covering diverse capabilities. Strong eval suites are foundational to building trustworthy AI products — they enable rapid iteration, catch regressions, and provide confidence before deployment. Building evals early and investing in a data flywheel is widely considered best practice in production LLM development.',
    'Evaluation',
    '{"LLM-as-Judge","Guardrails","Hallucination","Benchmarks"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'llmops',
    'LLMOps',
    'L',
    'The operational practices for deploying, monitoring, versioning, and maintaining LLM-based applications in production.',
    'LLMOps is the set of practices, tools, and culture for operationalizing LLM-based applications — analogous to MLOps for traditional machine learning. It covers prompt versioning, model versioning, A/B testing, monitoring (tracking input-output quality over time), logging, latency optimization, and cost management. The primary goal of LLMOps is to enable faster iteration cycles and maintain production reliability as models and prompts evolve.',
    'Operations',
    '{"Evals","Monitoring","Orchestration","Prompt Versioning"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'temperature',
    'Temperature',
    'T',
    'A sampling parameter that controls the randomness and creativity of LLM output.',
    'Temperature is a decoding parameter that controls the diversity of an LLM''s output by scaling the probability distribution over tokens before sampling. A temperature of 0 makes the model deterministic (always choosing the highest-probability token), while higher values (e.g., 0.7–1.0) introduce more randomness and creativity. Temperature is one of several decoding strategies; others include top-p (nucleus) sampling and top-k sampling. Choosing the right temperature depends on the task: low for factual tasks, higher for creative generation.',
    'Inference',
    '{"Sampling","Top-P Sampling","Inference","Decoding Strategy"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'top-p-sampling',
    'Top-P Sampling (Nucleus Sampling)',
    'T',
    'A sampling strategy that selects from the smallest set of tokens whose cumulative probability exceeds p.',
    'Top-P sampling, also called nucleus sampling, is a decoding strategy where the model samples from only the smallest set of tokens whose cumulative probability mass exceeds a threshold p (e.g., 0.9). This ensures that unlikely tokens are excluded from sampling while still allowing for variability. Top-P is often used alongside temperature to control output diversity, and it tends to produce more coherent outputs than pure temperature scaling at high values.',
    'Inference',
    '{"Temperature","Sampling","Decoding Strategy"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'beam-search',
    'Beam Search',
    'B',
    'A decoding strategy that explores multiple token sequences simultaneously to find the highest-probability output.',
    'Beam search is a decoding strategy that maintains a fixed number (the ''beam width'') of candidate sequences at each generation step, expanding each candidate with the most probable next tokens and keeping only the top candidates. Unlike greedy decoding (which always picks the single best token), beam search can find higher-quality overall sequences by exploring alternatives. It is commonly used in translation and summarization tasks but is less common in modern chat-oriented LLMs.',
    'Inference',
    '{"Temperature","Top-P Sampling","Decoding Strategy","Autoregressive Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'caching',
    'Caching',
    'C',
    'Storing LLM outputs or intermediate computations to avoid redundant, expensive model calls.',
    'Caching in LLM systems refers to storing the results of model calls or intermediate computations (such as KV cache for prefilled prompts) so they can be reused without re-running the model. Prompt caching can significantly reduce latency and API costs for applications with repetitive or large fixed-context prompts. Semantic caching goes further by retrieving cached responses to semantically similar (not just identical) queries. Caching is an underutilized but high-impact optimization in production LLM applications.',
    'Operations',
    '{"LLMOps","Inference","Latency","Prompt Engineering"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'alignment',
    'Alignment',
    'A',
    'The process of ensuring AI model behavior matches human values, intentions, and preferences.',
    'Alignment refers to the challenge and practice of ensuring that AI systems behave in ways that are consistent with human values, intentions, and preferences — particularly as models become more capable and autonomous. Alignment techniques include RLHF, Constitutional AI, and direct preference optimization (DPO). Misaligned AI can produce harmful, deceptive, or unintended behaviors even when technically capable. Alignment is considered one of the central problems in AI safety research.',
    'Safety',
    '{"RLHF","Safety","Guardrails","Constitutional AI"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'safety',
    'Safety',
    'S',
    'Practices and mechanisms to prevent AI models from generating harmful, dangerous, or inappropriate content.',
    'AI safety refers to the practices, techniques, and guidelines designed to prevent AI models from producing content or taking actions that are harmful, illegal, misleading, or dangerous. Safety measures in LLM applications include content moderation classifiers, guardrails, red-teaming, refusal training, and output filtering. Safety considerations are especially critical in agentic systems where the model can take real-world actions with irreversible consequences.',
    'Safety',
    '{"Alignment","Guardrails","Red-Teaming","Responsible AI"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'human-in-the-loop',
    'Human-in-the-Loop (HITL)',
    'H',
    'A design pattern where humans review or approve AI outputs at critical steps before they take effect.',
    'Human-in-the-Loop (HITL) is a design pattern where human oversight is incorporated at key points in an AI workflow — for example, requiring a human to review and approve a model-generated action before it is executed. HITL is especially important in high-stakes applications and agentic systems where errors can have significant consequences. Well-designed HITL interfaces keep humans informed and in control without creating excessive friction or bottlenecks.',
    'Operations',
    '{"AI Agent","Safety","Guardrails","Agentic Loop"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'agentic-ide',
    'Agentic IDE',
    'A',
    'A development environment powered by AI agents that can autonomously write, edit, and manage code.',
    'An agentic IDE is a software development environment that integrates AI agents deeply into the coding workflow, enabling the AI to not just suggest completions but to autonomously plan, generate, refactor, and manage code across a project. Examples include Kiro, which features capabilities like specs (structured feature planning), steering (custom AI rules), hooks (automated triggers), and MCP integration. Agentic IDEs represent a shift from AI as a code assistant to AI as a collaborative engineering partner.',
    'Tooling',
    '{"AI Agent","MCP","Specs","Steering","Hooks"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'specs',
    'Specs (Specifications)',
    'S',
    'Structured documents that define the requirements, design, and implementation plan for a feature before coding begins.',
    'In the context of agentic development tools like Kiro, specs (specifications) are structured documents that an AI agent helps generate before writing code. A spec typically includes a requirements document, a system design document, and a set of implementation tasks. This spec-driven approach ensures AI-generated code is aligned with user intent and project architecture before any implementation begins, reducing costly rework.',
    'Tooling',
    '{"Agentic IDE","Steering","Hooks","AI Agent"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'steering',
    'Steering',
    'S',
    'Custom rules and context provided to an AI system to guide its behavior within a specific project.',
    'Steering refers to the use of custom rules, constraints, and context to guide the behavior of an AI system within a specific environment. In Kiro, steering files allow developers to define project-specific conventions — such as coding standards, architectural patterns, or preferred libraries — that the AI agent follows consistently across all interactions. Steering is analogous to a persistent, project-scoped system prompt.',
    'Tooling',
    '{"Agentic IDE","Specs","System Prompt","Context Engineering"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'hooks',
    'Hooks',
    'H',
    'Automated triggers that execute AI actions in response to specific development events.',
    'Hooks are automated triggers in agentic development environments (like Kiro) that execute AI-driven actions in response to predefined events — such as a file being saved, a test failing, or a pull request being opened. Hooks enable repetitive tasks to be automated without manual intervention, embedding AI assistance directly into the development workflow. They are a form of event-driven automation for AI agent actions.',
    'Tooling',
    '{"Agentic IDE","Specs","AI Agent","Orchestration"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'model-selection',
    'Model Selection',
    'M',
    'The process of choosing the most appropriate LLM for a given task based on capability, cost, and latency.',
    'Model selection is the practice of choosing the most suitable LLM for a specific use case by balancing factors including task complexity, required quality, inference cost, and latency constraints. A key principle is to start with the smallest model that achieves acceptable quality — avoiding overpaying for capability that isn''t needed. Model selection also involves versioning and pinning model versions to avoid unexpected behavior changes when providers update their models.',
    'Operations',
    '{"Inference","LLMOps","Fine-Tuning","Latency"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'latency',
    'Latency',
    'L',
    'The time it takes for an LLM to begin or complete a response after receiving a prompt.',
    'Latency in LLM systems refers to the delay between submitting a prompt and receiving a response. It is commonly measured as Time to First Token (TTFT) — how long until the model starts streaming output — and total generation time. Latency is a critical factor in user experience and is influenced by model size, hardware, context length, and system architecture. Optimization techniques include caching, smaller models, streaming, and batching.',
    'Operations',
    '{"Inference","Caching","Model Selection","Streaming"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'streaming',
    'Streaming',
    'S',
    'Delivering LLM output token by token as it is generated, rather than waiting for the full response.',
    'Streaming is a technique where an LLM''s output is transmitted to the user token by token as it is generated, rather than waiting for the complete response. This significantly improves perceived latency since users see the response forming in real time. Streaming is supported by most major LLM APIs and is standard practice for chat interfaces and long-form generation tasks.',
    'Inference',
    '{"Latency","Inference","Autoregressive Model"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'agentic-rag',
    'Agentic RAG',
    'A',
    'A RAG architecture where an AI agent dynamically decides when and how to retrieve information.',
    'Agentic RAG combines the retrieval capabilities of Retrieval-Augmented Generation with the planning and decision-making capabilities of an AI agent. Instead of always retrieving documents as a fixed preprocessing step, the agent decides dynamically when retrieval is needed, what queries to issue, and how to synthesize results from multiple retrieval rounds. Agentic RAG enables more complex, multi-hop reasoning over large knowledge bases and is a key pattern in advanced agent architectures.',
    'Agents',
    '{"Retrieval-Augmented Generation","AI Agent","Agentic Loop","Tool Use","Memory"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'prompt-versioning',
    'Prompt Versioning',
    'P',
    'Tracking and managing changes to prompts over time, analogous to version control for code.',
    'Prompt versioning is the practice of tracking, storing, and managing different versions of prompts used in production LLM applications. Like software version control, it allows teams to roll back to previous prompt versions, compare performance across versions, and safely deploy changes. Because prompt changes can significantly affect model behavior, versioning is a critical part of LLMOps for maintaining stability and enabling systematic improvement.',
    'Operations',
    '{"LLMOps","Evals","Prompt Engineering"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'product-market-fit',
    'Product-Market Fit (PMF) for AI',
    'P',
    'Validating that an AI-powered product solves a real user need before investing heavily in infrastructure.',
    'In the context of LLM product development, Product-Market Fit (PMF) refers to the validation that an AI-powered product genuinely solves a real user need before making large infrastructure investments like training custom models or building complex ML pipelines. A widely cited heuristic is ''No GPUs before PMF'' — start with inference APIs, prompt engineering, and RAG before committing to custom training. Achieving PMF with minimal infrastructure reduces risk and accelerates learning.',
    'Strategy',
    '{"LLMOps","Fine-Tuning","Inference","Model Selection"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'data-flywheel',
    'Data Flywheel',
    'D',
    'A self-reinforcing cycle where product usage generates data that improves the AI, which attracts more users.',
    'A data flywheel is a strategic pattern in AI product development where product usage generates training and evaluation data, which is used to improve the AI model, which improves the product, which attracts more users, and so on. Building a data flywheel early — by instrumenting production to capture input-output pairs, user feedback, and edge cases — creates a compounding advantage over time. It is considered one of the most valuable strategic assets in AI product development.',
    'Strategy',
    '{"Evals","Fine-Tuning","LLMOps","Product-Market Fit"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'red-teaming',
    'Red-Teaming',
    'R',
    'The practice of adversarially testing an AI system to find safety and reliability failures before deployment.',
    'Red-teaming is a safety practice borrowed from cybersecurity where a dedicated team attempts to find flaws, jailbreaks, failure modes, and harmful behaviors in an AI system by adversarially probing it. Red-teaming uncovers issues that standard evals may miss, including edge cases, prompt injections, and content policy violations. It is considered a best practice for responsible AI deployment, especially for systems with broad user access or high-stakes applications.',
    'Safety',
    '{"Safety","Alignment","Guardrails","Evals"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'reward-model',
    'Reward Model',
    'R',
    'A model trained to score LLM outputs based on human preference, used in RLHF training.',
    'A reward model is a neural network trained to predict how much a human would prefer one model output over another. It is trained on human comparison data (e.g., annotators ranking pairs of model outputs) and produces a scalar score for any given output. In RLHF, the reward model serves as a proxy for human preference, guiding the LLM''s policy updates during reinforcement learning. The quality of the reward model is critical to alignment success.',
    'Training',
    '{"RLHF","Fine-Tuning","Alignment","Evals"}',
    'en'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'autonomous-agent',
    'Autonomous Agent',
    'A',
    'An AI agent capable of completing complex, multi-step tasks with minimal human intervention.',
    'An autonomous agent is an AI agent designed to operate with a high degree of independence, capable of planning, executing, and adapting across long sequences of actions to achieve a user-specified goal. Unlike simple chatbots or single-turn assistants, autonomous agents manage their own tool calls, memory, and decision-making over extended workflows. Kiro''s autonomous agent, for example, can execute agentic tasks end-to-end through a CLI or IDE interface.',
    'Agents',
    '{"AI Agent","Agentic Loop","Tool Use","Orchestration","Human-in-the-Loop"}',
    'en'
);

-- Glossary terms (es)
INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'attention-mechanism',
    'Mecanismo de Atención',
    'M',
    'Técnica que permite a los modelos enfocarse en partes relevantes de la entrada al producir una salida.',
    'El mecanismo de atención es un componente fundamental en las redes neuronales modernas, especialmente en las arquitecturas Transformer. Permite que un modelo se enfoque dinámicamente en diferentes partes de la secuencia de entrada al generar cada elemento de la salida. En lugar de comprimir toda la información de entrada en un vector de tamaño fijo, la atención calcula una suma ponderada de representaciones de entrada, donde los pesos reflejan la relevancia de cada elemento de entrada para el paso de salida actual. Esto permite a los modelos manejar dependencias de largo alcance y capturar relaciones complejas en los datos.',
    'Arquitectura',
    '{"Transformer","Auto-Atención","Atención Multi-Cabeza"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'autoregressive-model',
    'Modelo Autorregresivo',
    'M',
    'Modelo que genera salida de forma secuencial, donde cada token está condicionado por los tokens anteriores.',
    'Un modelo autorregresivo genera secuencias prediciendo un elemento a la vez, condicionando cada predicción en todos los elementos generados anteriormente. En modelos de lenguaje como GPT, esto significa generar texto token por token de izquierda a derecha. El modelo aprende la probabilidad conjunta de una secuencia descomponiéndola en un producto de probabilidades condicionales. Este enfoque es poderoso para tareas de generación, pero puede ser lento en inferencia ya que los tokens deben producirse secuencialmente.',
    'Tipo de Modelo',
    '{"GPT","Modelo de Lenguaje","Token"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'bert',
    'BERT',
    'B',
    'Representaciones de Codificador Bidireccional de Transformers — modelo de lenguaje preentrenado de Google.',
    'BERT (Bidirectional Encoder Representations from Transformers) es un modelo de lenguaje preentrenado desarrollado por Google en 2018. A diferencia de GPT, BERT utiliza un enfoque bidireccional, lo que significa que considera el contexto izquierdo y derecho simultáneamente al procesar texto. Se preentrenan usando dos tareas: Modelado de Lenguaje Enmascarado (MLM), donde se enmascaran tokens aleatorios y el modelo los predice, y Predicción de la Siguiente Oración (NSP). BERT estableció nuevos resultados de vanguardia en numerosos benchmarks de PLN y popularizó el paradigma de ajuste fino para tareas de PLN.',
    'Modelo',
    '{"Transformer","Ajuste Fino","Preentrenamiento","GPT"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'chain-of-thought',
    'Prompting Cadena de Pensamiento',
    'P',
    'Técnica de prompting que anima a los LLM a razonar paso a paso antes de responder.',
    'El prompting de Cadena de Pensamiento (CoT) es una técnica donde se guía al modelo para producir pasos de razonamiento intermedios antes de llegar a una respuesta final. Al incluir ejemplos que muestran razonamiento paso a paso en el prompt, o simplemente instruyendo al modelo a ''pensar paso a paso'', el CoT mejora significativamente el rendimiento en tareas de razonamiento complejo como problemas matemáticos, acertijos lógicos y preguntas de múltiples pasos. Fue introducido por investigadores de Google y se ha convertido en una técnica estándar para obtener mejor razonamiento de los grandes modelos de lenguaje.',
    'Prompting',
    '{"Ingeniería de Prompts","Aprendizaje Few-Shot","Razonamiento"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'context-window',
    'Ventana de Contexto',
    'V',
    'La cantidad máxima de texto (tokens) que un modelo de lenguaje puede procesar a la vez.',
    'La ventana de contexto se refiere al número máximo de tokens que un modelo de lenguaje puede considerar a la vez durante la inferencia. Los tokens dentro de la ventana de contexto pueden atenderse mutuamente a través del mecanismo de atención. Los primeros modelos como GPT-2 tenían ventanas de contexto de 1.024 tokens, mientras que los modelos modernos como GPT-4 Turbo admiten hasta 128.000 tokens. Una ventana de contexto más grande permite al modelo procesar documentos más largos, mantener conversaciones más largas y realizar tareas que requieren contexto extenso, pero también aumenta el costo computacional.',
    'Arquitectura',
    '{"Token","Mecanismo de Atención","Transformer"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'diffusion-model',
    'Modelo de Difusión',
    'M',
    'Modelo generativo que aprende a revertir un proceso de adición de ruido para generar datos.',
    'Los modelos de difusión son una clase de modelos generativos que aprenden a generar datos revirtiendo un proceso gradual de adición de ruido. Durante el entrenamiento, los datos (por ejemplo, imágenes) se corrompen progresivamente con ruido gaussiano a lo largo de muchos pasos. El modelo aprende a revertir este proceso, comenzando desde ruido puro y eliminando el ruido iterativamente para producir muestras realistas. Stable Diffusion, DALL-E 2 y Midjourney son ejemplos prominentes. Los modelos de difusión han logrado resultados de vanguardia en generación de imágenes, síntesis de audio y generación de video.',
    'IA Generativa',
    '{"Stable Diffusion","DALL-E","Modelo Generativo","Espacio Latente"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'embedding',
    'Embedding',
    'E',
    'Representación vectorial densa de datos (palabras, oraciones, imágenes) en un espacio continuo.',
    'Un embedding es una representación aprendida de datos como un vector denso en un espacio continuo de alta dimensión. Las palabras, oraciones, imágenes u otras entidades se mapean a vectores de modo que los elementos semánticamente similares estén cerca en el espacio de embedding. Los embeddings de palabras como Word2Vec y GloVe fueron ejemplos tempranos; los modelos modernos producen embeddings contextuales donde la misma palabra tiene diferentes representaciones según el contexto. Los embeddings son fundamentales para la mayoría de los sistemas de IA modernos, permitiendo búsqueda eficiente por similitud, agrupamiento y rendimiento en tareas posteriores.',
    'Representación',
    '{"Base de Datos Vectorial","Búsqueda Semántica","Word2Vec","Transformer"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'fine-tuning',
    'Ajuste Fino',
    'A',
    'Adaptar un modelo preentrenado a una tarea específica entrenándolo con datos específicos de esa tarea.',
    'El ajuste fino es el proceso de tomar un modelo preentrenado y continuar entrenándolo en un conjunto de datos más pequeño y específico de la tarea. Esto permite al modelo adaptar su conocimiento general a un dominio o tarea particular mientras retiene las amplias capacidades aprendidas durante el preentrenamiento. El ajuste fino puede ser completo (actualizando todos los parámetros) o eficiente en parámetros (por ejemplo, LoRA, adaptadores). Es una piedra angular del PLN moderno y la visión por computadora, permitiendo alto rendimiento en tareas especializadas sin entrenar desde cero.',
    'Entrenamiento',
    '{"Preentrenamiento","Aprendizaje por Transferencia","LoRA","RLHF"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'foundation-model',
    'Modelo Fundacional',
    'M',
    'Modelo grande entrenado con datos amplios que puede adaptarse a muchas tareas posteriores.',
    'Un modelo fundacional es un modelo de IA grande entrenado en vastas cantidades de datos diversos utilizando aprendizaje auto-supervisado. El término fue acuñado por investigadores de Stanford en 2021. Los modelos fundacionales sirven como base que puede ajustarse finamente o consultarse mediante prompts para una amplia gama de tareas posteriores. Ejemplos incluyen GPT-4, PaLM, LLaMA, CLIP y Stable Diffusion. Su escala y generalidad los hacen puntos de partida poderosos, pero también generan preocupaciones sobre sesgo, seguridad y la concentración de capacidades de IA.',
    'Tipo de Modelo',
    '{"Preentrenamiento","Ajuste Fino","Modelo de Lenguaje Grande","Aprendizaje por Transferencia"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'gpt',
    'GPT',
    'G',
    'Transformer Generativo Preentrenado — familia de grandes modelos de lenguaje autorregresivos de OpenAI.',
    'GPT (Generative Pre-trained Transformer) es una familia de grandes modelos de lenguaje desarrollados por OpenAI. Los modelos GPT se entrenan usando preentrenamiento no supervisado en corpus de texto masivos, seguido de ajuste fino para tareas específicas. Utilizan una arquitectura Transformer de solo decodificador y generan texto de forma autorregresiva. GPT-3 (175B parámetros) demostró notables capacidades de aprendizaje few-shot. GPT-4 es un modelo multimodal capaz de procesar tanto texto como imágenes. La serie GPT ha sido fundamental para demostrar el poder de escalar los modelos de lenguaje.',
    'Modelo',
    '{"Transformer","Modelo Autorregresivo","OpenAI","Modelo de Lenguaje Grande"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'generative-adversarial-network',
    'Red Generativa Adversarial',
    'R',
    'Marco donde dos redes neuronales compiten: un generador y un discriminador.',
    'Una Red Generativa Adversarial (GAN) consiste en dos redes neuronales entrenadas simultáneamente en un marco competitivo. La red generadora crea muestras de datos sintéticos, mientras que la red discriminadora intenta distinguir los datos reales de los generados. A través de este proceso adversarial, el generador aprende a producir salidas cada vez más realistas. Las GAN fueron introducidas por Ian Goodfellow en 2014 y se han utilizado para síntesis de imágenes, transferencia de estilo, aumento de datos y más. Las variantes incluyen DCGAN, StyleGAN y CycleGAN.',
    'IA Generativa',
    '{"Modelo de Difusión","Modelo Generativo","Espacio Latente"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'hallucination',
    'Alucinación',
    'A',
    'Cuando un modelo de IA genera información que suena plausible pero es factualmente incorrecta o fabricada.',
    'La alucinación en IA se refiere al fenómeno donde un modelo de lenguaje genera contenido que se afirma con confianza pero es factualmente incorrecto, sin sentido o completamente fabricado. Esto ocurre porque los LLM están entrenados para producir texto estadísticamente probable en lugar de hechos verificados. Las alucinaciones pueden variar desde errores sutiles (fechas, nombres incorrectos) hasta citas o eventos completamente inventados. Mitigar las alucinaciones es un gran desafío de investigación, con enfoques que incluyen la Generación Aumentada por Recuperación (RAG), mejores datos de entrenamiento y técnicas de alineación mejoradas.',
    'Seguridad y Alineación',
    '{"Generación Aumentada por Recuperación","Alineación","Modelo de Lenguaje Grande"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'inference',
    'Inferencia',
    'I',
    'El proceso de usar un modelo entrenado para generar predicciones o salidas sobre nuevos datos.',
    'La inferencia es la fase donde se usa un modelo de IA entrenado para hacer predicciones o generar salidas sobre datos nuevos no vistos. A diferencia del entrenamiento, la inferencia no actualiza los pesos del modelo. Implica un pase hacia adelante a través de la red. Para los grandes modelos de lenguaje, la inferencia puede ser computacionalmente costosa debido al tamaño de los modelos y la naturaleza autorregresiva de la generación de texto. Técnicas como la cuantización, el procesamiento por lotes, la decodificación especulativa y la aceleración por hardware (GPUs, TPUs) se utilizan para hacer la inferencia más rápida y rentable.',
    'Despliegue',
    '{"Entrenamiento","Cuantización","Modelo Autorregresivo"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'instruction-tuning',
    'Ajuste por Instrucciones',
    'A',
    'Ajuste fino de un modelo de lenguaje con pares instrucción-respuesta para mejorar el seguimiento de instrucciones.',
    'El ajuste por instrucciones es una técnica de ajuste fino donde un modelo de lenguaje preentrenado se entrena en un conjunto de datos de pares (instrucción, respuesta). Esto enseña al modelo a seguir instrucciones en lenguaje natural de manera más confiable. Modelos como InstructGPT, FLAN y Alpaca usan ajuste por instrucciones. Mejora significativamente la capacidad del modelo para generalizar a nuevas tareas descritas en lenguaje natural, sin requerir ejemplos específicos de la tarea. El ajuste por instrucciones a menudo se combina con RLHF para producir asistentes de IA útiles, inofensivos y honestos.',
    'Entrenamiento',
    '{"Ajuste Fino","RLHF","Ingeniería de Prompts","Alineación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'latent-space',
    'Espacio Latente',
    'E',
    'Espacio de representación comprimido y abstracto aprendido por un modelo para codificar datos.',
    'El espacio latente es el espacio multidimensional en el que un modelo representa características comprimidas y abstractas de los datos. En los autoencoders, el codificador mapea los datos de entrada a un punto en el espacio latente, y el decodificador reconstruye los datos desde ese punto. En los modelos de difusión y VAEs, el espacio latente captura la estructura subyacente de la distribución de datos. Navegar y manipular el espacio latente permite la generación controlada, la interpolación entre puntos de datos y la transferencia de estilo. Comprender el espacio latente es clave para entender cómo funcionan los modelos generativos.',
    'Arquitectura',
    '{"Embedding","Modelo de Difusión","Autoencoder Variacional","Modelo Generativo"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'large-language-model',
    'Modelo de Lenguaje Grande',
    'M',
    'Red neuronal con miles de millones de parámetros entrenada en corpus de texto masivos.',
    'Un Modelo de Lenguaje Grande (LLM) es un tipo de red neuronal con miles de millones a billones de parámetros, entrenada en vastas cantidades de datos de texto. Los LLM aprenden patrones estadísticos en el lenguaje y pueden realizar una amplia gama de tareas incluyendo generación de texto, traducción, resumen, respuesta a preguntas y generación de código. LLMs notables incluyen GPT-4, Claude, Gemini, LLaMA y Mistral. Lo ''grande'' se refiere tanto al número de parámetros como a la escala de los datos de entrenamiento. Los LLM exhiben capacidades emergentes que no fueron entrenadas explícitamente.',
    'Tipo de Modelo',
    '{"GPT","Transformer","Modelo Fundacional","Capacidades Emergentes"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'lora',
    'LoRA',
    'L',
    'Adaptación de Bajo Rango — método eficiente de ajuste fino que entrena solo pequeñas matrices adaptadoras.',
    'LoRA (Low-Rank Adaptation) es una técnica de ajuste fino eficiente en parámetros que congela los pesos del modelo preentrenado e inyecta matrices entrenables de descomposición de rango bajo en cada capa de la arquitectura Transformer. En lugar de actualizar todos los parámetros del modelo, LoRA entrena un número mucho menor de parámetros (a menudo <1% del original), haciendo el ajuste fino factible en hardware de consumo. LoRA se ha vuelto extremadamente popular para personalizar grandes modelos de lenguaje y modelos de generación de imágenes como Stable Diffusion, permitiendo adaptación específica del dominio sin los costos de ajuste fino completo.',
    'Entrenamiento',
    '{"Ajuste Fino","Ajuste Fino Eficiente en Parámetros","Transformer","Stable Diffusion"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'multimodal',
    'IA Multimodal',
    'I',
    'Sistemas de IA que pueden procesar y generar múltiples tipos de datos (texto, imágenes, audio, video).',
    'La IA multimodal se refiere a sistemas capaces de entender y generar múltiples modalidades de datos, como texto, imágenes, audio y video. A diferencia de los modelos unimodales que manejan solo un tipo de dato, los modelos multimodales pueden razonar entre modalidades. Ejemplos incluyen GPT-4V (texto + imágenes), Gemini (texto, imágenes, audio, video), CLIP (texto + imágenes) y Flamingo. Las capacidades multimodales permiten aplicaciones como respuesta a preguntas visuales, descripción de imágenes, transcripción de audio y comprensión de video. Esto se considera un paso clave hacia sistemas de IA más generales.',
    'Tipo de Modelo',
    '{"GPT","CLIP","Modelo Visión-Lenguaje","Modelo Fundacional"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'neural-network',
    'Red Neuronal',
    'R',
    'Modelo computacional inspirado en el cerebro, compuesto de capas interconectadas de nodos.',
    'Una red neuronal es un modelo de aprendizaje automático inspirado en la estructura de las redes neuronales biológicas del cerebro. Consiste en capas de nodos interconectados (neuronas), donde cada conexión tiene un peso aprendible. Los datos fluyen a través de la red (pase hacia adelante), y el modelo aprende ajustando los pesos para minimizar una función de pérdida mediante retropropagación. Las redes neuronales profundas con muchas capas son la base de la IA moderna, permitiendo avances en reconocimiento de imágenes, procesamiento de lenguaje natural, reconocimiento de voz e IA generativa.',
    'Fundamentos',
    '{"Aprendizaje Profundo","Retropropagación","Transformer","Red Neuronal Convolucional"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'prompt-engineering',
    'Ingeniería de Prompts',
    'I',
    'La práctica de diseñar entradas para guiar a los modelos de IA hacia las salidas deseadas.',
    'La ingeniería de prompts es la disciplina de diseñar y optimizar prompts de entrada para obtener comportamientos deseados de los modelos de lenguaje de IA. Dado que los LLM son sensibles a cómo se formulan las instrucciones, la ingeniería de prompts puede afectar significativamente la calidad de la salida. Las técnicas incluyen prompting zero-shot, prompting few-shot, prompting de cadena de pensamiento, prompting de rol y prompting de salida estructurada. La ingeniería de prompts ha surgido como una habilidad crítica para usar efectivamente los LLM en aplicaciones, y ha generado investigación en optimización automática de prompts y ataques de inyección de prompts.',
    'Prompting',
    '{"Prompting Cadena de Pensamiento","Aprendizaje Few-Shot","Modelo de Lenguaje Grande","Aprendizaje Zero-Shot"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'pre-training',
    'Preentrenamiento',
    'P',
    'Entrenar un modelo con datos a gran escala antes del ajuste fino en tareas específicas.',
    'El preentrenamiento es la fase inicial de entrenamiento de un modelo fundacional en un conjunto de datos grande y diverso utilizando objetivos auto-supervisados. Para los modelos de lenguaje, esto típicamente implica predecir tokens enmascarados (BERT) o tokens siguientes (GPT). Para los modelos de visión, puede implicar aprendizaje contrastivo o modelado de imágenes enmascaradas. El preentrenamiento permite al modelo aprender representaciones generales del lenguaje, la visión u otras modalidades. El modelo preentrenado luego se adapta para tareas específicas mediante ajuste fino o prompting, reduciendo drásticamente los datos y el cómputo necesarios para las tareas posteriores.',
    'Entrenamiento',
    '{"Ajuste Fino","Modelo Fundacional","Aprendizaje Auto-Supervisado","Aprendizaje por Transferencia"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'quantization',
    'Cuantización',
    'C',
    'Reducir el tamaño del modelo representando los pesos con números de menor precisión.',
    'La cuantización es una técnica de compresión de modelos que reduce la precisión numérica de los pesos y activaciones del modelo de formatos de alta precisión (por ejemplo, float32) a formatos de menor precisión (por ejemplo, int8, int4). Esto reduce los requisitos de memoria y acelera la inferencia con una pérdida mínima en la calidad del modelo. Las técnicas incluyen cuantización post-entrenamiento (PTQ) y entrenamiento con conciencia de cuantización (QAT). La cuantización ha sido crucial para desplegar grandes modelos de lenguaje en hardware de consumo, con herramientas como GPTQ, AWQ y llama.cpp que permiten ejecutar LLMs en laptops.',
    'Despliegue',
    '{"Inferencia","Compresión de Modelos","LoRA","Modelo de Lenguaje Grande"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'rag',
    'Generación Aumentada por Recuperación',
    'G',
    'Combinar un sistema de recuperación con un modelo generativo para fundamentar respuestas en conocimiento externo.',
    'La Generación Aumentada por Recuperación (RAG) es una arquitectura que mejora las respuestas de los modelos de lenguaje recuperando documentos relevantes de una base de conocimiento externa e incorporándolos en el prompt. El proceso implica: (1) codificar la consulta como un embedding, (2) buscar en una base de datos vectorial documentos relevantes, (3) incluir los documentos recuperados en el contexto del prompt, y (4) generar una respuesta fundamentada en la información recuperada. RAG reduce las alucinaciones, permite el acceso a información actualizada y permite a los LLM razonar sobre conocimiento privado o específico del dominio.',
    'Arquitectura',
    '{"Base de Datos Vectorial","Embedding","Alucinación","Modelo de Lenguaje Grande"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'rlhf',
    'RLHF',
    'R',
    'Aprendizaje por Refuerzo con Retroalimentación Humana — alinear modelos de IA usando datos de preferencias humanas.',
    'El Aprendizaje por Refuerzo con Retroalimentación Humana (RLHF) es una técnica de entrenamiento utilizada para alinear los modelos de lenguaje con los valores y preferencias humanas. El proceso implica: (1) ajuste fino supervisado con datos de demostración, (2) entrenamiento de un modelo de recompensa con comparaciones de preferencias humanas, y (3) optimización del modelo de lenguaje usando el modelo de recompensa mediante aprendizaje por refuerzo (típicamente PPO). RLHF se usó para entrenar InstructGPT y ChatGPT, mejorando dramáticamente su utilidad y seguridad. Es una técnica clave en la investigación de alineación de IA.',
    'Entrenamiento',
    '{"Alineación","Ajuste Fino","Ajuste por Instrucciones","Modelo de Recompensa"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'self-attention',
    'Auto-Atención',
    'A',
    'Mecanismo donde cada elemento en una secuencia atiende a todos los demás elementos de la misma secuencia.',
    'La auto-atención (también llamada atención intra-secuencia) es un mecanismo donde cada posición en una secuencia calcula pesos de atención sobre todas las demás posiciones en la misma secuencia. Para cada posición, se calculan consultas, claves y valores a partir de la entrada, y la salida es una suma ponderada de valores donde los pesos están determinados por la compatibilidad consulta-clave. La auto-atención permite al modelo capturar dependencias de largo alcance y relaciones entre cualquier dos posiciones independientemente de la distancia. Es la operación central en las arquitecturas Transformer y escala cuadráticamente con la longitud de la secuencia.',
    'Arquitectura',
    '{"Mecanismo de Atención","Transformer","Atención Multi-Cabeza","Ventana de Contexto"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'stable-diffusion',
    'Stable Diffusion',
    'S',
    'Modelo de difusión latente de código abierto para generación de imágenes de alta calidad a partir de texto.',
    'Stable Diffusion es un modelo de difusión latente de código abierto desarrollado por Stability AI, lanzado en 2022. Genera imágenes de alta calidad a partir de descripciones de texto realizando el proceso de difusión en un espacio latente comprimido en lugar del espacio de píxeles, haciéndolo computacionalmente eficiente. Utiliza un codificador de texto CLIP para condicionar la generación en prompts de texto y una red U-Net de eliminación de ruido. Stable Diffusion puede ejecutarse en GPUs de consumo y ha generado un gran ecosistema de modelos ajustados finamente, adaptaciones LoRA y herramientas como AUTOMATIC1111 y ComfyUI.',
    'IA Generativa',
    '{"Modelo de Difusión","Espacio Latente","LoRA","DALL-E"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'token',
    'Token',
    'T',
    'La unidad básica de texto que procesan los modelos de lenguaje — aproximadamente una palabra o fragmento de palabra.',
    'Un token es la unidad fundamental de texto que procesan los modelos de lenguaje. La tokenización divide el texto en tokens usando algoritmos como Byte-Pair Encoding (BPE) o WordPiece. Un token es típicamente una palabra, subpalabra o carácter, dependiendo del tokenizador. Por ejemplo, ''tokenización'' podría dividirse en [''token'', ''ización'']. El número de tokens en un texto afecta el costo de procesamiento y el uso de la ventana de contexto. En promedio, 1 token ≈ 4 caracteres o 0,75 palabras en inglés. Los precios de las API para LLMs típicamente se basan en el conteo de tokens.',
    'Fundamentos',
    '{"Ventana de Contexto","Tokenización","Modelo de Lenguaje Grande","Embedding"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'transformer',
    'Transformer',
    'T',
    'La arquitectura de red neuronal dominante para PLN, basada completamente en mecanismos de atención.',
    'El Transformer es una arquitectura de red neuronal introducida en el artículo de 2017 ''Attention Is All You Need'' de Vaswani et al. en Google. Reemplazó las redes recurrentes (RNN, LSTM) con mecanismos de auto-atención, permitiendo el procesamiento paralelo de secuencias y una mejor captura de dependencias de largo alcance. La arquitectura consiste en pilas de codificadores y decodificadores, cada uno conteniendo auto-atención multi-cabeza y capas de retroalimentación con conexiones residuales y normalización de capas. Los Transformers son la base de prácticamente todos los grandes modelos de lenguaje, modelos de visión y modelos multimodales modernos.',
    'Arquitectura',
    '{"Auto-Atención","Mecanismo de Atención","BERT","GPT","Modelo de Lenguaje Grande"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'transfer-learning',
    'Aprendizaje por Transferencia',
    'A',
    'Aplicar conocimiento aprendido de una tarea o dominio para mejorar el rendimiento en otro.',
    'El aprendizaje por transferencia es un paradigma de aprendizaje automático donde un modelo entrenado en una tarea se adapta para una tarea diferente pero relacionada. En el aprendizaje profundo, esto típicamente implica usar pesos de modelos preentrenados como inicialización para una nueva tarea. El modelo preentrenado ha aprendido representaciones de características útiles que se transfieren entre tareas. El aprendizaje por transferencia reduce drásticamente los datos y el cómputo necesarios para nuevas tareas. Es la base del flujo de trabajo moderno de desarrollo de IA: preentrenar con datos grandes, ajustar finamente con datos específicos de la tarea.',
    'Entrenamiento',
    '{"Preentrenamiento","Ajuste Fino","Modelo Fundacional","Adaptación de Dominio"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'vector-database',
    'Base de Datos Vectorial',
    'B',
    'Base de datos optimizada para almacenar y buscar vectores de embedding de alta dimensión.',
    'Una base de datos vectorial es una base de datos especializada diseñada para almacenar, indexar y buscar eficientemente vectores de embedding de alta dimensión. A diferencia de las bases de datos tradicionales que buscan por coincidencia exacta, las bases de datos vectoriales utilizan algoritmos de vecino más cercano aproximado (ANN) (por ejemplo, HNSW, IVF) para encontrar vectores más similares a un vector de consulta. Son infraestructura esencial para sistemas RAG, búsqueda semántica, motores de recomendación y aplicaciones basadas en similitud. Las bases de datos vectoriales populares incluyen Pinecone, Weaviate, Qdrant, Chroma y pgvector (extensión de PostgreSQL).',
    'Infraestructura',
    '{"Embedding","Generación Aumentada por Recuperación","Búsqueda Semántica"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'zero-shot-learning',
    'Aprendizaje Zero-Shot',
    'A',
    'La capacidad de un modelo para realizar tareas en las que nunca fue entrenado explícitamente.',
    'El aprendizaje zero-shot se refiere a la capacidad de un modelo para realizar una tarea sin haber visto ningún ejemplo de esa tarea durante el entrenamiento o en el prompt. Los grandes modelos de lenguaje exhiben capacidades zero-shot porque su preentrenamiento en datos diversos les da amplio conocimiento y habilidades de razonamiento. Por ejemplo, GPT-4 puede traducir texto a un idioma para el que nunca fue entrenado explícitamente, o resolver acertijos lógicos novedosos. El rendimiento zero-shot es una medida clave de la capacidad de generalización de un modelo y se contrasta con el aprendizaje few-shot, donde se proporcionan un pequeño número de ejemplos.',
    'Paradigma de Aprendizaje',
    '{"Aprendizaje Few-Shot","Ingeniería de Prompts","Modelo de Lenguaje Grande","Generalización"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'agent',
    'Agente',
    'A',
    'Sistema de IA que percibe su entorno y toma acciones para alcanzar objetivos.',
    'En IA, un agente es cualquier software o programa que interactúa con el mundo (o una simulación) recibiendo entradas y produciendo salidas o acciones. Los agentes operan dentro de un entorno, perciben su estado a través de sensores o entradas de datos, y actúan sobre él mediante actuadores o llamadas a APIs. Van desde sistemas simples basados en reglas (como un termostato) hasta agentes autónomos complejos impulsados por grandes modelos de lenguaje que pueden planificar, usar herramientas, navegar por la web, escribir código y ejecutar tareas de múltiples pasos. Los sistemas de IA agéntica se usan cada vez más en automatización, robótica y asistentes de IA.',
    'Arquitectura',
    '{"Aprendizaje por Refuerzo","Modelo de Lenguaje Grande","Uso de Herramientas","IA Autónoma"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'generative-ai',
    'IA Generativa',
    'I',
    'IA que crea contenido nuevo como texto, imágenes, audio o video.',
    'La IA generativa se refiere a una clase de modelos de IA diseñados para producir datos originales que se asemejan a los ejemplos con los que fueron entrenados. Estos sistemas aprenden patrones estadísticos de grandes conjuntos de datos y usan esos patrones para generar nuevo contenido — incluyendo texto natural, imágenes realistas, música, video, código y modelos 3D. Las arquitecturas clave incluyen Transformers (para texto), Modelos de Difusión (para imágenes) y GANs. Ejemplos prominentes son GPT-4 (texto), DALL-E y Stable Diffusion (imágenes), Sora (video) y MusicLM (audio). La IA generativa ha transformado las industrias creativas, el desarrollo de software y la investigación científica.',
    'IA Generativa',
    '{"Modelo de Lenguaje Grande","Modelo de Difusión","Red Generativa Adversarial","Modelo Fundacional"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'context',
    'Contexto',
    'C',
    'Información que rodea una entrada y ayuda al modelo a interpretar el significado con precisión.',
    'En IA, el contexto se refiere a cualquier información de fondo relevante que ayuda a un modelo a entender o responder adecuadamente. Para los modelos de lenguaje, el contexto incluye el historial de conversación, las instrucciones del sistema, la intención del usuario, el tema, el tono y cualquier documento proporcionado en el prompt. La cantidad de contexto que un modelo puede usar está limitada por su ventana de contexto. El uso efectivo del contexto es fundamental para respuestas precisas y relevantes — un modelo con contexto insuficiente puede malinterpretar consultas ambiguas. La ingeniería de contexto — la práctica de estructurar las entradas de contexto de forma óptima — se ha convertido en una habilidad clave para construir aplicaciones de IA.',
    'Fundamentos',
    '{"Ventana de Contexto","Ingeniería de Prompts","Ingeniería de Contexto","Generación Aumentada por Recuperación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'claude',
    'Claude',
    'C',
    'Familia de modelos de asistente de IA desarrollados por Anthropic, conocidos por su seguridad y utilidad.',
    'Claude es una familia de grandes modelos de lenguaje desarrollados por Anthropic, una empresa de seguridad en IA. Los modelos Claude están diseñados con un fuerte énfasis en ser útiles, inofensivos y honestos (el marco ''HHH''). Se entrenan usando técnicas de IA Constitucional (CAI) y RLHF para alinear el comportamiento del modelo con los valores humanos. Claude destaca en tareas como resumen, razonamiento, asistencia en programación, análisis y escritura creativa. La familia de modelos Claude incluye múltiples niveles (Haiku, Sonnet, Opus) optimizados para diferentes equilibrios de velocidad y capacidad. Claude se usa ampliamente a través de API y en los productos de consumo de Anthropic.',
    'Modelo',
    '{"Modelo de Lenguaje Grande","RLHF","Alineación","OpenAI","GPT"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'openai',
    'OpenAI',
    'O',
    'Organización de investigación en IA que desarrolla sistemas avanzados incluyendo la serie GPT.',
    'OpenAI es un laboratorio de investigación en IA y empresa tecnológica fundada en 2015, con la misión de garantizar que la inteligencia artificial general (AGI) beneficie a toda la humanidad. Es responsable de desarrollar algunos de los sistemas de IA más influyentes, incluyendo la serie GPT de modelos de lenguaje, los modelos de generación de imágenes DALL-E, Codex (generación de código), Whisper (reconocimiento de voz) y el modelo de generación de video Sora. OpenAI también creó ChatGPT, una de las aplicaciones de IA más utilizadas en la historia. La organización opera como una empresa de beneficio limitado con una organización sin fines de lucro como matriz, equilibrando operaciones comerciales con investigación de seguridad.',
    'Organización',
    '{"GPT","ChatGPT","DALL-E","Modelo de Lenguaje Grande","AGI"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'chatgpt',
    'ChatGPT',
    'C',
    'Asistente de IA conversacional de OpenAI, construido sobre los grandes modelos de lenguaje GPT.',
    'ChatGPT es una aplicación de IA conversacional desarrollada por OpenAI, lanzada en noviembre de 2022. Está construida sobre la serie GPT de grandes modelos de lenguaje (inicialmente GPT-3.5, luego GPT-4) y ajustada finamente con RLHF para ser un asistente conversacional útil. ChatGPT puede mantener diálogos de múltiples turnos, responder preguntas, escribir y depurar código, redactar documentos, resumir texto, traducir idiomas y realizar muchas otras tareas de lenguaje. Se convirtió en una de las aplicaciones de consumo de más rápido crecimiento en la historia, alcanzando 100 millones de usuarios en dos meses. Admite plugins, entrada de imágenes (GPT-4V) y GPTs personalizados.',
    'Modelo',
    '{"GPT","OpenAI","RLHF","Modelo de Lenguaje Grande","Ajuste por Instrucciones"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'deep-learning',
    'Aprendizaje Profundo',
    'A',
    'Subconjunto del aprendizaje automático que usa redes neuronales con muchas capas para aprender de los datos.',
    'El aprendizaje profundo es un subcampo del aprendizaje automático que utiliza redes neuronales artificiales con muchas capas (de ahí ''profundo'') para aprender automáticamente representaciones jerárquicas de datos brutos. Cada capa aprende características cada vez más abstractas — por ejemplo, en el reconocimiento de imágenes, las capas tempranas detectan bordes, las capas intermedias detectan formas y las capas posteriores detectan objetos. El aprendizaje profundo ha impulsado avances en visión por computadora (CNNs), procesamiento de lenguaje natural (Transformers), reconocimiento de voz (RNNs, atención) e IA generativa (GANs, modelos de difusión). Requiere grandes conjuntos de datos y cómputo significativo, típicamente usando GPUs o TPUs.',
    'Fundamentos',
    '{"Red Neuronal","Transformer","Red Neuronal Convolucional","Retropropagación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'reinforcement-learning',
    'Aprendizaje por Refuerzo',
    'A',
    'Paradigma de aprendizaje donde un agente aprende recibiendo recompensas o penalizaciones por sus acciones.',
    'El aprendizaje por refuerzo (RL) es un paradigma de aprendizaje automático donde un agente aprende a tomar decisiones interactuando con un entorno. El agente observa el estado actual, toma una acción, recibe una señal de recompensa (positiva o negativa) y actualiza su política para maximizar la recompensa acumulada a lo largo del tiempo. Los algoritmos clave incluyen Q-learning, SARSA y métodos de gradiente de política como PPO y A3C. El RL ha logrado rendimiento sobrehumano en juegos (AlphaGo, Atari), robótica y conducción autónoma. En el contexto de los LLMs, el RLHF usa RL para alinear las salidas del modelo con las preferencias humanas.',
    'Paradigma de Aprendizaje',
    '{"RLHF","Agente","Aprendizaje Profundo","Gradiente de Política"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'computer-vision',
    'Visión por Computadora',
    'V',
    'IA que permite a las máquinas interpretar y comprender información visual de imágenes y videos.',
    'La visión por computadora es un campo de la IA centrado en permitir que las máquinas extraigan información significativa de entradas visuales como imágenes y videos. Las tareas principales incluyen clasificación de imágenes (identificar qué hay en una imagen), detección de objetos (localizar objetos), segmentación semántica (etiquetar cada píxel) y generación de imágenes. La visión por computadora moderna depende en gran medida de las redes neuronales convolucionales (CNNs) y los Vision Transformers (ViT). Las aplicaciones abarcan vehículos autónomos, imágenes médicas, reconocimiento facial, realidad aumentada, control de calidad en manufactura y análisis de imágenes satelitales.',
    'Campo',
    '{"Red Neuronal","Red Neuronal Convolucional","Aprendizaje Profundo","IA Multimodal"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'nlp',
    'PLN',
    'P',
    'Procesamiento de Lenguaje Natural — IA que permite a las computadoras entender y generar lenguaje humano.',
    'El Procesamiento de Lenguaje Natural (PLN) es una rama de la IA que combina lingüística, ciencias de la computación y aprendizaje automático para permitir que las computadoras procesen, comprendan y generen lenguaje humano. El PLN abarca una amplia gama de tareas: clasificación de texto, análisis de sentimientos, traducción automática, respuesta a preguntas, resumen, reconocimiento de entidades nombradas y sistemas de diálogo. El PLN moderno está dominado por modelos basados en Transformers como BERT y GPT. El PLN impulsa aplicaciones incluyendo motores de búsqueda, asistentes virtuales, chatbots, correctores gramaticales y sistemas de moderación de contenido.',
    'Campo',
    '{"Comprensión del Lenguaje Natural","Generación de Lenguaje Natural","Transformer","BERT","Modelo de Lenguaje Grande"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'supervised-learning',
    'Aprendizaje Supervisado',
    'A',
    'Aprendizaje automático donde los modelos se entrenan con pares de entrada-salida etiquetados.',
    'El aprendizaje supervisado es un paradigma de aprendizaje automático donde un modelo se entrena en un conjunto de datos de ejemplos etiquetados — pares de entradas y sus salidas correctas. El modelo aprende un mapeo de entradas a salidas minimizando la diferencia entre sus predicciones y las etiquetas verdaderas. Las tareas comunes incluyen clasificación (predecir una categoría) y regresión (predecir un valor continuo). Ejemplos incluyen detección de spam, clasificación de imágenes y predicción de precios. El aprendizaje supervisado requiere datos etiquetados, que pueden ser costosos de obtener, pero es el paradigma de ML más utilizado en sistemas de producción.',
    'Paradigma de Aprendizaje',
    '{"Aprendizaje No Supervisado","Ajuste Fino","Red Neuronal","Aprendizaje por Transferencia"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'unsupervised-learning',
    'Aprendizaje No Supervisado',
    'A',
    'Aprendizaje automático sobre datos no etiquetados para descubrir patrones o estructura ocultos.',
    'El aprendizaje no supervisado es un paradigma de aprendizaje automático donde los modelos aprenden patrones y estructura de los datos sin salidas etiquetadas. El modelo debe descubrir por sí solo la organización subyacente de los datos. Las técnicas comunes incluyen agrupamiento (agrupar puntos de datos similares, por ejemplo, K-means), reducción de dimensionalidad (por ejemplo, PCA, t-SNE, autoencoders) y estimación de densidad. El aprendizaje no supervisado es valioso cuando los datos etiquetados son escasos o costosos. El aprendizaje auto-supervisado — donde los modelos generan sus propias etiquetas a partir de datos no etiquetados — es una variante poderosa utilizada para preentrenar grandes modelos de lenguaje.',
    'Paradigma de Aprendizaje',
    '{"Aprendizaje Auto-Supervisado","Aprendizaje Supervisado","Agrupamiento","Preentrenamiento"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'data-mining',
    'Minería de Datos',
    'M',
    'El proceso de descubrir patrones e información útil en grandes conjuntos de datos.',
    'La minería de datos es el proceso de aplicar técnicas estadísticas, matemáticas y computacionales para extraer patrones significativos, correlaciones e información de grandes colecciones de datos. Se sitúa en la intersección de la estadística, el aprendizaje automático y los sistemas de bases de datos. Las tareas comunes de minería de datos incluyen clasificación, agrupamiento, aprendizaje de reglas de asociación (por ejemplo, análisis de cesta de mercado), detección de anomalías y regresión. La minería de datos es fundamental para la inteligencia empresarial, la detección de fraudes, el descubrimiento científico y los sistemas de recomendación. La minería de datos moderna aprovecha cada vez más las técnicas de aprendizaje automático e IA.',
    'Ciencia de Datos',
    '{"Aprendizaje Automático","Aprendizaje Supervisado","Aprendizaje No Supervisado","Reconocimiento de Patrones"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'entity-annotation',
    'Anotación de Entidades',
    'A',
    'Etiquetar entidades significativas (nombres, lugares, fechas) en texto o datos para el entrenamiento de IA.',
    'La anotación de entidades es el proceso de marcar entidades — como nombres de personas, organizaciones, ubicaciones, fechas y nombres de productos — en conjuntos de datos de texto para que los modelos de IA puedan aprender a reconocerlos. Es un paso crítico en la creación de datos de entrenamiento para sistemas de Reconocimiento de Entidades Nombradas (NER) y otras tareas de PLN. La anotación puede realizarse manualmente por anotadores humanos o de forma semi-automática usando modelos preentrenados. La anotación de entidades de alta calidad es esencial para entrenar sistemas precisos de extracción de información utilizados en búsqueda, grafos de conocimiento y pipelines de procesamiento de documentos.',
    'PLN',
    '{"Extracción de Entidades","Reconocimiento de Entidades Nombradas","PLN","Aprendizaje Supervisado"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'entity-extraction',
    'Extracción de Entidades',
    'E',
    'Identificación y categorización automática de entidades clave en texto no estructurado.',
    'La extracción de entidades, también conocida como Reconocimiento de Entidades Nombradas (NER), es una tarea de PLN donde un modelo de IA identifica y clasifica automáticamente entidades nombradas en texto no estructurado en categorías predefinidas como personas, organizaciones, ubicaciones, fechas, valores monetarios y más. Por ejemplo, en la oración ''Apple fue fundada por Steve Jobs en Cupertino en 1976'', un modelo NER extraería Apple (organización), Steve Jobs (persona), Cupertino (ubicación) y 1976 (fecha). La extracción de entidades es fundamental para la recuperación de información, la construcción de grafos de conocimiento y la inteligencia documental.',
    'PLN',
    '{"Anotación de Entidades","PLN","Comprensión del Lenguaje Natural","Extracción de Información"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'intent',
    'Intención',
    'I',
    'El objetivo o propósito detrás de la entrada de un usuario en un sistema de IA conversacional.',
    'En la IA conversacional y el PLN, la intención se refiere al objetivo o propósito subyacente que un usuario pretende lograr con su entrada. Por ejemplo, la consulta ''¿Cómo estará el tiempo mañana?'' tiene la intención ''obtener pronóstico del tiempo''. El reconocimiento de intención (o clasificación de intención) es la tarea de identificar automáticamente la intención del usuario a partir de su enunciado. Es un componente central de los sistemas de diálogo, asistentes virtuales y chatbots. Los sistemas modernos usan clasificadores de aprendizaje automático o grandes modelos de lenguaje para detectar la intención, permitiendo el enrutamiento y la generación de respuestas apropiadas.',
    'PLN',
    '{"Comprensión del Lenguaje Natural","PLN","Sistema de Diálogo","Extracción de Entidades"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'model',
    'Modelo',
    'M',
    'Sistema matemático entrenado con datos para hacer predicciones, clasificaciones o generar salidas.',
    'En IA y aprendizaje automático, un modelo es un sistema computacional que ha aprendido patrones de datos de entrenamiento y puede aplicar ese conocimiento a nuevas entradas. Los modelos se definen por su arquitectura (la estructura del cómputo) y sus parámetros (los pesos aprendidos). Después del entrenamiento, un modelo puede hacer predicciones (regresión, clasificación), generar contenido (lenguaje, imágenes) o tomar acciones (agentes). El término abarca desde simples modelos de regresión lineal hasta redes neuronales con miles de millones de parámetros. La selección, entrenamiento, evaluación y despliegue de modelos son las etapas centrales del ciclo de vida del aprendizaje automático.',
    'Fundamentos',
    '{"Red Neuronal","Entrenamiento","Inferencia","Modelo Fundacional"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'nlu',
    'Comprensión del Lenguaje Natural',
    'C',
    'IA que interpreta el significado, la intención y el contexto del lenguaje humano.',
    'La Comprensión del Lenguaje Natural (CLN/NLU) es un subcampo del PLN centrado en permitir que las máquinas comprendan el significado, la intención, el sentimiento y el contexto del lenguaje humano — yendo más allá del procesamiento superficial de texto. Las tareas de CLN incluyen reconocimiento de intención, análisis de sentimientos, etiquetado de roles semánticos, resolución de correferencias y comprensión lectora. La CLN es el componente de ''comprensión'' de los sistemas de IA conversacional, permitiéndoles interpretar correctamente lo que los usuarios quieren decir en lugar de solo lo que dicen. La CLN moderna está impulsada por grandes modelos de lenguaje preentrenados como BERT y sus variantes.',
    'PLN',
    '{"PLN","Generación de Lenguaje Natural","Intención","BERT","Análisis de Sentimientos"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'nlg',
    'Generación de Lenguaje Natural',
    'G',
    'IA que produce texto o voz coherente y similar al humano a partir de datos u otras entradas.',
    'La Generación de Lenguaje Natural (GLN/NLG) es un subcampo del PLN centrado en producir automáticamente texto o voz coherente, fluido y contextualmente apropiado a partir de datos estructurados, conocimiento u otras entradas. Las tareas de GLN incluyen resumen de texto, generación de informes, generación de respuestas en diálogos, traducción automática y escritura creativa. La GLN moderna está dominada por modelos de lenguaje autorregresivos como GPT-4, que generan texto token por token. La GLN es el componente de ''generación'' de la IA conversacional y se usa en chatbots, periodismo automatizado, sistemas de datos a texto y asistentes virtuales.',
    'PLN',
    '{"PLN","Comprensión del Lenguaje Natural","Modelo de Lenguaje Grande","Modelo Autorregresivo"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'overfitting',
    'Sobreajuste',
    'S',
    'Cuando un modelo memoriza los datos de entrenamiento con demasiada precisión y no generaliza a datos nuevos.',
    'El sobreajuste ocurre cuando un modelo de aprendizaje automático aprende los datos de entrenamiento con demasiada precisión — incluyendo su ruido y fluctuaciones aleatorias — en lugar de los patrones generales subyacentes. Un modelo sobreajustado funciona muy bien en los datos de entrenamiento pero mal en datos de prueba no vistos, porque esencialmente ha memorizado los ejemplos de entrenamiento en lugar de aprender patrones transferibles. El sobreajuste es más probable con modelos complejos y conjuntos de datos pequeños. Las estrategias comunes de mitigación incluyen regularización (L1/L2), dropout, parada temprana, aumento de datos y validación cruzada.',
    'Fundamentos',
    '{"Aprendizaje Supervisado","Regularización","Generalización","Red Neuronal"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'pattern-recognition',
    'Reconocimiento de Patrones',
    'R',
    'La capacidad de los algoritmos para identificar estructuras o regularidades recurrentes en los datos.',
    'El reconocimiento de patrones es la capacidad de los algoritmos y sistemas de IA para detectar, clasificar y responder a estructuras, regularidades o relaciones recurrentes en los datos. Es una de las tareas fundamentales de la IA y el aprendizaje automático, subyacente a aplicaciones como el reconocimiento de imágenes (detectar rostros u objetos), el reconocimiento de voz (identificar fonemas y palabras), el reconocimiento de escritura a mano, la detección de anomalías y la identificación biométrica. El reconocimiento de patrones moderno se logra en gran medida mediante el aprendizaje profundo, donde las redes neuronales aprenden automáticamente representaciones de características jerárquicas de datos brutos.',
    'Fundamentos',
    '{"Aprendizaje Profundo","Visión por Computadora","Red Neuronal","Clasificación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'context-engineering',
    'Ingeniería de Contexto',
    'I',
    'Estructurar y optimizar la información proporcionada a los modelos de IA para mejorar la calidad de la salida.',
    'La ingeniería de contexto es la práctica de diseñar y estructurar deliberadamente la información proporcionada a un modelo de IA — incluyendo prompts del sistema, historial de conversación, documentos recuperados, ejemplos y datos del entorno — para maximizar la calidad y relevancia de sus salidas. Va más allá de la ingeniería de prompts básica para abarcar la arquitectura de información completa alrededor de una llamada al modelo: qué incluir, cómo formatearlo, qué recuperar y cómo priorizar. A medida que los sistemas de IA se vuelven más capaces, la ingeniería de contexto ha surgido como una disciplina crítica para construir aplicaciones de IA confiables y precisas.',
    'Prompting',
    '{"Ingeniería de Prompts","Generación Aumentada por Recuperación","Ventana de Contexto","Contexto"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'turing-test',
    'Test de Turing',
    'T',
    'Prueba propuesta por Alan Turing para evaluar si el comportamiento de una máquina es indistinguible del de un humano.',
    'El Test de Turing, propuesto por el matemático Alan Turing en su artículo de 1950 ''Computing Machinery and Intelligence'', es una prueba de la capacidad de una máquina para exhibir comportamiento inteligente indistinguible del de un humano. En la formulación original (el Juego de Imitación), un evaluador humano conversa por texto con un humano y una máquina sin saber cuál es cuál; si el evaluador no puede distinguir de manera confiable la máquina del humano, se dice que la máquina ha pasado la prueba. Aunque influyente como referencia filosófica, el Test de Turing ahora se considera insuficiente como medida de la verdadera inteligencia de la IA, ya que los LLMs modernos pueden pasarlo sin poseer comprensión genuina.',
    'Fundamentos',
    '{"Modelo de Lenguaje Grande","AGI","IA Estrecha","ChatGPT"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'narrow-ai',
    'IA Estrecha',
    'I',
    'IA diseñada para realizar un conjunto específico y limitado de tareas — también llamada IA Débil.',
    'La IA Estrecha (también llamada IA Débil o Inteligencia Artificial Estrecha, ANI) se refiere a sistemas de IA diseñados y entrenados para realizar una tarea específica o un conjunto limitado de tareas relacionadas. A diferencia de la hipotética Inteligencia Artificial General (AGI), la IA estrecha no puede transferir su conocimiento a dominios fuera de su entrenamiento. Ejemplos incluyen clasificadores de imágenes, filtros de spam, motores de recomendación, programas de ajedrez y sistemas de reconocimiento de voz. A pesar de la etiqueta ''estrecha'', los sistemas modernos de IA estrecha como GPT-4 pueden desempeñarse de manera impresionante en muchas tareas de lenguaje, difuminando la línea con capacidades más generales.',
    'Fundamentos',
    '{"AGI","Test de Turing","Modelo de Lenguaje Grande","Modelo Fundacional"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'spec-driven-development',
    'Desarrollo Guiado por Especificaciones',
    'D',
    'Metodología de desarrollo donde especificaciones detalladas guían el diseño, la implementación y las pruebas.',
    'El desarrollo guiado por especificaciones (DGE) es una metodología de ingeniería de software en la que se escriben especificaciones claras y estructuradas antes de que comience la implementación. Estas especificaciones definen el comportamiento esperado, entradas, salidas, casos extremos y criterios de aceptación. En el contexto de los sistemas de IA, el DGE es cada vez más importante para definir cómo deben comportarse los componentes de IA, qué salidas son aceptables y cómo evaluar la corrección. Se alinea estrechamente con el desarrollo guiado por pruebas (TDD) y el desarrollo guiado por comportamiento (BDD), y está ganando tracción como forma de construir aplicaciones de IA más confiables y auditables.',
    'Ingeniería',
    '{"Ingeniería de Contexto","Ingeniería de Prompts","Alineación","Evaluación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'nlu-nlg-nlp',
    'CLN / GLN / PLN',
    'C',
    'Los tres pilares de la IA del lenguaje: Comprensión, Generación y Procesamiento.',
    'La CLN (Comprensión del Lenguaje Natural), la GLN (Generación de Lenguaje Natural) y el PLN (Procesamiento de Lenguaje Natural) son tres subcampos estrechamente relacionados pero distintos de la IA del lenguaje. El PLN es el término más amplio, que abarca todas las técnicas computacionales para procesar el lenguaje humano. La CLN se centra específicamente en la comprensión — extraer significado, intención y estructura del texto. La GLN se centra en la producción — generar lenguaje coherente y contextualmente apropiado a partir de datos o conocimiento. Los grandes modelos de lenguaje modernos como GPT-4 integran las tres capacidades: procesan la entrada (PLN), la comprenden (CLN) y generan respuestas (GLN) en una arquitectura unificada.',
    'PLN',
    '{"PLN","Comprensión del Lenguaje Natural","Generación de Lenguaje Natural","Modelo de Lenguaje Grande"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'tokenization',
    'Tokenización',
    'T',
    'El proceso de convertir texto en tokens que un LLM puede procesar.',
    'La tokenización es el paso de preprocesamiento en el que el texto crudo se divide en tokens: las unidades atómicas que un LLM entiende. Los tokenizadores como Byte-Pair Encoding (BPE) o SentencePiece dividen el texto en unidades de subpalabras, equilibrando el tamaño del vocabulario con la cobertura. Diferentes modelos usan diferentes tokenizadores con diferentes vocabularios, por lo que el mismo texto puede producir conteos de tokens distintos entre modelos.',
    'Fundamentos',
    '{"Token","Ventana de Contexto","Token EOS"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'eos-token',
    'Token EOS (Fin de Secuencia)',
    'T',
    'Un token especial que indica el fin de la generación de un modelo.',
    'El token de Fin de Secuencia (EOS) es un token especial que indica que un LLM ha completado su generación. Cada familia de modelos usa su propio token EOS; por ejemplo, GPT-4 usa <|endoftext|>, Llama 3 usa <|eot_id|> y SmolLM2 usa <|im_end|>. El modelo deja de generar una vez que predice este token. Los tokens EOS forman parte de un conjunto más amplio de tokens especiales que estructuran las entradas y salidas del modelo.',
    'Fundamentos',
    '{"Token","Modelo Autorregresivo","Tokens Especiales"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'special-tokens',
    'Tokens Especiales',
    'T',
    'Tokens reservados usados para estructurar las entradas y salidas de un LLM, como marcar el inicio o fin de mensajes.',
    'Los tokens especiales son tokens reservados en el vocabulario de un LLM que tienen significado estructural en lugar de contenido lingüístico. Se usan para delimitar el inicio o fin de una secuencia, separar instrucciones del sistema de mensajes del usuario, o señalar el uso de herramientas y llamadas a funciones. Diferentes modelos usan diferentes conjuntos de tokens especiales, lo que hace que la migración de prompts entre modelos no sea trivial.',
    'Fundamentos',
    '{"Token EOS","Token","Prompt de Sistema","Plantilla de Chat"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'encoder',
    'Codificador (Encoder)',
    'C',
    'Un tipo de Transformer que convierte texto en representaciones vectoriales densas (embeddings).',
    'Un codificador es una variante del Transformer que procesa una secuencia de entrada y produce una representación vectorial densa (embedding) de esa entrada. Los modelos basados en codificadores como BERT son adecuados para tareas como clasificación de texto, búsqueda semántica y Reconocimiento de Entidades Nombradas (NER). A diferencia de los decodificadores, los codificadores no generan nuevo texto token a token; en cambio, producen representaciones de tamaño fijo de toda la entrada.',
    'Fundamentos',
    '{"Decodificador","Transformer","Embedding","Búsqueda Semántica"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'decoder',
    'Decodificador (Decoder)',
    'D',
    'Un tipo de Transformer diseñado para generar nuevos tokens, uno a la vez, para tareas como la generación de texto.',
    'Un decodificador es una variante del Transformer que genera texto nuevo prediciendo un token a la vez, condicionado por todos los tokens anteriores. Los modelos solo de decodificador como GPT-4, Llama y Mistral son la arquitectura más común para los LLM modernos. Se utilizan para generación de texto, chatbots, generación de código y razonamiento.',
    'Fundamentos',
    '{"Codificador","Transformer","Modelo Autorregresivo","Modelo de Lenguaje Grande"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'prompt',
    'Prompt',
    'P',
    'El texto de entrada proporcionado a un LLM para guiar su respuesta.',
    'Un prompt es el texto de entrada que se pasa a un LLM para instruirlo o guiar su generación. Los prompts pueden incluir instrucciones, ejemplos, contexto, historial de conversación, documentos recuperados y directivas de nivel sistema. La redacción, estructura y contenido de un prompt afectan significativamente la calidad y relevancia de la salida del modelo. El diseño del prompt es una de las formas más accesibles y poderosas de mejorar el rendimiento de los LLM sin reentrenamiento.',
    'Prompting',
    '{"Ingeniería de Prompts","Prompt de Sistema","Cadena de Pensamiento","Ingeniería de Contexto"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'system-prompt',
    'Prompt de Sistema',
    'P',
    'Un prompt especial que configura el comportamiento, la personalidad y las restricciones de un LLM antes de la conversación.',
    'Un prompt de sistema es un bloque de instrucciones que se pasa a un LLM antes de cualquier mensaje del usuario, utilizado para configurar el comportamiento, la personalidad, el tono y las restricciones del modelo. Típicamente contiene descripciones de roles, requisitos de formato de salida, reglas de seguridad e instrucciones específicas de la tarea. Los prompts de sistema son una herramienta principal para personalizar el comportamiento de los LLM en aplicaciones de producción.',
    'Prompting',
    '{"Ingeniería de Prompts","Prompt","Plantilla de Chat","Ingeniería de Contexto"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'few-shot-prompting',
    'Prompting de Pocos Ejemplos (Few-Shot)',
    'P',
    'Una técnica de prompting donde se incluyen unos pocos ejemplos de entrada-salida en el prompt para guiar al modelo.',
    'El prompting de pocos ejemplos (también llamado n-shot prompting) es una técnica donde un pequeño número de ejemplos de entrada-salida se integran directamente en el prompt para demostrar el formato y estilo de salida deseados. Aprovecha la capacidad de aprendizaje en contexto del LLM: el modelo infiere el patrón a partir de los ejemplos y lo aplica a nuevas entradas. Como regla general, proporcionar al menos 5 ejemplos ayuda al modelo a generalizar.',
    'Prompting',
    '{"Aprendizaje en Contexto","Prompting Sin Ejemplos","Ingeniería de Prompts","Cadena de Pensamiento"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'in-context-learning',
    'Aprendizaje en Contexto',
    'A',
    'La capacidad de un LLM de aprender una nueva tarea a partir de ejemplos dados directamente en el prompt, sin actualizar pesos.',
    'El aprendizaje en contexto es la capacidad de los LLM de adaptarse a nuevas tareas o comportamientos basándose únicamente en ejemplos e instrucciones proporcionados dentro del prompt, sin ningún cambio en los pesos subyacentents del modelo. Esto contrasta con el ajuste fino, que requiere actualizar los parámetros del modelo. El aprendizaje en contexto es habilitado por la capacidad del modelo de detectar y generalizar patrones a partir de los pocos ejemplos que ve.',
    'Prompting',
    '{"Prompting de Pocos Ejemplos","Prompting Sin Ejemplos","Ajuste Fino","Ingeniería de Prompts"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'zero-shot-prompting',
    'Prompting Sin Ejemplos (Zero-Shot)',
    'P',
    'Pedir a un LLM que realice una tarea solo con instrucciones, sin proporcionar ejemplos.',
    'El prompting sin ejemplos (zero-shot) es una técnica donde se pide a un LLM que realice una tarea basándose únicamente en una instrucción en lenguaje natural, sin proporcionar ejemplos de entrada-salida. El modelo se basa completamente en su conocimiento preentrenado y su capacidad de seguir instrucciones. Si bien es simple y flexible, el rendimiento sin ejemplos puede ser menos fiable que el prompting con pocos ejemplos para tareas complejas o especializadas.',
    'Prompting',
    '{"Prompting de Pocos Ejemplos","Aprendizaje en Contexto","Ingeniería de Prompts"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'semantic-search',
    'Búsqueda Semántica',
    'B',
    'Una técnica de búsqueda que encuentra documentos basándose en el significado e intención, no en coincidencias exactas de palabras.',
    'La búsqueda semántica es un enfoque de recuperación que encuentra documentos relevantes comparando el significado de una consulta con el significado de los documentos, usando vectores de embeddings y métricas de similitud. A diferencia de la búsqueda por palabras clave, la búsqueda semántica puede encontrar resultados relevantes incluso cuando las palabras exactas de la consulta no aparecen en el documento. Es un componente clave de los pipelines de RAG.',
    'Recuperación',
    '{"Embedding","Base de Datos Vectorial","Búsqueda Híbrida","Generación Aumentada por Recuperación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'hybrid-search',
    'Búsqueda Híbrida',
    'B',
    'Un método de recuperación que combina búsqueda semántica (vectorial) con búsqueda tradicional por palabras clave para mejores resultados.',
    'La búsqueda híbrida combina la búsqueda semántica (usando similitud de embeddings) con la búsqueda basada en palabras clave (como BM25 o TF-IDF) para recuperar documentos. Los dos rankings se fusionan típicamente usando técnicas como la Fusión de Rango Recíproco (RRF). La búsqueda híbrida frecuentemente supera a cualquiera de los métodos por sí solos, especialmente para consultas factuales precisas o terminología especializada. Es considerada una mejor práctica en sistemas RAG de producción.',
    'Recuperación',
    '{"Búsqueda Semántica","Generación Aumentada por Recuperación","Base de Datos Vectorial"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'grounding',
    'Anclaje (Grounding)',
    'A',
    'Anclar las salidas de los LLM a fuentes externas verificables para mejorar la precisión factual.',
    'El anclaje (grounding) es la práctica de conectar las salidas generadas por un LLM a fuentes externas confiables como documentos recuperados, bases de datos o datos en tiempo real. Una respuesta anclada es trazable hasta una fuente, reduciendo el riesgo de alucinación. RAG es la técnica de anclaje más común. Es especialmente importante en aplicaciones de IA de alto riesgo como sistemas médicos, legales o financieros donde la precisión factual es crítica.',
    'Evaluación',
    '{"Alucinación","Generación Aumentada por Recuperación","Guardianes"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'tool-use',
    'Uso de Herramientas (Tool Use)',
    'U',
    'La capacidad de un LLM o agente de IA de llamar funciones externas o APIs para ampliar sus capacidades.',
    'El uso de herramientas (también llamado llamada a funciones) es la capacidad de un LLM de invocar herramientas o APIs externas como parte de su proceso de razonamiento. Las herramientas pueden incluir búsqueda web, intérpretes de código, consultas a bases de datos, calculadoras y APIs personalizadas. El modelo recibe una descripción de las herramientas disponibles, decide cuál llamar y con qué argumentos, y procesa la salida de la herramienta para informar su siguiente acción.',
    'Agentes',
    '{"Agente de IA","Llamada a Funciones","Bucle Agéntico","ReAct","MCP"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'react',
    'ReAct (Razonar + Actuar)',
    'R',
    'Un marco donde un LLM intercala pasos de razonamiento con acciones para resolver tareas de forma iterativa.',
    'ReAct es un marco de agente que intercala el razonamiento (pensar qué hacer a continuación) con actuar (llamar a herramientas o tomar acciones) en un bucle iterativo. El modelo produce un Pensamiento describiendo su razonamiento, luego una Acción especificando una llamada a herramienta, luego una Observación reportando el resultado, y repite hasta completar la tarea. ReAct mejora los enfoques de razonamiento puro al anclar los pensamientos del agente en observaciones reales del entorno.',
    'Agentes',
    '{"Agente de IA","Uso de Herramientas","Bucle Agéntico","Cadena de Pensamiento","Ciclo Pensamiento-Acción-Observación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'agentic-loop',
    'Bucle Agéntico',
    'B',
    'El ciclo iterativo de razonamiento, acción y observación que impulsa el comportamiento de un agente de IA.',
    'El bucle agéntico es el ciclo operacional central de un agente de IA: el agente recibe un objetivo u observación, razona sobre la mejor siguiente acción, ejecuta esa acción (p. ej., llama a una herramienta), recibe una observación del resultado, actualiza su comprensión y repite el ciclo hasta lograr el objetivo o determinar que no puede avanzar. Este bucle permite a los agentes abordar tareas complejas de múltiples pasos que no pueden resolverse en una sola llamada al LLM.',
    'Agentes',
    '{"Agente de IA","ReAct","Uso de Herramientas","Ciclo Pensamiento-Acción-Observación","Orquestación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'multi-agent-system',
    'Sistema Multi-Agente',
    'S',
    'Un sistema compuesto por múltiples agentes de IA que colaboran para realizar tareas complejas.',
    'Un sistema multi-agente es una arquitectura donde múltiples agentes de IA — cada uno con su propio rol, herramientas y capacidades — trabajan juntos (o en una jerarquía estructurada) para realizar tareas demasiado complejas para un solo agente. Los patrones comunes incluyen un agente coordinador/orquestador que delega subtareas a agentes trabajadores especializados. Los sistemas multi-agente pueden mejorar el paralelismo, la especialización y la robustez, pero introducen complejidad de coordinación.',
    'Agentes',
    '{"Agente de IA","Orquestación","Bucle Agéntico","Uso de Herramientas"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'orchestration',
    'Orquestación',
    'O',
    'La coordinación de llamadas a LLM, invocaciones de herramientas y flujos de datos para construir pipelines de IA complejos.',
    'La orquestación se refiere al diseño y gestión de secuencias de llamadas a LLM, invocaciones de herramientas, recuperaciones de memoria y transformaciones de datos que implementan colectivamente un flujo de trabajo o agente de IA complejo. Frameworks de orquestación como LangChain, LlamaIndex y smolagents proporcionan abstracciones para construir estos pipelines. Un buen diseño de orquestación prioriza el determinismo, la observabilidad y el manejo elegante de errores.',
    'Agentes',
    '{"Agente de IA","Sistema Multi-Agente","LLMOps","Uso de Herramientas","Bucle Agéntico"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'memory',
    'Memoria',
    'M',
    'Mecanismos que permiten a los agentes de IA almacenar y recuperar información a través de múltiples pasos o sesiones.',
    'La memoria en sistemas de agentes de IA se refiere a mecanismos que permiten que la información persista y se recupere más allá de la ventana de contexto inmediata. Los tipos comunes de memoria incluyen: memoria a corto plazo (la ventana de contexto actual), memoria episódica (registros de interacciones pasadas), memoria semántica (conocimiento recuperado de almacenes vectoriales) y memoria procedimental (habilidades o planes aprendidos). La gestión efectiva de la memoria es esencial para agentes que manejan tareas largas.',
    'Agentes',
    '{"Agente de IA","Ventana de Contexto","Generación Aumentada por Recuperación","Bucle Agéntico"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'mcp',
    'Protocolo de Contexto de Modelo (MCP)',
    'P',
    'Un protocolo estándar abierto para conectar modelos de IA con herramientas externas y fuentes de datos.',
    'El Protocolo de Contexto de Modelo (MCP) es un protocolo abierto que estandariza cómo los modelos y agentes de IA se conectan con herramientas externas, APIs y fuentes de datos. Desarrollado por Anthropic y soportado por frameworks como Kiro, MCP permite a los sistemas de IA acceder a contexto desde diversas fuentes externas — bases de datos, sistemas de archivos, APIs — a través de una interfaz consistente. Los servidores MCP exponen capacidades que los agentes pueden descubrir y usar.',
    'Agentes',
    '{"Uso de Herramientas","Agente de IA","Llamada a Funciones","Orquestación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'function-calling',
    'Llamada a Funciones (Function Calling)',
    'L',
    'Una capacidad del modelo para generar llamadas estructuradas a funciones o APIs predefinidas como parte de su salida.',
    'La llamada a funciones (también conocida como uso de herramientas) es una capacidad en la que un LLM genera JSON estructurado o código para invocar una función o API externa predefinida, en lugar de generar texto sin formato. El modelo recibe un esquema que describe las funciones disponibles y sus parámetros, decide cuál función llamar y con qué argumentos, y devuelve una llamada estructurada que una aplicación puede ejecutar. La llamada a funciones es esencial para construir agentes confiables.',
    'Agentes',
    '{"Uso de Herramientas","Agente de IA","MCP","Salida Estructurada"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'structured-output',
    'Salida Estructurada',
    'S',
    'Restringir la salida de un LLM a un esquema definido (p. ej., JSON) para facilitar el procesamiento posterior.',
    'La salida estructurada es la práctica de restringir la generación de un LLM para que siga un formato específico — como JSON, XML o un esquema tipado — en lugar de producir texto libre. Esto simplifica el análisis posterior, la integración con otros sistemas y la validación automática. Las técnicas incluyen instrucciones de formato basadas en prompts, parsers de salida y características nativas del modelo como el modo JSON o el muestreo con restricciones gramaticales.',
    'Prompting',
    '{"Llamada a Funciones","Uso de Herramientas","Ingeniería de Prompts","Guardianes"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'guardrails',
    'Guardianes (Guardrails)',
    'G',
    'Reglas y mecanismos que restringen o validan las entradas y salidas de los LLM para garantizar un comportamiento seguro y apropiado.',
    'Los guardianes son mecanismos de seguridad y calidad aplicados a las entradas o salidas de los LLM para hacer cumplir restricciones como el cumplimiento de políticas de contenido, la validez del formato de salida, la precisión factual y la adherencia a instrucciones. Pueden implementarse como instrucciones basadas en prompts, clasificadores de salida, filtros basados en reglas o modelos de validación separados. Los guardianes son un componente central de los sistemas de IA de producción responsables.',
    'Evaluación',
    '{"Alucinación","LLM como Juez","Salida Estructurada","Anclaje","Seguridad"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'evals',
    'Evals (Evaluaciones)',
    'E',
    'Pruebas y benchmarks usados para medir el rendimiento de LLMs o sistemas de IA en tareas específicas.',
    'Los evals (abreviatura de evaluaciones) son pruebas sistemáticas diseñadas para medir el rendimiento, la precisión, la seguridad y la confiabilidad de los LLMs o sistemas de IA. Van desde pruebas unitarias sobre pares específicos de entrada-salida hasta grandes benchmarks que cubren capacidades diversas. Conjuntos de evals sólidos son fundamentales para construir productos de IA confiables: permiten iteración rápida, detectan regresiones y brindan confianza antes del despliegue.',
    'Evaluación',
    '{"LLM como Juez","Guardianes","Alucinación","Benchmarks"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'llmops',
    'LLMOps',
    'L',
    'Las prácticas operacionales para desplegar, monitorear, versionar y mantener aplicaciones basadas en LLMs en producción.',
    'LLMOps es el conjunto de prácticas, herramientas y cultura para operacionalizar aplicaciones basadas en LLMs — análogo a MLOps para el aprendizaje automático tradicional. Cubre el versionado de prompts, el versionado de modelos, las pruebas A/B, el monitoreo (seguimiento de la calidad de entrada-salida a lo largo del tiempo), el registro, la optimización de latencia y la gestión de costos. El objetivo principal de LLMOps es permitir ciclos de iteración más rápidos y mantener la confiabilidad en producción.',
    'Operaciones',
    '{"Evals","Monitoreo","Orquestación","Versionado de Prompts"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'temperature',
    'Temperatura',
    'T',
    'Un parámetro de muestreo que controla la aleatoriedad y creatividad de las salidas de los LLM.',
    'La temperatura es un parámetro de decodificación que controla la diversidad de la salida de un LLM escalando la distribución de probabilidad sobre los tokens antes del muestreo. Una temperatura de 0 hace al modelo determinista (siempre elige el token de mayor probabilidad), mientras que valores más altos (p. ej., 0.7–1.0) introducen más aleatoriedad y creatividad. Elegir la temperatura correcta depende de la tarea: baja para tareas factuales, más alta para generación creativa.',
    'Inferencia',
    '{"Muestreo","Muestreo Top-P","Inferencia","Estrategia de Decodificación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'top-p-sampling',
    'Muestreo Top-P (Muestreo de Núcleo)',
    'M',
    'Una estrategia de muestreo que selecciona del conjunto más pequeño de tokens cuya probabilidad acumulada supera p.',
    'El muestreo Top-P, también llamado muestreo de núcleo, es una estrategia de decodificación donde el modelo muestrea solo del conjunto más pequeño de tokens cuya masa de probabilidad acumulada supera un umbral p (p. ej., 0.9). Esto asegura que los tokens improbables queden excluidos del muestreo mientras se permite cierta variabilidad. Top-P se usa a menudo junto con la temperatura para controlar la diversidad de la salida.',
    'Inferencia',
    '{"Temperatura","Muestreo","Estrategia de Decodificación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'beam-search',
    'Búsqueda de Haz (Beam Search)',
    'B',
    'Una estrategia de decodificación que explora múltiples secuencias de tokens simultáneamente para encontrar la salida de mayor probabilidad.',
    'La búsqueda de haz es una estrategia de decodificación que mantiene un número fijo (el ''ancho del haz'') de secuencias candidatas en cada paso de generación, expandiendo cada candidato con los tokens más probables y manteniendo solo los mejores candidatos. A diferencia de la decodificación codiciosa, la búsqueda de haz puede encontrar secuencias de mayor calidad general al explorar alternativas. Es comúnmente usada en traducción y resumen.',
    'Inferencia',
    '{"Temperatura","Muestreo Top-P","Estrategia de Decodificación","Modelo Autorregresivo"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'caching',
    'Caché (Caching)',
    'C',
    'Almacenar salidas de LLM o cómputos intermedios para evitar llamadas redundantes y costosas al modelo.',
    'El caché en sistemas de LLMs se refiere a almacenar los resultados de llamadas al modelo o cómputos intermedios para que puedan reutilizarse sin volver a ejecutar el modelo. El caché de prompts puede reducir significativamente la latencia y los costos de API para aplicaciones con prompts fijos grandes o repetitivos. El caché semántico va más lejos al recuperar respuestas cacheadas para consultas semánticamente similares. El caché es una optimización subutilizada pero de alto impacto.',
    'Operaciones',
    '{"LLMOps","Inferencia","Latencia","Ingeniería de Prompts"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'alignment',
    'Alineación (Alignment)',
    'A',
    'El proceso de asegurar que el comportamiento del modelo de IA coincide con los valores, intenciones y preferencias humanas.',
    'La alineación se refiere al desafío y la práctica de asegurar que los sistemas de IA se comporten de manera consistente con los valores, intenciones y preferencias humanas, especialmente a medida que los modelos se vuelven más capaces y autónomos. Las técnicas de alineación incluyen RLHF, IA Constitucional y optimización directa de preferencias (DPO). Una IA desalineada puede producir comportamientos dañinos, engañosos o no deseados incluso cuando es técnicamente capaz.',
    'Seguridad',
    '{"RLHF","Seguridad","Guardianes","IA Constitucional"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'safety',
    'Seguridad (Safety)',
    'S',
    'Prácticas y mecanismos para evitar que los modelos de IA generen contenido dañino, peligroso o inapropiado.',
    'La seguridad en IA se refiere a las prácticas, técnicas y directrices diseñadas para evitar que los modelos de IA produzcan contenido o tomen acciones que sean dañinas, ilegales, engañosas o peligrosas. Las medidas de seguridad en aplicaciones de LLMs incluyen clasificadores de moderación de contenido, guardianes, red-teaming, entrenamiento de rechazo y filtrado de salidas. Las consideraciones de seguridad son especialmente críticas en sistemas agénticos donde el modelo puede tomar acciones del mundo real con consecuencias irreversibles.',
    'Seguridad',
    '{"Alineación","Guardianes","Red-Teaming","IA Responsable"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'human-in-the-loop',
    'Humano en el Bucle (HITL)',
    'H',
    'Un patrón de diseño donde los humanos revisan o aprueban las salidas de la IA en pasos críticos antes de que surtan efecto.',
    'El Humano en el Bucle (HITL) es un patrón de diseño donde la supervisión humana se incorpora en puntos clave de un flujo de trabajo de IA — por ejemplo, exigiendo que un humano revise y apruebe una acción generada por el modelo antes de que se ejecute. HITL es especialmente importante en aplicaciones de alto riesgo y sistemas agénticos donde los errores pueden tener consecuencias significativas. Las interfaces HITL bien diseñadas mantienen a los humanos informados y en control sin crear fricción excesiva.',
    'Operaciones',
    '{"Agente de IA","Seguridad","Guardianes","Bucle Agéntico"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'agentic-ide',
    'IDE Agéntico',
    'I',
    'Un entorno de desarrollo impulsado por agentes de IA capaces de escribir, editar y gestionar código de forma autónoma.',
    'Un IDE agéntico es un entorno de desarrollo de software que integra profundamente agentes de IA en el flujo de trabajo de codificación, permitiendo a la IA no solo sugerir completados sino planificar, generar, refactorizar y gestionar código de forma autónoma en un proyecto. Ejemplos incluyen Kiro, que cuenta con capacidades como specs (planificación estructurada de características), steering (reglas de IA personalizadas), hooks (disparadores automatizados) e integración MCP. Los IDEs agénticos representan un cambio de la IA como asistente de código a la IA como socio ingeniero colaborativo.',
    'Herramientas',
    '{"Agente de IA","MCP","Specs","Steering","Hooks"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'specs',
    'Specs (Especificaciones)',
    'S',
    'Documentos estructurados que definen los requisitos, diseño y plan de implementación de una característica antes de codificar.',
    'En el contexto de herramientas de desarrollo agéntico como Kiro, los specs (especificaciones) son documentos estructurados que un agente de IA ayuda a generar antes de escribir código. Un spec típicamente incluye un documento de requisitos, un documento de diseño del sistema y un conjunto de tareas de implementación. Este enfoque impulsado por specs asegura que el código generado por IA esté alineado con la intención del usuario y la arquitectura del proyecto antes de que comience cualquier implementación.',
    'Herramientas',
    '{"IDE Agéntico","Steering","Hooks","Agente de IA"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'steering',
    'Steering (Guía)',
    'S',
    'Reglas y contexto personalizados proporcionados a un sistema de IA para guiar su comportamiento en un proyecto específico.',
    'El steering se refiere al uso de reglas, restricciones y contexto personalizados para guiar el comportamiento de un sistema de IA dentro de un entorno específico. En Kiro, los archivos de steering permiten a los desarrolladores definir convenciones específicas del proyecto — como estándares de codificación, patrones arquitectónicos o bibliotecas preferidas — que el agente de IA sigue consistentemente en todas las interacciones. El steering es análogo a un prompt de sistema persistente con alcance de proyecto.',
    'Herramientas',
    '{"IDE Agéntico","Specs","Prompt de Sistema","Ingeniería de Contexto"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'hooks',
    'Hooks (Ganchos)',
    'H',
    'Disparadores automatizados que ejecutan acciones de IA en respuesta a eventos específicos de desarrollo.',
    'Los hooks son disparadores automatizados en entornos de desarrollo agéntico (como Kiro) que ejecutan acciones impulsadas por IA en respuesta a eventos predefinidos — como guardar un archivo, fallar una prueba o abrir un pull request. Los hooks permiten automatizar tareas repetitivas sin intervención manual, integrando la asistencia de IA directamente en el flujo de trabajo de desarrollo. Son una forma de automatización orientada a eventos para acciones de agentes de IA.',
    'Herramientas',
    '{"IDE Agéntico","Specs","Agente de IA","Orquestación"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'model-selection',
    'Selección de Modelo',
    'S',
    'El proceso de elegir el LLM más apropiado para una tarea dada según capacidad, costo y latencia.',
    'La selección de modelo es la práctica de elegir el LLM más adecuado para un caso de uso específico equilibrando factores como la complejidad de la tarea, la calidad requerida, el costo de inferencia y las restricciones de latencia. Un principio clave es comenzar con el modelo más pequeño que logre calidad aceptable, evitando pagar de más por capacidades que no se necesitan. La selección de modelo también implica versionar y fijar versiones de modelo para evitar cambios de comportamiento inesperados.',
    'Operaciones',
    '{"Inferencia","LLMOps","Ajuste Fino","Latencia"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'latency',
    'Latencia',
    'L',
    'El tiempo que tarda un LLM en comenzar o completar una respuesta después de recibir un prompt.',
    'La latencia en sistemas de LLMs se refiere al retraso entre enviar un prompt y recibir una respuesta. Se mide comúnmente como Tiempo al Primer Token (TTFT) — cuánto tarda el modelo en comenzar a transmitir salida — y el tiempo total de generación. La latencia es un factor crítico en la experiencia del usuario y está influenciada por el tamaño del modelo, el hardware, la longitud del contexto y la arquitectura del sistema. Las técnicas de optimización incluyen caché, modelos más pequeños, streaming y agrupación.',
    'Operaciones',
    '{"Inferencia","Caché","Selección de Modelo","Streaming"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'streaming',
    'Streaming',
    'S',
    'Entregar la salida de un LLM token a token a medida que se genera, en lugar de esperar la respuesta completa.',
    'El streaming es una técnica donde la salida de un LLM se transmite al usuario token a token a medida que se genera, en lugar de esperar la respuesta completa. Esto mejora significativamente la latencia percibida ya que los usuarios ven la respuesta formarse en tiempo real. El streaming es soportado por la mayoría de las principales APIs de LLM y es práctica estándar para interfaces de chat y tareas de generación de forma larga.',
    'Inferencia',
    '{"Latencia","Inferencia","Modelo Autorregresivo"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'agentic-rag',
    'RAG Agéntico',
    'R',
    'Una arquitectura RAG donde un agente de IA decide dinámicamente cuándo y cómo recuperar información.',
    'El RAG Agéntico combina las capacidades de recuperación de la Generación Aumentada por Recuperación con las capacidades de planificación y toma de decisiones de un agente de IA. En lugar de recuperar siempre documentos como un paso fijo de preprocesamiento, el agente decide dinámicamente cuándo se necesita la recuperación, qué consultas emitir y cómo sintetizar resultados de múltiples rondas de recuperación. El RAG Agéntico permite un razonamiento más complejo sobre grandes bases de conocimiento.',
    'Agentes',
    '{"Generación Aumentada por Recuperación","Agente de IA","Bucle Agéntico","Uso de Herramientas","Memoria"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'prompt-versioning',
    'Versionado de Prompts',
    'V',
    'Rastrear y gestionar los cambios en los prompts a lo largo del tiempo, análogo al control de versiones para código.',
    'El versionado de prompts es la práctica de rastrear, almacenar y gestionar diferentes versiones de los prompts usados en aplicaciones de LLMs de producción. Al igual que el control de versiones de software, permite a los equipos revertir a versiones anteriores de prompts, comparar el rendimiento entre versiones y desplegar cambios de forma segura. Dado que los cambios de prompts pueden afectar significativamente el comportamiento del modelo, el versionado es una parte crítica de LLMOps.',
    'Operaciones',
    '{"LLMOps","Evals","Ingeniería de Prompts"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'product-market-fit',
    'Ajuste Producto-Mercado (PMF) para IA',
    'A',
    'Validar que un producto impulsado por IA resuelve una necesidad real del usuario antes de invertir en infraestructura.',
    'En el contexto del desarrollo de productos LLM, el Ajuste Producto-Mercado (PMF) se refiere a la validación de que un producto impulsado por IA genuinamente resuelve una necesidad real del usuario antes de realizar grandes inversiones en infraestructura como entrenar modelos personalizados. Una heurística ampliamente citada es ''Sin GPUs antes del PMF'': comenzar con APIs de inferencia, ingeniería de prompts y RAG antes de comprometerse con entrenamiento personalizado.',
    'Estrategia',
    '{"LLMOps","Ajuste Fino","Inferencia","Selección de Modelo"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'data-flywheel',
    'Volante de Datos (Data Flywheel)',
    'V',
    'Un ciclo auto-reforzante donde el uso del producto genera datos que mejoran la IA, lo que atrae más usuarios.',
    'El volante de datos es un patrón estratégico en el desarrollo de productos de IA donde el uso del producto genera datos de entrenamiento y evaluación, que se usan para mejorar el modelo de IA, lo que mejora el producto, atrae más usuarios, y así sucesivamente. Construir un volante de datos temprano — instrumentando la producción para capturar pares de entrada-salida, retroalimentación de usuarios y casos extremos — crea una ventaja compuesta con el tiempo.',
    'Estrategia',
    '{"Evals","Ajuste Fino","LLMOps","Ajuste Producto-Mercado"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'red-teaming',
    'Red-Teaming',
    'R',
    'La práctica de probar adversarialmente un sistema de IA para encontrar fallos de seguridad y confiabilidad antes del despliegue.',
    'El red-teaming es una práctica de seguridad tomada de la ciberseguridad donde un equipo dedicado intenta encontrar fallas, jailbreaks, modos de fallo y comportamientos dañinos en un sistema de IA al sondearlo adversarialmente. El red-teaming descubre problemas que los evals estándar pueden pasar por alto, incluyendo casos extremos, inyecciones de prompts y violaciones de políticas de contenido. Se considera una mejor práctica para el despliegue responsable de IA.',
    'Seguridad',
    '{"Seguridad","Alineación","Guardianes","Evals"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'reward-model',
    'Modelo de Recompensa',
    'M',
    'Un modelo entrenado para puntuar salidas de LLMs según la preferencia humana, usado en el entrenamiento RLHF.',
    'Un modelo de recompensa es una red neuronal entrenada para predecir cuánto preferiría un humano una salida del modelo sobre otra. Se entrena con datos de comparación humana (p. ej., anotadores que clasifican pares de salidas del modelo) y produce una puntuación escalar para cualquier salida dada. En RLHF, el modelo de recompensa sirve como proxy de la preferencia humana, guiando las actualizaciones de la política del LLM durante el aprendizaje por refuerzo.',
    'Entrenamiento',
    '{"RLHF","Ajuste Fino","Alineación","Evals"}',
    'es'
);

INSERT INTO glossary_terms (slug, term, letter, summary, definition, category, related_terms, lang) VALUES (
    'autonomous-agent',
    'Agente Autónomo',
    'A',
    'Un agente de IA capaz de completar tareas complejas de múltiples pasos con mínima intervención humana.',
    'Un agente autónomo es un agente de IA diseñado para operar con un alto grado de independencia, capaz de planificar, ejecutar y adaptarse a través de largas secuencias de acciones para lograr un objetivo especificado por el usuario. A diferencia de los chatbots simples o asistentes de un solo turno, los agentes autónomos gestionan sus propias llamadas a herramientas, memoria y toma de decisiones en flujos de trabajo extendidos. El agente autónomo de Kiro, por ejemplo, puede ejecutar tareas agénticas de principio a fin a través de una interfaz CLI o IDE.',
    'Agentes',
    '{"Agente de IA","Bucle Agéntico","Uso de Herramientas","Orquestación","Humano en el Bucle"}',
    'es'
);

-- Learning items (en)
INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'agents-course',
    'Agents Course',
    'Hugging Face',
    'This free course will take you on a journey, from beginner to expert, in understanding, using and building AI agents.',
    'https://huggingface.co/learn/agents-course/unit1/what-are-llms',
    'Course',
    'en'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'llm-course',
    'LLM Course',
    'Hugging Face',
    'This course will teach you about large language models (LLMs) and natural language processing (NLP) using libraries from the Hugging Face ecosystem — 🤗 Transformers, 🤗 Datasets, 🤗 Tokenizers, and 🤗 Accelerate — as well as the Hugging Face Hub.',
    'https://huggingface.co/learn/llm-course/chapter1/1',
    'Course',
    'en'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'claude-code-action',
    'Claude Code in Action',
    'Anthropic',
    'This course provides comprehensive training on using Claude Code for software development tasks, covering the underlying architecture of AI coding assistants, practical implementation techniques, and advanced integration strategies. You''ll learn about Claude Code''s context management approaches, and how to extend functionality through MCP servers and GitHub integration.',
    'https://anthropic.skilljar.com/claude-code-in-action',
    'Course',
    'en'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'contextual-retrieval',
    'Contextual Retrieval',
    'Anthropic',
    'This article explores how Claude can retrieve information from external documents to improve the accuracy and relevance of its responses.',
    'https://www.anthropic.com/engineering/contextual-retrieval',
    'Article',
    'en'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'qlora',
    'QLORA: Efficient Finetuning of Quantized LLMs',
    'Tim Dettmers',
    'QLORA finetuning of quantized LLMs.',
    'https://arxiv.org/pdf/2305.14314',
    'Study',
    'en'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'transformer-math',
    'Transformer Math',
    'Quentin Anthony, Stella Biderman, Hailey Schoelkopf',
    'We present basic math related to computation and memory usage for transformers',
    'https://blog.eleuther.ai/transformer-math/',
    'Article',
    'en'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'agent-skills',
    'Agent Skills with Anthropic',
    'DeepLearning.AI',
    'Learn how to build agents that can use tools, access information, and perform complex tasks',
    'https://learn.deeplearning.ai/courses/agent-skills-with-anthropic',
    'Course',
    'en'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'generative-ai-with-llms',
    'Generative AI with LLMs',
    'DeepLearning.AI',
    'Understand the generative AI lifecycle. Describe transformer architecture powering LLMs. Apply training/tuning/inference methods. Hear from researchers on generative AI challenges/opportunities.',
    'https://learn.deeplearning.ai/courses/generative-ai-with-llms/lesson/rs5m7/course-introduction',
    'Course',
    'en'
);

-- Learning items (es)
INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'agents-course',
    'Curso de Agentes',
    'Hugging Face',
    'Este curso gratuito te llevará en un viaje, desde principiante hasta experto, para comprender, usar y construir agentes de IA.',
    'https://huggingface.co/learn/agents-course/unit1/what-are-llms',
    'Curso',
    'es'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'llm-course',
    'Curso de LLM',
    'Hugging Face',
    'Este curso te enseñará sobre modelos de lenguaje grandes (LLM) y procesamiento de lenguaje natural (NLP) utilizando bibliotecas del ecosistema Hugging Face — 🤗 Transformers, 🤗 Datasets, 🤗 Tokenizers y 🤗 Accelerate — así como el Hugging Face Hub.',
    'https://huggingface.co/learn/llm-course/chapter1/1',
    'Curso',
    'es'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'claude-code-action',
    'Claude Code en Acción',
    'Anthropic',
    'Este curso proporciona capacitación integral sobre el uso de Claude Code para tareas de desarrollo de software, cubriendo la arquitectura subyacente de los asistentes de codificación de IA, técnicas de implementación práctica y estrategias de integración avanzada. Aprenderás sobre los enfoques de gestión de contexto de Claude Code y cómo extender la funcionalidad a través de servidores MCP e integración con GitHub.',
    'https://anthropic.skilljar.com/claude-code-in-action',
    'Curso',
    'es'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'contextual-retrieval',
    'Recuperación Contextual',
    'Anthropic',
    'Este artículo explora cómo Claude puede recuperar información de documentos externos para mejorar la precisión y relevancia de sus respuestas.',
    'https://www.anthropic.com/engineering/contextual-retrieval',
    'Artículo',
    'es'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'qlora',
    'QLORA: Efficient Finetuning of Quantized LLMs',
    'Tim Dettmers',
    'Articulo donde se explica a fondo que es QLORA y como funciona el finetuning de LLMs.',
    'https://arxiv.org/pdf/2305.14314',
    'Estudio',
    'es'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'transformer-math',
    'Transformer Math',
    'Quentin Anthony, Stella Biderman, Hailey Schoelkopf',
    'Artículo donde se explican las matemáticas básicas relacionadas con la computación y el uso de memoria para transformers.',
    'https://blog.eleuther.ai/transformer-math/',
    'Artículo',
    'es'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'agent-skills',
    'Agent Skills with Anthropic',
    'DeepLearning.AI',
    'Aprende a construir agentes que puedan usar herramientas, acceder a información y realizar tareas complejas',
    'https://learn.deeplearning.ai/courses/agent-skills-with-anthropic',
    'Curso',
    'es'
);

INSERT INTO learning_items (slug, title, creator, summary, link, category, lang) VALUES (
    'generative-ai-with-llms',
    'Generative AI with LLMs',
    'DeepLearning.AI',
    'Comprende el ciclo de vida de la IA generativa. Describe la arquitectura transformer que potencia los LLM. Aplica métodos de entrenamiento/ajuste/inferencia. Escucha a investigadores sobre los desafíos/oportunidades de la IA generativa.',
    'https://learn.deeplearning.ai/courses/generative-ai-with-llms/lesson/rs5m7/course-introduction',
    'Curso',
    'es'
);

