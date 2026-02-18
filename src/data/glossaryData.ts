import { GlossaryTerm } from '../types/glossary';

const glossaryData: GlossaryTerm[] = [
    {
        id: "attention-mechanism",
        term: "Attention Mechanism",
        letter: "A",
        summary: "A technique that allows models to focus on relevant parts of the input when producing output.",
        definition: "The attention mechanism is a fundamental component in modern neural networks, particularly in Transformer architectures. It allows a model to dynamically focus on different parts of the input sequence when generating each element of the output. Instead of compressing all input information into a fixed-size vector, attention computes a weighted sum of input representations, where the weights reflect the relevance of each input element to the current output step. This enables models to handle long-range dependencies and capture complex relationships in data.",
        category: "Architecture",
        relatedTerms: ["Transformer", "Self-Attention", "Multi-Head Attention"]
    },
    {
        id: "autoregressive-model",
        term: "Autoregressive Model",
        letter: "A",
        summary: "A model that generates output sequentially, each token conditioned on previous tokens.",
        definition: "An autoregressive model generates sequences by predicting one element at a time, conditioning each prediction on all previously generated elements. In language models like GPT, this means generating text token by token from left to right. The model learns the joint probability of a sequence by decomposing it into a product of conditional probabilities. This approach is powerful for generation tasks but can be slow at inference time since tokens must be produced sequentially.",
        category: "Model Type",
        relatedTerms: ["GPT", "Language Model", "Token"]
    },
    {
        id: "bert",
        term: "BERT",
        letter: "B",
        summary: "Bidirectional Encoder Representations from Transformers — a pre-trained language model by Google.",
        definition: "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model developed by Google in 2018. Unlike GPT, BERT uses a bidirectional approach, meaning it considers both left and right context simultaneously when processing text. It is pre-trained using two tasks: Masked Language Modeling (MLM), where random tokens are masked and the model predicts them, and Next Sentence Prediction (NSP). BERT set new state-of-the-art results on numerous NLP benchmarks and popularized the fine-tuning paradigm for NLP tasks.",
        category: "Model",
        relatedTerms: ["Transformer", "Fine-tuning", "Pre-training", "GPT"]
    },
    {
        id: "chain-of-thought",
        term: "Chain-of-Thought Prompting",
        letter: "C",
        summary: "A prompting technique that encourages LLMs to reason step-by-step before answering.",
        definition: "Chain-of-Thought (CoT) prompting is a technique where the model is guided to produce intermediate reasoning steps before arriving at a final answer. By including examples that show step-by-step reasoning in the prompt, or by simply instructing the model to 'think step by step', CoT significantly improves performance on complex reasoning tasks such as math problems, logical puzzles, and multi-step question answering. It was introduced by Google researchers and has become a standard technique for eliciting better reasoning from large language models.",
        category: "Prompting",
        relatedTerms: ["Prompt Engineering", "Few-Shot Learning", "Reasoning"]
    },
    {
        id: "context-window",
        term: "Context Window",
        letter: "C",
        summary: "The maximum amount of text (tokens) a language model can process at once.",
        definition: "The context window refers to the maximum number of tokens that a language model can consider at one time during inference. Tokens within the context window can attend to each other via the attention mechanism. Early models like GPT-2 had context windows of 1,024 tokens, while modern models like GPT-4 Turbo support up to 128,000 tokens. A larger context window allows the model to process longer documents, maintain longer conversations, and perform tasks requiring extensive context, but it also increases computational cost.",
        category: "Architecture",
        relatedTerms: ["Token", "Attention Mechanism", "Transformer"]
    },
    {
        id: "diffusion-model",
        term: "Diffusion Model",
        letter: "D",
        summary: "A generative model that learns to reverse a noise-adding process to generate data.",
        definition: "Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process. During training, data (e.g., images) is progressively corrupted with Gaussian noise over many steps. The model learns to reverse this process, starting from pure noise and iteratively denoising to produce realistic samples. Stable Diffusion, DALL-E 2, and Midjourney are prominent examples. Diffusion models have achieved state-of-the-art results in image generation, audio synthesis, and video generation.",
        category: "Generative AI",
        relatedTerms: ["Stable Diffusion", "DALL-E", "Generative Model", "Latent Space"]
    },
    {
        id: "embedding",
        term: "Embedding",
        letter: "E",
        summary: "A dense vector representation of data (words, sentences, images) in a continuous space.",
        definition: "An embedding is a learned representation of data as a dense vector in a continuous, high-dimensional space. Words, sentences, images, or other entities are mapped to vectors such that semantically similar items are close together in the embedding space. Word embeddings like Word2Vec and GloVe were early examples; modern models produce contextual embeddings where the same word has different representations depending on context. Embeddings are foundational to most modern AI systems, enabling efficient similarity search, clustering, and downstream task performance.",
        category: "Representation",
        relatedTerms: ["Vector Database", "Semantic Search", "Word2Vec", "Transformer"]
    },
    {
        id: "fine-tuning",
        term: "Fine-Tuning",
        letter: "F",
        summary: "Adapting a pre-trained model to a specific task by training on task-specific data.",
        definition: "Fine-tuning is the process of taking a pre-trained model and continuing to train it on a smaller, task-specific dataset. This allows the model to adapt its general knowledge to a particular domain or task while retaining the broad capabilities learned during pre-training. Fine-tuning can be full (updating all parameters) or parameter-efficient (e.g., LoRA, adapters). It is a cornerstone of modern NLP and computer vision, enabling high performance on specialized tasks without training from scratch.",
        category: "Training",
        relatedTerms: ["Pre-training", "Transfer Learning", "LoRA", "RLHF"]
    },
    {
        id: "foundation-model",
        term: "Foundation Model",
        letter: "F",
        summary: "A large model trained on broad data that can be adapted to many downstream tasks.",
        definition: "A foundation model is a large AI model trained on vast amounts of diverse data using self-supervised learning. The term was coined by Stanford researchers in 2021. Foundation models serve as a base that can be fine-tuned or prompted for a wide range of downstream tasks. Examples include GPT-4, PaLM, LLaMA, CLIP, and Stable Diffusion. Their scale and generality make them powerful starting points, but also raise concerns about bias, safety, and the concentration of AI capabilities.",
        category: "Model Type",
        relatedTerms: ["Pre-training", "Fine-tuning", "Large Language Model", "Transfer Learning"]
    },
    {
        id: "gpt",
        term: "GPT",
        letter: "G",
        summary: "Generative Pre-trained Transformer — OpenAI's family of large autoregressive language models.",
        definition: "GPT (Generative Pre-trained Transformer) is a family of large language models developed by OpenAI. GPT models are trained using unsupervised pre-training on massive text corpora, followed by fine-tuning for specific tasks. They use a decoder-only Transformer architecture and generate text autoregressively. GPT-3 (175B parameters) demonstrated remarkable few-shot learning capabilities. GPT-4 is a multimodal model capable of processing both text and images. The GPT series has been foundational in demonstrating the power of scaling language models.",
        category: "Model",
        relatedTerms: ["Transformer", "Autoregressive Model", "OpenAI", "Large Language Model"]
    },
    {
        id: "generative-adversarial-network",
        term: "Generative Adversarial Network",
        letter: "G",
        summary: "A framework where two neural networks compete: a generator and a discriminator.",
        definition: "A Generative Adversarial Network (GAN) consists of two neural networks trained simultaneously in a competitive framework. The generator network creates synthetic data samples, while the discriminator network tries to distinguish real data from generated data. Through this adversarial process, the generator learns to produce increasingly realistic outputs. GANs were introduced by Ian Goodfellow in 2014 and have been used for image synthesis, style transfer, data augmentation, and more. Variants include DCGAN, StyleGAN, and CycleGAN.",
        category: "Generative AI",
        relatedTerms: ["Diffusion Model", "Generative Model", "Latent Space"]
    },
    {
        id: "hallucination",
        term: "Hallucination",
        letter: "H",
        summary: "When an AI model generates plausible-sounding but factually incorrect or fabricated information.",
        definition: "Hallucination in AI refers to the phenomenon where a language model generates content that is confidently stated but factually incorrect, nonsensical, or entirely fabricated. This occurs because LLMs are trained to produce statistically likely text rather than verified facts. Hallucinations can range from subtle errors (wrong dates, names) to completely invented citations or events. Mitigating hallucinations is a major research challenge, with approaches including Retrieval-Augmented Generation (RAG), better training data, and improved alignment techniques.",
        category: "Safety & Alignment",
        relatedTerms: ["Retrieval-Augmented Generation", "Alignment", "Large Language Model"]
    },
    {
        id: "inference",
        term: "Inference",
        letter: "I",
        summary: "The process of using a trained model to generate predictions or outputs on new data.",
        definition: "Inference is the phase where a trained AI model is used to make predictions or generate outputs on new, unseen data. Unlike training, inference does not update model weights. It involves a forward pass through the network. For large language models, inference can be computationally expensive due to the size of the models and the autoregressive nature of text generation. Techniques like quantization, batching, speculative decoding, and hardware acceleration (GPUs, TPUs) are used to make inference faster and more cost-effective.",
        category: "Deployment",
        relatedTerms: ["Training", "Quantization", "Autoregressive Model"]
    },
    {
        id: "instruction-tuning",
        term: "Instruction Tuning",
        letter: "I",
        summary: "Fine-tuning a language model on instruction-response pairs to improve instruction-following.",
        definition: "Instruction tuning is a fine-tuning technique where a pre-trained language model is trained on a dataset of (instruction, response) pairs. This teaches the model to follow natural language instructions more reliably. Models like InstructGPT, FLAN, and Alpaca use instruction tuning. It significantly improves the model's ability to generalize to new tasks described in natural language, without requiring task-specific examples. Instruction tuning is often combined with RLHF to produce helpful, harmless, and honest AI assistants.",
        category: "Training",
        relatedTerms: ["Fine-tuning", "RLHF", "Prompt Engineering", "Alignment"]
    },
    {
        id: "latent-space",
        term: "Latent Space",
        letter: "L",
        summary: "A compressed, abstract representation space learned by a model to encode data.",
        definition: "Latent space is the multi-dimensional space in which a model represents compressed, abstract features of data. In autoencoders, the encoder maps input data to a point in latent space, and the decoder reconstructs the data from that point. In diffusion models and VAEs, the latent space captures the underlying structure of the data distribution. Navigating and manipulating latent space allows for controlled generation, interpolation between data points, and style transfer. Understanding latent space is key to understanding how generative models work.",
        category: "Architecture",
        relatedTerms: ["Embedding", "Diffusion Model", "Variational Autoencoder", "Generative Model"]
    },
    {
        id: "large-language-model",
        term: "Large Language Model",
        letter: "L",
        summary: "A neural network with billions of parameters trained on massive text corpora.",
        definition: "A Large Language Model (LLM) is a type of neural network with billions to trillions of parameters, trained on vast amounts of text data. LLMs learn statistical patterns in language and can perform a wide range of tasks including text generation, translation, summarization, question answering, and code generation. Notable LLMs include GPT-4, Claude, Gemini, LLaMA, and Mistral. The 'large' refers to both the number of parameters and the scale of training data. LLMs exhibit emergent capabilities that were not explicitly trained for.",
        category: "Model Type",
        relatedTerms: ["GPT", "Transformer", "Foundation Model", "Emergent Capabilities"]
    },
    {
        id: "lora",
        term: "LoRA",
        letter: "L",
        summary: "Low-Rank Adaptation — an efficient fine-tuning method that trains only small adapter matrices.",
        definition: "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable rank-decomposition matrices into each layer of the Transformer architecture. Instead of updating all model parameters, LoRA trains a much smaller number of parameters (often <1% of the original), making fine-tuning feasible on consumer hardware. LoRA has become extremely popular for customizing large language models and image generation models like Stable Diffusion, enabling domain-specific adaptation without full fine-tuning costs.",
        category: "Training",
        relatedTerms: ["Fine-tuning", "Parameter-Efficient Fine-Tuning", "Transformer", "Stable Diffusion"]
    },
    {
        id: "multimodal",
        term: "Multimodal AI",
        letter: "M",
        summary: "AI systems that can process and generate multiple types of data (text, images, audio, video).",
        definition: "Multimodal AI refers to systems capable of understanding and generating multiple modalities of data, such as text, images, audio, and video. Unlike unimodal models that handle only one data type, multimodal models can reason across modalities. Examples include GPT-4V (text + images), Gemini (text, images, audio, video), CLIP (text + images), and Flamingo. Multimodal capabilities enable applications like visual question answering, image captioning, audio transcription, and video understanding. This is considered a key step toward more general AI systems.",
        category: "Model Type",
        relatedTerms: ["GPT", "CLIP", "Vision-Language Model", "Foundation Model"]
    },
    {
        id: "neural-network",
        term: "Neural Network",
        letter: "N",
        summary: "A computational model inspired by the brain, composed of interconnected layers of nodes.",
        definition: "A neural network is a machine learning model inspired by the structure of biological neural networks in the brain. It consists of layers of interconnected nodes (neurons), where each connection has a learnable weight. Data flows through the network (forward pass), and the model learns by adjusting weights to minimize a loss function via backpropagation. Deep neural networks with many layers are the foundation of modern AI, enabling breakthroughs in image recognition, natural language processing, speech recognition, and generative AI.",
        category: "Fundamentals",
        relatedTerms: ["Deep Learning", "Backpropagation", "Transformer", "Convolutional Neural Network"]
    },
    {
        id: "prompt-engineering",
        term: "Prompt Engineering",
        letter: "P",
        summary: "The practice of crafting inputs to guide AI models toward desired outputs.",
        definition: "Prompt engineering is the discipline of designing and optimizing input prompts to elicit desired behaviors from AI language models. Since LLMs are sensitive to how instructions are phrased, prompt engineering can significantly affect output quality. Techniques include zero-shot prompting, few-shot prompting, chain-of-thought prompting, role prompting, and structured output prompting. Prompt engineering has emerged as a critical skill for effectively using LLMs in applications, and has spawned research into automatic prompt optimization and prompt injection attacks.",
        category: "Prompting",
        relatedTerms: ["Chain-of-Thought Prompting", "Few-Shot Learning", "Large Language Model", "Zero-Shot Learning"]
    },
    {
        id: "pre-training",
        term: "Pre-training",
        letter: "P",
        summary: "Training a model on large-scale data before fine-tuning on specific tasks.",
        definition: "Pre-training is the initial phase of training a foundation model on a large, diverse dataset using self-supervised objectives. For language models, this typically involves predicting masked tokens (BERT) or next tokens (GPT). For vision models, it may involve contrastive learning or masked image modeling. Pre-training allows the model to learn general representations of language, vision, or other modalities. The pre-trained model is then adapted for specific tasks via fine-tuning or prompting, dramatically reducing the data and compute needed for downstream tasks.",
        category: "Training",
        relatedTerms: ["Fine-tuning", "Foundation Model", "Self-Supervised Learning", "Transfer Learning"]
    },
    {
        id: "quantization",
        term: "Quantization",
        letter: "Q",
        summary: "Reducing model size by representing weights with lower-precision numbers.",
        definition: "Quantization is a model compression technique that reduces the numerical precision of model weights and activations from high-precision formats (e.g., float32) to lower-precision formats (e.g., int8, int4). This reduces memory requirements and speeds up inference with minimal loss in model quality. Techniques include post-training quantization (PTQ) and quantization-aware training (QAT). Quantization has been crucial for deploying large language models on consumer hardware, with tools like GPTQ, AWQ, and llama.cpp enabling running LLMs on laptops.",
        category: "Deployment",
        relatedTerms: ["Inference", "Model Compression", "LoRA", "Large Language Model"]
    },
    {
        id: "rag",
        term: "Retrieval-Augmented Generation",
        letter: "R",
        summary: "Combining a retrieval system with a generative model to ground responses in external knowledge.",
        definition: "Retrieval-Augmented Generation (RAG) is an architecture that enhances language model responses by retrieving relevant documents from an external knowledge base and incorporating them into the prompt. The process involves: (1) encoding the query as an embedding, (2) searching a vector database for relevant documents, (3) including retrieved documents in the prompt context, and (4) generating a response grounded in the retrieved information. RAG reduces hallucinations, enables access to up-to-date information, and allows LLMs to reason over private or domain-specific knowledge.",
        category: "Architecture",
        relatedTerms: ["Vector Database", "Embedding", "Hallucination", "Large Language Model"]
    },
    {
        id: "rlhf",
        term: "RLHF",
        letter: "R",
        summary: "Reinforcement Learning from Human Feedback — aligning AI models using human preference data.",
        definition: "Reinforcement Learning from Human Feedback (RLHF) is a training technique used to align language models with human values and preferences. The process involves: (1) supervised fine-tuning on demonstration data, (2) training a reward model on human preference comparisons, and (3) optimizing the language model using the reward model via reinforcement learning (typically PPO). RLHF was used to train InstructGPT and ChatGPT, dramatically improving their helpfulness and safety. It is a key technique in AI alignment research.",
        category: "Training",
        relatedTerms: ["Alignment", "Fine-tuning", "Instruction Tuning", "Reward Model"]
    },
    {
        id: "self-attention",
        term: "Self-Attention",
        letter: "S",
        summary: "A mechanism where each element in a sequence attends to all other elements in the same sequence.",
        definition: "Self-attention (also called intra-attention) is a mechanism where each position in a sequence computes attention weights over all other positions in the same sequence. For each position, queries, keys, and values are computed from the input, and the output is a weighted sum of values where weights are determined by query-key compatibility. Self-attention enables the model to capture long-range dependencies and relationships between any two positions regardless of distance. It is the core operation in Transformer architectures and scales quadratically with sequence length.",
        category: "Architecture",
        relatedTerms: ["Attention Mechanism", "Transformer", "Multi-Head Attention", "Context Window"]
    },
    {
        id: "stable-diffusion",
        term: "Stable Diffusion",
        letter: "S",
        summary: "An open-source latent diffusion model for high-quality text-to-image generation.",
        definition: "Stable Diffusion is an open-source latent diffusion model developed by Stability AI, released in 2022. It generates high-quality images from text descriptions by performing the diffusion process in a compressed latent space rather than pixel space, making it computationally efficient. It uses a CLIP text encoder to condition generation on text prompts and a U-Net denoising network. Stable Diffusion can run on consumer GPUs and has spawned a large ecosystem of fine-tuned models, LoRA adaptations, and tools like AUTOMATIC1111 and ComfyUI.",
        category: "Generative AI",
        relatedTerms: ["Diffusion Model", "Latent Space", "LoRA", "DALL-E"]
    },
    {
        id: "token",
        term: "Token",
        letter: "T",
        summary: "The basic unit of text that language models process — roughly a word or word fragment.",
        definition: "A token is the fundamental unit of text that language models process. Tokenization splits text into tokens using algorithms like Byte-Pair Encoding (BPE) or WordPiece. A token is typically a word, subword, or character, depending on the tokenizer. For example, 'tokenization' might be split into ['token', 'ization']. The number of tokens in a text affects processing cost and context window usage. On average, 1 token ≈ 4 characters or 0.75 words in English. API pricing for LLMs is typically based on token count.",
        category: "Fundamentals",
        relatedTerms: ["Context Window", "Tokenization", "Large Language Model", "Embedding"]
    },
    {
        id: "transformer",
        term: "Transformer",
        letter: "T",
        summary: "The dominant neural network architecture for NLP, based entirely on attention mechanisms.",
        definition: "The Transformer is a neural network architecture introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al. at Google. It replaced recurrent networks (RNNs, LSTMs) with self-attention mechanisms, enabling parallel processing of sequences and better capture of long-range dependencies. The architecture consists of encoder and decoder stacks, each containing multi-head self-attention and feed-forward layers with residual connections and layer normalization. Transformers are the foundation of virtually all modern large language models, vision models, and multimodal models.",
        category: "Architecture",
        relatedTerms: ["Self-Attention", "Attention Mechanism", "BERT", "GPT", "Large Language Model"]
    },
    {
        id: "transfer-learning",
        term: "Transfer Learning",
        letter: "T",
        summary: "Applying knowledge learned from one task or domain to improve performance on another.",
        definition: "Transfer learning is a machine learning paradigm where a model trained on one task is adapted for a different but related task. In deep learning, this typically involves using pre-trained model weights as initialization for a new task. The pre-trained model has learned useful feature representations that transfer across tasks. Transfer learning dramatically reduces the data and compute needed for new tasks. It is the foundation of the modern AI development workflow: pre-train on large data, fine-tune on task-specific data.",
        category: "Training",
        relatedTerms: ["Pre-training", "Fine-tuning", "Foundation Model", "Domain Adaptation"]
    },
    {
        id: "vector-database",
        term: "Vector Database",
        letter: "V",
        summary: "A database optimized for storing and searching high-dimensional embedding vectors.",
        definition: "A vector database is a specialized database designed to store, index, and efficiently search high-dimensional embedding vectors. Unlike traditional databases that search by exact match, vector databases use approximate nearest neighbor (ANN) algorithms (e.g., HNSW, IVF) to find vectors most similar to a query vector. They are essential infrastructure for RAG systems, semantic search, recommendation engines, and similarity-based applications. Popular vector databases include Pinecone, Weaviate, Qdrant, Chroma, and pgvector (PostgreSQL extension).",
        category: "Infrastructure",
        relatedTerms: ["Embedding", "Retrieval-Augmented Generation", "Semantic Search"]
    },
    {
        id: "zero-shot-learning",
        term: "Zero-Shot Learning",
        letter: "Z",
        summary: "A model's ability to perform tasks it has never explicitly been trained on.",
        definition: "Zero-shot learning refers to a model's ability to perform a task without having seen any examples of that task during training or in the prompt. Large language models exhibit zero-shot capabilities because their pre-training on diverse data gives them broad knowledge and reasoning abilities. For example, GPT-4 can translate text to a language it was never explicitly trained to translate, or solve novel logic puzzles. Zero-shot performance is a key measure of a model's generalization ability and is contrasted with few-shot learning, where a small number of examples are provided.",
        category: "Learning Paradigm",
        relatedTerms: ["Few-Shot Learning", "Prompt Engineering", "Large Language Model", "Generalization"]
    },
    // --- New terms ---
    {
        id: "agent",
        term: "Agent",
        letter: "A",
        summary: "An AI system that perceives its environment and takes actions to achieve goals.",
        definition: "In AI, an agent is any software or program that interacts with the world (or a simulation) by receiving inputs and producing outputs or actions. Agents operate within an environment, perceive its state through sensors or data inputs, and act upon it through actuators or API calls. They range from simple rule-based systems (like a thermostat) to complex autonomous agents powered by large language models that can plan, use tools, browse the web, write code, and execute multi-step tasks. Agentic AI systems are increasingly used in automation, robotics, and AI assistants.",
        category: "Architecture",
        relatedTerms: ["Reinforcement Learning", "Large Language Model", "Tool Use", "Autonomous AI"]
    },
    {
        id: "generative-ai",
        term: "Generative AI",
        letter: "G",
        summary: "AI that creates new content such as text, images, audio, or video.",
        definition: "Generative AI refers to a class of AI models designed to produce original data that resembles the examples they were trained on. These systems learn statistical patterns from large datasets and use those patterns to generate new content — including natural-sounding text, realistic images, music, video, code, and 3D models. Key architectures include Transformers (for text), Diffusion Models (for images), and GANs. Prominent examples are GPT-4 (text), DALL-E and Stable Diffusion (images), Sora (video), and MusicLM (audio). Generative AI has transformed creative industries, software development, and scientific research.",
        category: "Generative AI",
        relatedTerms: ["Large Language Model", "Diffusion Model", "Generative Adversarial Network", "Foundation Model"]
    },
    {
        id: "context",
        term: "Context",
        letter: "C",
        summary: "Information surrounding an input that helps a model interpret meaning accurately.",
        definition: "In AI, context refers to any relevant background information that helps a model understand or respond appropriately. For language models, context includes the conversation history, system instructions, user intent, topic, tone, and any documents provided in the prompt. The amount of context a model can use is bounded by its context window. Effective use of context is critical for accurate, relevant responses — a model with insufficient context may misinterpret ambiguous queries. Context engineering — the practice of structuring context inputs optimally — has become a key skill in building AI applications.",
        category: "Fundamentals",
        relatedTerms: ["Context Window", "Prompt Engineering", "Context Engineering", "Retrieval-Augmented Generation"]
    },
    {
        id: "claude",
        term: "Claude",
        letter: "C",
        summary: "A family of AI assistant models developed by Anthropic, known for safety and helpfulness.",
        definition: "Claude is a family of large language models developed by Anthropic, an AI safety company. Claude models are designed with a strong emphasis on being helpful, harmless, and honest (the 'HHH' framework). They are trained using Constitutional AI (CAI) and RLHF techniques to align model behavior with human values. Claude excels at tasks including summarization, reasoning, coding assistance, analysis, and creative writing. The Claude model family includes multiple tiers (Haiku, Sonnet, Opus) optimized for different speed/capability trade-offs. Claude is widely used via API and in Anthropic's consumer products.",
        category: "Model",
        relatedTerms: ["Large Language Model", "RLHF", "Alignment", "OpenAI", "GPT"]
    },
    {
        id: "openai",
        term: "OpenAI",
        letter: "O",
        summary: "An AI research organization that develops advanced AI systems including the GPT series.",
        definition: "OpenAI is an AI research laboratory and technology company founded in 2015, with a mission to ensure that artificial general intelligence (AGI) benefits all of humanity. It is responsible for developing some of the most influential AI systems, including the GPT series of language models, DALL-E image generation models, Codex (code generation), Whisper (speech recognition), and the Sora video generation model. OpenAI also created ChatGPT, one of the most widely used AI applications in history. The organization operates as a capped-profit company with a non-profit parent, balancing commercial operations with safety research.",
        category: "Organization",
        relatedTerms: ["GPT", "ChatGPT", "DALL-E", "Large Language Model", "AGI"]
    },
    {
        id: "chatgpt",
        term: "ChatGPT",
        letter: "C",
        summary: "A conversational AI assistant by OpenAI, built on GPT large language models.",
        definition: "ChatGPT is a conversational AI application developed by OpenAI, launched in November 2022. It is built on top of the GPT series of large language models (initially GPT-3.5, later GPT-4) and fine-tuned with RLHF to be a helpful, conversational assistant. ChatGPT can engage in multi-turn dialogue, answer questions, write and debug code, draft documents, summarize text, translate languages, and perform many other language tasks. It became one of the fastest-growing consumer applications in history, reaching 100 million users in two months. It supports plugins, image input (GPT-4V), and custom GPTs.",
        category: "Model",
        relatedTerms: ["GPT", "OpenAI", "RLHF", "Large Language Model", "Instruction Tuning"]
    },
    {
        id: "deep-learning",
        term: "Deep Learning",
        letter: "D",
        summary: "A subset of machine learning using neural networks with many layers to learn from data.",
        definition: "Deep learning is a subfield of machine learning that uses artificial neural networks with many layers (hence 'deep') to automatically learn hierarchical representations from raw data. Each layer learns increasingly abstract features — for example, in image recognition, early layers detect edges, middle layers detect shapes, and later layers detect objects. Deep learning has driven breakthroughs in computer vision (CNNs), natural language processing (Transformers), speech recognition (RNNs, attention), and generative AI (GANs, diffusion models). It requires large datasets and significant compute, typically using GPUs or TPUs.",
        category: "Fundamentals",
        relatedTerms: ["Neural Network", "Transformer", "Convolutional Neural Network", "Backpropagation"]
    },
    {
        id: "reinforcement-learning",
        term: "Reinforcement Learning",
        letter: "R",
        summary: "A learning paradigm where an agent learns by receiving rewards or penalties for its actions.",
        definition: "Reinforcement learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent observes the current state, takes an action, receives a reward signal (positive or negative), and updates its policy to maximize cumulative reward over time. Key algorithms include Q-learning, SARSA, and policy gradient methods like PPO and A3C. RL has achieved superhuman performance in games (AlphaGo, Atari), robotics, and autonomous driving. In the context of LLMs, RLHF uses RL to align model outputs with human preferences.",
        category: "Learning Paradigm",
        relatedTerms: ["RLHF", "Agent", "Deep Learning", "Policy Gradient"]
    },
    {
        id: "computer-vision",
        term: "Computer Vision",
        letter: "C",
        summary: "AI that enables machines to interpret and understand visual information from images and videos.",
        definition: "Computer vision is a field of AI focused on enabling machines to extract meaningful information from visual inputs such as images and videos. Core tasks include image classification (identifying what is in an image), object detection (locating objects), semantic segmentation (labeling each pixel), and image generation. Modern computer vision relies heavily on convolutional neural networks (CNNs) and Vision Transformers (ViT). Applications span autonomous vehicles, medical imaging, facial recognition, augmented reality, quality control in manufacturing, and satellite imagery analysis.",
        category: "Field",
        relatedTerms: ["Neural Network", "Convolutional Neural Network", "Deep Learning", "Multimodal AI"]
    },
    {
        id: "nlp",
        term: "NLP",
        letter: "N",
        summary: "Natural Language Processing — AI that enables computers to understand and generate human language.",
        definition: "Natural Language Processing (NLP) is a branch of AI that combines linguistics, computer science, and machine learning to enable computers to process, understand, and generate human language. NLP encompasses a wide range of tasks: text classification, sentiment analysis, machine translation, question answering, summarization, named entity recognition, and dialogue systems. Modern NLP is dominated by Transformer-based models like BERT and GPT. NLP powers applications including search engines, virtual assistants, chatbots, grammar checkers, and content moderation systems.",
        category: "Field",
        relatedTerms: ["Natural Language Understanding", "Natural Language Generation", "Transformer", "BERT", "Large Language Model"]
    },
    {
        id: "supervised-learning",
        term: "Supervised Learning",
        letter: "S",
        summary: "Machine learning where models are trained on labeled input-output pairs.",
        definition: "Supervised learning is a machine learning paradigm where a model is trained on a dataset of labeled examples — pairs of inputs and their correct outputs. The model learns a mapping from inputs to outputs by minimizing the difference between its predictions and the true labels. Common tasks include classification (predicting a category) and regression (predicting a continuous value). Examples include spam detection, image classification, and price prediction. Supervised learning requires labeled data, which can be expensive to obtain, but it is the most widely used ML paradigm in production systems.",
        category: "Learning Paradigm",
        relatedTerms: ["Unsupervised Learning", "Fine-tuning", "Neural Network", "Transfer Learning"]
    },
    {
        id: "unsupervised-learning",
        term: "Unsupervised Learning",
        letter: "U",
        summary: "Machine learning on unlabeled data to discover hidden patterns or structure.",
        definition: "Unsupervised learning is a machine learning paradigm where models learn patterns and structure from data without labeled outputs. The model must discover the underlying organization of the data on its own. Common techniques include clustering (grouping similar data points, e.g., K-means), dimensionality reduction (e.g., PCA, t-SNE, autoencoders), and density estimation. Unsupervised learning is valuable when labeled data is scarce or expensive. Self-supervised learning — where models generate their own labels from unlabeled data — is a powerful variant used to pre-train large language models.",
        category: "Learning Paradigm",
        relatedTerms: ["Self-Supervised Learning", "Supervised Learning", "Clustering", "Pre-training"]
    },
    {
        id: "data-mining",
        term: "Data Mining",
        letter: "D",
        summary: "The process of discovering useful patterns and insights from large datasets.",
        definition: "Data mining is the process of applying statistical, mathematical, and computational techniques to extract meaningful patterns, correlations, and insights from large collections of data. It sits at the intersection of statistics, machine learning, and database systems. Common data mining tasks include classification, clustering, association rule learning (e.g., market basket analysis), anomaly detection, and regression. Data mining is foundational to business intelligence, fraud detection, scientific discovery, and recommendation systems. Modern data mining increasingly leverages machine learning and AI techniques.",
        category: "Data Science",
        relatedTerms: ["Machine Learning", "Supervised Learning", "Unsupervised Learning", "Pattern Recognition"]
    },
    {
        id: "entity-annotation",
        term: "Entity Annotation",
        letter: "E",
        summary: "Labeling meaningful entities (names, places, dates) in text or data for AI training.",
        definition: "Entity annotation is the process of marking up entities — such as person names, organizations, locations, dates, and product names — in text datasets so that AI models can learn to recognize them. It is a critical step in creating training data for Named Entity Recognition (NER) systems and other NLP tasks. Annotation can be done manually by human annotators or semi-automatically using pre-trained models. High-quality entity annotation is essential for training accurate information extraction systems used in search, knowledge graphs, and document processing pipelines.",
        category: "NLP",
        relatedTerms: ["Entity Extraction", "Named Entity Recognition", "NLP", "Supervised Learning"]
    },
    {
        id: "entity-extraction",
        term: "Entity Extraction",
        letter: "E",
        summary: "Automatically identifying and categorizing key entities from unstructured text.",
        definition: "Entity extraction, also known as Named Entity Recognition (NER), is an NLP task where an AI model automatically identifies and classifies named entities in unstructured text into predefined categories such as people, organizations, locations, dates, monetary values, and more. For example, in the sentence 'Apple was founded by Steve Jobs in Cupertino in 1976,' a NER model would extract Apple (organization), Steve Jobs (person), Cupertino (location), and 1976 (date). Entity extraction is fundamental to information retrieval, knowledge graph construction, and document intelligence.",
        category: "NLP",
        relatedTerms: ["Entity Annotation", "NLP", "Natural Language Understanding", "Information Extraction"]
    },
    {
        id: "intent",
        term: "Intent",
        letter: "I",
        summary: "The goal or purpose behind a user's input in a conversational AI system.",
        definition: "In conversational AI and NLP, intent refers to the underlying goal or purpose that a user aims to achieve with their input. For example, the query 'What's the weather like tomorrow?' has the intent 'get weather forecast.' Intent recognition (or intent classification) is the task of automatically identifying the user's intent from their utterance. It is a core component of dialogue systems, virtual assistants, and chatbots. Modern systems use machine learning classifiers or large language models to detect intent, enabling appropriate routing and response generation.",
        category: "NLP",
        relatedTerms: ["Natural Language Understanding", "NLP", "Dialogue System", "Entity Extraction"]
    },
    {
        id: "model",
        term: "Model",
        letter: "M",
        summary: "A mathematical system trained on data to make predictions, classifications, or generate outputs.",
        definition: "In AI and machine learning, a model is a computational system that has learned patterns from training data and can apply that knowledge to new inputs. Models are defined by their architecture (the structure of the computation) and their parameters (the learned weights). After training, a model can make predictions (regression, classification), generate content (language, images), or take actions (agents). The term encompasses everything from simple linear regression models to billion-parameter neural networks. Model selection, training, evaluation, and deployment are the core stages of the machine learning lifecycle.",
        category: "Fundamentals",
        relatedTerms: ["Neural Network", "Training", "Inference", "Foundation Model"]
    },
    {
        id: "nlu",
        term: "Natural Language Understanding",
        letter: "N",
        summary: "AI that interprets the meaning, intent, and context of human language.",
        definition: "Natural Language Understanding (NLU) is a subfield of NLP focused on enabling machines to comprehend the meaning, intent, sentiment, and context of human language — going beyond surface-level text processing. NLU tasks include intent recognition, sentiment analysis, semantic role labeling, coreference resolution, and reading comprehension. NLU is the 'understanding' component of conversational AI systems, enabling them to correctly interpret what users mean rather than just what they say. Modern NLU is powered by large pre-trained language models like BERT and its variants.",
        category: "NLP",
        relatedTerms: ["NLP", "Natural Language Generation", "Intent", "BERT", "Sentiment Analysis"]
    },
    {
        id: "nlg",
        term: "Natural Language Generation",
        letter: "N",
        summary: "AI that produces coherent, human-like text or speech from data or other inputs.",
        definition: "Natural Language Generation (NLG) is a subfield of NLP focused on automatically producing coherent, fluent, and contextually appropriate text or speech from structured data, knowledge, or other inputs. NLG tasks include text summarization, report generation, dialogue response generation, machine translation, and creative writing. Modern NLG is dominated by autoregressive language models like GPT-4, which generate text token by token. NLG is the 'generation' component of conversational AI and is used in chatbots, automated journalism, data-to-text systems, and virtual assistants.",
        category: "NLP",
        relatedTerms: ["NLP", "Natural Language Understanding", "Large Language Model", "Autoregressive Model"]
    },
    {
        id: "overfitting",
        term: "Overfitting",
        letter: "O",
        summary: "When a model memorizes training data too closely and fails to generalize to new data.",
        definition: "Overfitting occurs when a machine learning model learns the training data too precisely — including its noise and random fluctuations — rather than the underlying general patterns. An overfitted model performs very well on training data but poorly on unseen test data, because it has essentially memorized the training examples rather than learning transferable patterns. Overfitting is more likely with complex models and small datasets. Common mitigation strategies include regularization (L1/L2), dropout, early stopping, data augmentation, and cross-validation.",
        category: "Fundamentals",
        relatedTerms: ["Supervised Learning", "Regularization", "Generalization", "Neural Network"]
    },
    {
        id: "pattern-recognition",
        term: "Pattern Recognition",
        letter: "P",
        summary: "The ability of algorithms to identify recurring structures or regularities in data.",
        definition: "Pattern recognition is the ability of algorithms and AI systems to detect, classify, and respond to recurring structures, regularities, or relationships in data. It is one of the foundational tasks of AI and machine learning, underlying applications such as image recognition (detecting faces or objects), speech recognition (identifying phonemes and words), handwriting recognition, anomaly detection, and biometric identification. Modern pattern recognition is largely achieved through deep learning, where neural networks automatically learn hierarchical feature representations from raw data.",
        category: "Fundamentals",
        relatedTerms: ["Deep Learning", "Computer Vision", "Neural Network", "Classification"]
    },
    {
        id: "context-engineering",
        term: "Context Engineering",
        letter: "C",
        summary: "Structuring and optimizing the information provided to AI models to improve output quality.",
        definition: "Context engineering is the practice of deliberately designing and structuring the information provided to an AI model — including system prompts, conversation history, retrieved documents, examples, and environmental data — to maximize the quality and relevance of its outputs. It goes beyond basic prompt engineering to encompass the full information architecture around a model call: what to include, how to format it, what to retrieve, and how to prioritize. As AI systems become more capable, context engineering has emerged as a critical discipline for building reliable, accurate AI applications.",
        category: "Prompting",
        relatedTerms: ["Prompt Engineering", "Retrieval-Augmented Generation", "Context Window", "Context"]
    },
    {
        id: "turing-test",
        term: "Turing Test",
        letter: "T",
        summary: "A test proposed by Alan Turing to assess whether a machine's behavior is indistinguishable from a human's.",
        definition: "The Turing Test, proposed by mathematician Alan Turing in his 1950 paper 'Computing Machinery and Intelligence,' is a test of a machine's ability to exhibit intelligent behavior indistinguishable from that of a human. In the original formulation (the Imitation Game), a human evaluator converses via text with both a human and a machine without knowing which is which; if the evaluator cannot reliably distinguish the machine from the human, the machine is said to have passed the test. While influential as a philosophical benchmark, the Turing Test is now considered insufficient as a measure of true AI intelligence, as modern LLMs can pass it without possessing genuine understanding.",
        category: "Fundamentals",
        relatedTerms: ["Large Language Model", "AGI", "Narrow AI", "ChatGPT"]
    },
    {
        id: "narrow-ai",
        term: "Narrow AI",
        letter: "N",
        summary: "AI designed to perform a specific, limited set of tasks — also called Weak AI.",
        definition: "Narrow AI (also called Weak AI or Artificial Narrow Intelligence, ANI) refers to AI systems designed and trained to perform a specific task or a limited set of related tasks. Unlike hypothetical Artificial General Intelligence (AGI), narrow AI cannot transfer its knowledge to domains outside its training. Examples include image classifiers, spam filters, recommendation engines, chess-playing programs, and speech recognition systems. Despite the 'narrow' label, modern narrow AI systems like GPT-4 can perform impressively across many language tasks, blurring the line with more general capabilities.",
        category: "Fundamentals",
        relatedTerms: ["AGI", "Turing Test", "Large Language Model", "Foundation Model"]
    },
    {
        id: "spec-driven-development",
        term: "Spec-Driven Development",
        letter: "S",
        summary: "A development methodology where detailed specifications guide design, implementation, and testing.",
        definition: "Spec-driven development (SDD) is a software engineering methodology in which clear, structured specifications are written before implementation begins. These specs define expected behavior, inputs, outputs, edge cases, and acceptance criteria. In the context of AI systems, SDD is increasingly important for defining how AI components should behave, what outputs are acceptable, and how to evaluate correctness. It aligns closely with test-driven development (TDD) and behavior-driven development (BDD), and is gaining traction as a way to build more reliable, auditable AI-powered applications.",
        category: "Engineering",
        relatedTerms: ["Context Engineering", "Prompt Engineering", "Alignment", "Evaluation"]
    },
    {
        id: "nlu-nlg-nlp",
        term: "NLU / NLG / NLP",
        letter: "N",
        summary: "The three pillars of language AI: Understanding, Generation, and Processing.",
        definition: "NLU (Natural Language Understanding), NLG (Natural Language Generation), and NLP (Natural Language Processing) are three closely related but distinct subfields of language AI. NLP is the broadest term, covering all computational techniques for processing human language. NLU focuses specifically on comprehension — extracting meaning, intent, and structure from text. NLG focuses on production — generating coherent, contextually appropriate language from data or knowledge. Modern large language models like GPT-4 integrate all three capabilities: they process input (NLP), understand it (NLU), and generate responses (NLG) in a unified architecture.",
        category: "NLP",
        relatedTerms: ["NLP", "Natural Language Understanding", "Natural Language Generation", "Large Language Model"]
    }
];

export default glossaryData;

