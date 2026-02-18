import { GlossaryTerm } from '../types/glossary'

const glossaryDataEs: GlossaryTerm[] = [
    {
        id: "attention-mechanism",
        term: "Mecanismo de Atención",
        letter: "M",
        summary: "Técnica que permite a los modelos enfocarse en partes relevantes de la entrada al producir una salida.",
        definition: "El mecanismo de atención es un componente fundamental en las redes neuronales modernas, especialmente en las arquitecturas Transformer. Permite que un modelo se enfoque dinámicamente en diferentes partes de la secuencia de entrada al generar cada elemento de la salida. En lugar de comprimir toda la información de entrada en un vector de tamaño fijo, la atención calcula una suma ponderada de representaciones de entrada, donde los pesos reflejan la relevancia de cada elemento de entrada para el paso de salida actual. Esto permite a los modelos manejar dependencias de largo alcance y capturar relaciones complejas en los datos.",
        category: "Arquitectura",
        relatedTerms: ["Transformer", "Auto-Atención", "Atención Multi-Cabeza"]
    },
    {
        id: "autoregressive-model",
        term: "Modelo Autorregresivo",
        letter: "M",
        summary: "Modelo que genera salida de forma secuencial, donde cada token está condicionado por los tokens anteriores.",
        definition: "Un modelo autorregresivo genera secuencias prediciendo un elemento a la vez, condicionando cada predicción en todos los elementos generados anteriormente. En modelos de lenguaje como GPT, esto significa generar texto token por token de izquierda a derecha. El modelo aprende la probabilidad conjunta de una secuencia descomponiéndola en un producto de probabilidades condicionales. Este enfoque es poderoso para tareas de generación, pero puede ser lento en inferencia ya que los tokens deben producirse secuencialmente.",
        category: "Tipo de Modelo",
        relatedTerms: ["GPT", "Modelo de Lenguaje", "Token"]
    },
    {
        id: "bert",
        term: "BERT",
        letter: "B",
        summary: "Representaciones de Codificador Bidireccional de Transformers — modelo de lenguaje preentrenado de Google.",
        definition: "BERT (Bidirectional Encoder Representations from Transformers) es un modelo de lenguaje preentrenado desarrollado por Google en 2018. A diferencia de GPT, BERT utiliza un enfoque bidireccional, lo que significa que considera el contexto izquierdo y derecho simultáneamente al procesar texto. Se preentrenan usando dos tareas: Modelado de Lenguaje Enmascarado (MLM), donde se enmascaran tokens aleatorios y el modelo los predice, y Predicción de la Siguiente Oración (NSP). BERT estableció nuevos resultados de vanguardia en numerosos benchmarks de PLN y popularizó el paradigma de ajuste fino para tareas de PLN.",
        category: "Modelo",
        relatedTerms: ["Transformer", "Ajuste Fino", "Preentrenamiento", "GPT"]
    },
    {
        id: "chain-of-thought",
        term: "Prompting Cadena de Pensamiento",
        letter: "P",
        summary: "Técnica de prompting que anima a los LLM a razonar paso a paso antes de responder.",
        definition: "El prompting de Cadena de Pensamiento (CoT) es una técnica donde se guía al modelo para producir pasos de razonamiento intermedios antes de llegar a una respuesta final. Al incluir ejemplos que muestran razonamiento paso a paso en el prompt, o simplemente instruyendo al modelo a 'pensar paso a paso', el CoT mejora significativamente el rendimiento en tareas de razonamiento complejo como problemas matemáticos, acertijos lógicos y preguntas de múltiples pasos. Fue introducido por investigadores de Google y se ha convertido en una técnica estándar para obtener mejor razonamiento de los grandes modelos de lenguaje.",
        category: "Prompting",
        relatedTerms: ["Ingeniería de Prompts", "Aprendizaje Few-Shot", "Razonamiento"]
    },
    {
        id: "context-window",
        term: "Ventana de Contexto",
        letter: "V",
        summary: "La cantidad máxima de texto (tokens) que un modelo de lenguaje puede procesar a la vez.",
        definition: "La ventana de contexto se refiere al número máximo de tokens que un modelo de lenguaje puede considerar a la vez durante la inferencia. Los tokens dentro de la ventana de contexto pueden atenderse mutuamente a través del mecanismo de atención. Los primeros modelos como GPT-2 tenían ventanas de contexto de 1.024 tokens, mientras que los modelos modernos como GPT-4 Turbo admiten hasta 128.000 tokens. Una ventana de contexto más grande permite al modelo procesar documentos más largos, mantener conversaciones más largas y realizar tareas que requieren contexto extenso, pero también aumenta el costo computacional.",
        category: "Arquitectura",
        relatedTerms: ["Token", "Mecanismo de Atención", "Transformer"]
    },
    {
        id: "diffusion-model",
        term: "Modelo de Difusión",
        letter: "M",
        summary: "Modelo generativo que aprende a revertir un proceso de adición de ruido para generar datos.",
        definition: "Los modelos de difusión son una clase de modelos generativos que aprenden a generar datos revirtiendo un proceso gradual de adición de ruido. Durante el entrenamiento, los datos (por ejemplo, imágenes) se corrompen progresivamente con ruido gaussiano a lo largo de muchos pasos. El modelo aprende a revertir este proceso, comenzando desde ruido puro y eliminando el ruido iterativamente para producir muestras realistas. Stable Diffusion, DALL-E 2 y Midjourney son ejemplos prominentes. Los modelos de difusión han logrado resultados de vanguardia en generación de imágenes, síntesis de audio y generación de video.",
        category: "IA Generativa",
        relatedTerms: ["Stable Diffusion", "DALL-E", "Modelo Generativo", "Espacio Latente"]
    },
    {
        id: "embedding",
        term: "Embedding",
        letter: "E",
        summary: "Representación vectorial densa de datos (palabras, oraciones, imágenes) en un espacio continuo.",
        definition: "Un embedding es una representación aprendida de datos como un vector denso en un espacio continuo de alta dimensión. Las palabras, oraciones, imágenes u otras entidades se mapean a vectores de modo que los elementos semánticamente similares estén cerca en el espacio de embedding. Los embeddings de palabras como Word2Vec y GloVe fueron ejemplos tempranos; los modelos modernos producen embeddings contextuales donde la misma palabra tiene diferentes representaciones según el contexto. Los embeddings son fundamentales para la mayoría de los sistemas de IA modernos, permitiendo búsqueda eficiente por similitud, agrupamiento y rendimiento en tareas posteriores.",
        category: "Representación",
        relatedTerms: ["Base de Datos Vectorial", "Búsqueda Semántica", "Word2Vec", "Transformer"]
    },
    {
        id: "fine-tuning",
        term: "Ajuste Fino",
        letter: "A",
        summary: "Adaptar un modelo preentrenado a una tarea específica entrenándolo con datos específicos de esa tarea.",
        definition: "El ajuste fino es el proceso de tomar un modelo preentrenado y continuar entrenándolo en un conjunto de datos más pequeño y específico de la tarea. Esto permite al modelo adaptar su conocimiento general a un dominio o tarea particular mientras retiene las amplias capacidades aprendidas durante el preentrenamiento. El ajuste fino puede ser completo (actualizando todos los parámetros) o eficiente en parámetros (por ejemplo, LoRA, adaptadores). Es una piedra angular del PLN moderno y la visión por computadora, permitiendo alto rendimiento en tareas especializadas sin entrenar desde cero.",
        category: "Entrenamiento",
        relatedTerms: ["Preentrenamiento", "Aprendizaje por Transferencia", "LoRA", "RLHF"]
    },
    {
        id: "foundation-model",
        term: "Modelo Fundacional",
        letter: "M",
        summary: "Modelo grande entrenado con datos amplios que puede adaptarse a muchas tareas posteriores.",
        definition: "Un modelo fundacional es un modelo de IA grande entrenado en vastas cantidades de datos diversos utilizando aprendizaje auto-supervisado. El término fue acuñado por investigadores de Stanford en 2021. Los modelos fundacionales sirven como base que puede ajustarse finamente o consultarse mediante prompts para una amplia gama de tareas posteriores. Ejemplos incluyen GPT-4, PaLM, LLaMA, CLIP y Stable Diffusion. Su escala y generalidad los hacen puntos de partida poderosos, pero también generan preocupaciones sobre sesgo, seguridad y la concentración de capacidades de IA.",
        category: "Tipo de Modelo",
        relatedTerms: ["Preentrenamiento", "Ajuste Fino", "Modelo de Lenguaje Grande", "Aprendizaje por Transferencia"]
    },
    {
        id: "gpt",
        term: "GPT",
        letter: "G",
        summary: "Transformer Generativo Preentrenado — familia de grandes modelos de lenguaje autorregresivos de OpenAI.",
        definition: "GPT (Generative Pre-trained Transformer) es una familia de grandes modelos de lenguaje desarrollados por OpenAI. Los modelos GPT se entrenan usando preentrenamiento no supervisado en corpus de texto masivos, seguido de ajuste fino para tareas específicas. Utilizan una arquitectura Transformer de solo decodificador y generan texto de forma autorregresiva. GPT-3 (175B parámetros) demostró notables capacidades de aprendizaje few-shot. GPT-4 es un modelo multimodal capaz de procesar tanto texto como imágenes. La serie GPT ha sido fundamental para demostrar el poder de escalar los modelos de lenguaje.",
        category: "Modelo",
        relatedTerms: ["Transformer", "Modelo Autorregresivo", "OpenAI", "Modelo de Lenguaje Grande"]
    },
    {
        id: "generative-adversarial-network",
        term: "Red Generativa Adversarial",
        letter: "R",
        summary: "Marco donde dos redes neuronales compiten: un generador y un discriminador.",
        definition: "Una Red Generativa Adversarial (GAN) consiste en dos redes neuronales entrenadas simultáneamente en un marco competitivo. La red generadora crea muestras de datos sintéticos, mientras que la red discriminadora intenta distinguir los datos reales de los generados. A través de este proceso adversarial, el generador aprende a producir salidas cada vez más realistas. Las GAN fueron introducidas por Ian Goodfellow en 2014 y se han utilizado para síntesis de imágenes, transferencia de estilo, aumento de datos y más. Las variantes incluyen DCGAN, StyleGAN y CycleGAN.",
        category: "IA Generativa",
        relatedTerms: ["Modelo de Difusión", "Modelo Generativo", "Espacio Latente"]
    },
    {
        id: "hallucination",
        term: "Alucinación",
        letter: "A",
        summary: "Cuando un modelo de IA genera información que suena plausible pero es factualmente incorrecta o fabricada.",
        definition: "La alucinación en IA se refiere al fenómeno donde un modelo de lenguaje genera contenido que se afirma con confianza pero es factualmente incorrecto, sin sentido o completamente fabricado. Esto ocurre porque los LLM están entrenados para producir texto estadísticamente probable en lugar de hechos verificados. Las alucinaciones pueden variar desde errores sutiles (fechas, nombres incorrectos) hasta citas o eventos completamente inventados. Mitigar las alucinaciones es un gran desafío de investigación, con enfoques que incluyen la Generación Aumentada por Recuperación (RAG), mejores datos de entrenamiento y técnicas de alineación mejoradas.",
        category: "Seguridad y Alineación",
        relatedTerms: ["Generación Aumentada por Recuperación", "Alineación", "Modelo de Lenguaje Grande"]
    },
    {
        id: "inference",
        term: "Inferencia",
        letter: "I",
        summary: "El proceso de usar un modelo entrenado para generar predicciones o salidas sobre nuevos datos.",
        definition: "La inferencia es la fase donde se usa un modelo de IA entrenado para hacer predicciones o generar salidas sobre datos nuevos no vistos. A diferencia del entrenamiento, la inferencia no actualiza los pesos del modelo. Implica un pase hacia adelante a través de la red. Para los grandes modelos de lenguaje, la inferencia puede ser computacionalmente costosa debido al tamaño de los modelos y la naturaleza autorregresiva de la generación de texto. Técnicas como la cuantización, el procesamiento por lotes, la decodificación especulativa y la aceleración por hardware (GPUs, TPUs) se utilizan para hacer la inferencia más rápida y rentable.",
        category: "Despliegue",
        relatedTerms: ["Entrenamiento", "Cuantización", "Modelo Autorregresivo"]
    },
    {
        id: "instruction-tuning",
        term: "Ajuste por Instrucciones",
        letter: "A",
        summary: "Ajuste fino de un modelo de lenguaje con pares instrucción-respuesta para mejorar el seguimiento de instrucciones.",
        definition: "El ajuste por instrucciones es una técnica de ajuste fino donde un modelo de lenguaje preentrenado se entrena en un conjunto de datos de pares (instrucción, respuesta). Esto enseña al modelo a seguir instrucciones en lenguaje natural de manera más confiable. Modelos como InstructGPT, FLAN y Alpaca usan ajuste por instrucciones. Mejora significativamente la capacidad del modelo para generalizar a nuevas tareas descritas en lenguaje natural, sin requerir ejemplos específicos de la tarea. El ajuste por instrucciones a menudo se combina con RLHF para producir asistentes de IA útiles, inofensivos y honestos.",
        category: "Entrenamiento",
        relatedTerms: ["Ajuste Fino", "RLHF", "Ingeniería de Prompts", "Alineación"]
    },
    {
        id: "latent-space",
        term: "Espacio Latente",
        letter: "E",
        summary: "Espacio de representación comprimido y abstracto aprendido por un modelo para codificar datos.",
        definition: "El espacio latente es el espacio multidimensional en el que un modelo representa características comprimidas y abstractas de los datos. En los autoencoders, el codificador mapea los datos de entrada a un punto en el espacio latente, y el decodificador reconstruye los datos desde ese punto. En los modelos de difusión y VAEs, el espacio latente captura la estructura subyacente de la distribución de datos. Navegar y manipular el espacio latente permite la generación controlada, la interpolación entre puntos de datos y la transferencia de estilo. Comprender el espacio latente es clave para entender cómo funcionan los modelos generativos.",
        category: "Arquitectura",
        relatedTerms: ["Embedding", "Modelo de Difusión", "Autoencoder Variacional", "Modelo Generativo"]
    },
    {
        id: "large-language-model",
        term: "Modelo de Lenguaje Grande",
        letter: "M",
        summary: "Red neuronal con miles de millones de parámetros entrenada en corpus de texto masivos.",
        definition: "Un Modelo de Lenguaje Grande (LLM) es un tipo de red neuronal con miles de millones a billones de parámetros, entrenada en vastas cantidades de datos de texto. Los LLM aprenden patrones estadísticos en el lenguaje y pueden realizar una amplia gama de tareas incluyendo generación de texto, traducción, resumen, respuesta a preguntas y generación de código. LLMs notables incluyen GPT-4, Claude, Gemini, LLaMA y Mistral. Lo 'grande' se refiere tanto al número de parámetros como a la escala de los datos de entrenamiento. Los LLM exhiben capacidades emergentes que no fueron entrenadas explícitamente.",
        category: "Tipo de Modelo",
        relatedTerms: ["GPT", "Transformer", "Modelo Fundacional", "Capacidades Emergentes"]
    },
    {
        id: "lora",
        term: "LoRA",
        letter: "L",
        summary: "Adaptación de Bajo Rango — método eficiente de ajuste fino que entrena solo pequeñas matrices adaptadoras.",
        definition: "LoRA (Low-Rank Adaptation) es una técnica de ajuste fino eficiente en parámetros que congela los pesos del modelo preentrenado e inyecta matrices entrenables de descomposición de rango bajo en cada capa de la arquitectura Transformer. En lugar de actualizar todos los parámetros del modelo, LoRA entrena un número mucho menor de parámetros (a menudo <1% del original), haciendo el ajuste fino factible en hardware de consumo. LoRA se ha vuelto extremadamente popular para personalizar grandes modelos de lenguaje y modelos de generación de imágenes como Stable Diffusion, permitiendo adaptación específica del dominio sin los costos de ajuste fino completo.",
        category: "Entrenamiento",
        relatedTerms: ["Ajuste Fino", "Ajuste Fino Eficiente en Parámetros", "Transformer", "Stable Diffusion"]
    },
    {
        id: "multimodal",
        term: "IA Multimodal",
        letter: "I",
        summary: "Sistemas de IA que pueden procesar y generar múltiples tipos de datos (texto, imágenes, audio, video).",
        definition: "La IA multimodal se refiere a sistemas capaces de entender y generar múltiples modalidades de datos, como texto, imágenes, audio y video. A diferencia de los modelos unimodales que manejan solo un tipo de dato, los modelos multimodales pueden razonar entre modalidades. Ejemplos incluyen GPT-4V (texto + imágenes), Gemini (texto, imágenes, audio, video), CLIP (texto + imágenes) y Flamingo. Las capacidades multimodales permiten aplicaciones como respuesta a preguntas visuales, descripción de imágenes, transcripción de audio y comprensión de video. Esto se considera un paso clave hacia sistemas de IA más generales.",
        category: "Tipo de Modelo",
        relatedTerms: ["GPT", "CLIP", "Modelo Visión-Lenguaje", "Modelo Fundacional"]
    },
    {
        id: "neural-network",
        term: "Red Neuronal",
        letter: "R",
        summary: "Modelo computacional inspirado en el cerebro, compuesto de capas interconectadas de nodos.",
        definition: "Una red neuronal es un modelo de aprendizaje automático inspirado en la estructura de las redes neuronales biológicas del cerebro. Consiste en capas de nodos interconectados (neuronas), donde cada conexión tiene un peso aprendible. Los datos fluyen a través de la red (pase hacia adelante), y el modelo aprende ajustando los pesos para minimizar una función de pérdida mediante retropropagación. Las redes neuronales profundas con muchas capas son la base de la IA moderna, permitiendo avances en reconocimiento de imágenes, procesamiento de lenguaje natural, reconocimiento de voz e IA generativa.",
        category: "Fundamentos",
        relatedTerms: ["Aprendizaje Profundo", "Retropropagación", "Transformer", "Red Neuronal Convolucional"]
    },
    {
        id: "prompt-engineering",
        term: "Ingeniería de Prompts",
        letter: "I",
        summary: "La práctica de diseñar entradas para guiar a los modelos de IA hacia las salidas deseadas.",
        definition: "La ingeniería de prompts es la disciplina de diseñar y optimizar prompts de entrada para obtener comportamientos deseados de los modelos de lenguaje de IA. Dado que los LLM son sensibles a cómo se formulan las instrucciones, la ingeniería de prompts puede afectar significativamente la calidad de la salida. Las técnicas incluyen prompting zero-shot, prompting few-shot, prompting de cadena de pensamiento, prompting de rol y prompting de salida estructurada. La ingeniería de prompts ha surgido como una habilidad crítica para usar efectivamente los LLM en aplicaciones, y ha generado investigación en optimización automática de prompts y ataques de inyección de prompts.",
        category: "Prompting",
        relatedTerms: ["Prompting Cadena de Pensamiento", "Aprendizaje Few-Shot", "Modelo de Lenguaje Grande", "Aprendizaje Zero-Shot"]
    },
    {
        id: "pre-training",
        term: "Preentrenamiento",
        letter: "P",
        summary: "Entrenar un modelo con datos a gran escala antes del ajuste fino en tareas específicas.",
        definition: "El preentrenamiento es la fase inicial de entrenamiento de un modelo fundacional en un conjunto de datos grande y diverso utilizando objetivos auto-supervisados. Para los modelos de lenguaje, esto típicamente implica predecir tokens enmascarados (BERT) o tokens siguientes (GPT). Para los modelos de visión, puede implicar aprendizaje contrastivo o modelado de imágenes enmascaradas. El preentrenamiento permite al modelo aprender representaciones generales del lenguaje, la visión u otras modalidades. El modelo preentrenado luego se adapta para tareas específicas mediante ajuste fino o prompting, reduciendo drásticamente los datos y el cómputo necesarios para las tareas posteriores.",
        category: "Entrenamiento",
        relatedTerms: ["Ajuste Fino", "Modelo Fundacional", "Aprendizaje Auto-Supervisado", "Aprendizaje por Transferencia"]
    },
    {
        id: "quantization",
        term: "Cuantización",
        letter: "C",
        summary: "Reducir el tamaño del modelo representando los pesos con números de menor precisión.",
        definition: "La cuantización es una técnica de compresión de modelos que reduce la precisión numérica de los pesos y activaciones del modelo de formatos de alta precisión (por ejemplo, float32) a formatos de menor precisión (por ejemplo, int8, int4). Esto reduce los requisitos de memoria y acelera la inferencia con una pérdida mínima en la calidad del modelo. Las técnicas incluyen cuantización post-entrenamiento (PTQ) y entrenamiento con conciencia de cuantización (QAT). La cuantización ha sido crucial para desplegar grandes modelos de lenguaje en hardware de consumo, con herramientas como GPTQ, AWQ y llama.cpp que permiten ejecutar LLMs en laptops.",
        category: "Despliegue",
        relatedTerms: ["Inferencia", "Compresión de Modelos", "LoRA", "Modelo de Lenguaje Grande"]
    },
    {
        id: "rag",
        term: "Generación Aumentada por Recuperación",
        letter: "G",
        summary: "Combinar un sistema de recuperación con un modelo generativo para fundamentar respuestas en conocimiento externo.",
        definition: "La Generación Aumentada por Recuperación (RAG) es una arquitectura que mejora las respuestas de los modelos de lenguaje recuperando documentos relevantes de una base de conocimiento externa e incorporándolos en el prompt. El proceso implica: (1) codificar la consulta como un embedding, (2) buscar en una base de datos vectorial documentos relevantes, (3) incluir los documentos recuperados en el contexto del prompt, y (4) generar una respuesta fundamentada en la información recuperada. RAG reduce las alucinaciones, permite el acceso a información actualizada y permite a los LLM razonar sobre conocimiento privado o específico del dominio.",
        category: "Arquitectura",
        relatedTerms: ["Base de Datos Vectorial", "Embedding", "Alucinación", "Modelo de Lenguaje Grande"]
    },
    {
        id: "rlhf",
        term: "RLHF",
        letter: "R",
        summary: "Aprendizaje por Refuerzo con Retroalimentación Humana — alinear modelos de IA usando datos de preferencias humanas.",
        definition: "El Aprendizaje por Refuerzo con Retroalimentación Humana (RLHF) es una técnica de entrenamiento utilizada para alinear los modelos de lenguaje con los valores y preferencias humanas. El proceso implica: (1) ajuste fino supervisado con datos de demostración, (2) entrenamiento de un modelo de recompensa con comparaciones de preferencias humanas, y (3) optimización del modelo de lenguaje usando el modelo de recompensa mediante aprendizaje por refuerzo (típicamente PPO). RLHF se usó para entrenar InstructGPT y ChatGPT, mejorando dramáticamente su utilidad y seguridad. Es una técnica clave en la investigación de alineación de IA.",
        category: "Entrenamiento",
        relatedTerms: ["Alineación", "Ajuste Fino", "Ajuste por Instrucciones", "Modelo de Recompensa"]
    },
    {
        id: "self-attention",
        term: "Auto-Atención",
        letter: "A",
        summary: "Mecanismo donde cada elemento en una secuencia atiende a todos los demás elementos de la misma secuencia.",
        definition: "La auto-atención (también llamada atención intra-secuencia) es un mecanismo donde cada posición en una secuencia calcula pesos de atención sobre todas las demás posiciones en la misma secuencia. Para cada posición, se calculan consultas, claves y valores a partir de la entrada, y la salida es una suma ponderada de valores donde los pesos están determinados por la compatibilidad consulta-clave. La auto-atención permite al modelo capturar dependencias de largo alcance y relaciones entre cualquier dos posiciones independientemente de la distancia. Es la operación central en las arquitecturas Transformer y escala cuadráticamente con la longitud de la secuencia.",
        category: "Arquitectura",
        relatedTerms: ["Mecanismo de Atención", "Transformer", "Atención Multi-Cabeza", "Ventana de Contexto"]
    },
    {
        id: "stable-diffusion",
        term: "Stable Diffusion",
        letter: "S",
        summary: "Modelo de difusión latente de código abierto para generación de imágenes de alta calidad a partir de texto.",
        definition: "Stable Diffusion es un modelo de difusión latente de código abierto desarrollado por Stability AI, lanzado en 2022. Genera imágenes de alta calidad a partir de descripciones de texto realizando el proceso de difusión en un espacio latente comprimido en lugar del espacio de píxeles, haciéndolo computacionalmente eficiente. Utiliza un codificador de texto CLIP para condicionar la generación en prompts de texto y una red U-Net de eliminación de ruido. Stable Diffusion puede ejecutarse en GPUs de consumo y ha generado un gran ecosistema de modelos ajustados finamente, adaptaciones LoRA y herramientas como AUTOMATIC1111 y ComfyUI.",
        category: "IA Generativa",
        relatedTerms: ["Modelo de Difusión", "Espacio Latente", "LoRA", "DALL-E"]
    },
    {
        id: "token",
        term: "Token",
        letter: "T",
        summary: "La unidad básica de texto que procesan los modelos de lenguaje — aproximadamente una palabra o fragmento de palabra.",
        definition: "Un token es la unidad fundamental de texto que procesan los modelos de lenguaje. La tokenización divide el texto en tokens usando algoritmos como Byte-Pair Encoding (BPE) o WordPiece. Un token es típicamente una palabra, subpalabra o carácter, dependiendo del tokenizador. Por ejemplo, 'tokenización' podría dividirse en ['token', 'ización']. El número de tokens en un texto afecta el costo de procesamiento y el uso de la ventana de contexto. En promedio, 1 token ≈ 4 caracteres o 0,75 palabras en inglés. Los precios de las API para LLMs típicamente se basan en el conteo de tokens.",
        category: "Fundamentos",
        relatedTerms: ["Ventana de Contexto", "Tokenización", "Modelo de Lenguaje Grande", "Embedding"]
    },
    {
        id: "transformer",
        term: "Transformer",
        letter: "T",
        summary: "La arquitectura de red neuronal dominante para PLN, basada completamente en mecanismos de atención.",
        definition: "El Transformer es una arquitectura de red neuronal introducida en el artículo de 2017 'Attention Is All You Need' de Vaswani et al. en Google. Reemplazó las redes recurrentes (RNN, LSTM) con mecanismos de auto-atención, permitiendo el procesamiento paralelo de secuencias y una mejor captura de dependencias de largo alcance. La arquitectura consiste en pilas de codificadores y decodificadores, cada uno conteniendo auto-atención multi-cabeza y capas de retroalimentación con conexiones residuales y normalización de capas. Los Transformers son la base de prácticamente todos los grandes modelos de lenguaje, modelos de visión y modelos multimodales modernos.",
        category: "Arquitectura",
        relatedTerms: ["Auto-Atención", "Mecanismo de Atención", "BERT", "GPT", "Modelo de Lenguaje Grande"]
    },
    {
        id: "transfer-learning",
        term: "Aprendizaje por Transferencia",
        letter: "A",
        summary: "Aplicar conocimiento aprendido de una tarea o dominio para mejorar el rendimiento en otro.",
        definition: "El aprendizaje por transferencia es un paradigma de aprendizaje automático donde un modelo entrenado en una tarea se adapta para una tarea diferente pero relacionada. En el aprendizaje profundo, esto típicamente implica usar pesos de modelos preentrenados como inicialización para una nueva tarea. El modelo preentrenado ha aprendido representaciones de características útiles que se transfieren entre tareas. El aprendizaje por transferencia reduce drásticamente los datos y el cómputo necesarios para nuevas tareas. Es la base del flujo de trabajo moderno de desarrollo de IA: preentrenar con datos grandes, ajustar finamente con datos específicos de la tarea.",
        category: "Entrenamiento",
        relatedTerms: ["Preentrenamiento", "Ajuste Fino", "Modelo Fundacional", "Adaptación de Dominio"]
    },
    {
        id: "vector-database",
        term: "Base de Datos Vectorial",
        letter: "B",
        summary: "Base de datos optimizada para almacenar y buscar vectores de embedding de alta dimensión.",
        definition: "Una base de datos vectorial es una base de datos especializada diseñada para almacenar, indexar y buscar eficientemente vectores de embedding de alta dimensión. A diferencia de las bases de datos tradicionales que buscan por coincidencia exacta, las bases de datos vectoriales utilizan algoritmos de vecino más cercano aproximado (ANN) (por ejemplo, HNSW, IVF) para encontrar vectores más similares a un vector de consulta. Son infraestructura esencial para sistemas RAG, búsqueda semántica, motores de recomendación y aplicaciones basadas en similitud. Las bases de datos vectoriales populares incluyen Pinecone, Weaviate, Qdrant, Chroma y pgvector (extensión de PostgreSQL).",
        category: "Infraestructura",
        relatedTerms: ["Embedding", "Generación Aumentada por Recuperación", "Búsqueda Semántica"]
    },
    {
        id: "zero-shot-learning",
        term: "Aprendizaje Zero-Shot",
        letter: "A",
        summary: "La capacidad de un modelo para realizar tareas en las que nunca fue entrenado explícitamente.",
        definition: "El aprendizaje zero-shot se refiere a la capacidad de un modelo para realizar una tarea sin haber visto ningún ejemplo de esa tarea durante el entrenamiento o en el prompt. Los grandes modelos de lenguaje exhiben capacidades zero-shot porque su preentrenamiento en datos diversos les da amplio conocimiento y habilidades de razonamiento. Por ejemplo, GPT-4 puede traducir texto a un idioma para el que nunca fue entrenado explícitamente, o resolver acertijos lógicos novedosos. El rendimiento zero-shot es una medida clave de la capacidad de generalización de un modelo y se contrasta con el aprendizaje few-shot, donde se proporcionan un pequeño número de ejemplos.",
        category: "Paradigma de Aprendizaje",
        relatedTerms: ["Aprendizaje Few-Shot", "Ingeniería de Prompts", "Modelo de Lenguaje Grande", "Generalización"]
    },
    // --- Nuevos términos ---
    {
        id: "agent",
        term: "Agente",
        letter: "A",
        summary: "Sistema de IA que percibe su entorno y toma acciones para alcanzar objetivos.",
        definition: "En IA, un agente es cualquier software o programa que interactúa con el mundo (o una simulación) recibiendo entradas y produciendo salidas o acciones. Los agentes operan dentro de un entorno, perciben su estado a través de sensores o entradas de datos, y actúan sobre él mediante actuadores o llamadas a APIs. Van desde sistemas simples basados en reglas (como un termostato) hasta agentes autónomos complejos impulsados por grandes modelos de lenguaje que pueden planificar, usar herramientas, navegar por la web, escribir código y ejecutar tareas de múltiples pasos. Los sistemas de IA agéntica se usan cada vez más en automatización, robótica y asistentes de IA.",
        category: "Arquitectura",
        relatedTerms: ["Aprendizaje por Refuerzo", "Modelo de Lenguaje Grande", "Uso de Herramientas", "IA Autónoma"]
    },
    {
        id: "generative-ai",
        term: "IA Generativa",
        letter: "I",
        summary: "IA que crea contenido nuevo como texto, imágenes, audio o video.",
        definition: "La IA generativa se refiere a una clase de modelos de IA diseñados para producir datos originales que se asemejan a los ejemplos con los que fueron entrenados. Estos sistemas aprenden patrones estadísticos de grandes conjuntos de datos y usan esos patrones para generar nuevo contenido — incluyendo texto natural, imágenes realistas, música, video, código y modelos 3D. Las arquitecturas clave incluyen Transformers (para texto), Modelos de Difusión (para imágenes) y GANs. Ejemplos prominentes son GPT-4 (texto), DALL-E y Stable Diffusion (imágenes), Sora (video) y MusicLM (audio). La IA generativa ha transformado las industrias creativas, el desarrollo de software y la investigación científica.",
        category: "IA Generativa",
        relatedTerms: ["Modelo de Lenguaje Grande", "Modelo de Difusión", "Red Generativa Adversarial", "Modelo Fundacional"]
    },
    {
        id: "context",
        term: "Contexto",
        letter: "C",
        summary: "Información que rodea una entrada y ayuda al modelo a interpretar el significado con precisión.",
        definition: "En IA, el contexto se refiere a cualquier información de fondo relevante que ayuda a un modelo a entender o responder adecuadamente. Para los modelos de lenguaje, el contexto incluye el historial de conversación, las instrucciones del sistema, la intención del usuario, el tema, el tono y cualquier documento proporcionado en el prompt. La cantidad de contexto que un modelo puede usar está limitada por su ventana de contexto. El uso efectivo del contexto es fundamental para respuestas precisas y relevantes — un modelo con contexto insuficiente puede malinterpretar consultas ambiguas. La ingeniería de contexto — la práctica de estructurar las entradas de contexto de forma óptima — se ha convertido en una habilidad clave para construir aplicaciones de IA.",
        category: "Fundamentos",
        relatedTerms: ["Ventana de Contexto", "Ingeniería de Prompts", "Ingeniería de Contexto", "Generación Aumentada por Recuperación"]
    },
    {
        id: "claude",
        term: "Claude",
        letter: "C",
        summary: "Familia de modelos de asistente de IA desarrollados por Anthropic, conocidos por su seguridad y utilidad.",
        definition: "Claude es una familia de grandes modelos de lenguaje desarrollados por Anthropic, una empresa de seguridad en IA. Los modelos Claude están diseñados con un fuerte énfasis en ser útiles, inofensivos y honestos (el marco 'HHH'). Se entrenan usando técnicas de IA Constitucional (CAI) y RLHF para alinear el comportamiento del modelo con los valores humanos. Claude destaca en tareas como resumen, razonamiento, asistencia en programación, análisis y escritura creativa. La familia de modelos Claude incluye múltiples niveles (Haiku, Sonnet, Opus) optimizados para diferentes equilibrios de velocidad y capacidad. Claude se usa ampliamente a través de API y en los productos de consumo de Anthropic.",
        category: "Modelo",
        relatedTerms: ["Modelo de Lenguaje Grande", "RLHF", "Alineación", "OpenAI", "GPT"]
    },
    {
        id: "openai",
        term: "OpenAI",
        letter: "O",
        summary: "Organización de investigación en IA que desarrolla sistemas avanzados incluyendo la serie GPT.",
        definition: "OpenAI es un laboratorio de investigación en IA y empresa tecnológica fundada en 2015, con la misión de garantizar que la inteligencia artificial general (AGI) beneficie a toda la humanidad. Es responsable de desarrollar algunos de los sistemas de IA más influyentes, incluyendo la serie GPT de modelos de lenguaje, los modelos de generación de imágenes DALL-E, Codex (generación de código), Whisper (reconocimiento de voz) y el modelo de generación de video Sora. OpenAI también creó ChatGPT, una de las aplicaciones de IA más utilizadas en la historia. La organización opera como una empresa de beneficio limitado con una organización sin fines de lucro como matriz, equilibrando operaciones comerciales con investigación de seguridad.",
        category: "Organización",
        relatedTerms: ["GPT", "ChatGPT", "DALL-E", "Modelo de Lenguaje Grande", "AGI"]
    },
    {
        id: "chatgpt",
        term: "ChatGPT",
        letter: "C",
        summary: "Asistente de IA conversacional de OpenAI, construido sobre los grandes modelos de lenguaje GPT.",
        definition: "ChatGPT es una aplicación de IA conversacional desarrollada por OpenAI, lanzada en noviembre de 2022. Está construida sobre la serie GPT de grandes modelos de lenguaje (inicialmente GPT-3.5, luego GPT-4) y ajustada finamente con RLHF para ser un asistente conversacional útil. ChatGPT puede mantener diálogos de múltiples turnos, responder preguntas, escribir y depurar código, redactar documentos, resumir texto, traducir idiomas y realizar muchas otras tareas de lenguaje. Se convirtió en una de las aplicaciones de consumo de más rápido crecimiento en la historia, alcanzando 100 millones de usuarios en dos meses. Admite plugins, entrada de imágenes (GPT-4V) y GPTs personalizados.",
        category: "Modelo",
        relatedTerms: ["GPT", "OpenAI", "RLHF", "Modelo de Lenguaje Grande", "Ajuste por Instrucciones"]
    },
    {
        id: "deep-learning",
        term: "Aprendizaje Profundo",
        letter: "A",
        summary: "Subconjunto del aprendizaje automático que usa redes neuronales con muchas capas para aprender de los datos.",
        definition: "El aprendizaje profundo es un subcampo del aprendizaje automático que utiliza redes neuronales artificiales con muchas capas (de ahí 'profundo') para aprender automáticamente representaciones jerárquicas de datos brutos. Cada capa aprende características cada vez más abstractas — por ejemplo, en el reconocimiento de imágenes, las capas tempranas detectan bordes, las capas intermedias detectan formas y las capas posteriores detectan objetos. El aprendizaje profundo ha impulsado avances en visión por computadora (CNNs), procesamiento de lenguaje natural (Transformers), reconocimiento de voz (RNNs, atención) e IA generativa (GANs, modelos de difusión). Requiere grandes conjuntos de datos y cómputo significativo, típicamente usando GPUs o TPUs.",
        category: "Fundamentos",
        relatedTerms: ["Red Neuronal", "Transformer", "Red Neuronal Convolucional", "Retropropagación"]
    },
    {
        id: "reinforcement-learning",
        term: "Aprendizaje por Refuerzo",
        letter: "A",
        summary: "Paradigma de aprendizaje donde un agente aprende recibiendo recompensas o penalizaciones por sus acciones.",
        definition: "El aprendizaje por refuerzo (RL) es un paradigma de aprendizaje automático donde un agente aprende a tomar decisiones interactuando con un entorno. El agente observa el estado actual, toma una acción, recibe una señal de recompensa (positiva o negativa) y actualiza su política para maximizar la recompensa acumulada a lo largo del tiempo. Los algoritmos clave incluyen Q-learning, SARSA y métodos de gradiente de política como PPO y A3C. El RL ha logrado rendimiento sobrehumano en juegos (AlphaGo, Atari), robótica y conducción autónoma. En el contexto de los LLMs, el RLHF usa RL para alinear las salidas del modelo con las preferencias humanas.",
        category: "Paradigma de Aprendizaje",
        relatedTerms: ["RLHF", "Agente", "Aprendizaje Profundo", "Gradiente de Política"]
    },
    {
        id: "computer-vision",
        term: "Visión por Computadora",
        letter: "V",
        summary: "IA que permite a las máquinas interpretar y comprender información visual de imágenes y videos.",
        definition: "La visión por computadora es un campo de la IA centrado en permitir que las máquinas extraigan información significativa de entradas visuales como imágenes y videos. Las tareas principales incluyen clasificación de imágenes (identificar qué hay en una imagen), detección de objetos (localizar objetos), segmentación semántica (etiquetar cada píxel) y generación de imágenes. La visión por computadora moderna depende en gran medida de las redes neuronales convolucionales (CNNs) y los Vision Transformers (ViT). Las aplicaciones abarcan vehículos autónomos, imágenes médicas, reconocimiento facial, realidad aumentada, control de calidad en manufactura y análisis de imágenes satelitales.",
        category: "Campo",
        relatedTerms: ["Red Neuronal", "Red Neuronal Convolucional", "Aprendizaje Profundo", "IA Multimodal"]
    },
    {
        id: "nlp",
        term: "PLN",
        letter: "P",
        summary: "Procesamiento de Lenguaje Natural — IA que permite a las computadoras entender y generar lenguaje humano.",
        definition: "El Procesamiento de Lenguaje Natural (PLN) es una rama de la IA que combina lingüística, ciencias de la computación y aprendizaje automático para permitir que las computadoras procesen, comprendan y generen lenguaje humano. El PLN abarca una amplia gama de tareas: clasificación de texto, análisis de sentimientos, traducción automática, respuesta a preguntas, resumen, reconocimiento de entidades nombradas y sistemas de diálogo. El PLN moderno está dominado por modelos basados en Transformers como BERT y GPT. El PLN impulsa aplicaciones incluyendo motores de búsqueda, asistentes virtuales, chatbots, correctores gramaticales y sistemas de moderación de contenido.",
        category: "Campo",
        relatedTerms: ["Comprensión del Lenguaje Natural", "Generación de Lenguaje Natural", "Transformer", "BERT", "Modelo de Lenguaje Grande"]
    },
    {
        id: "supervised-learning",
        term: "Aprendizaje Supervisado",
        letter: "A",
        summary: "Aprendizaje automático donde los modelos se entrenan con pares de entrada-salida etiquetados.",
        definition: "El aprendizaje supervisado es un paradigma de aprendizaje automático donde un modelo se entrena en un conjunto de datos de ejemplos etiquetados — pares de entradas y sus salidas correctas. El modelo aprende un mapeo de entradas a salidas minimizando la diferencia entre sus predicciones y las etiquetas verdaderas. Las tareas comunes incluyen clasificación (predecir una categoría) y regresión (predecir un valor continuo). Ejemplos incluyen detección de spam, clasificación de imágenes y predicción de precios. El aprendizaje supervisado requiere datos etiquetados, que pueden ser costosos de obtener, pero es el paradigma de ML más utilizado en sistemas de producción.",
        category: "Paradigma de Aprendizaje",
        relatedTerms: ["Aprendizaje No Supervisado", "Ajuste Fino", "Red Neuronal", "Aprendizaje por Transferencia"]
    },
    {
        id: "unsupervised-learning",
        term: "Aprendizaje No Supervisado",
        letter: "A",
        summary: "Aprendizaje automático sobre datos no etiquetados para descubrir patrones o estructura ocultos.",
        definition: "El aprendizaje no supervisado es un paradigma de aprendizaje automático donde los modelos aprenden patrones y estructura de los datos sin salidas etiquetadas. El modelo debe descubrir por sí solo la organización subyacente de los datos. Las técnicas comunes incluyen agrupamiento (agrupar puntos de datos similares, por ejemplo, K-means), reducción de dimensionalidad (por ejemplo, PCA, t-SNE, autoencoders) y estimación de densidad. El aprendizaje no supervisado es valioso cuando los datos etiquetados son escasos o costosos. El aprendizaje auto-supervisado — donde los modelos generan sus propias etiquetas a partir de datos no etiquetados — es una variante poderosa utilizada para preentrenar grandes modelos de lenguaje.",
        category: "Paradigma de Aprendizaje",
        relatedTerms: ["Aprendizaje Auto-Supervisado", "Aprendizaje Supervisado", "Agrupamiento", "Preentrenamiento"]
    },
    {
        id: "data-mining",
        term: "Minería de Datos",
        letter: "M",
        summary: "El proceso de descubrir patrones e información útil en grandes conjuntos de datos.",
        definition: "La minería de datos es el proceso de aplicar técnicas estadísticas, matemáticas y computacionales para extraer patrones significativos, correlaciones e información de grandes colecciones de datos. Se sitúa en la intersección de la estadística, el aprendizaje automático y los sistemas de bases de datos. Las tareas comunes de minería de datos incluyen clasificación, agrupamiento, aprendizaje de reglas de asociación (por ejemplo, análisis de cesta de mercado), detección de anomalías y regresión. La minería de datos es fundamental para la inteligencia empresarial, la detección de fraudes, el descubrimiento científico y los sistemas de recomendación. La minería de datos moderna aprovecha cada vez más las técnicas de aprendizaje automático e IA.",
        category: "Ciencia de Datos",
        relatedTerms: ["Aprendizaje Automático", "Aprendizaje Supervisado", "Aprendizaje No Supervisado", "Reconocimiento de Patrones"]
    },
    {
        id: "entity-annotation",
        term: "Anotación de Entidades",
        letter: "A",
        summary: "Etiquetar entidades significativas (nombres, lugares, fechas) en texto o datos para el entrenamiento de IA.",
        definition: "La anotación de entidades es el proceso de marcar entidades — como nombres de personas, organizaciones, ubicaciones, fechas y nombres de productos — en conjuntos de datos de texto para que los modelos de IA puedan aprender a reconocerlos. Es un paso crítico en la creación de datos de entrenamiento para sistemas de Reconocimiento de Entidades Nombradas (NER) y otras tareas de PLN. La anotación puede realizarse manualmente por anotadores humanos o de forma semi-automática usando modelos preentrenados. La anotación de entidades de alta calidad es esencial para entrenar sistemas precisos de extracción de información utilizados en búsqueda, grafos de conocimiento y pipelines de procesamiento de documentos.",
        category: "PLN",
        relatedTerms: ["Extracción de Entidades", "Reconocimiento de Entidades Nombradas", "PLN", "Aprendizaje Supervisado"]
    },
    {
        id: "entity-extraction",
        term: "Extracción de Entidades",
        letter: "E",
        summary: "Identificación y categorización automática de entidades clave en texto no estructurado.",
        definition: "La extracción de entidades, también conocida como Reconocimiento de Entidades Nombradas (NER), es una tarea de PLN donde un modelo de IA identifica y clasifica automáticamente entidades nombradas en texto no estructurado en categorías predefinidas como personas, organizaciones, ubicaciones, fechas, valores monetarios y más. Por ejemplo, en la oración 'Apple fue fundada por Steve Jobs en Cupertino en 1976', un modelo NER extraería Apple (organización), Steve Jobs (persona), Cupertino (ubicación) y 1976 (fecha). La extracción de entidades es fundamental para la recuperación de información, la construcción de grafos de conocimiento y la inteligencia documental.",
        category: "PLN",
        relatedTerms: ["Anotación de Entidades", "PLN", "Comprensión del Lenguaje Natural", "Extracción de Información"]
    },
    {
        id: "intent",
        term: "Intención",
        letter: "I",
        summary: "El objetivo o propósito detrás de la entrada de un usuario en un sistema de IA conversacional.",
        definition: "En la IA conversacional y el PLN, la intención se refiere al objetivo o propósito subyacente que un usuario pretende lograr con su entrada. Por ejemplo, la consulta '¿Cómo estará el tiempo mañana?' tiene la intención 'obtener pronóstico del tiempo'. El reconocimiento de intención (o clasificación de intención) es la tarea de identificar automáticamente la intención del usuario a partir de su enunciado. Es un componente central de los sistemas de diálogo, asistentes virtuales y chatbots. Los sistemas modernos usan clasificadores de aprendizaje automático o grandes modelos de lenguaje para detectar la intención, permitiendo el enrutamiento y la generación de respuestas apropiadas.",
        category: "PLN",
        relatedTerms: ["Comprensión del Lenguaje Natural", "PLN", "Sistema de Diálogo", "Extracción de Entidades"]
    },
    {
        id: "model",
        term: "Modelo",
        letter: "M",
        summary: "Sistema matemático entrenado con datos para hacer predicciones, clasificaciones o generar salidas.",
        definition: "En IA y aprendizaje automático, un modelo es un sistema computacional que ha aprendido patrones de datos de entrenamiento y puede aplicar ese conocimiento a nuevas entradas. Los modelos se definen por su arquitectura (la estructura del cómputo) y sus parámetros (los pesos aprendidos). Después del entrenamiento, un modelo puede hacer predicciones (regresión, clasificación), generar contenido (lenguaje, imágenes) o tomar acciones (agentes). El término abarca desde simples modelos de regresión lineal hasta redes neuronales con miles de millones de parámetros. La selección, entrenamiento, evaluación y despliegue de modelos son las etapas centrales del ciclo de vida del aprendizaje automático.",
        category: "Fundamentos",
        relatedTerms: ["Red Neuronal", "Entrenamiento", "Inferencia", "Modelo Fundacional"]
    },
    {
        id: "nlu",
        term: "Comprensión del Lenguaje Natural",
        letter: "C",
        summary: "IA que interpreta el significado, la intención y el contexto del lenguaje humano.",
        definition: "La Comprensión del Lenguaje Natural (CLN/NLU) es un subcampo del PLN centrado en permitir que las máquinas comprendan el significado, la intención, el sentimiento y el contexto del lenguaje humano — yendo más allá del procesamiento superficial de texto. Las tareas de CLN incluyen reconocimiento de intención, análisis de sentimientos, etiquetado de roles semánticos, resolución de correferencias y comprensión lectora. La CLN es el componente de 'comprensión' de los sistemas de IA conversacional, permitiéndoles interpretar correctamente lo que los usuarios quieren decir en lugar de solo lo que dicen. La CLN moderna está impulsada por grandes modelos de lenguaje preentrenados como BERT y sus variantes.",
        category: "PLN",
        relatedTerms: ["PLN", "Generación de Lenguaje Natural", "Intención", "BERT", "Análisis de Sentimientos"]
    },
    {
        id: "nlg",
        term: "Generación de Lenguaje Natural",
        letter: "G",
        summary: "IA que produce texto o voz coherente y similar al humano a partir de datos u otras entradas.",
        definition: "La Generación de Lenguaje Natural (GLN/NLG) es un subcampo del PLN centrado en producir automáticamente texto o voz coherente, fluido y contextualmente apropiado a partir de datos estructurados, conocimiento u otras entradas. Las tareas de GLN incluyen resumen de texto, generación de informes, generación de respuestas en diálogos, traducción automática y escritura creativa. La GLN moderna está dominada por modelos de lenguaje autorregresivos como GPT-4, que generan texto token por token. La GLN es el componente de 'generación' de la IA conversacional y se usa en chatbots, periodismo automatizado, sistemas de datos a texto y asistentes virtuales.",
        category: "PLN",
        relatedTerms: ["PLN", "Comprensión del Lenguaje Natural", "Modelo de Lenguaje Grande", "Modelo Autorregresivo"]
    },
    {
        id: "overfitting",
        term: "Sobreajuste",
        letter: "S",
        summary: "Cuando un modelo memoriza los datos de entrenamiento con demasiada precisión y no generaliza a datos nuevos.",
        definition: "El sobreajuste ocurre cuando un modelo de aprendizaje automático aprende los datos de entrenamiento con demasiada precisión — incluyendo su ruido y fluctuaciones aleatorias — en lugar de los patrones generales subyacentes. Un modelo sobreajustado funciona muy bien en los datos de entrenamiento pero mal en datos de prueba no vistos, porque esencialmente ha memorizado los ejemplos de entrenamiento en lugar de aprender patrones transferibles. El sobreajuste es más probable con modelos complejos y conjuntos de datos pequeños. Las estrategias comunes de mitigación incluyen regularización (L1/L2), dropout, parada temprana, aumento de datos y validación cruzada.",
        category: "Fundamentos",
        relatedTerms: ["Aprendizaje Supervisado", "Regularización", "Generalización", "Red Neuronal"]
    },
    {
        id: "pattern-recognition",
        term: "Reconocimiento de Patrones",
        letter: "R",
        summary: "La capacidad de los algoritmos para identificar estructuras o regularidades recurrentes en los datos.",
        definition: "El reconocimiento de patrones es la capacidad de los algoritmos y sistemas de IA para detectar, clasificar y responder a estructuras, regularidades o relaciones recurrentes en los datos. Es una de las tareas fundamentales de la IA y el aprendizaje automático, subyacente a aplicaciones como el reconocimiento de imágenes (detectar rostros u objetos), el reconocimiento de voz (identificar fonemas y palabras), el reconocimiento de escritura a mano, la detección de anomalías y la identificación biométrica. El reconocimiento de patrones moderno se logra en gran medida mediante el aprendizaje profundo, donde las redes neuronales aprenden automáticamente representaciones de características jerárquicas de datos brutos.",
        category: "Fundamentos",
        relatedTerms: ["Aprendizaje Profundo", "Visión por Computadora", "Red Neuronal", "Clasificación"]
    },
    {
        id: "context-engineering",
        term: "Ingeniería de Contexto",
        letter: "I",
        summary: "Estructurar y optimizar la información proporcionada a los modelos de IA para mejorar la calidad de la salida.",
        definition: "La ingeniería de contexto es la práctica de diseñar y estructurar deliberadamente la información proporcionada a un modelo de IA — incluyendo prompts del sistema, historial de conversación, documentos recuperados, ejemplos y datos del entorno — para maximizar la calidad y relevancia de sus salidas. Va más allá de la ingeniería de prompts básica para abarcar la arquitectura de información completa alrededor de una llamada al modelo: qué incluir, cómo formatearlo, qué recuperar y cómo priorizar. A medida que los sistemas de IA se vuelven más capaces, la ingeniería de contexto ha surgido como una disciplina crítica para construir aplicaciones de IA confiables y precisas.",
        category: "Prompting",
        relatedTerms: ["Ingeniería de Prompts", "Generación Aumentada por Recuperación", "Ventana de Contexto", "Contexto"]
    },
    {
        id: "turing-test",
        term: "Test de Turing",
        letter: "T",
        summary: "Prueba propuesta por Alan Turing para evaluar si el comportamiento de una máquina es indistinguible del de un humano.",
        definition: "El Test de Turing, propuesto por el matemático Alan Turing en su artículo de 1950 'Computing Machinery and Intelligence', es una prueba de la capacidad de una máquina para exhibir comportamiento inteligente indistinguible del de un humano. En la formulación original (el Juego de Imitación), un evaluador humano conversa por texto con un humano y una máquina sin saber cuál es cuál; si el evaluador no puede distinguir de manera confiable la máquina del humano, se dice que la máquina ha pasado la prueba. Aunque influyente como referencia filosófica, el Test de Turing ahora se considera insuficiente como medida de la verdadera inteligencia de la IA, ya que los LLMs modernos pueden pasarlo sin poseer comprensión genuina.",
        category: "Fundamentos",
        relatedTerms: ["Modelo de Lenguaje Grande", "AGI", "IA Estrecha", "ChatGPT"]
    },
    {
        id: "narrow-ai",
        term: "IA Estrecha",
        letter: "I",
        summary: "IA diseñada para realizar un conjunto específico y limitado de tareas — también llamada IA Débil.",
        definition: "La IA Estrecha (también llamada IA Débil o Inteligencia Artificial Estrecha, ANI) se refiere a sistemas de IA diseñados y entrenados para realizar una tarea específica o un conjunto limitado de tareas relacionadas. A diferencia de la hipotética Inteligencia Artificial General (AGI), la IA estrecha no puede transferir su conocimiento a dominios fuera de su entrenamiento. Ejemplos incluyen clasificadores de imágenes, filtros de spam, motores de recomendación, programas de ajedrez y sistemas de reconocimiento de voz. A pesar de la etiqueta 'estrecha', los sistemas modernos de IA estrecha como GPT-4 pueden desempeñarse de manera impresionante en muchas tareas de lenguaje, difuminando la línea con capacidades más generales.",
        category: "Fundamentos",
        relatedTerms: ["AGI", "Test de Turing", "Modelo de Lenguaje Grande", "Modelo Fundacional"]
    },
    {
        id: "spec-driven-development",
        term: "Desarrollo Guiado por Especificaciones",
        letter: "D",
        summary: "Metodología de desarrollo donde especificaciones detalladas guían el diseño, la implementación y las pruebas.",
        definition: "El desarrollo guiado por especificaciones (DGE) es una metodología de ingeniería de software en la que se escriben especificaciones claras y estructuradas antes de que comience la implementación. Estas especificaciones definen el comportamiento esperado, entradas, salidas, casos extremos y criterios de aceptación. En el contexto de los sistemas de IA, el DGE es cada vez más importante para definir cómo deben comportarse los componentes de IA, qué salidas son aceptables y cómo evaluar la corrección. Se alinea estrechamente con el desarrollo guiado por pruebas (TDD) y el desarrollo guiado por comportamiento (BDD), y está ganando tracción como forma de construir aplicaciones de IA más confiables y auditables.",
        category: "Ingeniería",
        relatedTerms: ["Ingeniería de Contexto", "Ingeniería de Prompts", "Alineación", "Evaluación"]
    },
    {
        id: "nlu-nlg-nlp",
        term: "CLN / GLN / PLN",
        letter: "C",
        summary: "Los tres pilares de la IA del lenguaje: Comprensión, Generación y Procesamiento.",
        definition: "La CLN (Comprensión del Lenguaje Natural), la GLN (Generación de Lenguaje Natural) y el PLN (Procesamiento de Lenguaje Natural) son tres subcampos estrechamente relacionados pero distintos de la IA del lenguaje. El PLN es el término más amplio, que abarca todas las técnicas computacionales para procesar el lenguaje humano. La CLN se centra específicamente en la comprensión — extraer significado, intención y estructura del texto. La GLN se centra en la producción — generar lenguaje coherente y contextualmente apropiado a partir de datos o conocimiento. Los grandes modelos de lenguaje modernos como GPT-4 integran las tres capacidades: procesan la entrada (PLN), la comprenden (CLN) y generan respuestas (GLN) en una arquitectura unificada.",
        category: "PLN",
        relatedTerms: ["PLN", "Comprensión del Lenguaje Natural", "Generación de Lenguaje Natural", "Modelo de Lenguaje Grande"]
    }
]

export default glossaryDataEs

