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
    },
    {
        "id": "tokenization",
        "term": "Tokenización",
        "letter": "T",
        "summary": "El proceso de convertir texto en tokens que un LLM puede procesar.",
        "definition": "La tokenización es el paso de preprocesamiento en el que el texto crudo se divide en tokens: las unidades atómicas que un LLM entiende. Los tokenizadores como Byte-Pair Encoding (BPE) o SentencePiece dividen el texto en unidades de subpalabras, equilibrando el tamaño del vocabulario con la cobertura. Diferentes modelos usan diferentes tokenizadores con diferentes vocabularios, por lo que el mismo texto puede producir conteos de tokens distintos entre modelos.",
        "category": "Fundamentos",
        "relatedTerms": ["Token", "Ventana de Contexto", "Token EOS"]
    },
    {
        "id": "eos-token",
        "term": "Token EOS (Fin de Secuencia)",
        "letter": "T",
        "summary": "Un token especial que indica el fin de la generación de un modelo.",
        "definition": "El token de Fin de Secuencia (EOS) es un token especial que indica que un LLM ha completado su generación. Cada familia de modelos usa su propio token EOS; por ejemplo, GPT-4 usa <|endoftext|>, Llama 3 usa <|eot_id|> y SmolLM2 usa <|im_end|>. El modelo deja de generar una vez que predice este token. Los tokens EOS forman parte de un conjunto más amplio de tokens especiales que estructuran las entradas y salidas del modelo.",
        "category": "Fundamentos",
        "relatedTerms": ["Token", "Modelo Autorregresivo", "Tokens Especiales"]
    },
    {
        "id": "special-tokens",
        "term": "Tokens Especiales",
        "letter": "T",
        "summary": "Tokens reservados usados para estructurar las entradas y salidas de un LLM, como marcar el inicio o fin de mensajes.",
        "definition": "Los tokens especiales son tokens reservados en el vocabulario de un LLM que tienen significado estructural en lugar de contenido lingüístico. Se usan para delimitar el inicio o fin de una secuencia, separar instrucciones del sistema de mensajes del usuario, o señalar el uso de herramientas y llamadas a funciones. Diferentes modelos usan diferentes conjuntos de tokens especiales, lo que hace que la migración de prompts entre modelos no sea trivial.",
        "category": "Fundamentos",
        "relatedTerms": ["Token EOS", "Token", "Prompt de Sistema", "Plantilla de Chat"]
    },
    {
        "id": "encoder",
        "term": "Codificador (Encoder)",
        "letter": "C",
        "summary": "Un tipo de Transformer que convierte texto en representaciones vectoriales densas (embeddings).",
        "definition": "Un codificador es una variante del Transformer que procesa una secuencia de entrada y produce una representación vectorial densa (embedding) de esa entrada. Los modelos basados en codificadores como BERT son adecuados para tareas como clasificación de texto, búsqueda semántica y Reconocimiento de Entidades Nombradas (NER). A diferencia de los decodificadores, los codificadores no generan nuevo texto token a token; en cambio, producen representaciones de tamaño fijo de toda la entrada.",
        "category": "Fundamentos",
        "relatedTerms": ["Decodificador", "Transformer", "Embedding", "Búsqueda Semántica"]
    },
    {
        "id": "decoder",
        "term": "Decodificador (Decoder)",
        "letter": "D",
        "summary": "Un tipo de Transformer diseñado para generar nuevos tokens, uno a la vez, para tareas como la generación de texto.",
        "definition": "Un decodificador es una variante del Transformer que genera texto nuevo prediciendo un token a la vez, condicionado por todos los tokens anteriores. Los modelos solo de decodificador como GPT-4, Llama y Mistral son la arquitectura más común para los LLM modernos. Se utilizan para generación de texto, chatbots, generación de código y razonamiento.",
        "category": "Fundamentos",
        "relatedTerms": ["Codificador", "Transformer", "Modelo Autorregresivo", "Modelo de Lenguaje Grande"]
    },
    {
        "id": "prompt",
        "term": "Prompt",
        "letter": "P",
        "summary": "El texto de entrada proporcionado a un LLM para guiar su respuesta.",
        "definition": "Un prompt es el texto de entrada que se pasa a un LLM para instruirlo o guiar su generación. Los prompts pueden incluir instrucciones, ejemplos, contexto, historial de conversación, documentos recuperados y directivas de nivel sistema. La redacción, estructura y contenido de un prompt afectan significativamente la calidad y relevancia de la salida del modelo. El diseño del prompt es una de las formas más accesibles y poderosas de mejorar el rendimiento de los LLM sin reentrenamiento.",
        "category": "Prompting",
        "relatedTerms": ["Ingeniería de Prompts", "Prompt de Sistema", "Cadena de Pensamiento", "Ingeniería de Contexto"]
    },
    {
        "id": "system-prompt",
        "term": "Prompt de Sistema",
        "letter": "P",
        "summary": "Un prompt especial que configura el comportamiento, la personalidad y las restricciones de un LLM antes de la conversación.",
        "definition": "Un prompt de sistema es un bloque de instrucciones que se pasa a un LLM antes de cualquier mensaje del usuario, utilizado para configurar el comportamiento, la personalidad, el tono y las restricciones del modelo. Típicamente contiene descripciones de roles, requisitos de formato de salida, reglas de seguridad e instrucciones específicas de la tarea. Los prompts de sistema son una herramienta principal para personalizar el comportamiento de los LLM en aplicaciones de producción.",
        "category": "Prompting",
        "relatedTerms": ["Ingeniería de Prompts", "Prompt", "Plantilla de Chat", "Ingeniería de Contexto"]
    },
    {
        "id": "few-shot-prompting",
        "term": "Prompting de Pocos Ejemplos (Few-Shot)",
        "letter": "P",
        "summary": "Una técnica de prompting donde se incluyen unos pocos ejemplos de entrada-salida en el prompt para guiar al modelo.",
        "definition": "El prompting de pocos ejemplos (también llamado n-shot prompting) es una técnica donde un pequeño número de ejemplos de entrada-salida se integran directamente en el prompt para demostrar el formato y estilo de salida deseados. Aprovecha la capacidad de aprendizaje en contexto del LLM: el modelo infiere el patrón a partir de los ejemplos y lo aplica a nuevas entradas. Como regla general, proporcionar al menos 5 ejemplos ayuda al modelo a generalizar.",
        "category": "Prompting",
        "relatedTerms": ["Aprendizaje en Contexto", "Prompting Sin Ejemplos", "Ingeniería de Prompts", "Cadena de Pensamiento"]
    },
    {
        "id": "in-context-learning",
        "term": "Aprendizaje en Contexto",
        "letter": "A",
        "summary": "La capacidad de un LLM de aprender una nueva tarea a partir de ejemplos dados directamente en el prompt, sin actualizar pesos.",
        "definition": "El aprendizaje en contexto es la capacidad de los LLM de adaptarse a nuevas tareas o comportamientos basándose únicamente en ejemplos e instrucciones proporcionados dentro del prompt, sin ningún cambio en los pesos subyacentents del modelo. Esto contrasta con el ajuste fino, que requiere actualizar los parámetros del modelo. El aprendizaje en contexto es habilitado por la capacidad del modelo de detectar y generalizar patrones a partir de los pocos ejemplos que ve.",
        "category": "Prompting",
        "relatedTerms": ["Prompting de Pocos Ejemplos", "Prompting Sin Ejemplos", "Ajuste Fino", "Ingeniería de Prompts"]
    },
    {
        "id": "zero-shot-prompting",
        "term": "Prompting Sin Ejemplos (Zero-Shot)",
        "letter": "P",
        "summary": "Pedir a un LLM que realice una tarea solo con instrucciones, sin proporcionar ejemplos.",
        "definition": "El prompting sin ejemplos (zero-shot) es una técnica donde se pide a un LLM que realice una tarea basándose únicamente en una instrucción en lenguaje natural, sin proporcionar ejemplos de entrada-salida. El modelo se basa completamente en su conocimiento preentrenado y su capacidad de seguir instrucciones. Si bien es simple y flexible, el rendimiento sin ejemplos puede ser menos fiable que el prompting con pocos ejemplos para tareas complejas o especializadas.",
        "category": "Prompting",
        "relatedTerms": ["Prompting de Pocos Ejemplos", "Aprendizaje en Contexto", "Ingeniería de Prompts"]
    },
    {
        "id": "semantic-search",
        "term": "Búsqueda Semántica",
        "letter": "B",
        "summary": "Una técnica de búsqueda que encuentra documentos basándose en el significado e intención, no en coincidencias exactas de palabras.",
        "definition": "La búsqueda semántica es un enfoque de recuperación que encuentra documentos relevantes comparando el significado de una consulta con el significado de los documentos, usando vectores de embeddings y métricas de similitud. A diferencia de la búsqueda por palabras clave, la búsqueda semántica puede encontrar resultados relevantes incluso cuando las palabras exactas de la consulta no aparecen en el documento. Es un componente clave de los pipelines de RAG.",
        "category": "Recuperación",
        "relatedTerms": ["Embedding", "Base de Datos Vectorial", "Búsqueda Híbrida", "Generación Aumentada por Recuperación"]
    },
    {
        "id": "hybrid-search",
        "term": "Búsqueda Híbrida",
        "letter": "B",
        "summary": "Un método de recuperación que combina búsqueda semántica (vectorial) con búsqueda tradicional por palabras clave para mejores resultados.",
        "definition": "La búsqueda híbrida combina la búsqueda semántica (usando similitud de embeddings) con la búsqueda basada en palabras clave (como BM25 o TF-IDF) para recuperar documentos. Los dos rankings se fusionan típicamente usando técnicas como la Fusión de Rango Recíproco (RRF). La búsqueda híbrida frecuentemente supera a cualquiera de los métodos por sí solos, especialmente para consultas factuales precisas o terminología especializada. Es considerada una mejor práctica en sistemas RAG de producción.",
        "category": "Recuperación",
        "relatedTerms": ["Búsqueda Semántica", "Generación Aumentada por Recuperación", "Base de Datos Vectorial"]
    },
    {
        "id": "grounding",
        "term": "Anclaje (Grounding)",
        "letter": "A",
        "summary": "Anclar las salidas de los LLM a fuentes externas verificables para mejorar la precisión factual.",
        "definition": "El anclaje (grounding) es la práctica de conectar las salidas generadas por un LLM a fuentes externas confiables como documentos recuperados, bases de datos o datos en tiempo real. Una respuesta anclada es trazable hasta una fuente, reduciendo el riesgo de alucinación. RAG es la técnica de anclaje más común. Es especialmente importante en aplicaciones de IA de alto riesgo como sistemas médicos, legales o financieros donde la precisión factual es crítica.",
        "category": "Evaluación",
        "relatedTerms": ["Alucinación", "Generación Aumentada por Recuperación", "Guardianes"]
    },
    {
        "id": "tool-use",
        "term": "Uso de Herramientas (Tool Use)",
        "letter": "U",
        "summary": "La capacidad de un LLM o agente de IA de llamar funciones externas o APIs para ampliar sus capacidades.",
        "definition": "El uso de herramientas (también llamado llamada a funciones) es la capacidad de un LLM de invocar herramientas o APIs externas como parte de su proceso de razonamiento. Las herramientas pueden incluir búsqueda web, intérpretes de código, consultas a bases de datos, calculadoras y APIs personalizadas. El modelo recibe una descripción de las herramientas disponibles, decide cuál llamar y con qué argumentos, y procesa la salida de la herramienta para informar su siguiente acción.",
        "category": "Agentes",
        "relatedTerms": ["Agente de IA", "Llamada a Funciones", "Bucle Agéntico", "ReAct", "MCP"]
    },
    {
        "id": "react",
        "term": "ReAct (Razonar + Actuar)",
        "letter": "R",
        "summary": "Un marco donde un LLM intercala pasos de razonamiento con acciones para resolver tareas de forma iterativa.",
        "definition": "ReAct es un marco de agente que intercala el razonamiento (pensar qué hacer a continuación) con actuar (llamar a herramientas o tomar acciones) en un bucle iterativo. El modelo produce un Pensamiento describiendo su razonamiento, luego una Acción especificando una llamada a herramienta, luego una Observación reportando el resultado, y repite hasta completar la tarea. ReAct mejora los enfoques de razonamiento puro al anclar los pensamientos del agente en observaciones reales del entorno.",
        "category": "Agentes",
        "relatedTerms": ["Agente de IA", "Uso de Herramientas", "Bucle Agéntico", "Cadena de Pensamiento", "Ciclo Pensamiento-Acción-Observación"]
    },
    {
        "id": "agentic-loop",
        "term": "Bucle Agéntico",
        "letter": "B",
        "summary": "El ciclo iterativo de razonamiento, acción y observación que impulsa el comportamiento de un agente de IA.",
        "definition": "El bucle agéntico es el ciclo operacional central de un agente de IA: el agente recibe un objetivo u observación, razona sobre la mejor siguiente acción, ejecuta esa acción (p. ej., llama a una herramienta), recibe una observación del resultado, actualiza su comprensión y repite el ciclo hasta lograr el objetivo o determinar que no puede avanzar. Este bucle permite a los agentes abordar tareas complejas de múltiples pasos que no pueden resolverse en una sola llamada al LLM.",
        "category": "Agentes",
        "relatedTerms": ["Agente de IA", "ReAct", "Uso de Herramientas", "Ciclo Pensamiento-Acción-Observación", "Orquestación"]
    },
    {
        "id": "multi-agent-system",
        "term": "Sistema Multi-Agente",
        "letter": "S",
        "summary": "Un sistema compuesto por múltiples agentes de IA que colaboran para realizar tareas complejas.",
        "definition": "Un sistema multi-agente es una arquitectura donde múltiples agentes de IA — cada uno con su propio rol, herramientas y capacidades — trabajan juntos (o en una jerarquía estructurada) para realizar tareas demasiado complejas para un solo agente. Los patrones comunes incluyen un agente coordinador/orquestador que delega subtareas a agentes trabajadores especializados. Los sistemas multi-agente pueden mejorar el paralelismo, la especialización y la robustez, pero introducen complejidad de coordinación.",
        "category": "Agentes",
        "relatedTerms": ["Agente de IA", "Orquestación", "Bucle Agéntico", "Uso de Herramientas"]
    },
    {
        "id": "orchestration",
        "term": "Orquestación",
        "letter": "O",
        "summary": "La coordinación de llamadas a LLM, invocaciones de herramientas y flujos de datos para construir pipelines de IA complejos.",
        "definition": "La orquestación se refiere al diseño y gestión de secuencias de llamadas a LLM, invocaciones de herramientas, recuperaciones de memoria y transformaciones de datos que implementan colectivamente un flujo de trabajo o agente de IA complejo. Frameworks de orquestación como LangChain, LlamaIndex y smolagents proporcionan abstracciones para construir estos pipelines. Un buen diseño de orquestación prioriza el determinismo, la observabilidad y el manejo elegante de errores.",
        "category": "Agentes",
        "relatedTerms": ["Agente de IA", "Sistema Multi-Agente", "LLMOps", "Uso de Herramientas", "Bucle Agéntico"]
    },
    {
        "id": "memory",
        "term": "Memoria",
        "letter": "M",
        "summary": "Mecanismos que permiten a los agentes de IA almacenar y recuperar información a través de múltiples pasos o sesiones.",
        "definition": "La memoria en sistemas de agentes de IA se refiere a mecanismos que permiten que la información persista y se recupere más allá de la ventana de contexto inmediata. Los tipos comunes de memoria incluyen: memoria a corto plazo (la ventana de contexto actual), memoria episódica (registros de interacciones pasadas), memoria semántica (conocimiento recuperado de almacenes vectoriales) y memoria procedimental (habilidades o planes aprendidos). La gestión efectiva de la memoria es esencial para agentes que manejan tareas largas.",
        "category": "Agentes",
        "relatedTerms": ["Agente de IA", "Ventana de Contexto", "Generación Aumentada por Recuperación", "Bucle Agéntico"]
    },
    {
        "id": "mcp",
        "term": "Protocolo de Contexto de Modelo (MCP)",
        "letter": "P",
        "summary": "Un protocolo estándar abierto para conectar modelos de IA con herramientas externas y fuentes de datos.",
        "definition": "El Protocolo de Contexto de Modelo (MCP) es un protocolo abierto que estandariza cómo los modelos y agentes de IA se conectan con herramientas externas, APIs y fuentes de datos. Desarrollado por Anthropic y soportado por frameworks como Kiro, MCP permite a los sistemas de IA acceder a contexto desde diversas fuentes externas — bases de datos, sistemas de archivos, APIs — a través de una interfaz consistente. Los servidores MCP exponen capacidades que los agentes pueden descubrir y usar.",
        "category": "Agentes",
        "relatedTerms": ["Uso de Herramientas", "Agente de IA", "Llamada a Funciones", "Orquestación"]
    },
    {
        "id": "function-calling",
        "term": "Llamada a Funciones (Function Calling)",
        "letter": "L",
        "summary": "Una capacidad del modelo para generar llamadas estructuradas a funciones o APIs predefinidas como parte de su salida.",
        "definition": "La llamada a funciones (también conocida como uso de herramientas) es una capacidad en la que un LLM genera JSON estructurado o código para invocar una función o API externa predefinida, en lugar de generar texto sin formato. El modelo recibe un esquema que describe las funciones disponibles y sus parámetros, decide cuál función llamar y con qué argumentos, y devuelve una llamada estructurada que una aplicación puede ejecutar. La llamada a funciones es esencial para construir agentes confiables.",
        "category": "Agentes",
        "relatedTerms": ["Uso de Herramientas", "Agente de IA", "MCP", "Salida Estructurada"]
    },
    {
        "id": "structured-output",
        "term": "Salida Estructurada",
        "letter": "S",
        "summary": "Restringir la salida de un LLM a un esquema definido (p. ej., JSON) para facilitar el procesamiento posterior.",
        "definition": "La salida estructurada es la práctica de restringir la generación de un LLM para que siga un formato específico — como JSON, XML o un esquema tipado — en lugar de producir texto libre. Esto simplifica el análisis posterior, la integración con otros sistemas y la validación automática. Las técnicas incluyen instrucciones de formato basadas en prompts, parsers de salida y características nativas del modelo como el modo JSON o el muestreo con restricciones gramaticales.",
        "category": "Prompting",
        "relatedTerms": ["Llamada a Funciones", "Uso de Herramientas", "Ingeniería de Prompts", "Guardianes"]
    },
    {
        "id": "guardrails",
        "term": "Guardianes (Guardrails)",
        "letter": "G",
        "summary": "Reglas y mecanismos que restringen o validan las entradas y salidas de los LLM para garantizar un comportamiento seguro y apropiado.",
        "definition": "Los guardianes son mecanismos de seguridad y calidad aplicados a las entradas o salidas de los LLM para hacer cumplir restricciones como el cumplimiento de políticas de contenido, la validez del formato de salida, la precisión factual y la adherencia a instrucciones. Pueden implementarse como instrucciones basadas en prompts, clasificadores de salida, filtros basados en reglas o modelos de validación separados. Los guardianes son un componente central de los sistemas de IA de producción responsables.",
        "category": "Evaluación",
        "relatedTerms": ["Alucinación", "LLM como Juez", "Salida Estructurada", "Anclaje", "Seguridad"]
    },
    {
        "id": "evals",
        "term": "Evals (Evaluaciones)",
        "letter": "E",
        "summary": "Pruebas y benchmarks usados para medir el rendimiento de LLMs o sistemas de IA en tareas específicas.",
        "definition": "Los evals (abreviatura de evaluaciones) son pruebas sistemáticas diseñadas para medir el rendimiento, la precisión, la seguridad y la confiabilidad de los LLMs o sistemas de IA. Van desde pruebas unitarias sobre pares específicos de entrada-salida hasta grandes benchmarks que cubren capacidades diversas. Conjuntos de evals sólidos son fundamentales para construir productos de IA confiables: permiten iteración rápida, detectan regresiones y brindan confianza antes del despliegue.",
        "category": "Evaluación",
        "relatedTerms": ["LLM como Juez", "Guardianes", "Alucinación", "Benchmarks"]
    },
    {
        "id": "llmops",
        "term": "LLMOps",
        "letter": "L",
        "summary": "Las prácticas operacionales para desplegar, monitorear, versionar y mantener aplicaciones basadas en LLMs en producción.",
        "definition": "LLMOps es el conjunto de prácticas, herramientas y cultura para operacionalizar aplicaciones basadas en LLMs — análogo a MLOps para el aprendizaje automático tradicional. Cubre el versionado de prompts, el versionado de modelos, las pruebas A/B, el monitoreo (seguimiento de la calidad de entrada-salida a lo largo del tiempo), el registro, la optimización de latencia y la gestión de costos. El objetivo principal de LLMOps es permitir ciclos de iteración más rápidos y mantener la confiabilidad en producción.",
        "category": "Operaciones",
        "relatedTerms": ["Evals", "Monitoreo", "Orquestación", "Versionado de Prompts"]
    },
    {
        "id": "temperature",
        "term": "Temperatura",
        "letter": "T",
        "summary": "Un parámetro de muestreo que controla la aleatoriedad y creatividad de las salidas de los LLM.",
        "definition": "La temperatura es un parámetro de decodificación que controla la diversidad de la salida de un LLM escalando la distribución de probabilidad sobre los tokens antes del muestreo. Una temperatura de 0 hace al modelo determinista (siempre elige el token de mayor probabilidad), mientras que valores más altos (p. ej., 0.7–1.0) introducen más aleatoriedad y creatividad. Elegir la temperatura correcta depende de la tarea: baja para tareas factuales, más alta para generación creativa.",
        "category": "Inferencia",
        "relatedTerms": ["Muestreo", "Muestreo Top-P", "Inferencia", "Estrategia de Decodificación"]
    },
    {
        "id": "top-p-sampling",
        "term": "Muestreo Top-P (Muestreo de Núcleo)",
        "letter": "M",
        "summary": "Una estrategia de muestreo que selecciona del conjunto más pequeño de tokens cuya probabilidad acumulada supera p.",
        "definition": "El muestreo Top-P, también llamado muestreo de núcleo, es una estrategia de decodificación donde el modelo muestrea solo del conjunto más pequeño de tokens cuya masa de probabilidad acumulada supera un umbral p (p. ej., 0.9). Esto asegura que los tokens improbables queden excluidos del muestreo mientras se permite cierta variabilidad. Top-P se usa a menudo junto con la temperatura para controlar la diversidad de la salida.",
        "category": "Inferencia",
        "relatedTerms": ["Temperatura", "Muestreo", "Estrategia de Decodificación"]
    },
    {
        "id": "beam-search",
        "term": "Búsqueda de Haz (Beam Search)",
        "letter": "B",
        "summary": "Una estrategia de decodificación que explora múltiples secuencias de tokens simultáneamente para encontrar la salida de mayor probabilidad.",
        "definition": "La búsqueda de haz es una estrategia de decodificación que mantiene un número fijo (el 'ancho del haz') de secuencias candidatas en cada paso de generación, expandiendo cada candidato con los tokens más probables y manteniendo solo los mejores candidatos. A diferencia de la decodificación codiciosa, la búsqueda de haz puede encontrar secuencias de mayor calidad general al explorar alternativas. Es comúnmente usada en traducción y resumen.",
        "category": "Inferencia",
        "relatedTerms": ["Temperatura", "Muestreo Top-P", "Estrategia de Decodificación", "Modelo Autorregresivo"]
    },
    {
        "id": "caching",
        "term": "Caché (Caching)",
        "letter": "C",
        "summary": "Almacenar salidas de LLM o cómputos intermedios para evitar llamadas redundantes y costosas al modelo.",
        "definition": "El caché en sistemas de LLMs se refiere a almacenar los resultados de llamadas al modelo o cómputos intermedios para que puedan reutilizarse sin volver a ejecutar el modelo. El caché de prompts puede reducir significativamente la latencia y los costos de API para aplicaciones con prompts fijos grandes o repetitivos. El caché semántico va más lejos al recuperar respuestas cacheadas para consultas semánticamente similares. El caché es una optimización subutilizada pero de alto impacto.",
        "category": "Operaciones",
        "relatedTerms": ["LLMOps", "Inferencia", "Latencia", "Ingeniería de Prompts"]
    },
    {
        "id": "alignment",
        "term": "Alineación (Alignment)",
        "letter": "A",
        "summary": "El proceso de asegurar que el comportamiento del modelo de IA coincide con los valores, intenciones y preferencias humanas.",
        "definition": "La alineación se refiere al desafío y la práctica de asegurar que los sistemas de IA se comporten de manera consistente con los valores, intenciones y preferencias humanas, especialmente a medida que los modelos se vuelven más capaces y autónomos. Las técnicas de alineación incluyen RLHF, IA Constitucional y optimización directa de preferencias (DPO). Una IA desalineada puede producir comportamientos dañinos, engañosos o no deseados incluso cuando es técnicamente capaz.",
        "category": "Seguridad",
        "relatedTerms": ["RLHF", "Seguridad", "Guardianes", "IA Constitucional"]
    },
    {
        "id": "safety",
        "term": "Seguridad (Safety)",
        "letter": "S",
        "summary": "Prácticas y mecanismos para evitar que los modelos de IA generen contenido dañino, peligroso o inapropiado.",
        "definition": "La seguridad en IA se refiere a las prácticas, técnicas y directrices diseñadas para evitar que los modelos de IA produzcan contenido o tomen acciones que sean dañinas, ilegales, engañosas o peligrosas. Las medidas de seguridad en aplicaciones de LLMs incluyen clasificadores de moderación de contenido, guardianes, red-teaming, entrenamiento de rechazo y filtrado de salidas. Las consideraciones de seguridad son especialmente críticas en sistemas agénticos donde el modelo puede tomar acciones del mundo real con consecuencias irreversibles.",
        "category": "Seguridad",
        "relatedTerms": ["Alineación", "Guardianes", "Red-Teaming", "IA Responsable"]
    },
    {
        "id": "human-in-the-loop",
        "term": "Humano en el Bucle (HITL)",
        "letter": "H",
        "summary": "Un patrón de diseño donde los humanos revisan o aprueban las salidas de la IA en pasos críticos antes de que surtan efecto.",
        "definition": "El Humano en el Bucle (HITL) es un patrón de diseño donde la supervisión humana se incorpora en puntos clave de un flujo de trabajo de IA — por ejemplo, exigiendo que un humano revise y apruebe una acción generada por el modelo antes de que se ejecute. HITL es especialmente importante en aplicaciones de alto riesgo y sistemas agénticos donde los errores pueden tener consecuencias significativas. Las interfaces HITL bien diseñadas mantienen a los humanos informados y en control sin crear fricción excesiva.",
        "category": "Operaciones",
        "relatedTerms": ["Agente de IA", "Seguridad", "Guardianes", "Bucle Agéntico"]
    },
    {
        "id": "agentic-ide",
        "term": "IDE Agéntico",
        "letter": "I",
        "summary": "Un entorno de desarrollo impulsado por agentes de IA capaces de escribir, editar y gestionar código de forma autónoma.",
        "definition": "Un IDE agéntico es un entorno de desarrollo de software que integra profundamente agentes de IA en el flujo de trabajo de codificación, permitiendo a la IA no solo sugerir completados sino planificar, generar, refactorizar y gestionar código de forma autónoma en un proyecto. Ejemplos incluyen Kiro, que cuenta con capacidades como specs (planificación estructurada de características), steering (reglas de IA personalizadas), hooks (disparadores automatizados) e integración MCP. Los IDEs agénticos representan un cambio de la IA como asistente de código a la IA como socio ingeniero colaborativo.",
        "category": "Herramientas",
        "relatedTerms": ["Agente de IA", "MCP", "Specs", "Steering", "Hooks"]
    },
    {
        "id": "specs",
        "term": "Specs (Especificaciones)",
        "letter": "S",
        "summary": "Documentos estructurados que definen los requisitos, diseño y plan de implementación de una característica antes de codificar.",
        "definition": "En el contexto de herramientas de desarrollo agéntico como Kiro, los specs (especificaciones) son documentos estructurados que un agente de IA ayuda a generar antes de escribir código. Un spec típicamente incluye un documento de requisitos, un documento de diseño del sistema y un conjunto de tareas de implementación. Este enfoque impulsado por specs asegura que el código generado por IA esté alineado con la intención del usuario y la arquitectura del proyecto antes de que comience cualquier implementación.",
        "category": "Herramientas",
        "relatedTerms": ["IDE Agéntico", "Steering", "Hooks", "Agente de IA"]
    },
    {
        "id": "steering",
        "term": "Steering (Guía)",
        "letter": "S",
        "summary": "Reglas y contexto personalizados proporcionados a un sistema de IA para guiar su comportamiento en un proyecto específico.",
        "definition": "El steering se refiere al uso de reglas, restricciones y contexto personalizados para guiar el comportamiento de un sistema de IA dentro de un entorno específico. En Kiro, los archivos de steering permiten a los desarrolladores definir convenciones específicas del proyecto — como estándares de codificación, patrones arquitectónicos o bibliotecas preferidas — que el agente de IA sigue consistentemente en todas las interacciones. El steering es análogo a un prompt de sistema persistente con alcance de proyecto.",
        "category": "Herramientas",
        "relatedTerms": ["IDE Agéntico", "Specs", "Prompt de Sistema", "Ingeniería de Contexto"]
    },
    {
        "id": "hooks",
        "term": "Hooks (Ganchos)",
        "letter": "H",
        "summary": "Disparadores automatizados que ejecutan acciones de IA en respuesta a eventos específicos de desarrollo.",
        "definition": "Los hooks son disparadores automatizados en entornos de desarrollo agéntico (como Kiro) que ejecutan acciones impulsadas por IA en respuesta a eventos predefinidos — como guardar un archivo, fallar una prueba o abrir un pull request. Los hooks permiten automatizar tareas repetitivas sin intervención manual, integrando la asistencia de IA directamente en el flujo de trabajo de desarrollo. Son una forma de automatización orientada a eventos para acciones de agentes de IA.",
        "category": "Herramientas",
        "relatedTerms": ["IDE Agéntico", "Specs", "Agente de IA", "Orquestación"]
    },
    {
        "id": "model-selection",
        "term": "Selección de Modelo",
        "letter": "S",
        "summary": "El proceso de elegir el LLM más apropiado para una tarea dada según capacidad, costo y latencia.",
        "definition": "La selección de modelo es la práctica de elegir el LLM más adecuado para un caso de uso específico equilibrando factores como la complejidad de la tarea, la calidad requerida, el costo de inferencia y las restricciones de latencia. Un principio clave es comenzar con el modelo más pequeño que logre calidad aceptable, evitando pagar de más por capacidades que no se necesitan. La selección de modelo también implica versionar y fijar versiones de modelo para evitar cambios de comportamiento inesperados.",
        "category": "Operaciones",
        "relatedTerms": ["Inferencia", "LLMOps", "Ajuste Fino", "Latencia"]
    },
    {
        "id": "latency",
        "term": "Latencia",
        "letter": "L",
        "summary": "El tiempo que tarda un LLM en comenzar o completar una respuesta después de recibir un prompt.",
        "definition": "La latencia en sistemas de LLMs se refiere al retraso entre enviar un prompt y recibir una respuesta. Se mide comúnmente como Tiempo al Primer Token (TTFT) — cuánto tarda el modelo en comenzar a transmitir salida — y el tiempo total de generación. La latencia es un factor crítico en la experiencia del usuario y está influenciada por el tamaño del modelo, el hardware, la longitud del contexto y la arquitectura del sistema. Las técnicas de optimización incluyen caché, modelos más pequeños, streaming y agrupación.",
        "category": "Operaciones",
        "relatedTerms": ["Inferencia", "Caché", "Selección de Modelo", "Streaming"]
    },
    {
        "id": "streaming",
        "term": "Streaming",
        "letter": "S",
        "summary": "Entregar la salida de un LLM token a token a medida que se genera, en lugar de esperar la respuesta completa.",
        "definition": "El streaming es una técnica donde la salida de un LLM se transmite al usuario token a token a medida que se genera, en lugar de esperar la respuesta completa. Esto mejora significativamente la latencia percibida ya que los usuarios ven la respuesta formarse en tiempo real. El streaming es soportado por la mayoría de las principales APIs de LLM y es práctica estándar para interfaces de chat y tareas de generación de forma larga.",
        "category": "Inferencia",
        "relatedTerms": ["Latencia", "Inferencia", "Modelo Autorregresivo"]
    },
    {
        "id": "agentic-rag",
        "term": "RAG Agéntico",
        "letter": "R",
        "summary": "Una arquitectura RAG donde un agente de IA decide dinámicamente cuándo y cómo recuperar información.",
        "definition": "El RAG Agéntico combina las capacidades de recuperación de la Generación Aumentada por Recuperación con las capacidades de planificación y toma de decisiones de un agente de IA. En lugar de recuperar siempre documentos como un paso fijo de preprocesamiento, el agente decide dinámicamente cuándo se necesita la recuperación, qué consultas emitir y cómo sintetizar resultados de múltiples rondas de recuperación. El RAG Agéntico permite un razonamiento más complejo sobre grandes bases de conocimiento.",
        "category": "Agentes",
        "relatedTerms": ["Generación Aumentada por Recuperación", "Agente de IA", "Bucle Agéntico", "Uso de Herramientas", "Memoria"]
    },
    {
        "id": "prompt-versioning",
        "term": "Versionado de Prompts",
        "letter": "V",
        "summary": "Rastrear y gestionar los cambios en los prompts a lo largo del tiempo, análogo al control de versiones para código.",
        "definition": "El versionado de prompts es la práctica de rastrear, almacenar y gestionar diferentes versiones de los prompts usados en aplicaciones de LLMs de producción. Al igual que el control de versiones de software, permite a los equipos revertir a versiones anteriores de prompts, comparar el rendimiento entre versiones y desplegar cambios de forma segura. Dado que los cambios de prompts pueden afectar significativamente el comportamiento del modelo, el versionado es una parte crítica de LLMOps.",
        "category": "Operaciones",
        "relatedTerms": ["LLMOps", "Evals", "Ingeniería de Prompts"]
    },
    {
        "id": "product-market-fit",
        "term": "Ajuste Producto-Mercado (PMF) para IA",
        "letter": "A",
        "summary": "Validar que un producto impulsado por IA resuelve una necesidad real del usuario antes de invertir en infraestructura.",
        "definition": "En el contexto del desarrollo de productos LLM, el Ajuste Producto-Mercado (PMF) se refiere a la validación de que un producto impulsado por IA genuinamente resuelve una necesidad real del usuario antes de realizar grandes inversiones en infraestructura como entrenar modelos personalizados. Una heurística ampliamente citada es 'Sin GPUs antes del PMF': comenzar con APIs de inferencia, ingeniería de prompts y RAG antes de comprometerse con entrenamiento personalizado.",
        "category": "Estrategia",
        "relatedTerms": ["LLMOps", "Ajuste Fino", "Inferencia", "Selección de Modelo"]
    },
    {
        "id": "data-flywheel",
        "term": "Volante de Datos (Data Flywheel)",
        "letter": "V",
        "summary": "Un ciclo auto-reforzante donde el uso del producto genera datos que mejoran la IA, lo que atrae más usuarios.",
        "definition": "El volante de datos es un patrón estratégico en el desarrollo de productos de IA donde el uso del producto genera datos de entrenamiento y evaluación, que se usan para mejorar el modelo de IA, lo que mejora el producto, atrae más usuarios, y así sucesivamente. Construir un volante de datos temprano — instrumentando la producción para capturar pares de entrada-salida, retroalimentación de usuarios y casos extremos — crea una ventaja compuesta con el tiempo.",
        "category": "Estrategia",
        "relatedTerms": ["Evals", "Ajuste Fino", "LLMOps", "Ajuste Producto-Mercado"]
    },
    {
        "id": "red-teaming",
        "term": "Red-Teaming",
        "letter": "R",
        "summary": "La práctica de probar adversarialmente un sistema de IA para encontrar fallos de seguridad y confiabilidad antes del despliegue.",
        "definition": "El red-teaming es una práctica de seguridad tomada de la ciberseguridad donde un equipo dedicado intenta encontrar fallas, jailbreaks, modos de fallo y comportamientos dañinos en un sistema de IA al sondearlo adversarialmente. El red-teaming descubre problemas que los evals estándar pueden pasar por alto, incluyendo casos extremos, inyecciones de prompts y violaciones de políticas de contenido. Se considera una mejor práctica para el despliegue responsable de IA.",
        "category": "Seguridad",
        "relatedTerms": ["Seguridad", "Alineación", "Guardianes", "Evals"]
    },
    {
        "id": "reward-model",
        "term": "Modelo de Recompensa",
        "letter": "M",
        "summary": "Un modelo entrenado para puntuar salidas de LLMs según la preferencia humana, usado en el entrenamiento RLHF.",
        "definition": "Un modelo de recompensa es una red neuronal entrenada para predecir cuánto preferiría un humano una salida del modelo sobre otra. Se entrena con datos de comparación humana (p. ej., anotadores que clasifican pares de salidas del modelo) y produce una puntuación escalar para cualquier salida dada. En RLHF, el modelo de recompensa sirve como proxy de la preferencia humana, guiando las actualizaciones de la política del LLM durante el aprendizaje por refuerzo.",
        "category": "Entrenamiento",
        "relatedTerms": ["RLHF", "Ajuste Fino", "Alineación", "Evals"]
    },
    {
        "id": "autonomous-agent",
        "term": "Agente Autónomo",
        "letter": "A",
        "summary": "Un agente de IA capaz de completar tareas complejas de múltiples pasos con mínima intervención humana.",
        "definition": "Un agente autónomo es un agente de IA diseñado para operar con un alto grado de independencia, capaz de planificar, ejecutar y adaptarse a través de largas secuencias de acciones para lograr un objetivo especificado por el usuario. A diferencia de los chatbots simples o asistentes de un solo turno, los agentes autónomos gestionan sus propias llamadas a herramientas, memoria y toma de decisiones en flujos de trabajo extendidos. El agente autónomo de Kiro, por ejemplo, puede ejecutar tareas agénticas de principio a fin a través de una interfaz CLI o IDE.",
        "category": "Agentes",
        "relatedTerms": ["Agente de IA", "Bucle Agéntico", "Uso de Herramientas", "Orquestación", "Humano en el Bucle"]
    }
];

export default glossaryDataEs

