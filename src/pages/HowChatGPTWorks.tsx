import { useEffect, useRef } from 'react'
import { Box } from '@chakra-ui/react'
import { useLanguage } from '../context/LanguageContext'
import { useColorMode } from '../context/ThemeContext'

// ── Types ──────────────────────────────────────────────────────────────────────

interface ComponentCard {
    icon: string
    title: string
    desc: string
    bullets?: string[]
    wide?: boolean
}

interface LoopStep {
    text: string
    bold?: string
}

interface AnalogyItem {
    icon: string
    label: string
    desc: string
}

// ── Mermaid diagram (same structure for both languages, labels differ) ─────────

const DIAGRAM_EN = `graph TD
    User["👤 User / Web or App Client"] -->|1. Sends Prompt| APIGW["🌐 API Gateway / Load Balancer"]
    subgraph Capa_Orq["Application & Orchestration Layer (Backend)"]
        APIGW -->|2. Routes Request| Backend["⚙️ Backend Service / Orchestrator"]
        Backend -->|3. Auth & Rate Limiting| AuthDB[("🗄️ Users Database")]
        Backend -.-|4. Retrieves Prior Context| HistDB[("🗄️ Conversation History")]
        Backend -->|5. Safety Check| ModAPI["🛡️ Moderation API"]
        ModAPI --x|If unsafe| Backend
        ModAPI -->|If safe| Backend
    end
    subgraph Capa_IA["AI Layer (Model Serving)"]
        Backend -->|6. Sends Prompt + Context| Tokenizer["🔤 Tokenizer"]
        Tokenizer -->|7. Converts to Tokens| ModelEngine["🧠 GPT Inference Engine"]
        subgraph Modelo_GPT["GPT Model (e.g. GPT-4)"]
            ModelEngine --> Layers["Attention Layers & Neural Networks"]
            Layers --> Prob["Probability Distribution"]
        end
        Prob -->|8. Selection Sampling| Decode["Token Decoder"]
    end
    Decode -->|9. Generated Token Stream| Backend
    Backend -->|10. Stream to User| APIGW
    APIGW -->|Stream| User
    Backend -.-|11. Output Verification| ModAPI
    Backend -.-|12. Saves Conversation| HistDB
    style Backend fill:#e0e7ff,stroke:#4f46e5,stroke-width:2px,color:#1e1b4b
    style ModelEngine fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    style ModAPI fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d`

const DIAGRAM_ES = `graph TD
    User["👤 Usuario / Cliente Web/App"] -->|1. Envía Prompt| APIGW["🌐 API Gateway / Load Balancer"]
    subgraph Capa_Orq["Capa de Aplicación y Orquestación (Backend)"]
        APIGW -->|2. Enruta Petición| Backend["⚙️ Servicio de Backend / Orquestador"]
        Backend -->|3. Autenticación y Rate Limiting| AuthDB[("🗄️ Base de Datos de Usuarios")]
        Backend -.-|4. Recupera Contexto Previo| HistDB[("🗄️ Historial de Conversaciones")]
        Backend -->|5. Verificación de Seguridad| ModAPI["🛡️ API de Moderación"]
        ModAPI --x|Si es inseguro| Backend
        ModAPI -->|Si es seguro| Backend
    end
    subgraph Capa_IA["Capa de Inteligencia Artificial (Model Serving)"]
        Backend -->|6. Envía Prompt + Contexto| Tokenizer["🔤 Tokenizador"]
        Tokenizer -->|7. Convierte a Tokens| ModelEngine["🧠 Motor de Inferencia GPT"]
        subgraph Modelo_GPT["El Modelo GPT (Ej. GPT-4)"]
            ModelEngine --> Layers["Capas de Atención y Redes Neuronales"]
            Layers --> Prob["Distribución de Probabilidad"]
        end
        Prob -->|8. Selección Sampling| Decode["Decodificador de Tokens"]
    end
    Decode -->|9. Stream del Token Generado| Backend
    Backend -->|10. Stream al Usuario| APIGW
    APIGW -->|Stream| User
    Backend -.-|11. Verificación Output| ModAPI
    Backend -.-|12. Guarda Conversación| HistDB
    style Backend fill:#e0e7ff,stroke:#4f46e5,stroke-width:2px,color:#1e1b4b
    style ModelEngine fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    style ModAPI fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d`

// ── Component ──────────────────────────────────────────────────────────────────

export default function HowChatGPTWorks() {
    const { language } = useLanguage()
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'
    const containerRef = useRef<HTMLDivElement>(null)
    const mermaidInitRef = useRef(false)

    const isEs = language === 'es'

    const L = isEs ? {
        badge: '// Cómo Funciona',
        title: 'Arquitectura de ChatGPT',
        subtitle: 'Una visualización conceptual del flujo de datos y los componentes del sistema, desde que el usuario envía un prompt hasta que recibe una respuesta generada por IA.',
        diagramTitle: 'Diagrama de Flujo del Sistema',
        componentsTitle: 'Desglose de los Componentes',
        components: [
            { icon: '👤', title: '1. El Cliente (Usuario)', desc: 'Es la interfaz (web de chat.openai.com o app móvil). Captura el texto y muestra la respuesta en tiempo real (streaming), dando la sensación de que la IA está "escribiendo".' },
            { icon: '🌐', title: '2. API Gateway', desc: 'La puerta de entrada. Maneja el tráfico masivo, distribuye las peticiones entre servidores para evitar colapsos y brinda seguridad básica (protección DDoS).' },
            { icon: '⚙️', title: '3. Orquestación (Backend)', desc: 'El "director de orquesta". Gestiona autenticación de usuarios, selección del modelo y recupera el historial de la base de datos para darle "memoria" a la conversación.' },
            { icon: '🛡️', title: '4. API de Moderación', desc: 'La capa de seguridad (Safety Layer). Analiza inputs y outputs buscando contenido ilegal, odio o peligroso. Si detecta algo grave, bloquea la respuesta.' },
            {
                icon: '🧠', title: '5. Capa de IA (Model Serving)',
                desc: 'Donde ocurre la magia, ejecutado en clusters masivos de GPUs.',
                bullets: [
                    'Tokenizador: Corta el texto en fragmentos ("tokens") y los convierte a números.',
                    'Motor de Inferencia (Transformer): Pasa los tokens por capas de atención para entender el contexto.',
                    'Predicción & Sampling: Calcula la probabilidad de la siguiente palabra y usa "sampling" (temperatura) para añadir creatividad.',
                ],
                wide: true,
            },
        ] as ComponentCard[],
        loopTitle: 'El Bucle de Generación',
        loopIntro: 'ChatGPT genera texto token por token en un bucle ultrarrápido:',
        loopSteps: [
            { text: 'Envías: "¿Cuál es la capital de Francia?"' },
            { text: 'El backend añade contexto y lo envía al modelo.' },
            { text: 'El modelo predice el siguiente token:', bold: '"La"' },
            { text: 'Se envía "La" al usuario (streaming).' },
            { text: 'El Bucle:', bold: 'Se toma la entrada original + el nuevo token y se reintroduce al modelo.' },
            { text: 'Se repite hasta predecir un "token de parada".' },
        ] as LoopStep[],
        analogyTitle: 'La Analogía de la Biblioteca',
        analogyItems: [
            { icon: '👤', label: 'Tú:', desc: 'Escribes una pregunta en papel y la entregas.' },
            { icon: '🌐', label: 'Recepcionista (API Gateway):', desc: 'Recibe tu papel.' },
            { icon: '⚙️', label: 'Administrador (Backend):', desc: 'Comprueba tu carnet (Auth) y busca tu archivo (Historial).' },
            { icon: '🛡️', label: 'Escáner (Moderación):', desc: 'Revisa que la pregunta no sea peligrosa.' },
            { icon: '🧠', label: 'Bibliotecario Genio (Modelo GPT):', desc: 'Vive en la sala de GPUs. Lee en códigos numéricos (Tokens) y calcula la siguiente "sílaba" lógica.' },
            { icon: '🗣️', label: 'Respuesta:', desc: 'Dice la sílaba en voz alta y repite el proceso hasta terminar.' },
        ] as AnalogyItem[],
        diagram: DIAGRAM_ES,
    } : {
        badge: '// How It Works',
        title: 'ChatGPT Architecture',
        subtitle: 'A conceptual visualization of the data flow and system components — from the user sending a prompt to receiving an AI-generated response.',
        diagramTitle: 'System Flow Diagram',
        componentsTitle: 'Component Breakdown',
        components: [
            { icon: '👤', title: '1. The Client (User)', desc: 'The interface (chat.openai.com or mobile app). Captures the text and displays the response in real time (streaming), giving the feeling that the AI is "typing".' },
            { icon: '🌐', title: '2. API Gateway', desc: 'The entry point. Handles massive traffic, distributes requests across servers to prevent overload, and provides basic security (DDoS protection).' },
            { icon: '⚙️', title: '3. Orchestration (Backend)', desc: 'The "conductor". Manages user authentication, model selection, and retrieves conversation history from the database to give the AI "memory".' },
            { icon: '🛡️', title: '4. Moderation API', desc: 'The safety layer. Analyzes inputs and outputs looking for illegal, hateful, or dangerous content. If it detects something severe, it blocks the response.' },
            {
                icon: '🧠', title: '5. AI Layer (Model Serving)',
                desc: 'Where the magic happens, running on massive GPU clusters.',
                bullets: [
                    'Tokenizer: Splits text into fragments ("tokens") and converts them to numbers.',
                    'Inference Engine (Transformer): Passes tokens through attention layers to understand context.',
                    'Prediction & Sampling: Calculates the probability of the next word and uses "sampling" (temperature) to add creativity.',
                ],
                wide: true,
            },
        ] as ComponentCard[],
        loopTitle: 'The Generation Loop',
        loopIntro: 'ChatGPT generates text token by token in an ultrafast loop:',
        loopSteps: [
            { text: 'You send: "What is the capital of France?"' },
            { text: 'The backend adds context and sends it to the model.' },
            { text: 'The model predicts the next token:', bold: '"The"' },
            { text: 'Sends "The" to the user (streaming).' },
            { text: 'The Loop:', bold: 'It takes the original input + the new token and feeds it back to the model.' },
            { text: 'Repeats until predicting a "stop token".' },
        ] as LoopStep[],
        analogyTitle: 'The Library Analogy',
        analogyItems: [
            { icon: '👤', label: 'You:', desc: 'Write a question on paper and hand it in.' },
            { icon: '🌐', label: 'Receptionist (API Gateway):', desc: 'Receives your paper.' },
            { icon: '⚙️', label: 'Administrator (Backend):', desc: 'Checks your ID (Auth) and looks up your file (History).' },
            { icon: '🛡️', label: 'Scanner (Moderation):', desc: 'Checks that the question is not dangerous.' },
            { icon: '🧠', label: 'Genius Librarian (GPT Model):', desc: 'Lives in the GPU room. Reads in numeric codes (Tokens) and calculates the next logical "syllable".' },
            { icon: '🗣️', label: 'Response:', desc: 'Says the syllable out loud and repeats the process until done.' },
        ] as AnalogyItem[],
        diagram: DIAGRAM_EN,
    }

    useEffect(() => {
        const container = containerRef.current
        if (!container) return

        // ── Build static HTML ──────────────────────────────────────────────────────
        container.innerHTML = `
      <div class="ht-root">
        <header class="ht-header">
          <div class="ht-badge">${L.badge}</div>
          <h1 class="ht-h1">${L.title}</h1>
          <p class="ht-subtitle">${L.subtitle}</p>
        </header>

        <!-- Diagram -->
        <div class="ht-card ht-diagram-card" style="margin-bottom:24px">
          <div class="ht-section-lbl">${L.diagramTitle}</div>
          <div class="ht-mermaid-wrap">
            <div class="mermaid ht-mermaid" id="ht-mermaid-el">${L.diagram}</div>
          </div>
        </div>

        <!-- Component cards -->
        <div class="ht-section-title">${L.componentsTitle}</div>
        <div class="ht-cards-grid">
          ${L.components.map(c => `
            <div class="ht-comp-card${c.wide ? ' ht-wide' : ''}">
              <div class="ht-comp-icon">${c.icon}</div>
              <h3 class="ht-comp-title">${c.title}</h3>
              <p class="ht-comp-desc">${c.desc}</p>
              ${c.bullets ? `<ul class="ht-comp-bullets">${c.bullets.map(b => `<li>${b}</li>`).join('')}</ul>` : ''}
            </div>
          `).join('')}
        </div>

        <!-- Loop & Analogy -->
        <div class="ht-two-col" style="margin-top:24px;margin-bottom:0">
          <div class="ht-tinted-card ht-blue">
            <h3 class="ht-tint-title">🔄 ${L.loopTitle}</h3>
            <p class="ht-tint-intro">${L.loopIntro}</p>
            <ol class="ht-loop-list">
              ${L.loopSteps.map(s => `<li>${s.text}${s.bold ? ` <strong>${s.bold}</strong>` : ''}</li>`).join('')}
            </ol>
          </div>
          <div class="ht-tinted-card ht-amber">
            <h3 class="ht-tint-title">📚 ${L.analogyTitle}</h3>
            <ul class="ht-analogy-list">
              ${L.analogyItems.map(a => `<li><span class="ht-a-icon">${a.icon}</span><span><strong>${a.label}</strong> ${a.desc}</span></li>`).join('')}
            </ul>
          </div>
        </div>
      </div>
    `

        // ── Init / re-render Mermaid ───────────────────────────────────────────────
        let cancelled = false
        ;(async () => {
            try {
                const { default: mermaid } = await import('mermaid')
                if (cancelled) return

                const isDark = dark

                mermaid.initialize({
                    startOnLoad: false,
                    theme: isDark ? 'dark' : 'base',
                    themeVariables: isDark ? {
                        fontFamily: "'Space Grotesk', 'Inter', sans-serif",
                        primaryColor: '#1a2a3a',
                        primaryTextColor: '#e2e8f0',
                        primaryBorderColor: '#2d4a6a',
                        lineColor: '#4a7090',
                        secondaryColor: '#0d1520',
                        tertiaryColor: '#111d2d',
                        background: '#0d1520',
                        edgeLabelBackground: '#0d1520',
                        clusterBkg: '#111d2d',
                        fontSize: '16px',
                    } : {
                        fontFamily: "'Space Grotesk', 'Inter', sans-serif",
                        primaryColor: '#f8fafc',
                        primaryTextColor: '#1e293b',
                        primaryBorderColor: '#cbd5e1',
                        lineColor: '#64748b',
                        secondaryColor: '#f1f5f9',
                        tertiaryColor: '#e2e8f0',
                        fontSize: '16px',
                    },
                    flowchart: { curve: 'basis', useMaxWidth: false, htmlLabels: true },
                })

                const el = container.querySelector('#ht-mermaid-el') as HTMLElement
                if (!el || cancelled) return

                // Clear previous render
                el.removeAttribute('data-processed')
                el.innerHTML = L.diagram

                const id = 'ht-diagram-' + Date.now()
                const { svg } = await mermaid.render(id, L.diagram)
                if (!cancelled && el) {
                    el.innerHTML = svg
                    // Let the SVG render at its natural intrinsic size
                    const svgEl = el.querySelector('svg')
                    if (svgEl) {
                        // Remove any max-width/height constraints mermaid may inject
                        svgEl.removeAttribute('style')
                        svgEl.style.display = 'block'
                        svgEl.style.width = '100%'
                        svgEl.style.minWidth = '1200px'
                        svgEl.style.height = 'auto'
                        svgEl.style.maxHeight = 'none'
                    }
                }
            } catch (err) {
                console.error('Mermaid render error:', err)
            }
        })()

        return () => { cancelled = true }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [language, dark])

    mermaidInitRef.current = true

    const CSS = `
    .ht-root { font-family: 'Space Grotesk','Inter',sans-serif; color: #e2e8f0; padding: 32px 28px 60px; max-width: 1600px; margin: 0 auto; }
    .ht-header { text-align: center; margin-bottom: 32px; }
    .ht-badge { display: inline-block; background: rgba(0,229,255,0.08); border: 1px solid rgba(0,229,255,0.25); color: #00e5ff; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; padding: 6px 16px; border-radius: 2px; margin-bottom: 14px; font-family: 'JetBrains Mono',monospace; }
    .ht-h1 { font-size: clamp(24px,4vw,44px); font-weight: 700; line-height: 1.1; margin-bottom: 10px; background: linear-gradient(135deg,#fff 30%,#00e5ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .ht-subtitle { color: #4a6080; font-size: 14px; max-width: 580px; margin: 0 auto; line-height: 1.7; }

    /* Cards */
    .ht-card { background: #0d1520; border: 1px solid #1a2a3a; border-radius: 12px; padding: 24px; }
    .ht-diagram-card { overflow-x: auto; }
    .ht-section-lbl { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #00e5ff; font-family: 'JetBrains Mono',monospace; margin-bottom: 20px; display: flex; align-items: center; gap: 8px; }
    .ht-section-lbl::after { content: ''; flex: 1; height: 1px; background: #1a2a3a; }
    .ht-section-title { font-size: 22px; font-weight: 700; color: #e2e8f0; margin-bottom: 16px; }

    .ht-mermaid-wrap { overflow-x: auto; padding-bottom: 12px; }
    .ht-mermaid { display: block; background: transparent; min-width: 1200px; }
    .ht-mermaid svg { display: block; width: 100%; min-width: 1200px; height: auto !important; max-height: none !important; }

    /* Component grid */
    .ht-cards-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    @media(max-width:900px){ .ht-cards-grid { grid-template-columns: repeat(2,1fr); } }
    @media(max-width:600px){ .ht-cards-grid { grid-template-columns: 1fr; } }
    .ht-comp-card { background: #0d1520; border: 1px solid #1a2a3a; border-radius: 10px; padding: 20px; transition: border-color 0.2s, box-shadow 0.2s; }
    .ht-comp-card:hover { border-color: rgba(0,229,255,0.3); box-shadow: 0 4px 20px rgba(0,229,255,0.06); }
    .ht-wide { grid-column: span 2; }
    @media(max-width:900px){ .ht-wide { grid-column: span 1; } }
    .ht-comp-icon { font-size: 28px; margin-bottom: 10px; }
    .ht-comp-title { font-size: 15px; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
    .ht-comp-desc { font-size: 13px; color: #4a6080; line-height: 1.65; }
    .ht-comp-bullets { margin-top: 10px; padding-left: 16px; font-size: 13px; color: #4a6080; line-height: 1.75; }
    .ht-comp-bullets li { margin-bottom: 4px; }

    /* Two col */
    .ht-two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media(max-width:700px){ .ht-two-col { grid-template-columns: 1fr; } }
    .ht-tinted-card { border-radius: 14px; padding: 28px; border: 1px solid transparent; }
    .ht-blue { background: rgba(99,102,241,0.08); border-color: rgba(99,102,241,0.2); }
    .ht-amber { background: rgba(251,191,36,0.07); border-color: rgba(251,191,36,0.18); }
    .ht-tint-title { font-size: 18px; font-weight: 700; margin-bottom: 12px; }
    .ht-blue .ht-tint-title { color: #a5b4fc; }
    .ht-amber .ht-tint-title { color: #fcd34d; }
    .ht-tint-intro { font-size: 13px; margin-bottom: 12px; }
    .ht-blue .ht-tint-intro { color: #c7d2fe; }
    .ht-loop-list { padding-left: 18px; font-size: 13px; line-height: 1.8; }
    .ht-blue .ht-loop-list { color: #c7d2fe; }
    .ht-loop-list li { margin-bottom: 4px; }
    .ht-analogy-list { list-style: none; padding: 0; font-size: 13px; line-height: 1.8; display: flex; flex-direction: column; gap: 8px; }
    .ht-amber .ht-analogy-list { color: #fde68a; }
    .ht-analogy-list li { display: flex; align-items: flex-start; gap: 8px; }
    .ht-a-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }
  `

    return (
        <>
            <style>{CSS}</style>
            <Box
                ref={containerRef}
                sx={{
                    'body.light &': {
                        background: 'rgba(13,21,32,0.95)',
                        borderRadius: '12px',
                        margin: '20px',
                        boxShadow: '0 8px 40px rgba(0,0,0,0.15)',
                    },
                }}
            />
        </>
    )
}
