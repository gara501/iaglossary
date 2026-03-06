import { useEffect, useRef } from 'react'
import { Box } from '@chakra-ui/react'
import { useLanguage } from '../context/LanguageContext'
import { useColorMode } from '../context/ThemeContext'

// ── Diagrams ───────────────────────────────────────────────────────────────────

const DIAGRAM_EN = `graph TD
    User["👤 Developer"] -->|1. Types prompt in CLI| CLI["💻 Claude Code CLI"]
    subgraph Local_Env["Local Environment (Your Computer)"]
        CLI
        CLI -->|Runs and captures output| Tools["🛠️ Local Tools"]
        Tools -->|Reads / Edits| Files["📁 Project Files"]
        Tools -->|Executes| Bash["📟 Terminal / Bash / Git"]
        CLI -.-|Reads base instructions| CLAUDEMD["📄 CLAUDE.md"]
    end
    subgraph Anthropic_Cloud["Anthropic Cloud (The Brain)"]
        API["🌐 API Interface"]
        Model["🧠 Claude Model"]
        API <--> Model
    end
    CLI -->|2. Sends Prompt + Context + Tool List| API
    API -->|3. Responds: Requests tool execution e.g. find file| CLI
    CLI -->|4. Executes and returns tool results| API
    API -->|5. Agentic Loop evaluates if more tools needed| API
    API -->|6. Final response or completed action| CLI
    CLI -->|Shows changes / Asks confirmation| User
    style CLI fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#082f49
    style Model fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#4c1d95
    style Tools fill:#f1f5f9,stroke:#64748b,stroke-width:2px,color:#0f172a
    style API fill:#fff1f2,stroke:#e11d48,stroke-width:2px,color:#881337`

const DIAGRAM_ES = `graph TD
    User["👤 Desarrollador"] -->|1. Escribe prompt en CLI| CLI["💻 Claude Code CLI"]
    subgraph Entorno_Local["Entorno Local (Tu Computadora)"]
        CLI
        CLI -->|Ejecuta y captura salida| Tools["🛠️ Herramientas Locales"]
        Tools -->|Lee / Edita| Files["📁 Archivos del Proyecto"]
        Tools -->|Ejecuta| Bash["📟 Terminal / Bash / Git"]
        CLI -.-|Lee instrucciones base| CLAUDEMD["📄 CLAUDE.md"]
    end
    subgraph Nube_Anthropic["Anthropic Cloud (El Cerebro)"]
        API["🌐 Interfaz API"]
        Model["🧠 Modelo Claude"]
        API <--> Model
    end
    CLI -->|2. Envía Prompt + Contexto + Lista de Herramientas| API
    API -->|3. Responde: Pide ejecutar herramienta Ej: buscar archivo| CLI
    CLI -->|4. Ejecuta y devuelve los resultados de la herramienta| API
    API -->|5. Bucle Inteligente evalúa si necesita más herramientas| API
    API -->|6. Respuesta final o acción completada| CLI
    CLI -->|Muestra cambios / Pide confirmación| User
    style CLI fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#082f49
    style Model fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#4c1d95
    style Tools fill:#f1f5f9,stroke:#64748b,stroke-width:2px,color:#0f172a
    style API fill:#fff1f2,stroke:#e11d48,stroke-width:2px,color:#881337`

// ── Component ──────────────────────────────────────────────────────────────────

export default function HowClaudeCodeWorks() {
    const { language } = useLanguage()
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'
    const containerRef = useRef<HTMLDivElement>(null)

    const isEs = language === 'es'

    const L = isEs ? {
        badge: '// Cómo Funciona',
        title: 'Cómo funciona Claude Code',
        subtitle: 'Una visualización de la arquitectura "Cliente-Servidor Agéntico". Muestra cómo un entorno local (CLI) colabora con un modelo de IA remoto mediante un bucle de razonamiento y uso de herramientas.',
        diagramTitle: 'Arquitectura del Bucle Agéntico',
        componentsTitle: 'Desglose de los Componentes',
        components: [
            {
                icon: '💻', title: 'Claude Code (El Arnés Local)',
                desc: 'La herramienta de línea de comandos (CLI) que instalas. Actúa como el intermediario: gestiona las sesiones, envía peticiones a la API y <strong>ejecuta físicamente las herramientas</strong> en tu máquina (lee archivos, corre comandos).',
            },
            {
                icon: '🧠', title: 'El Modelo Claude',
                desc: 'La inteligencia en la nube. No tiene acceso directo a tus archivos. En su lugar, lee un "menú" de herramientas que el CLI le envía y razona: <em>"Para resolver esto, primero necesito pedirle al CLI que ejecute un \'ls\' en esta carpeta"</em>.',
            },
            {
                icon: '🛠️', title: 'Las Herramientas (Tools)',
                desc: 'Lo que transforma a un chatbot en un <strong>agente</strong>. Las herramientas permiten a Claude interactuar con la realidad: buscar en el código, editar líneas específicas, ejecutar tests (bash), leer logs de errores o hacer commits en Git.',
            },
            {
                icon: '📄', title: 'CLAUDE.md',
                desc: 'Tu archivo de memoria y reglas. El CLI no lo "entiende", simplemente lo adjunta como texto en cada petición para que el modelo sepa cuáles son los estándares de tu equipo, comandos de build o arquitectura preferida.',
            },
        ],
        loopTitle: 'Las 3 Fases del Bucle Agéntico',
        loopIntro: 'A diferencia de una simple respuesta de chat, Claude Code opera en un ciclo continuo hasta resolver el problema:',
        loopSteps: [
            { label: '1. Recopilar Contexto (Explore):', text: 'Ante una petición (ej. "Arregla el bug de login"), Claude primero pide ejecutar búsquedas en los archivos para entender cómo funciona la autenticación en tu proyecto específico.' },
            { label: '2. Tomar Acción (Plan & Implement):', text: 'Con la información recopilada, Claude decide qué código cambiar y envía una petición estructurada para que la herramienta local edite los archivos exactos.' },
            { label: '3. Verificar Resultados (Verify):', text: 'Claude Code se distingue por su capacidad de autoevaluación. Puede pedir ejecutar los tests de tu proyecto. Si fallan, lee el error, vuelve a razonar y corrige su propio código.' },
        ],
        analogyTitle: 'La Analogía del Asesor Remoto',
        analogyIntro: 'Imagina cómo funcionan juntos el modelo y la herramienta local:',
        analogyItems: [
            { icon: '👤', label: 'Tú:', desc: 'Le pides a tu asistente local que resuelva un problema complejo.' },
            { icon: '💻', label: 'Asistente Local (CLI):', desc: 'Está en tu oficina. Tiene manos para abrir archivos y teclear, pero necesita que le digan qué hacer. Llama por teléfono a un Experto.' },
            { icon: '🧠', label: 'Experto Remoto (Claude):', desc: 'Es un genio, pero no tiene acceso físico a tu oficina. Le dice al Asistente: "Abre la carpeta de cuentas y dítame qué archivos hay".' },
            { icon: '🔄', label: 'El Bucle:', desc: 'El Asistente lo hace y se lo lee por teléfono. El Experto piensa y dice: "Bien, ahora abre el archivo auth.js y cambia la línea 5". Este proceso continúa hasta solucionar el bug.' },
        ],
        diagram: DIAGRAM_ES,
    } : {
        badge: '// How It Works',
        title: 'How Claude Code Works',
        subtitle: 'A visualization of the "Agentic Client-Server" architecture. Shows how a local environment (CLI) collaborates with a remote AI model through a reasoning and tool-use loop.',
        diagramTitle: 'Agentic Loop Architecture',
        componentsTitle: 'Component Breakdown',
        components: [
            {
                icon: '💻', title: 'Claude Code (The Local Harness)',
                desc: 'The command-line tool (CLI) you install. Acts as the intermediary: manages sessions, sends requests to the API, and <strong>physically executes tools</strong> on your machine (reads files, runs commands).',
            },
            {
                icon: '🧠', title: 'The Claude Model',
                desc: 'The intelligence in the cloud. It has no direct access to your files. Instead, it reads a "menu" of tools the CLI sends it and reasons: <em>"To solve this, I first need to ask the CLI to run \'ls\' on this folder"</em>.',
            },
            {
                icon: '🛠️', title: 'The Tools',
                desc: 'What transforms a chatbot into an <strong>agent</strong>. Tools let Claude interact with reality: search code, edit specific lines, run tests (bash), read error logs, or commit to Git.',
            },
            {
                icon: '📄', title: 'CLAUDE.md',
                desc: 'Your memory and rules file. The CLI doesn\'t "understand" it — it simply attaches it as text in every request so the model knows your team\'s standards, build commands, or preferred architecture.',
            },
        ],
        loopTitle: 'The 3 Phases of the Agentic Loop',
        loopIntro: 'Unlike a simple chat response, Claude Code operates in a continuous cycle until the problem is solved:',
        loopSteps: [
            { label: '1. Gather Context (Explore):', text: 'Given a request (e.g. "Fix the login bug"), Claude first requests file searches to understand how authentication works in your specific project.' },
            { label: '2. Take Action (Plan & Implement):', text: 'With the gathered information, Claude decides what code to change and sends a structured request for the local tool to edit the exact files.' },
            { label: '3. Verify Results (Verify):', text: 'Claude Code stands out for its self-evaluation capability. It can request your project\'s tests be run. If they fail, it reads the error, reasons again, and fixes its own code.' },
        ],
        analogyTitle: 'The Remote Advisor Analogy',
        analogyIntro: 'Imagine how the model and the local tool work together:',
        analogyItems: [
            { icon: '👤', label: 'You:', desc: 'Ask your local assistant to solve a complex problem.' },
            { icon: '💻', label: 'Local Assistant (CLI):', desc: 'Is in your office. Has hands to open files and type, but needs to be told what to do. Calls a Remote Expert.' },
            { icon: '🧠', label: 'Remote Expert (Claude):', desc: 'Is a genius but has no physical access to your office. Tells the Assistant: "Open the accounts folder and read me what files are there".' },
            { icon: '🔄', label: 'The Loop:', desc: 'The Assistant does it and reads it aloud. The Expert thinks and says: "Good, now open auth.js and change line 5". This continues until the bug is fixed.' },
        ],
        diagram: DIAGRAM_EN,
    }

    useEffect(() => {
        const container = containerRef.current
        if (!container) return

        container.innerHTML = `
      <div class="hcc-root">
        <header class="hcc-header">
          <div class="hcc-badge">${L.badge}</div>
          <h1 class="hcc-h1">${L.title}</h1>
          <p class="hcc-subtitle">${L.subtitle}</p>
        </header>

        <!-- Diagram -->
        <div class="hcc-card hcc-diagram-card">
          <div class="hcc-section-lbl">${L.diagramTitle}</div>
          <div class="hcc-mermaid-wrap">
            <div class="mermaid hcc-mermaid" id="hcc-mermaid-el">${L.diagram}</div>
          </div>
        </div>

        <!-- Component cards -->
        <div class="hcc-section-title">${L.componentsTitle}</div>
        <div class="hcc-cards-grid">
          ${L.components.map(c => `
            <div class="hcc-comp-card">
              <div class="hcc-comp-icon">${c.icon}</div>
              <h3 class="hcc-comp-title">${c.title}</h3>
              <p class="hcc-comp-desc">${c.desc}</p>
            </div>
          `).join('')}
        </div>

        <!-- Loop & Analogy -->
        <div class="hcc-two-col">
          <div class="hcc-tinted-card hcc-green">
            <h3 class="hcc-tint-title">🔄 ${L.loopTitle}</h3>
            <p class="hcc-tint-intro">${L.loopIntro}</p>
            <ul class="hcc-loop-list">
              ${L.loopSteps.map(s => `<li><strong>${s.label}</strong> ${s.text}</li>`).join('')}
            </ul>
          </div>
          <div class="hcc-tinted-card hcc-indigo">
            <h3 class="hcc-tint-title">📞 ${L.analogyTitle}</h3>
            <p class="hcc-tint-intro">${L.analogyIntro}</p>
            <ul class="hcc-analogy-list">
              ${L.analogyItems.map(a => `<li><span class="hcc-a-icon">${a.icon}</span><span><strong>${a.label}</strong> ${a.desc}</span></li>`).join('')}
            </ul>
          </div>
        </div>
      </div>
    `

        let cancelled = false
        ;(async () => {
            try {
                const { default: mermaid } = await import('mermaid')
                if (cancelled) return

                mermaid.initialize({
                    startOnLoad: false,
                    theme: dark ? 'dark' : 'base',
                    themeVariables: dark ? {
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

                const el = container.querySelector('#hcc-mermaid-el') as HTMLElement
                if (!el || cancelled) return

                el.removeAttribute('data-processed')
                el.innerHTML = L.diagram

                const id = 'hcc-diagram-' + Date.now()
                const { svg } = await mermaid.render(id, L.diagram)
                if (!cancelled && el) {
                    el.innerHTML = svg
                    const svgEl = el.querySelector('svg')
                    if (svgEl) {
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

    const CSS = `
    .hcc-root { font-family: 'Space Grotesk','Inter',sans-serif; color: #e2e8f0; padding: 32px 28px 60px; max-width: 1600px; margin: 0 auto; }
    .hcc-header { text-align: center; margin-bottom: 32px; }
    .hcc-badge { display: inline-block; background: rgba(147,51,234,0.10); border: 1px solid rgba(147,51,234,0.30); color: #c084fc; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; padding: 6px 16px; border-radius: 2px; margin-bottom: 14px; font-family: 'JetBrains Mono',monospace; }
    .hcc-h1 { font-size: clamp(24px,4vw,44px); font-weight: 700; line-height: 1.1; margin-bottom: 10px; background: linear-gradient(135deg,#fff 30%,#c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .hcc-subtitle { color: #4a6080; font-size: 14px; max-width: 620px; margin: 0 auto; line-height: 1.7; }

    .hcc-card { background: #0d1520; border: 1px solid #1a2a3a; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
    .hcc-diagram-card { overflow-x: auto; }
    .hcc-section-lbl { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #c084fc; font-family: 'JetBrains Mono',monospace; margin-bottom: 20px; display: flex; align-items: center; gap: 8px; }
    .hcc-section-lbl::after { content: ''; flex: 1; height: 1px; background: #1a2a3a; }
    .hcc-section-title { font-size: 22px; font-weight: 700; color: #e2e8f0; margin-bottom: 16px; }

    .hcc-mermaid-wrap { overflow-x: auto; padding-bottom: 12px; }
    .hcc-mermaid { display: block; background: transparent; min-width: 1200px; }
    .hcc-mermaid svg { display: block; width: 100%; min-width: 1200px; height: auto !important; max-height: none !important; }

    .hcc-cards-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    @media(max-width:1000px){ .hcc-cards-grid { grid-template-columns: repeat(2,1fr); } }
    @media(max-width:600px){ .hcc-cards-grid { grid-template-columns: 1fr; } }
    .hcc-comp-card { background: #0d1520; border: 1px solid #1a2a3a; border-radius: 10px; padding: 20px; transition: border-color 0.2s, box-shadow 0.2s; }
    .hcc-comp-card:hover { border-color: rgba(192,132,252,0.35); box-shadow: 0 4px 20px rgba(147,51,234,0.08); }
    .hcc-comp-icon { font-size: 28px; margin-bottom: 10px; }
    .hcc-comp-title { font-size: 15px; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
    .hcc-comp-desc { font-size: 13px; color: #4a6080; line-height: 1.65; }

    .hcc-two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media(max-width:700px){ .hcc-two-col { grid-template-columns: 1fr; } }
    .hcc-tinted-card { border-radius: 14px; padding: 28px; border: 1px solid transparent; }
    .hcc-green { background: rgba(16,185,129,0.07); border-color: rgba(16,185,129,0.20); }
    .hcc-indigo { background: rgba(99,102,241,0.07); border-color: rgba(99,102,241,0.20); }
    .hcc-tint-title { font-size: 18px; font-weight: 700; margin-bottom: 12px; }
    .hcc-green .hcc-tint-title { color: #6ee7b7; }
    .hcc-indigo .hcc-tint-title { color: #a5b4fc; }
    .hcc-tint-intro { font-size: 13px; margin-bottom: 12px; }
    .hcc-green .hcc-tint-intro { color: #a7f3d0; }
    .hcc-indigo .hcc-tint-intro { color: #c7d2fe; }
    .hcc-loop-list { list-style: none; padding: 0; display: flex; flex-direction: column; gap: 14px; }
    .hcc-green .hcc-loop-list { color: #a7f3d0; }
    .hcc-loop-list li { font-size: 13px; line-height: 1.7; border-left: 2px solid rgba(16,185,129,0.4); padding-left: 12px; }
    .hcc-analogy-list { list-style: none; padding: 0; font-size: 13px; line-height: 1.8; display: flex; flex-direction: column; gap: 10px; }
    .hcc-indigo .hcc-analogy-list { color: #c7d2fe; }
    .hcc-analogy-list li { display: flex; align-items: flex-start; gap: 8px; }
    .hcc-a-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }
  `

    return (
        <>
            <style>{CSS}</style>
            <Box ref={containerRef} />
        </>
    )
}
