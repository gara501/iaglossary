import { useEffect, useRef } from 'react'
import { Box } from '@chakra-ui/react'
import { useLanguage } from '../context/LanguageContext'

// ── Types ──────────────────────────────────────────────────────────────────────

interface KBDoc {
    icon: string
    name: string
    type: string
    topics: string[]
}

interface StageDef {
    id: string
    color: string
    num: string
    title: string
    desc: string
}

interface VectorPoint {
    x: number
    y: number
    doc: number
    r: number
}

// ── Data (bilingual) ──────────────────────────────────────────────────────────

const KB_DOCS: KBDoc[] = [
    { icon: '📋', name: 'HR Policy',       type: 'PDF',  topics: ['vacation','leave','holiday','days','license','schedule','benefits','vacaciones','días','licencia','permisos'] },
    { icon: '💻', name: 'Software Manual', type: 'DOCX', topics: ['install','software','configure','system','download','update','instalar','configurar','descarga','actualizar'] },
    { icon: '📰', name: 'Newsletter Q1',   type: 'HTML', topics: ['news','company','updates','quarter','results','announcement','noticias','empresa','novedades','trimestre'] },
    { icon: '🕐', name: 'Support SLA',     type: 'TXT',  topics: ['support','schedule','service','ticket','help','soporte','horario','atención','ayuda'] },
    { icon: '📊', name: 'KPIs 2024',       type: 'XLSX', topics: ['metrics','results','data','statistics','sales','métricas','resultados','datos','ventas'] },
    { icon: '🔐', name: 'IT Security',     type: 'PDF',  topics: ['security','password','access','privacy','seguridad','contraseña','acceso','privacidad'] },
    { icon: '🏢', name: 'Org Chart',       type: 'PNG',  topics: ['team','organization','structure','department','equipo','organización','estructura','departamento'] },
    { icon: '📝', name: 'Onboarding',      type: 'PDF',  topics: ['welcome','new','process','employee','bienvenida','nuevo','proceso','empleado','incorporación'] },
]

const STAGE_DEFS_EN: StageDef[] = [
    { id: 'index',    color: '#5ab4ff', num: '01', title: 'Indexing',          desc: 'Documents chunked, embedded and stored in vector DB.' },
    { id: 'embed',    color: '#c084fc', num: '02', title: 'Query Embedding',   desc: 'Query converted to a high-dimensional numeric vector.' },
    { id: 'search',   color: '#f5a623', num: '03', title: 'Semantic Search',   desc: 'Cosine similarity against all vectors in the index.' },
    { id: 'augment',  color: '#3ddc84', num: '04', title: 'Augmentation',      desc: 'Retrieved context injected into the LLM prompt.' },
    { id: 'generate', color: '#ff7eb3', num: '05', title: 'LLM Generation',    desc: 'Model generates a response grounded in documents.' },
]

const STAGE_DEFS_ES: StageDef[] = [
    { id: 'index',    color: '#5ab4ff', num: '01', title: 'Indexación',         desc: 'Documentos procesados, chunkeados y embedidos en la base vectorial.' },
    { id: 'embed',    color: '#c084fc', num: '02', title: 'Embedding Query',    desc: 'La consulta se convierte en un vector numérico de alta dimensión.' },
    { id: 'search',   color: '#f5a623', num: '03', title: 'Búsqueda Semántica', desc: 'Similitud coseno vs. todos los vectores del índice.' },
    { id: 'augment',  color: '#3ddc84', num: '04', title: 'Augmentación',       desc: 'Contexto recuperado se inyecta al prompt del LLM.' },
    { id: 'generate', color: '#ff7eb3', num: '05', title: 'Generación LLM',     desc: 'El modelo genera respuesta grounded en los documentos.' },
]

const EXAMPLES_EN = [
    'How many vacation days do I have?',
    'What is the sick leave policy?',
    'How do I install the software?',
    'What are the support hours?',
]

const EXAMPLES_ES = [
    '¿Cuántos días de vacaciones tengo disponibles?',
    '¿Cuál es la política de bajas?',
    '¿Cómo instalo el software?',
    '¿Cuál es el horario de soporte?',
]

// ── Helpers ───────────────────────────────────────────────────────────────────

function hexRGB(hex: string) {
    const r = parseInt(hex.slice(1, 3), 16)
    const g = parseInt(hex.slice(3, 5), 16)
    const b = parseInt(hex.slice(5, 7), 16)
    return `${r},${g},${b}`
}

function delay(ms: number) { return new Promise(r => setTimeout(r, ms)) }

// ── Component ─────────────────────────────────────────────────────────────────

export default function RagPipelineSim() {
    const { language } = useLanguage()
    const containerRef = useRef<HTMLDivElement>(null)
    const runningRef = useRef(false)
    const vectorPointsRef = useRef<VectorPoint[]>([])

    const L = language === 'es' ? {
        sysTag: 'Sistema RAG // Visualización',
        title1: 'Retrieval-', titleAccent: 'Augmented', title2: 'Generation',
        subtitle: 'Simulación interactiva del pipeline RAG — desde la consulta del usuario hasta la respuesta aumentada con contexto externo.',
        queryLabel: '// Consulta del usuario',
        placeholder: 'Escribe tu consulta...',
        runBtn: '▶ Ejecutar RAG',
        examplesLabel: 'EJEMPLOS →',
        examples: EXAMPLES_ES,
        mDocsLabel: 'Docs Indexados',
        mRetrievedLabel: 'Recuperados',
        mTokensLabel: 'Tokens Prompt',
        mLatencyLabel: 'Latencia (ms)',
        vectorTitle: 'Base Vectorial',
        kbTitle: 'Knowledge Base',
        retrievedTitle: 'Documentos Recuperados',
        waitingQuery: 'Esperando consulta...',
        augTitle: 'Prompt Aumentado (RAG)',
        waitingPipeline: 'Esperando pipeline...',
        llmTitle: 'Respuesta del LLM',
        llmIdle: 'El LLM generará la respuesta aquí...',
        logTitle: 'System Log',
        stageDefs: STAGE_DEFS_ES,
        stageStatus: { idle: 'idle', running: 'procesando...', done: 'completado ✓' },
        logInit: 'Sistema RAG inicializado. Knowledge base cargada.',
        logDocs: (n: number) => `${n} documentos disponibles para recuperación.`,
        logQuery: (q: string) => `Query: "${q}"`,
        logNewRag: 'Nueva consulta RAG iniciada',
        logIndex: 'Cargando índice vectorial...',
        logIndexDone: (n: number) => `${n} documentos indexados. Dimensión: 768d`,
        logEmbed: 'Embedding de la query con sentence-transformers...',
        logVector: 'Vector generado: [0.231, -0.445, 0.891, ... +764 dims]',
        logSearch: 'Búsqueda por similitud coseno en VectorDB...',
        logAugment: 'Construyendo prompt aumentado...',
        logPromptBuilt: (t: number) => `Prompt construido: ${t} tokens`,
        logGenerate: 'Enviando prompt aumentado al LLM...',
        logDone: (ms: number) => `Respuesta generada en ${ms}ms ✓`,
        generating: 'Generando...',
        instrPrefix: 'INSTRUCCIÓN: Responde solo usando el contexto provisto. Cita fuentes.\n\n',
        ctxPrefix: 'CONTEXTO RECUPERADO:\n',
        qPrefix: '\nPREGUNTA DEL USUARIO:\n',
        docSnippet: (topics: string[]) => `Fragmento relevante extraído del documento indexado. Contiene información sobre ${topics.slice(0,3).join(', ')}...`,
        responses: {
            vacation: (src: string) => `Según la **Política de RRHH**, tienes derecho a 22 días hábiles de vacaciones al año. Con base en tu fecha de inicio, te quedan **14 días** disponibles para el 2024.\n\nFuente: ${src}`,
            sick: (src: string) => `La política de bajas por enfermedad cubre hasta **30 días** con sueldo completo. Para bajas prolongadas, se activa el protocolo de incapacidad médica.\n\nFuente: ${src}`,
            software: (src: string) => `Para instalar el software interno: 1) Descarga desde el portal IT. 2) Ejecuta como administrador. 3) Introduce tu credencial corporativa.\n\nFuente: ${src}`,
            support: (src: string) => `El equipo de soporte atiende de **Lunes a Viernes de 8:00 a 20:00 hrs** (UTC-5). Para urgencias: extensión 911.\n\nFuente: ${src}`,
            default: (src: string) => `Basándome en los documentos recuperados (${src}), he encontrado información relevante para tu consulta. Los documentos contienen datos actualizados sobre este tema.\n\nEsta respuesta fue generada con contexto de tu Knowledge Base.`,
        },
    } : {
        sysTag: 'RAG System // Visualization',
        title1: 'Retrieval-', titleAccent: 'Augmented', title2: 'Generation',
        subtitle: 'Interactive simulation of the RAG pipeline — from user query to context-augmented LLM response.',
        queryLabel: '// User query',
        placeholder: 'Type your query...',
        runBtn: '▶ Run RAG',
        examplesLabel: 'EXAMPLES →',
        examples: EXAMPLES_EN,
        mDocsLabel: 'Indexed Docs',
        mRetrievedLabel: 'Retrieved',
        mTokensLabel: 'Prompt Tokens',
        mLatencyLabel: 'Latency (ms)',
        vectorTitle: 'Vector Database',
        kbTitle: 'Knowledge Base',
        retrievedTitle: 'Retrieved Documents',
        waitingQuery: 'Waiting for query...',
        augTitle: 'Augmented Prompt (RAG)',
        waitingPipeline: 'Waiting for pipeline...',
        llmTitle: 'LLM Response',
        llmIdle: 'The LLM will generate the response here...',
        logTitle: 'System Log',
        stageDefs: STAGE_DEFS_EN,
        stageStatus: { idle: 'idle', running: 'processing...', done: 'done ✓' },
        logInit: 'RAG system initialized. Knowledge base loaded.',
        logDocs: (n: number) => `${n} documents available for retrieval.`,
        logQuery: (q: string) => `Query: "${q}"`,
        logNewRag: 'New RAG query started',
        logIndex: 'Loading vector index...',
        logIndexDone: (n: number) => `${n} documents indexed. Dimension: 768d`,
        logEmbed: 'Embedding query with sentence-transformers...',
        logVector: 'Vector generated: [0.231, -0.445, 0.891, ... +764 dims]',
        logSearch: 'Cosine similarity search in VectorDB...',
        logAugment: 'Building augmented prompt...',
        logPromptBuilt: (t: number) => `Prompt built: ${t} tokens`,
        logGenerate: 'Sending augmented prompt to LLM...',
        logDone: (ms: number) => `Response generated in ${ms}ms ✓`,
        generating: 'Generating...',
        instrPrefix: 'INSTRUCTION: Answer only using the provided context. Cite sources.\n\n',
        ctxPrefix: 'RETRIEVED CONTEXT:\n',
        qPrefix: '\nUSER QUESTION:\n',
        docSnippet: (topics: string[]) => `Relevant fragment extracted from indexed document. Contains information about ${topics.slice(0,3).join(', ')}...`,
        responses: {
            vacation: (src: string) => `According to the **HR Policy**, you have 22 business days of vacation per year. Based on your start date, you have **14 days** remaining for 2024.\n\nSource: ${src}`,
            sick: (src: string) => `The sick leave policy covers up to **30 days** with full pay. For extended leave, the medical disability protocol is activated.\n\nSource: ${src}`,
            software: (src: string) => `To install the internal software: 1) Download from the IT portal. 2) Run as administrator. 3) Enter your corporate credentials.\n\nSource: ${src}`,
            support: (src: string) => `The support team is available **Monday–Friday 8:00 AM to 8:00 PM** (UTC-5). For emergencies: extension 911.\n\nSource: ${src}`,
            default: (src: string) => `Based on the retrieved documents (${src}), I found relevant information for your query. The documents contain up-to-date data on this topic.\n\nThis response was generated with context from your Knowledge Base.`,
        },
    }

    useEffect(() => {
        const container = containerRef.current
        if (!container) return
        runningRef.current = false

        // ── Build HTML ──────────────────────────────────────────────────────────
        container.innerHTML = `
      <header class="rag-header">
        <div class="rag-sys-tag">${L.sysTag}</div>
        <h1 class="rag-h1">${L.title1}<span>${L.titleAccent}</span><br>${L.title2}</h1>
        <p class="rag-subtitle">${L.subtitle}</p>
      </header>

      <!-- PIPELINE SVG -->
      <div class="rag-pipe-wrap">
        <svg id="rag-pipeline-svg" height="110" viewBox="0 0 860 110"></svg>
      </div>

      <!-- QUERY INPUT -->
      <div class="rag-query-panel">
        <div class="rag-qlabel">${L.queryLabel}</div>
        <div class="rag-qrow">
          <input class="rag-qinput" id="rag-q-input" type="text" value="${L.examples[0]}" placeholder="${L.placeholder}" />
          <button class="rag-run-btn" id="rag-run-btn">${L.runBtn}</button>
        </div>
        <div class="rag-examples">
          <span class="rag-ex-label">${L.examplesLabel}</span>
          ${L.examples.map((e, i) => `<span class="rag-ex-chip" data-ex="${i}">${e}</span>`).join('')}
        </div>
      </div>

      <!-- STAGE CARDS -->
      <div class="rag-stages-grid" id="rag-stages-grid"></div>

      <!-- METRICS -->
      <div class="rag-metrics-row">
        <div class="rag-metric"><div class="rag-metric-val" id="rag-m-docs" style="color:#5ab4ff">—</div><div class="rag-metric-label">${L.mDocsLabel}</div></div>
        <div class="rag-metric"><div class="rag-metric-val" id="rag-m-retrieved" style="color:#f5a623">—</div><div class="rag-metric-label">${L.mRetrievedLabel}</div></div>
        <div class="rag-metric"><div class="rag-metric-val" id="rag-m-tokens" style="color:#c084fc">—</div><div class="rag-metric-label">${L.mTokensLabel}</div></div>
        <div class="rag-metric"><div class="rag-metric-val" id="rag-m-latency" style="color:#3ddc84">—</div><div class="rag-metric-label">${L.mLatencyLabel}</div></div>
      </div>

      <!-- TWO COL: vector db + kb -->
      <div class="rag-two-col">
        <div class="rag-panel">
          <div class="rag-panel-title">${L.vectorTitle}</div>
          <canvas id="rag-vector-canvas" width="280" height="200"></canvas>
        </div>
        <div class="rag-panel">
          <div class="rag-panel-title">${L.kbTitle}</div>
          <div class="rag-kb-grid" id="rag-kb-grid"></div>
        </div>
      </div>

      <!-- TWO COL: retrieved docs + augmented prompt -->
      <div class="rag-two-col">
        <div class="rag-panel">
          <div class="rag-panel-title">${L.retrievedTitle}</div>
          <div id="rag-retrieved-docs"><p style="color:#4a6b4a;font-size:11px;">${L.waitingQuery}</p></div>
        </div>
        <div class="rag-panel">
          <div class="rag-panel-title">${L.augTitle}</div>
          <div class="rag-prompt-box" id="rag-aug-prompt">${L.waitingPipeline}</div>
        </div>
      </div>

      <!-- LLM RESPONSE -->
      <div class="rag-panel" style="margin-bottom:20px;">
        <div class="rag-panel-title">${L.llmTitle}</div>
        <div id="rag-llm-response" style="padding:16px;border:1px solid #3ddc84;border-radius:5px;font-size:12px;line-height:1.9;min-height:80px;color:#4a6b4a;background:rgba(0,0,0,0.4);box-shadow:0 0 30px rgba(61,220,132,0.07);">${L.llmIdle}</div>
      </div>

      <!-- LOG -->
      <div class="rag-panel">
        <div class="rag-panel-title">${L.logTitle}</div>
        <div id="rag-log-feed" class="rag-log-feed"></div>
      </div>
    `

        // ── Stage cards ──────────────────────────────────────────────────────────
        const stagesGrid = container.querySelector('#rag-stages-grid')!
        stagesGrid.innerHTML = L.stageDefs.map(s => `
      <div class="rag-stage-card" id="rag-sc-${s.id}" style="--stage-color:${s.color}">
        <div class="rag-sc-num">// STAGE ${s.num}</div>
        <div class="rag-sc-title">${s.title}</div>
        <div class="rag-sc-desc">${s.desc}</div>
        <div class="rag-sc-status idle" id="rag-ss-${s.id}"><span>●</span> <span>idle</span></div>
      </div>
    `).join('')

        // ── Knowledge base ────────────────────────────────────────────────────────
        const kbGrid = container.querySelector('#rag-kb-grid')!
        kbGrid.innerHTML = KB_DOCS.map((d, i) => `
      <div class="rag-kb-doc" id="rag-kb-${i}">
        <div class="rag-kb-icon">${d.icon}</div>
        <div class="rag-kb-name">${d.name}</div>
        <div class="rag-kb-type">${d.type}</div>
      </div>
    `).join('')
        ;(container.querySelector('#rag-m-docs') as HTMLElement).textContent = String(KB_DOCS.length)

        // ── Vector canvas ─────────────────────────────────────────────────────────
        function initVectors() {
            const canvas = container.querySelector('#rag-vector-canvas') as HTMLCanvasElement
            const W = canvas.offsetWidth || 280
            canvas.width = W
            vectorPointsRef.current = KB_DOCS.map(() => ({
                x: 20 + Math.random() * (W - 40),
                y: 20 + Math.random() * 160,
                doc: 0,
                r: 4 + Math.random() * 3,
            }))
            drawVectors(null, [])
        }

        function drawVectors(qVec: { x: number; y: number } | null, retrieved: number[]) {
            const canvas = container.querySelector('#rag-vector-canvas') as HTMLCanvasElement
            if (!canvas) return
            const ctx = canvas.getContext('2d')!
            const W = canvas.width, H = canvas.height
            ctx.clearRect(0, 0, W, H)
            ctx.fillStyle = 'rgba(0,0,0,0.3)'
            ctx.fillRect(0, 0, W, H)
            ctx.strokeStyle = 'rgba(61,220,132,0.04)'
            ctx.lineWidth = 1
            for (let x = 0; x < W; x += 28) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke() }
            for (let y = 0; y < H; y += 28) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke() }

            if (qVec && retrieved.length) {
                retrieved.forEach(ri => {
                    const p = vectorPointsRef.current[ri]
                    ctx.beginPath(); ctx.moveTo(qVec.x, qVec.y); ctx.lineTo(p.x, p.y)
                    ctx.strokeStyle = 'rgba(245,166,35,0.3)'; ctx.lineWidth = 1
                    ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([])
                })
            }

            vectorPointsRef.current.forEach((p, i) => {
                const isRetrieved = retrieved.includes(i)
                ctx.beginPath(); ctx.arc(p.x, p.y, p.r + (isRetrieved ? 3 : 0), 0, Math.PI * 2)
                ctx.fillStyle = isRetrieved ? 'rgba(61,220,132,0.8)' : 'rgba(90,180,255,0.4)'
                ctx.fill()
                ctx.strokeStyle = isRetrieved ? '#3ddc84' : 'rgba(90,180,255,0.2)'
                ctx.lineWidth = isRetrieved ? 2 : 1; ctx.stroke()
                if (isRetrieved) {
                    ctx.fillStyle = '#3ddc84'; ctx.font = '8px IBM Plex Mono'
                    ctx.fillText(KB_DOCS[i].icon + ' ' + KB_DOCS[i].name.slice(0, 8), p.x + 7, p.y + 3)
                }
            })

            if (qVec) {
                ctx.beginPath(); ctx.arc(qVec.x, qVec.y, 7, 0, Math.PI * 2)
                ctx.fillStyle = 'rgba(245,166,35,0.9)'; ctx.fill()
                ctx.strokeStyle = '#f5a623'; ctx.lineWidth = 2; ctx.stroke()
                ctx.beginPath(); ctx.arc(qVec.x, qVec.y, 12, 0, Math.PI * 2)
                ctx.strokeStyle = 'rgba(245,166,35,0.3)'; ctx.lineWidth = 1; ctx.stroke()
                ctx.fillStyle = '#f5a623'; ctx.font = 'bold 9px IBM Plex Mono'
                ctx.fillText('QUERY', qVec.x - 15, qVec.y - 12)
            }
        }

        // ── Pipeline SVG ──────────────────────────────────────────────────────────
        function drawPipeline(activeIdx: number) {
            const svg = container.querySelector('#rag-pipeline-svg') as SVGSVGElement
            if (!svg) return
            const nodes = [
                { label: 'USER',      sub: 'Query',    x: 60 },
                { label: 'EMBEDDING', sub: 'Vectorize', x: 200 },
                { label: 'VECTOR DB', sub: 'Search',   x: 370 },
                { label: 'CONTEXT',   sub: 'Top-K',    x: 540 },
                { label: 'LLM',       sub: 'Generate', x: 700 },
                { label: 'RESPONSE',  sub: 'Output',   x: 820 },
            ]
            const stageColors = ['#4a6b4a', '#5ab4ff', '#f5a623', '#3ddc84', '#c084fc', '#ff7eb3']
            let html = ''

            nodes.forEach((n, i) => {
                if (i === nodes.length - 1) return
                const n2 = nodes[i + 1]
                const active = i < activeIdx
                const col = active ? stageColors[i + 1] : '#1e2e1e'
                html += `<line x1="${n.x + 30}" y1="55" x2="${n2.x - 30}" y2="55" stroke="${col}" stroke-width="${active ? 2 : 1}" stroke-dasharray="${active ? '0' : '6 4'}"/>
                <polygon points="${n2.x - 32},50 ${n2.x - 22},55 ${n2.x - 32},60" fill="${col}"/>`
            })

            nodes.forEach((n, i) => {
                const active = i <= activeIdx
                const current = i === activeIdx
                const col = stageColors[i]
                const fill = active ? `rgba(${hexRGB(col)},.12)` : 'rgba(14,20,14,.8)'
                const stroke = active ? col : '#1e2e1e'
                html += `<rect x="${n.x - 30}" y="30" width="60" height="50" rx="5" fill="${fill}" stroke="${stroke}" stroke-width="${current ? 2 : 1}"/>`
                if (current) {
                    html += `<rect x="${n.x - 34}" y="26" width="68" height="58" rx="7" fill="none" stroke="${col}" stroke-width="1" opacity="0.4">
                    <animate attributeName="opacity" values="0.4;0.1;0.4" dur="1.2s" repeatCount="indefinite"/>
                  </rect>`
                }
                html += `<text x="${n.x}" y="52" text-anchor="middle" dominant-baseline="middle" font-family="Syne, sans-serif" font-weight="800" font-size="9" fill="${active ? col : '#2d452d'}">${n.label}</text>`
                html += `<text x="${n.x}" y="66" text-anchor="middle" font-family="IBM Plex Mono" font-size="8" fill="${active ? 'rgba(200,240,200,.5)' : '#1e2e1e'}">${n.sub}</text>`
            })

            svg.innerHTML = html
        }

        // ── Log ───────────────────────────────────────────────────────────────────
        function log(msg: string, type = 'info') {
            const feed = container.querySelector('#rag-log-feed')!
            const now = new Date()
            const ts = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`
            const div = document.createElement('div')
            div.className = `rag-log-line ${type}`
            div.innerHTML = `<span class="rag-ts">[${ts}]</span><span class="rag-msg">${msg}</span>`
            feed.appendChild(div)
            ;(feed as HTMLElement).scrollTop = (feed as HTMLElement).scrollHeight
        }

        // ── Stage helpers ─────────────────────────────────────────────────────────
        function setStage(id: string, state: 'idle' | 'running' | 'done') {
            const card = container.querySelector(`#rag-sc-${id}`)
            const status = container.querySelector(`#rag-ss-${id}`)
            if (!card || !status) return
            card.className = 'rag-stage-card ' + (state === 'running' ? 'rag-active' : state === 'done' ? 'rag-done' : '')
            status.className = 'rag-sc-status ' + state
            status.innerHTML = `<span>●</span> <span>${L.stageStatus[state]}</span>`
        }

        // ── Retrieve docs ─────────────────────────────────────────────────────────
        function retrieveDocs(query: string) {
            const q = query.toLowerCase()
            return KB_DOCS.map((doc, i) => {
                const score = doc.topics.reduce((s, t) => s + (q.includes(t) ? 1 : 0), 0)
                return { idx: i, score: score + Math.random() * 0.3 }
            }).sort((a, b) => b.score - a.score).slice(0, 3)
        }

        // ── Build response ────────────────────────────────────────────────────────
        function buildResponse(query: string, docs: { idx: number }[]) {
            const src = docs.map(d => KB_DOCS[d.idx].name).join(', ')
            const q = query.toLowerCase()
            if (q.includes('vacation') || q.includes('vacacion') || q.includes('días') || q.includes('days'))
                return L.responses.vacation(src)
            if (q.includes('sick') || q.includes('baja') || q.includes('leave') || q.includes('enfermedad'))
                return L.responses.sick(src)
            if (q.includes('install') || q.includes('software') || q.includes('instala'))
                return L.responses.software(src)
            if (q.includes('support') || q.includes('soporte') || q.includes('horario') || q.includes('hours'))
                return L.responses.support(src)
            return L.responses.default(src)
        }

        // ── Typewriter ────────────────────────────────────────────────────────────
        function typewrite(el: HTMLElement, text: string, done?: () => void) {
            el.innerHTML = ''
            let i = 0
            const cursor = document.createElement('span')
            cursor.className = 'rag-typing-cursor'
            el.appendChild(cursor)
            const interval = setInterval(() => {
                if (i >= text.length) {
                    clearInterval(interval); cursor.remove(); done?.(); return
                }
                cursor.insertAdjacentText('beforebegin', text[i]); i++
            }, 18)
        }

        // ── Main RAG pipeline ─────────────────────────────────────────────────────
        async function runRAG() {
            if (runningRef.current) return
            runningRef.current = true
            const qInput = container.querySelector('#rag-q-input') as HTMLInputElement
            const query = qInput.value.trim()
            if (!query) { runningRef.current = false; return }

            const runBtn = container.querySelector('#rag-run-btn') as HTMLButtonElement
            runBtn.disabled = true

            const t0 = Date.now()
            L.stageDefs.forEach(s => setStage(s.id, 'idle'))
            ;(container.querySelector('#rag-retrieved-docs') as HTMLElement).innerHTML = ''
            ;(container.querySelector('#rag-aug-prompt') as HTMLElement).innerHTML = L.waitingPipeline
            const llmEl = container.querySelector('#rag-llm-response') as HTMLElement
            llmEl.innerHTML = `<span style="color:#4a6b4a;font-size:11px;">${L.generating}</span>`
            container.querySelectorAll('.rag-kb-doc').forEach(el => el.classList.remove('rag-retrieved'))

            log(L.logNewRag, 'ok')
            log(L.logQuery(query), 'data')

            // Stage 1: Index
            drawPipeline(0); setStage('index', 'running')
            log(L.logIndex, 'info')
            await delay(700); setStage('index', 'done')
            log(L.logIndexDone(KB_DOCS.length), 'ok')
            drawVectors(null, [])

            // Stage 2: Embed
            drawPipeline(1); setStage('embed', 'running')
            log(L.logEmbed, 'info')
            await delay(800)
            const qVec = { x: 20 + Math.random() * 220, y: 20 + Math.random() * 160 }
            drawVectors(qVec, []); setStage('embed', 'done')
            log(L.logVector, 'data')

            // Stage 3: Search
            drawPipeline(2); setStage('search', 'running')
            log(L.logSearch, 'info')
            await delay(900)
            const scored = retrieveDocs(query)
            drawVectors(qVec, scored.map(s => s.idx))
            scored.forEach(s => {
                container.querySelector(`#rag-kb-${s.idx}`)?.classList.add('rag-retrieved')
                log(`  ↳ ${KB_DOCS[s.idx].name} — score: ${(0.7 + Math.random() * 0.25).toFixed(3)}`, 'data')
            })
            setStage('search', 'done')
            ;(container.querySelector('#rag-m-retrieved') as HTMLElement).textContent = String(scored.length)

            const rdEl = container.querySelector('#rag-retrieved-docs')!
            rdEl.innerHTML = ''
            scored.forEach((s, ri) => {
                const doc = KB_DOCS[s.idx]
                const score = (0.7 + Math.random() * 0.25).toFixed(3)
                const div = document.createElement('div')
                div.className = 'rag-doc-item'
                div.innerHTML = `<span class="rag-doc-score">sim: ${score}</span>
          <div class="rag-doc-title">${doc.icon} ${doc.name} <span style="opacity:.5;font-size:9px">[${doc.type}]</span></div>
          <div class="rag-doc-snippet">${L.docSnippet(doc.topics)}</div>`
                rdEl.appendChild(div)
                setTimeout(() => div.classList.add('rag-doc-visible'), ri * 200)
            })

            // Stage 4: Augment
            await delay(400); drawPipeline(3); setStage('augment', 'running')
            log(L.logAugment, 'info')
            await delay(700)
            const ctxDocs = scored.map(s => `[Doc: ${KB_DOCS[s.idx].name}] ${KB_DOCS[s.idx].topics.slice(0, 4).join(', ')}.`).join('\n')
            const promptTokens = Math.floor(ctxDocs.length / 4 + query.length / 4 + 40)
            ;(container.querySelector('#rag-m-tokens') as HTMLElement).textContent = String(promptTokens)
            const augEl = container.querySelector('#rag-aug-prompt') as HTMLElement
            augEl.innerHTML = `<span class="rag-p-inst">${L.instrPrefix}</span><span class="rag-p-ctx">${L.ctxPrefix}${ctxDocs}\n</span><span class="rag-p-query">${L.qPrefix}${query}</span>`
            setStage('augment', 'done'); log(L.logPromptBuilt(promptTokens), 'ok')

            // Stage 5: Generate
            await delay(300); drawPipeline(4); setStage('generate', 'running')
            log(L.logGenerate, 'info')
            await delay(600)
            const responseText = buildResponse(query, scored)
            llmEl.style.color = '#d4e8d4'
            typewrite(llmEl, responseText, () => {
                setStage('generate', 'done'); drawPipeline(5)
                const latency = Date.now() - t0
                ;(container.querySelector('#rag-m-latency') as HTMLElement).textContent = String(latency)
                log(L.logDone(latency), 'ok')
                runningRef.current = false; runBtn.disabled = false
            })
        }

        // ── Wire events ───────────────────────────────────────────────────────────
        container.querySelector('#rag-run-btn')?.addEventListener('click', runRAG)
        container.querySelectorAll('.rag-ex-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const i = parseInt((chip as HTMLElement).dataset.ex ?? '0')
                ;(container.querySelector('#rag-q-input') as HTMLInputElement).value = L.examples[i]
            })
        })

        // ── Init ──────────────────────────────────────────────────────────────────
        drawPipeline(-1)
        setTimeout(initVectors, 100)
        log(L.logInit, 'ok')
        log(L.logDocs(KB_DOCS.length), 'data')

        return () => { runningRef.current = false }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [language])

    // ── Scoped styles ─────────────────────────────────────────────────────────────
    const styles = `
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Space+Grotesk:wght@300;500;700&display=swap');
    .rag-root { font-family: 'Space Grotesk', 'Inter', sans-serif; color: #d4e8d4; padding: 28px 24px 56px; max-width: 1060px; margin: 0 auto; position: relative; }
    .rag-header { margin-bottom: 36px; }
    .rag-sys-tag { font-size: 10px; letter-spacing: 4px; text-transform: uppercase; color: #3ddc84; opacity: .7; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
    .rag-sys-tag::before { content:''; width:24px; height:1px; background:#3ddc84; }
    .rag-h1 { font-family: 'Space Grotesk', sans-serif; font-size: clamp(28px, 5vw, 56px); font-weight: 700; line-height: .95; letter-spacing: -1px; color: #fff; margin-bottom: 8px; }
    .rag-h1 span { color: #f5a623; }
    .rag-subtitle { color: #4a6b4a; font-size: 12px; max-width: 480px; line-height: 1.7; }

    .rag-pipe-wrap { margin-bottom: 28px; border: 1px solid #263026; border-radius: 10px; background: #0e140e; padding: 8px; overflow-x: auto; }
    #rag-pipeline-svg { display: block; min-width: 700px; }

    .rag-query-panel { background: #111811; border: 1px solid #263026; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    .rag-qlabel { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #4a6b4a; margin-bottom: 10px; }
    .rag-qrow { display: flex; gap: 10px; }
    .rag-qinput { flex: 1; background: rgba(0,0,0,0.5); border: 1px solid #263026; border-radius: 5px; color: #d4e8d4; font-family: 'JetBrains Mono', monospace; font-size: 13px; padding: 11px 14px; outline: none; transition: border-color .2s; }
    .rag-qinput:focus { border-color: #f5a623; }
    .rag-run-btn { background: #f5a623; border: none; border-radius: 5px; color: #0a0a00; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 12px; letter-spacing: 1px; padding: 11px 22px; cursor: pointer; text-transform: uppercase; transition: transform .15s, box-shadow .2s; white-space: nowrap; }
    .rag-run-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(245,166,35,0.35); }
    .rag-run-btn:active { transform: none; }
    .rag-run-btn:disabled { opacity: .4; cursor: not-allowed; transform: none; }
    .rag-examples { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 10px; align-items: center; }
    .rag-ex-label { font-size: 9px; color: #4a6b4a; letter-spacing: 1px; }
    .rag-ex-chip { background: rgba(61,220,132,0.06); border: 1px solid rgba(61,220,132,0.2); color: #3ddc84; font-size: 10px; letter-spacing: .5px; padding: 4px 10px; border-radius: 3px; cursor: pointer; transition: all .2s; font-family: 'JetBrains Mono', monospace; }
    .rag-ex-chip:hover { background: rgba(61,220,132,0.14); border-color: #3ddc84; }

    .rag-stages-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
    .rag-stage-card { background: #111811; border: 1px solid #1e2e1e; border-radius: 7px; padding: 16px; transition: border-color .3s, background .3s; position: relative; overflow: hidden; }
    .rag-stage-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--stage-color, #2d452d); }
    .rag-stage-card.rag-active { border-color: var(--stage-color, #1e2e1e); background: rgba(255,255,255,.02); }
    .rag-stage-card.rag-done { border-color: #263026; opacity: .7; }
    .rag-sc-num { font-size: 9px; color: #4a6b4a; letter-spacing: 2px; margin-bottom: 6px; }
    .rag-sc-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 14px; color: #fff; margin-bottom: 6px; }
    .rag-sc-desc { font-size: 10px; color: #4a6b4a; line-height: 1.6; }
    .rag-sc-status { display: inline-flex; align-items: center; gap: 5px; font-size: 9px; letter-spacing: 1px; text-transform: uppercase; margin-top: 10px; padding: 3px 8px; border-radius: 2px; border: 1px solid currentColor; opacity: .6; font-family: 'JetBrains Mono', monospace; }
    .rag-sc-status.idle { color: #4a6b4a; }
    .rag-sc-status.running { color: #f5a623; animation: ragBlink .8s infinite; }
    .rag-sc-status.done { color: #3ddc84; opacity: 1; }
    @keyframes ragBlink { 0%,100%{opacity:.6} 50%{opacity:1} }

    .rag-metrics-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
    .rag-metric { flex: 1; min-width: 120px; background: #111811; border: 1px solid #263026; border-radius: 6px; padding: 14px; text-align: center; }
    .rag-metric-val { font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; line-height: 1; margin-bottom: 4px; }
    .rag-metric-label { font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: #4a6b4a; }

    .rag-two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
    @media(max-width:620px){ .rag-two-col { grid-template-columns: 1fr; } }

    .rag-panel { background: #111811; border: 1px solid #263026; border-radius: 8px; padding: 20px; }
    .rag-panel-title { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #3ddc84; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
    .rag-panel-title::after { content:''; flex:1; height:1px; background:#263026; }

    #rag-vector-canvas { display: block; border-radius: 6px; background: rgba(0,0,0,0.3); max-width: 100%; }
    .rag-kb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; }
    .rag-kb-doc { background: rgba(0,0,0,.35); border: 1px solid #1e2e1e; border-radius: 5px; padding: 10px; font-size: 10px; cursor: default; transition: border-color .2s, background .2s; }
    .rag-kb-doc:hover { border-color: #f5a623; background: rgba(245,166,35,.12); }
    .rag-kb-doc.rag-retrieved { border-color: #3ddc84 !important; background: rgba(61,220,132,.1) !important; animation: ragKbPulse .5s ease; }
    @keyframes ragKbPulse { 0%{transform:scale(1)} 50%{transform:scale(1.04)} 100%{transform:scale(1)} }
    .rag-kb-icon { font-size: 18px; margin-bottom: 4px; }
    .rag-kb-name { color: #d4e8d4; font-weight: 600; margin-bottom: 2px; }
    .rag-kb-type { color: #4a6b4a; font-size: 9px; letter-spacing: 1px; }

    .rag-doc-item { background: rgba(0,0,0,.3); border: 1px solid #263026; border-left: 3px solid #f5a623; border-radius: 4px; padding: 12px; margin-bottom: 8px; opacity: 0; transform: translateX(-8px); transition: opacity .4s, transform .4s; font-size: 11px; }
    .rag-doc-item.rag-doc-visible { opacity: 1; transform: none; }
    .rag-doc-score { font-size: 9px; color: #f5a623; float: right; letter-spacing: 1px; }
    .rag-doc-title { color: #3ddc84; font-size: 11px; margin-bottom: 4px; font-weight: 600; }
    .rag-doc-snippet { color: #4a6b4a; line-height: 1.6; font-size: 10px; }

    .rag-prompt-box { background: rgba(0,0,0,.4); border: 1px solid #263026; border-radius: 5px; padding: 14px; font-size: 11px; line-height: 1.8; min-height: 100px; white-space: pre-wrap; color: #4a6b4a; font-family: 'JetBrains Mono', monospace; }
    .rag-p-inst { color: #c084fc; }
    .rag-p-ctx { color: #5ab4ff; }
    .rag-p-query { color: #f5a623; }

    .rag-typing-cursor { display: inline-block; width: 8px; height: 14px; background: #3ddc84; vertical-align: middle; animation: ragBlink .6s infinite; margin-left: 2px; }

    .rag-log-feed { height: 140px; overflow-y: auto; background: rgba(0,0,0,0.5); border: 1px solid #1e2e1e; border-radius: 5px; padding: 10px; font-size: 10px; line-height: 1.8; scrollbar-width: thin; scrollbar-color: #263026 transparent; font-family: 'JetBrains Mono', monospace; }
    .rag-log-line { color: #4a6b4a; }
    .rag-ts { color: #2d452d; margin-right: 8px; }
    .rag-log-line.info .rag-msg { color: #d4e8d4; }
    .rag-log-line.ok .rag-msg { color: #3ddc84; }
    .rag-log-line.warn .rag-msg { color: #f5a623; }
    .rag-log-line.data .rag-msg { color: #5ab4ff; }
  `

    return (
        <>
            <style>{styles}</style>
            <Box
                ref={containerRef}
                className="rag-root"
                sx={{
                    'body.light &': {
                        background: 'rgba(8,11,8,0.97)',
                        borderRadius: '12px',
                        margin: '20px',
                        boxShadow: '0 8px 40px rgba(0,0,0,0.20)',
                    },
                }}
            />
        </>
    )
}
