import { useEffect, useRef } from 'react'
import { Box } from '@chakra-ui/react'
import { useLanguage } from '../context/LanguageContext'
import { useColorMode } from '../context/ThemeContext'

// ── Helpers ────────────────────────────────────────────────────────────────────

function softmax(arr: number[]): number[] {
    const max = Math.max(...arr)
    const exps = arr.map(x => Math.exp(x - max))
    const sum = exps.reduce((a, b) => a + b, 0)
    return exps.map(x => x / sum)
}

function dot(a: number[], b: number[]): number {
    return a.reduce((s, v, i) => s + v * b[i], 0)
}

function scaleColor(weight: number, dark: boolean): string {
    // weight 0..1 → color from muted to cyan
    const r = Math.round(0 + weight * 0)
    const g = Math.round((dark ? 30 : 180) + weight * (229 - (dark ? 30 : 180)))
    const blue = Math.round((dark ? 40 : 200) + weight * (255 - (dark ? 40 : 200)))
    const alpha = 0.08 + weight * 0.72
    return `rgba(${r},${g},${blue},${alpha})`
}

// ── Pretend embeddings: 4-dim for visual simplicity ───────────────────────────
// Each token gets a unique embedding vector

function makeEmbedding(tokenIdx: number, vocabSize: number): number[] {
    // Deterministic pseudo-embedding based on token index
    const t = tokenIdx / Math.max(vocabSize - 1, 1)
    return [
        Math.cos(t * Math.PI),
        Math.sin(t * Math.PI * 2),
        Math.cos(t * Math.PI * 3 + 0.5),
        Math.sin(t * Math.PI / 2 + 1),
    ]
}

// Q, K, V weight matrices (4×4), fixed but different per head
function makeWMatrix(seed: number): number[][] {
    const m: number[][] = []
    for (let i = 0; i < 4; i++) {
        m.push([])
        for (let j = 0; j < 4; j++) {
            const v = Math.sin((i + 1) * (j + 1) * seed * 0.7) * 0.8
            m[i].push(v)
        }
    }
    return m
}

function matVec(M: number[][], v: number[]): number[] {
    return M.map(row => dot(row, v))
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function TransformerAttentionSim() {
    const { language } = useLanguage()
    const { colorMode } = useColorMode()
    const dark = colorMode === 'dark'
    const containerRef = useRef<HTMLDivElement>(null)
    const headRef = useRef(0) // currently selected attention head (0..3)
    const tokensRef = useRef<string[]>([])

    const L = language === 'es' ? {
        badge: '// Simulación Interactiva',
        title: 'Atención en Transformers',
        subtitle: 'Explora cómo la auto-atención pondera cada token en relación al resto de la secuencia',
        inputLabel: '// Oración de entrada',
        placeholder: 'Escribe una oración...',
        defaultText: 'El gato se sentó en la alfombra',
        btnAnalyze: '▶ Analizar Atención',
        examples: 'Ejemplos:',
        presets: ['El banco del río', 'El modelo aprende patrones', 'La IA transforma el mundo'],
        stepTokens: '① Tokens',
        stepEmbed: '② Embeddings',
        stepQKV: '③ Matrices Q · K · V',
        stepScores: '④ Puntuaciones de Atención',
        stepWeights: '⑤ Pesos Softmax',
        stepOutput: '⑥ Vector de Salida',
        headLabel: 'Cabeza de Atención',
        heads: ['Cabeza 1', 'Cabeza 2', 'Cabeza 3', 'Cabeza 4'],
        heatmapTitle: 'Mapa de Calor de Atención',
        heatmapSub: 'Cada celda [fila, col] muestra cuánto atiende el token de la fila al de la columna',
        queryRow: 'fila = token que atiende',
        keyCol: 'columna = token al que atiende',
        qkvTitle: 'Proyecciones Q, K, V',
        qkvSub: 'Cada token se proyecta en tres espacios: Consulta, Clave y Valor',
        multiheadTitle: 'Atención Multi-Cabeza',
        multiheadSub: 'Las 4 cabezas aprenden a atender a diferentes relaciones en paralelo',
        formulaTitle: 'Fórmula de Atención',
        conceptsTitle: 'Conceptos Clave',
        concepts: [
            { term: 'Query (Q)', def: 'Representa "qué busca" cada token. Se crea multiplicando el embedding por la matriz W_Q.' },
            { term: 'Key (K)', def: 'Representa "qué ofrece" cada token. Determina cuán relevante es un token para las queries de los demás.' },
            { term: 'Value (V)', def: 'Contiene la información real que se transfiere. Una vez conocidos los pesos de atención, se pondera V.' },
            { term: 'Escala √d_k', def: 'Los scores se dividen por √d_k para evitar gradientes muy pequeños en el softmax.' },
            { term: 'Multi-Head', def: 'Múltiples cabezas en paralelo permiten que el modelo atienda a diferentes tipos de relaciones (sintácticas, semánticas, etc.).' },
        ],
        statTokens: 'Tokens',
        statHeads: 'Cabezas',
        statDim: 'Dimensión',
        statParams: 'Parámetros Q·K·V',
        emptyState: '⚡ Ingresa texto y presiona Analizar',
        formulaDesc: 'Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V',
    } : {
        badge: '// Interactive Simulation',
        title: 'Transformer Attention',
        subtitle: 'Explore how self-attention weights each token relative to every other token in the sequence',
        inputLabel: '// Input sentence',
        placeholder: 'Type a sentence...',
        defaultText: 'The cat sat on the mat',
        btnAnalyze: '▶ Analyze Attention',
        examples: 'Examples:',
        presets: ['The bank by the river', 'The model learns patterns', 'AI transforms the world'],
        stepTokens: '① Tokens',
        stepEmbed: '② Embeddings',
        stepQKV: '③ Q · K · V Matrices',
        stepScores: '④ Attention Scores',
        stepWeights: '⑤ Softmax Weights',
        stepOutput: '⑥ Output Vector',
        headLabel: 'Attention Head',
        heads: ['Head 1', 'Head 2', 'Head 3', 'Head 4'],
        heatmapTitle: 'Attention Heat Map',
        heatmapSub: 'Each cell [row, col] shows how much the row token attends to the column token',
        queryRow: 'row = attending token',
        keyCol: 'column = attended token',
        qkvTitle: 'Q, K, V Projections',
        qkvSub: 'Each token is projected into three separate spaces: Query, Key, and Value',
        multiheadTitle: 'Multi-Head Attention',
        multiheadSub: 'All 4 heads run in parallel, each learning to attend to different token relationships',
        formulaTitle: 'Attention Formula',
        conceptsTitle: 'Key Concepts',
        concepts: [
            { term: 'Query (Q)', def: 'Represents "what each token is looking for." Created by multiplying the embedding by weight matrix W_Q.' },
            { term: 'Key (K)', def: 'Represents "what each token offers." Determines how relevant a token is to other tokens\' queries.' },
            { term: 'Value (V)', def: 'Contains the actual information to be transferred. Weighted by the attention scores after softmax.' },
            { term: 'Scale √d_k', def: 'Scores are divided by √d_k to prevent vanishing gradients in the softmax for large dimensions.' },
            { term: 'Multi-Head', def: 'Multiple parallel heads let the model simultaneously attend to different relationship types (syntactic, semantic, etc.).' },
        ],
        statTokens: 'Tokens',
        statHeads: 'Heads',
        statDim: 'Dimension',
        statParams: 'Q·K·V Params',
        emptyState: '⚡ Enter text and press Analyze',
        formulaDesc: 'Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V',
    }

    useEffect(() => {
        if (!containerRef.current) return
        const container = containerRef.current!
        headRef.current = 0

        // ── WQ WK WV matrices per head ─────────────────────────────────────────────
        const WQ = [makeWMatrix(1.1), makeWMatrix(2.3), makeWMatrix(3.7), makeWMatrix(4.2)]
        const WK = [makeWMatrix(5.5), makeWMatrix(6.9), makeWMatrix(7.3), makeWMatrix(8.1)]
        const WV = [makeWMatrix(9.4), makeWMatrix(10.7), makeWMatrix(11.2), makeWMatrix(12.6)]
        const DK = 4

        // ── Build HTML ─────────────────────────────────────────────────────────────
        container.innerHTML = `
      <header class="at-header">
        <div class="at-badge">${L.badge}</div>
        <h1 class="at-h1">${L.title}</h1>
        <p class="at-subtitle">${L.subtitle}</p>
      </header>

      <!-- Stats row -->
      <div class="at-stats-row">
        <div class="at-stat"><div class="at-stat-val" id="at-s-tokens">0</div><div class="at-stat-lbl">${L.statTokens}</div></div>
        <div class="at-stat"><div class="at-stat-val" id="at-s-heads" style="color:#a78bfa">4</div><div class="at-stat-lbl">${L.statHeads}</div></div>
        <div class="at-stat"><div class="at-stat-val" id="at-s-dim" style="color:#10b981">4</div><div class="at-stat-lbl">${L.statDim}</div></div>
        <div class="at-stat"><div class="at-stat-val" id="at-s-params" style="color:#fbbf24">0</div><div class="at-stat-lbl">${L.statParams}</div></div>
      </div>

      <!-- Input -->
      <div class="at-input-card">
        <div class="at-input-lbl">${L.inputLabel}</div>
        <div class="at-input-row">
          <textarea id="at-input" placeholder="${L.placeholder}">${L.defaultText}</textarea>
          <button class="at-btn at-pulse" id="at-btn">${L.btnAnalyze}</button>
        </div>
        <div class="at-presets">
          <span class="at-presets-lbl">${L.examples}</span>
          ${L.presets.map((p, i) => `<span class="at-preset" data-i="${i}">${p}</span>`).join('')}
        </div>
      </div>

      <!-- Pipeline steps -->
      <div class="at-pipeline" id="at-pipeline">
        ${[L.stepTokens, L.stepEmbed, L.stepQKV, L.stepScores, L.stepWeights, L.stepOutput].map((s, i) => `
          <div class="at-pipe-step" id="at-pipe-${i}">
            <div class="at-pipe-dot"></div>
            <div class="at-pipe-label">${s}</div>
          </div>
        `).join('<div class="at-pipe-arrow">→</div>')}
      </div>

      <!-- Head selector -->
      <div class="at-head-row">
        <span class="at-head-lbl">${L.headLabel}:</span>
        ${L.heads.map((h, i) => `<button class="at-head-btn${i === 0 ? ' active' : ''}" data-head="${i}">${h}</button>`).join('')}
      </div>

      <!-- Heatmap + QKV side by side -->
      <div class="at-two-col">
        <div class="at-card">
          <div class="at-section-lbl">${L.heatmapTitle}</div>
          <div class="at-section-sub">${L.heatmapSub}</div>
          <div id="at-heatmap"><div class="at-empty">${L.emptyState}</div></div>
          <div class="at-heatmap-meta">
            <span class="at-meta-q">↕ ${L.queryRow}</span>
            <span class="at-meta-k">↔ ${L.keyCol}</span>
          </div>
        </div>
        <div class="at-card">
          <div class="at-section-lbl">${L.qkvTitle}</div>
          <div class="at-section-sub">${L.qkvSub}</div>
          <div id="at-qkv"><div class="at-empty">${L.emptyState}</div></div>
        </div>
      </div>

      <!-- Multi-head overview -->
      <div class="at-card" style="margin-bottom:20px">
        <div class="at-section-lbl">${L.multiheadTitle}</div>
        <div class="at-section-sub">${L.multiheadSub}</div>
        <div id="at-multihead"><div class="at-empty">${L.emptyState}</div></div>
      </div>

      <!-- Formula + concepts -->
      <div class="at-two-col">
        <div class="at-card">
          <div class="at-section-lbl">${L.formulaTitle}</div>
          <div class="at-formula-box">
            <div class="at-formula-main">Attention(Q,K,V)</div>
            <div class="at-formula-eq">= softmax</div>
            <div class="at-formula-frac">
              <span class="at-formula-num">QK<sup>T</sup></span>
              <span class="at-formula-bar"></span>
              <span class="at-formula-den">√d<sub>k</sub></span>
            </div>
            <div class="at-formula-eq">· V</div>
          </div>
          <div class="at-formula-desc">${L.formulaDesc}</div>
        </div>
        <div class="at-card">
          <div class="at-section-lbl">${L.conceptsTitle}</div>
          <div id="at-concepts">
            ${L.concepts.map(c => `
              <div class="at-concept">
                <div class="at-concept-term">${c.term}</div>
                <div class="at-concept-def">${c.def}</div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>

      <div id="at-tooltip" class="at-tooltip"></div>
    `

        // ── Core compute ───────────────────────────────────────────────────────────
        function tokenize(text: string): string[] {
            return text.trim().split(/\s+/).filter(Boolean).slice(0, 12)
        }

        function computeAttention(tokens: string[], headIdx: number) {
            const N = tokens.length
            const embeds = tokens.map((_, i) => makeEmbedding(i, N))
            const Qs = embeds.map(e => matVec(WQ[headIdx], e))
            const Ks = embeds.map(e => matVec(WK[headIdx], e))
            const Vs = embeds.map(e => matVec(WV[headIdx], e))

            // Score matrix N×N
            const scores: number[][] = []
            for (let i = 0; i < N; i++) {
                scores.push([])
                for (let j = 0; j < N; j++) {
                    scores[i].push(dot(Qs[i], Ks[j]) / Math.sqrt(DK))
                }
            }

            // Softmax per row
            const weights = scores.map(row => softmax(row))

            // Output: weighted sum of V
            const output = weights.map(row =>
                Vs[0].map((_, d) => row.reduce((s, w, j) => s + w * Vs[j][d], 0))
            )

            return { Qs, Ks, Vs, scores, weights, output }
        }

        // ── Render heatmap ─────────────────────────────────────────────────────────
        function renderHeatmap(tokens: string[], weights: number[][], headIdx: number) {
            const el = container.querySelector('#at-heatmap')!
            el.innerHTML = ''
            const N = tokens.length
            const headColors = ['#00e5ff', '#a78bfa', '#10b981', '#fbbf24']
            const hCol = headColors[headIdx]

            const grid = document.createElement('div')
            grid.className = 'at-hm-grid'
            grid.style.gridTemplateColumns = `80px repeat(${N}, 1fr)`

            // Corner blank
            const corner = document.createElement('div')
            corner.className = 'at-hm-corner'
            grid.appendChild(corner)

            // Column headers
            tokens.forEach(tok => {
                const h = document.createElement('div')
                h.className = 'at-hm-col-header'
                h.textContent = tok
                grid.appendChild(h)
            })

            // Rows
            tokens.forEach((rowTok, ri) => {
                const rh = document.createElement('div')
                rh.className = 'at-hm-row-header'
                rh.textContent = rowTok
                grid.appendChild(rh)

                tokens.forEach((_, ci) => {
                    const w = weights[ri][ci]
                    const cell = document.createElement('div')
                    cell.className = 'at-hm-cell'
                    cell.style.background = scaleColor(w, dark)
                    cell.style.borderColor = `rgba(${hCol === '#00e5ff' ? '0,229,255' : hCol === '#a78bfa' ? '167,139,250' : hCol === '#10b981' ? '16,185,129' : '251,191,36'},${0.1 + w * 0.5})`
                    cell.textContent = w.toFixed(2)
                    cell.style.color = w > 0.3 ? '#fff' : 'rgba(255,255,255,0.35)'
                    cell.style.fontSize = N > 8 ? '9px' : '11px'

                    // Tooltip
                    cell.addEventListener('mouseenter', (e) => {
                        const tip = container.querySelector('#at-tooltip') as HTMLElement
                        tip.innerHTML = `<strong>"${tokens[ri]}"</strong> → <strong>"${tokens[ci]}"</strong><br>weight: <span style="color:${hCol}">${w.toFixed(4)}</span>`
                        tip.style.opacity = '1'
                        tip.style.left = ((e as MouseEvent).clientX + 12) + 'px'
                        tip.style.top = ((e as MouseEvent).clientY - 8) + 'px'
                    })
                    cell.addEventListener('mouseleave', () => {
                        const tip = container.querySelector('#at-tooltip') as HTMLElement
                        tip.style.opacity = '0'
                    })
                    grid.appendChild(cell)
                })
            })

            el.appendChild(grid)
        }

        // ── Render Q K V bars ──────────────────────────────────────────────────────
        function renderQKV(tokens: string[], Qs: number[][], Ks: number[][], Vs: number[][]) {
            const el = container.querySelector('#at-qkv')!
            el.innerHTML = ''
            const MAX_TOKENS = Math.min(tokens.length, 6)

            tokens.slice(0, MAX_TOKENS).forEach((tok, i) => {
                const row = document.createElement('div')
                row.className = 'at-qkv-row'
                row.innerHTML = `<div class="at-qkv-tok">${tok}</div>`

                const bars = document.createElement('div')
                bars.className = 'at-qkv-bars'

                ;[['Q', Qs[i], '#00e5ff'], ['K', Ks[i], '#a78bfa'], ['V', Vs[i], '#10b981']].forEach(([label, vec, color]) => {
                    const group = document.createElement('div')
                    group.className = 'at-qkv-group'
                    group.innerHTML = `<div class="at-qkv-label" style="color:${color}">${label}</div>`
                    const barRow = document.createElement('div')
                    barRow.className = 'at-qkv-barrow'
                    ;(vec as number[]).forEach(v => {
                        const b = document.createElement('div')
                        b.className = 'at-qkv-bar'
                        const h = Math.abs(v) * 30
                        b.style.height = Math.max(4, h) + 'px'
                        b.style.background = v >= 0 ? color as string : 'rgba(255,255,255,0.15)'
                        b.style.opacity = String(0.3 + Math.abs(v) * 0.7)
                        barRow.appendChild(b)
                    })
                    group.appendChild(barRow)
                    bars.appendChild(group)
                })

                row.appendChild(bars)
                el.appendChild(row)
            })
        }

        // ── Render multi-head overview ─────────────────────────────────────────────
        function renderMultiHead(tokens: string[]) {
            const el = container.querySelector('#at-multihead')!
            el.innerHTML = ''
            const N = tokens.length
            const headColors = ['#00e5ff', '#a78bfa', '#10b981', '#fbbf24']

            const grid = document.createElement('div')
            grid.className = 'at-mh-grid'

            ;[0, 1, 2, 3].forEach(hi => {
                const { weights } = computeAttention(tokens, hi)
                const card = document.createElement('div')
                card.className = 'at-mh-card' + (hi === headRef.current ? ' at-mh-active' : '')
                card.style.setProperty('--h-color', headColors[hi])

                const title = document.createElement('div')
                title.className = 'at-mh-title'
                title.textContent = L.heads[hi]
                title.style.color = headColors[hi]
                card.appendChild(title)

                // Mini heatmap
                const mini = document.createElement('div')
                mini.className = 'at-mh-mini'
                mini.style.gridTemplateColumns = `repeat(${N}, 1fr)`

                for (let r = 0; r < N; r++) {
                    for (let c = 0; c < N; c++) {
                        const cell = document.createElement('div')
                        cell.className = 'at-mh-cell'
                        cell.style.background = scaleColor(weights[r][c], dark)
                        mini.appendChild(cell)
                    }
                }
                card.appendChild(mini)

                // Top attention span label
                const maxW = Math.max(...weights.flat())
                const maxIdx = weights.flat().indexOf(maxW)
                const ri = Math.floor(maxIdx / N), ci = maxIdx % N
                const span = document.createElement('div')
                span.className = 'at-mh-span'
                span.innerHTML = `<span style="color:${headColors[hi]}">"${tokens[ri]}"</span> → <span style="color:${headColors[hi]}">"${tokens[ci]}"</span>`
                card.appendChild(span)

                card.addEventListener('click', () => {
                    headRef.current = hi
                    container.querySelectorAll('.at-head-btn').forEach((b, bi) => {
                        b.classList.toggle('active', bi === hi)
                    })
                    analyze()
                })

                grid.appendChild(card)
            })

            el.appendChild(grid)
        }

        // ── Animate pipeline step ──────────────────────────────────────────────────
        function highlightStep(stepIdx: number) {
            container.querySelectorAll('.at-pipe-step').forEach((el, i) => {
                el.classList.toggle('at-pipe-active', i <= stepIdx)
            })
        }

        // ── Main analyze ───────────────────────────────────────────────────────────
        function analyze() {
            const textarea = container.querySelector('#at-input') as HTMLTextAreaElement
            const text = textarea.value.trim()
            if (!text) return
            const tokens = tokenize(text)
            tokensRef.current = tokens
            const hi = headRef.current
            const { Qs, Ks, Vs, weights } = computeAttention(tokens, hi)

            // Stats
            const statEl = (id: string) => container.querySelector(id) as HTMLElement
            animateStat(statEl('#at-s-tokens'), tokens.length)
            animateStat(statEl('#at-s-params'), tokens.length * 4 * 3)
            highlightStep(5)
            renderHeatmap(tokens, weights, hi)
            renderQKV(tokens, Qs, Ks, Vs)
            renderMultiHead(tokens)
        }

        function animateStat(el: HTMLElement, target: number) {
            const start = parseInt(el.textContent ?? '0') || 0
            const dur = 500; const t0 = performance.now()
            const step = (now: number) => {
                const p = Math.min((now - t0) / dur, 1)
                const ease = 1 - Math.pow(1 - p, 3)
                el.textContent = String(Math.round(start + (target - start) * ease))
                if (p < 1) requestAnimationFrame(step)
            }
            requestAnimationFrame(step)
        }

        // ── Wire events ────────────────────────────────────────────────────────────
        container.querySelector('#at-btn')?.addEventListener('click', analyze)

        container.querySelectorAll('.at-preset').forEach(chip => {
            chip.addEventListener('click', () => {
                const i = parseInt((chip as HTMLElement).dataset.i ?? '0')
                const ta = container.querySelector('#at-input') as HTMLTextAreaElement
                ta.value = L.presets[i]
                analyze()
            })
        })

        container.querySelectorAll('.at-head-btn').forEach((btn, bi) => {
            btn.addEventListener('click', () => {
                headRef.current = bi
                container.querySelectorAll('.at-head-btn').forEach((b, i) => b.classList.toggle('active', i === bi))
                if (tokensRef.current.length > 0) analyze()
            })
        })

        // Tooltip follow
        const mouseMoveHandler = (e: Event) => {
            const tip = container.querySelector('#at-tooltip') as HTMLElement
            if (tip?.style.opacity === '1') {
                tip.style.left = ((e as MouseEvent).clientX + 12) + 'px'
                tip.style.top = ((e as MouseEvent).clientY - 8) + 'px'
            }
        }
        document.addEventListener('mousemove', mouseMoveHandler)

        // Initial run
        analyze()

        return () => document.removeEventListener('mousemove', mouseMoveHandler)
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [language, dark])

    // ── Scoped styles ──────────────────────────────────────────────────────────────
    const CSS = `
    .at-root { font-family: 'Space Grotesk','Inter',sans-serif; color: #e2e8f0; padding: 32px 28px 56px; max-width: 1060px; margin: 0 auto; position: relative; }
    .at-header { text-align: center; margin-bottom: 32px; }
    .at-badge { display: inline-block; background: rgba(0,229,255,0.08); border: 1px solid rgba(0,229,255,0.25); color: #00e5ff; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; padding: 6px 16px; border-radius: 2px; margin-bottom: 14px; font-family: 'JetBrains Mono',monospace; }
    .at-h1 { font-size: clamp(24px,4vw,44px); font-weight: 700; line-height: 1.1; margin-bottom: 10px; background: linear-gradient(135deg,#fff 30%,#00e5ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .at-subtitle { color: #4a6080; font-size: 14px; max-width: 520px; margin: 0 auto; }

    .at-stats-row { display: grid; grid-template-columns: repeat(auto-fit,minmax(130px,1fr)); gap: 12px; margin-bottom: 20px; }
    .at-stat { background: #0d1520; border: 1px solid #1a2a3a; border-radius: 6px; padding: 16px; text-align: center; }
    .at-stat-val { font-size: 30px; font-weight: 700; font-family: 'JetBrains Mono',monospace; color: #00e5ff; line-height: 1; margin-bottom: 4px; }
    .at-stat-lbl { font-size: 11px; color: #4a6080; letter-spacing: 1px; text-transform: uppercase; }

    .at-input-card { background: #0d1520; border: 1px solid #1a2a3a; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    .at-input-lbl { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #4a6080; font-family: 'JetBrains Mono',monospace; margin-bottom: 10px; }
    .at-input-row { display: flex; gap: 12px; }
    .at-input-card textarea { flex: 1; background: rgba(0,0,0,0.4); border: 1px solid #1a2a3a; border-radius: 6px; color: #e2e8f0; font-family: 'JetBrains Mono',monospace; font-size: 14px; padding: 12px; resize: vertical; min-height: 60px; outline: none; transition: border-color 0.2s; }
    .at-input-card textarea:focus { border-color: #00e5ff; }
    .at-btn { background: linear-gradient(135deg,#00e5ff,#0090a8); border: none; color: #000; font-weight: 700; font-family: 'JetBrains Mono',monospace; font-size: 12px; letter-spacing: 2px; padding: 12px 22px; border-radius: 6px; cursor: pointer; text-transform: uppercase; transition: transform 0.15s,box-shadow 0.2s; white-space: nowrap; align-self: flex-start; }
    .at-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,229,255,0.35); }
    .at-btn:active { transform: translateY(0); }
    .at-presets { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; align-items: center; }
    .at-presets-lbl { font-size: 11px; color: #4a6080; font-family: monospace; }
    .at-preset { background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.2); color: #a78bfa; font-size: 11px; font-family: 'JetBrains Mono',monospace; padding: 4px 10px; border-radius: 3px; cursor: pointer; transition: all 0.2s; }
    .at-preset:hover { background: rgba(167,139,250,0.15); border-color: #a78bfa; }
    @keyframes atPulse { 0%,100%{box-shadow:0 0 0 0 rgba(0,229,255,0)} 50%{box-shadow:0 0 0 6px rgba(0,229,255,0.10)} }
    .at-pulse { animation: atPulse 2s infinite; }

    /* Pipeline */
    .at-pipeline { display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 4px; margin-bottom: 20px; }
    .at-pipe-step { display: flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 4px; border: 1px solid #1a2a3a; background: #0d1520; transition: all 0.3s; }
    .at-pipe-step.at-pipe-active { border-color: #00e5ff; background: rgba(0,229,255,0.08); }
    .at-pipe-dot { width: 6px; height: 6px; border-radius: 50%; background: #1a2a3a; transition: background 0.3s; }
    .at-pipe-step.at-pipe-active .at-pipe-dot { background: #00e5ff; box-shadow: 0 0 6px rgba(0,229,255,0.6); }
    .at-pipe-label { font-size: 11px; font-family: 'JetBrains Mono',monospace; color: #4a6080; transition: color 0.3s; white-space: nowrap; }
    .at-pipe-step.at-pipe-active .at-pipe-label { color: #00e5ff; }
    .at-pipe-arrow { color: #1a2a3a; font-size: 14px; }

    /* Head selector */
    .at-head-row { display: flex; align-items: center; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
    .at-head-lbl { font-size: 11px; color: #4a6080; font-family: 'JetBrains Mono',monospace; letter-spacing: 1px; }
    .at-head-btn { background: #0d1520; border: 1px solid #1a2a3a; color: #4a6080; font-family: 'JetBrains Mono',monospace; font-size: 11px; padding: 6px 14px; border-radius: 4px; cursor: pointer; transition: all 0.2s; }
    .at-head-btn:hover { border-color: #00e5ff; color: #00e5ff; }
    .at-head-btn.active { background: rgba(0,229,255,0.10); border-color: #00e5ff; color: #00e5ff; }

    /* Cards */
    .at-two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
    @media(max-width:700px){.at-two-col{grid-template-columns:1fr;}}
    .at-card { background: #0d1520; border: 1px solid #1a2a3a; border-radius: 8px; padding: 22px; }
    .at-section-lbl { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #00e5ff; font-family: 'JetBrains Mono',monospace; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }
    .at-section-lbl::after { content: ''; flex: 1; height: 1px; background: #1a2a3a; }
    .at-section-sub { font-size: 11px; color: #4a6080; margin-bottom: 16px; font-family: 'JetBrains Mono',monospace; line-height: 1.6; }
    .at-empty { display: flex; align-items: center; justify-content: center; height: 100px; color: #4a6080; font-family: 'JetBrains Mono',monospace; font-size: 13px; }

    /* Heatmap */
    .at-hm-grid { display: grid; gap: 3px; }
    .at-hm-corner { width: 80px; }
    .at-hm-col-header { font-size: 10px; font-family: 'JetBrains Mono',monospace; color: #4a6080; text-align: center; padding: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; writing-mode: vertical-lr; text-orientation: mixed; height: 60px; display: flex; align-items: center; justify-content: center; }
    .at-hm-row-header { font-size: 11px; font-family: 'JetBrains Mono',monospace; color: #4a6080; display: flex; align-items: center; padding-right: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 80px; }
    .at-hm-cell { border-radius: 3px; border: 1px solid rgba(0,229,255,0.05); display: flex; align-items: center; justify-content: center; font-size: 11px; font-family: 'JetBrains Mono',monospace; min-height: 32px; transition: filter 0.15s; cursor: default; }
    .at-hm-cell:hover { filter: brightness(1.4); }
    .at-heatmap-meta { display: flex; gap: 16px; margin-top: 10px; font-size: 10px; font-family: 'JetBrains Mono',monospace; }
    .at-meta-q { color: #00e5ff; } .at-meta-k { color: #a78bfa; }

    /* QKV bars */
    .at-qkv-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #1a2a3a; }
    .at-qkv-row:last-child { border-bottom: none; margin-bottom: 0; }
    .at-qkv-tok { font-family: 'JetBrains Mono',monospace; font-size: 12px; color: #e2e8f0; min-width: 60px; }
    .at-qkv-bars { display: flex; gap: 12px; flex: 1; }
    .at-qkv-group { display: flex; flex-direction: column; gap: 4px; }
    .at-qkv-label { font-size: 9px; font-family: 'JetBrains Mono',monospace; font-weight: 700; }
    .at-qkv-barrow { display: flex; gap: 2px; align-items: flex-end; height: 40px; }
    .at-qkv-bar { width: 12px; border-radius: 2px 2px 0 0; transition: height 0.4s cubic-bezier(0.34,1.56,0.64,1); }

    /* Multi-head grid */
    .at-mh-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
    @media(max-width:700px){.at-mh-grid{grid-template-columns:repeat(2,1fr);}}
    .at-mh-card { background: rgba(0,0,0,0.3); border: 1px solid #1a2a3a; border-radius: 6px; padding: 12px; cursor: pointer; transition: all 0.2s; }
    .at-mh-card:hover { border-color: var(--h-color,#00e5ff); background: rgba(255,255,255,0.02); }
    .at-mh-card.at-mh-active { border-color: var(--h-color,#00e5ff); background: rgba(0,229,255,0.05); }
    .at-mh-title { font-size: 11px; font-family: 'JetBrains Mono',monospace; font-weight: 700; margin-bottom: 8px; }
    .at-mh-mini { display: grid; gap: 2px; margin-bottom: 8px; }
    .at-mh-cell { width: 100%; aspect-ratio: 1; border-radius: 1px; min-width: 4px; }
    .at-mh-span { font-size: 10px; font-family: 'JetBrains Mono',monospace; color: #4a6080; }

    /* Formula */
    .at-formula-box { display: flex; align-items: center; gap: 8px; padding: 20px; background: rgba(0,0,0,0.3); border-radius: 6px; border: 1px solid #1a2a3a; margin-bottom: 12px; flex-wrap: wrap; justify-content: center; }
    .at-formula-main { font-family: 'JetBrains Mono',monospace; font-size: 13px; color: #00e5ff; }
    .at-formula-eq { font-family: 'JetBrains Mono',monospace; font-size: 13px; color: #4a6080; }
    .at-formula-frac { display: flex; flex-direction: column; align-items: center; gap: 2px; }
    .at-formula-num { font-family: 'JetBrains Mono',monospace; font-size: 13px; color: #a78bfa; }
    .at-formula-bar { width: 100%; height: 1px; background: #4a6080; }
    .at-formula-den { font-family: 'JetBrains Mono',monospace; font-size: 13px; color: #fbbf24; }
    .at-formula-desc { font-family: 'JetBrains Mono',monospace; font-size: 11px; color: #4a6080; text-align: center; }

    /* Concepts */
    #at-concepts { display: flex; flex-direction: column; gap: 12px; }
    .at-concept { border-left: 2px solid rgba(0,229,255,0.3); padding-left: 12px; }
    .at-concept-term { font-size: 13px; font-weight: 700; color: #00e5ff; font-family: 'JetBrains Mono',monospace; margin-bottom: 3px; }
    .at-concept-def { font-size: 12px; color: #4a6080; line-height: 1.6; }

    /* Tooltip */
    .at-tooltip { position: fixed; background: #111d2d; border: 1px solid #00e5ff; border-radius: 6px; padding: 8px 12px; font-size: 12px; font-family: 'JetBrains Mono',monospace; pointer-events: none; opacity: 0; transition: opacity 0.15s; z-index: 9999; color: #e2e8f0; max-width: 240px; }
  `

    return (
        <>
            <style>{CSS}</style>
            <Box
                ref={containerRef}
                className="at-root"
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
