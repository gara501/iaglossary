import { useEffect, useRef, useCallback } from 'react'
import { Box } from '@chakra-ui/react'
import { useLanguage } from '../context/LanguageContext'
import { useColorMode } from '../context/ThemeContext'

// ── Helpers ──────────────────────────────────────────────────────────────────

function escapeHtml(str: string) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
}

function hexToRgb(hex: string) {
    const r = parseInt(hex.slice(1, 3), 16)
    const g = parseInt(hex.slice(3, 5), 16)
    const b = parseInt(hex.slice(5, 7), 16)
    return `${r},${g},${b}`
}

const COLORS = ['tc0', 'tc1', 'tc2', 'tc3', 'tc4', 'tc5']
const BG_COLORS = [
    'rgba(0,229,255,0.15)', 'rgba(255,107,53,0.15)', 'rgba(124,58,237,0.15)',
    'rgba(16,185,129,0.15)', 'rgba(245,158,11,0.15)', 'rgba(236,72,153,0.15)',
]
const FG_COLORS = ['#00e5ff', '#ff6b35', '#a78bfa', '#10b981', '#fbbf24', '#f472b6']

function colorForToken(text: string) {
    const hash = [...text].reduce((a, c) => a + c.charCodeAt(0), 0)
    return hash % 6
}

// ── Tokenizers ────────────────────────────────────────────────────────────────

interface Token {
    text: string
    start: number
    end: number
    byte?: number
    word?: string
}

function tokenizeWord(text: string): Token[] {
    const parts: Token[] = []
    const re = /(\s+|[^\w\sáéíóúüñÁÉÍÓÚÜÑ]|[\w\sáéíóúüñÁÉÍÓÚÜÑ]+)/gu
    let m: RegExpExecArray | null
    while ((m = re.exec(text)) !== null) {
        if (m[0].trim()) parts.push({ text: m[0].trim(), start: m.index, end: m.index + m[0].length })
    }
    return parts
}

function tokenizeChar(text: string): Token[] {
    return [...text].map((c, i) => ({ text: c === ' ' ? '·' : c, start: i, end: i + 1 }))
}

function tokenizeSubword(text: string): Token[] {
    const merges: Record<string, string[]> = {
        'tokenización': ['token', '##ización'],
        'tokenization': ['token', '##ization'],
        'inteligencia': ['intel', '##igencia'],
        'artificial': ['art', '##ificial'],
        'increíblemente': ['increíble', '##mente'],
        'incrementally': ['increment', '##ally'],
        'útiles': ['útil', '##es'],
        'chatbots': ['chat', '##bots'],
        'procesamiento': ['proces', '##amiento'],
        'processing': ['process', '##ing'],
        'extraordinario': ['extra', '##ordinario'],
        'poderoso': ['poder', '##oso'],
        'lenguaje': ['len', '##guaje'],
        'language': ['lang', '##uage'],
    }
    const result: Token[] = []
    const words = text.split(/(\s+|[^\w\sáéíóúüñÁÉÍÓÚÜÑ])/gu).filter(w => w.trim())
    let pos = 0
    for (const w of words) {
        const lower = w.toLowerCase()
        if (merges[lower]) {
            merges[lower].forEach(part => {
                result.push({ text: part, start: pos, end: pos + w.length, word: w })
                pos += w.length
            })
        } else if (w.length > 8) {
            const mid = Math.floor(w.length / 2)
            result.push({ text: w.slice(0, mid), start: pos, end: pos + mid, word: w })
            result.push({ text: '##' + w.slice(mid), start: pos + mid, end: pos + w.length, word: w })
        } else {
            result.push({ text: w, start: pos, end: pos + w.length })
        }
        pos += w.length + 1
    }
    return result
}

function tokenizeByte(text: string): Token[] {
    const enc = new TextEncoder()
    const bytes = enc.encode(text)
    return Array.from(bytes).map((b, i) => ({
        text: '0x' + b.toString(16).toUpperCase().padStart(2, '0'),
        start: i, end: i + 1, byte: b,
    }))
}

function runTokenizer(mode: string, text: string): Token[] {
    switch (mode) {
        case 'word': return tokenizeWord(text)
        case 'char': return tokenizeChar(text)
        case 'subword': return tokenizeSubword(text)
        case 'byte': return tokenizeByte(text)
        default: return tokenizeWord(text)
    }
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function TokenizationSim() {
    const { language } = useLanguage()
    const { colorMode } = useColorMode()
    void colorMode // used for light-mode CSS override via body class

    // DOM refs
    const containerRef = useRef<HTMLDivElement>(null)
    const modeRef = useRef('word')
    const tokensRef = useRef<Token[]>([])

    // ── i18n labels ──────────────────────────────────────────────────────────
    const labels = {
        en: {
            badge: '// Interactive Simulation',
            title: 'Tokenization in AI',
            subtitle: 'Explore how language models break text into semantic units called <em>tokens</em>',
            inputLabel: '// Input text',
            placeholder: 'Type or paste text here...',
            defaultText: 'AI chatbots and large language models rely on tokenization.',
            btnTokenize: '▶ Tokenize',
            examples: 'Examples:',
            presets: ['Hello world', 'GPT-4 is an LLM', 'tokenization NLP', '¡Extraordinary!'],
            tabWord: 'Word', tabChar: 'Character', tabSubword: 'Subword (BPE)', tabByte: 'Byte',
            statTokens: 'Tokens', statChars: 'Characters', statRatio: 'Chars/Token', statUnique: 'Unique',
            pipelineLabel: 'Tokenization Pipeline',
            pipelineStages: ['Raw Text', 'Normalization', 'Segmentation', 'Numeric IDs', 'Embeddings'],
            tokensLabel: 'Generated tokens',
            segmentedLabel: 'Segmented text',
            bpeLabel: 'BPE Decomposition (Byte Pair Encoding)',
            freqLabel: 'Token frequency',
            aboutLabel: 'About current mode',
            emptyPress: '⚡ Press Tokenize',
            emptyViz: 'Original text visualization',
            noData: 'No data',
            moreItems: (n: number) => `+${n} more...`,
            modeInfo: {
                word: `<strong style="color:#00e5ff">Word Tokenization</strong><br>Text is split by spaces and punctuation. The most intuitive method. Each word becomes a token. Simple but has issues with unknown vocabulary (OOV — Out of Vocabulary).`,
                char: `<strong style="color:#f472b6">Character Tokenization</strong><br>Each individual character is a token. Vocabulary is minimal (~100 tokens) but sequences are very long. Useful for languages without clear separators (Chinese, Japanese) or morphological analysis tasks.`,
                subword: `<strong style="color:#a78bfa">BPE — Byte Pair Encoding</strong><br>The method used by GPT, BERT and most modern LLMs. Common words remain intact; rare ones are split into known morphemes. Optimal balance between small vocabulary and manageable sequences.`,
                byte: `<strong style="color:#10b981">Byte Tokenization</strong><br>Each UTF-8 byte is a token (0–255). No OOV possible — works for any language or binary file. GPT-2 and mT5 use it as a base layer. Sequences are long but vocabulary is perfectly fixed.`,
            },
        },
        es: {
            badge: '// Simulación Interactiva',
            title: 'Tokenización en IA',
            subtitle: 'Explora cómo los modelos de lenguaje descomponen el texto en unidades semánticas llamadas <em>tokens</em>',
            inputLabel: '// Texto de entrada',
            placeholder: 'Escribe o pega texto aquí...',
            defaultText: 'Los chatbots de inteligencia artificial son increíblemente útiles para tokenización.',
            btnTokenize: '▶ Tokenizar',
            examples: 'Ejemplos:',
            presets: ['Hola mundo', 'GPT-4 es un LLM', 'tokenización NLP', '¡Extraordinario!'],
            tabWord: 'Palabra', tabChar: 'Carácter', tabSubword: 'Subpalabra (BPE)', tabByte: 'Byte',
            statTokens: 'Tokens', statChars: 'Caracteres', statRatio: 'Chars/Token', statUnique: 'Únicos',
            pipelineLabel: 'Pipeline de Tokenización',
            pipelineStages: ['Texto Raw', 'Normalización', 'Segmentación', 'IDs numéricos', 'Embeddings'],
            tokensLabel: 'Tokens generados',
            segmentedLabel: 'Texto segmentado',
            bpeLabel: 'Descomposición BPE (Byte Pair Encoding)',
            freqLabel: 'Frecuencia de tokens',
            aboutLabel: 'Acerca del modo actual',
            emptyPress: '⚡ Presiona Tokenizar',
            emptyViz: 'Visualización del texto original',
            noData: 'Sin datos',
            moreItems: (n: number) => `+${n} más...`,
            modeInfo: {
                word: `<strong style="color:#00e5ff">Tokenización por Palabra</strong><br>El texto se divide por espacios y signos de puntuación. Es el método más intuitivo. Cada palabra se convierte en un token. Simple pero tiene problemas con vocabulario desconocido (OOV - Out of Vocabulary).`,
                char: `<strong style="color:#f472b6">Tokenización por Carácter</strong><br>Cada carácter individual es un token. El vocabulario es mínimo (~100 tokens) pero las secuencias son muy largas. Útil para idiomas sin separadores claros (chino, japonés) o tareas de análisis morfológico.`,
                subword: `<strong style="color:#a78bfa">BPE — Byte Pair Encoding</strong><br>El método usado por GPT, BERT y la mayoría de LLMs modernos. Las palabras frecuentes quedan intactas; las raras se descomponen en morfemas conocidos. Balance óptimo entre vocabulario pequeño y secuencias manejables.`,
                byte: `<strong style="color:#10b981">Tokenización por Byte</strong><br>Cada byte UTF-8 es un token (0–255). Sin OOV posible — funciona para cualquier idioma o archivo binario. GPT-2 y mT5 lo usan como capa base. Las secuencias son largas pero el vocabulario es perfectamente fijo.`,
            },
        },
    }

    // ── Core functions (working inside the shadow DOM refs) ──────────────────

    const getEl = useCallback((id: string) => {
        return containerRef.current?.querySelector(`#${id}`) as HTMLElement | null
    }, [])

    const animateStat = useCallback((id: string, target: number) => {
        const el = getEl(id)
        if (!el) return
        const start = parseInt(el.textContent ?? '0') || 0
        const dur = 500
        const t0 = performance.now()
        const step = (now: number) => {
            const p = Math.min((now - t0) / dur, 1)
            const ease = 1 - Math.pow(1 - p, 3)
            el.textContent = String(Math.round(start + (target - start) * ease))
            if (p < 1) requestAnimationFrame(step)
        }
        requestAnimationFrame(step)
    }, [getEl])

    const renderTokenStream = useCallback((tokens: Token[]) => {
        const el = getEl('sim-token-stream')
        if (!el) return
        el.innerHTML = ''
        tokens.forEach((tok, i) => {
            const ci = colorForToken(tok.text)
            const chip = document.createElement('div')
            chip.className = `sim-token-chip ${COLORS[ci]}`
            chip.style.animationDelay = `${Math.min(i * 30, 600)}ms`
            chip.innerHTML = `<span class="sim-tid">#${i}</span>${escapeHtml(tok.text)}`
            chip.addEventListener('mouseenter', (e) => {
                const tip = getEl('sim-tooltip')
                if (!tip) return
                const lines = [
                    `Token #${i}: <span style="color:#00e5ff">${escapeHtml(tok.text)}</span>`,
                    `Length: ${tok.text.length} chars`,
                    tok.byte !== undefined ? `Byte: ${tok.byte} (0x${tok.byte.toString(16).toUpperCase()})` : '',
                ].filter(Boolean)
                tip.innerHTML = lines.join('<br>')
                tip.style.opacity = '1'
                tip.style.left = ((e as MouseEvent).clientX + 12) + 'px'
                tip.style.top = ((e as MouseEvent).clientY - 10) + 'px'
            })
            chip.addEventListener('mouseleave', () => {
                const tip = getEl('sim-tooltip')
                if (tip) tip.style.opacity = '0'
            })
            el.appendChild(chip)
        })
    }, [getEl])

    const renderHighlightedText = useCallback((_text: string, tokens: Token[], mode: string, L: typeof labels['en']) => {
        const el = getEl('sim-highlighted-text')
        if (!el) return
        el.innerHTML = ''
        if (mode === 'word' || mode === 'subword') {
            tokens.forEach((tok, idx) => {
                const ci = colorForToken(tok.text)
                const span = document.createElement('span')
                span.className = 'sim-ht-span'
                span.style.background = BG_COLORS[ci]
                span.style.color = FG_COLORS[ci]
                span.textContent = tok.word ?? tok.text
                span.addEventListener('mouseenter', (e) => {
                    const tip = getEl('sim-tooltip')
                    if (!tip) return
                    tip.innerHTML = `Token #${idx}: <span style="color:#00e5ff">${escapeHtml(tok.text)}</span>`
                    tip.style.opacity = '1'
                    tip.style.left = ((e as MouseEvent).clientX + 12) + 'px'
                    tip.style.top = ((e as MouseEvent).clientY - 10) + 'px'
                })
                span.addEventListener('mouseleave', () => {
                    const tip = getEl('sim-tooltip')
                    if (tip) tip.style.opacity = '0'
                })
                el.appendChild(span)
                el.appendChild(document.createTextNode(' '))
            })
        } else {
            tokens.slice(0, 100).forEach((tok) => {
                const ci = colorForToken(tok.text)
                const span = document.createElement('span')
                span.className = 'sim-ht-span'
                span.style.background = BG_COLORS[ci]
                span.style.color = FG_COLORS[ci]
                span.textContent = tok.text.replace('·', ' ')
                el.appendChild(span)
            })
            if (tokens.length > 100) {
                const more = document.createElement('span')
                more.style.color = 'rgba(74,96,128,1)'
                more.style.fontSize = '12px'
                more.textContent = ` ${L.moreItems(tokens.length - 100)}` // L used here
                el.appendChild(more)
            }
        }
    }, [getEl])

    const renderPipeline = useCallback((text: string, tokens: Token[], stages: string[]) => {
        const svg = document.getElementById('sim-svg-pipeline') as SVGSVGElement | null
        if (!svg) return
        const stageColors = ['#00e5ff', '#ff6b35', '#a78bfa', '#10b981', '#fbbf24']
        const W = Math.max(600, stages.length * 140)
        svg.setAttribute('viewBox', `0 0 ${W} 90`)
        ;(svg as unknown as HTMLElement).style.minWidth = W + 'px'
        const stageW = W / stages.length
        let html = ''
        stages.forEach((label, i) => {
            const x = i * stageW + stageW / 2
            const color = stageColors[i]
            const isActive = i <= 3
            if (i < stages.length - 1) {
                const ax = x + stageW / 2 - 20
                html += `<line x1="${x + 44}" y1="45" x2="${ax}" y2="45" stroke="${isActive ? color : '#1a2a3a'}" stroke-width="1.5" stroke-dasharray="${isActive ? '0' : '4'}"/>`
                html += `<polygon points="${ax},40 ${ax + 8},45 ${ax},50" fill="${isActive ? stageColors[i + 1] : '#1a2a3a'}"/>`
            }
            html += `<circle cx="${x}" cy="45" r="30" fill="${isActive ? `rgba(${hexToRgb(color)},0.08)` : 'rgba(26,42,58,0.5)'}" stroke="${isActive ? color : '#1a2a3a'}" stroke-width="1.5"/>`
            if (isActive && i === 2) {
                html += `<circle cx="${x}" cy="45" r="34" fill="none" stroke="${color}" stroke-width="1" opacity="0.4"><animate attributeName="r" from="30" to="40" dur="2s" repeatCount="indefinite"/><animate attributeName="opacity" from="0.5" to="0" dur="2s" repeatCount="indefinite"/></circle>`
            }
            const icons = ['T', '≋', '{ }', '#', '⊕']
            html += `<text x="${x}" y="40" text-anchor="middle" dominant-baseline="middle" font-family="JetBrains Mono" font-size="14" fill="${isActive ? color : '#1a2a3a'}" font-weight="bold">${icons[i]}</text>`
            html += `<text x="${x}" y="54" text-anchor="middle" dominant-baseline="middle" font-family="JetBrains Mono" font-size="8" fill="${isActive ? color : '#2a4050'}">${label}</text>`
            const vals = [`${text.length}c`, `${text.length}c`, `${tokens.length}t`, `${tokens.length} ids`, '→ LLM']
            html += `<text x="${x}" y="76" text-anchor="middle" font-family="JetBrains Mono" font-size="9" fill="${isActive ? 'rgba(200,220,240,0.7)' : '#2a4050'}">${vals[i]}</text>`
        })
        svg.innerHTML = html
    }, [getEl])

    const renderFrequency = useCallback((tokens: Token[]) => {
        const el = getEl('sim-freq-chart')
        if (!el) return
        const freq: Record<string, number> = {}
        tokens.forEach(t => { freq[t.text] = (freq[t.text] || 0) + 1 })
        const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 20)
        const maxF = sorted[0]?.[1] || 1
        el.innerHTML = ''
        sorted.forEach(([tok, count]) => {
            const h = Math.max(4, (count / maxF) * 80)
            const ci = colorForToken(tok)
            const wrap = document.createElement('div')
            wrap.className = 'sim-freq-bar-wrap'
            wrap.innerHTML = `
        <div class="sim-freq-count">${count}</div>
        <div class="sim-freq-bar" style="height:${h}px;background:linear-gradient(to top,${FG_COLORS[ci]},${FG_COLORS[ci]}44)"></div>
        <div class="sim-freq-label" title="${tok}">${escapeHtml(tok.length > 5 ? tok.slice(0, 5) + '…' : tok)}</div>
      `
            el.appendChild(wrap)
        })
    }, [getEl])

    const renderSubwordView = useCallback((tokens: Token[], mode: string) => {
        if (mode !== 'subword') return
        const el = getEl('sim-subword-view')
        if (!el) return
        const words: Record<string, string[]> = {}
        tokens.forEach(t => {
            const w = t.word ?? t.text
            if (!words[w]) words[w] = []
            words[w].push(t.text)
        })
        el.innerHTML = Object.entries(words).map(([word, parts]) => {
            const partsHtml = parts.map((p, i) => {
                const cls = p.startsWith('##') ? 'suffix' : (i === 0 ? 'prefix' : '')
                return `<span class="sim-sw-part ${cls}">${escapeHtml(p)}</span>`
            }).join('')
            return `<div class="sim-sw-row">
        <span class="sim-sw-original">${escapeHtml(word)}</span>
        <span class="sim-sw-arrow">→</span>
        <div class="sim-sw-parts">${partsHtml}</div>
      </div>`
        }).join('')
    }, [getEl])

    const doTokenize = useCallback((L: typeof labels['en']) => {
        const textarea = getEl('sim-input-text') as HTMLTextAreaElement | null
        const inputText = textarea?.value ?? ''
        if (!inputText.trim()) return
        const mode = modeRef.current
        const tokens = runTokenizer(mode, inputText)
        tokensRef.current = tokens

        animateStat('sim-stat-tokens', tokens.length)
        animateStat('sim-stat-chars', inputText.length)
        const ratioEl = getEl('sim-stat-ratio')
        if (ratioEl) ratioEl.textContent = (inputText.length / Math.max(tokens.length, 1)).toFixed(1)
        const unique = new Set(tokens.map(t => t.text)).size
        animateStat('sim-stat-unique', unique)

        renderTokenStream(tokens)
        renderHighlightedText(inputText, tokens, mode, L)
        renderFrequency(tokens)
        renderPipeline(inputText, tokens, L.pipelineStages)
        renderSubwordView(tokens, mode)

        // Update mode info
        const infoEl = getEl('sim-mode-info')
        if (infoEl) infoEl.innerHTML = L.modeInfo[mode as keyof typeof L.modeInfo] ?? ''

        // Show/hide BPE card
        const bpeCard = getEl('sim-subword-card')
        if (bpeCard) bpeCard.style.display = mode === 'subword' ? 'block' : 'none'
    }, [animateStat, renderTokenStream, renderHighlightedText, renderFrequency, renderPipeline, renderSubwordView, getEl])

    const setMode = useCallback((mode: string, L: typeof labels['en']) => {
        modeRef.current = mode
        const tabs = containerRef.current?.querySelectorAll('.sim-mode-tab')
        tabs?.forEach(t => t.classList.remove('active'))
        const activeTab = getEl(`sim-tab-${mode}`)
        if (activeTab) activeTab.classList.add('active')
        doTokenize(L)
    }, [doTokenize, getEl])

    // ── Build initial HTML into container ────────────────────────────────────

    useEffect(() => {
        const container = containerRef.current
        if (!container) return

        const L = labels[language] ?? labels.en

        container.innerHTML = `
      <div id="sim-tooltip"></div>

      <header class="sim-header">
        <div class="sim-badge">${L.badge}</div>
        <h1 class="sim-h1">${L.title}</h1>
        <p class="sim-subtitle">${L.subtitle}</p>
      </header>

      <!-- Mode tabs -->
      <div class="sim-mode-tabs">
        <button class="sim-mode-tab active" id="sim-tab-word">${L.tabWord}</button>
        <button class="sim-mode-tab" id="sim-tab-char">${L.tabChar}</button>
        <button class="sim-mode-tab" id="sim-tab-subword">${L.tabSubword}</button>
        <button class="sim-mode-tab" id="sim-tab-byte">${L.tabByte}</button>
      </div>

      <!-- Input -->
      <div class="sim-input-card">
        <div class="sim-input-label">${L.inputLabel}</div>
        <div class="sim-input-row">
          <textarea id="sim-input-text" placeholder="${L.placeholder}">${L.defaultText}</textarea>
          <button class="sim-tokenize-btn sim-pulse" id="sim-tokenize-btn">${L.btnTokenize}</button>
        </div>
        <div class="sim-presets">
          <span class="sim-presets-label">${L.examples}</span>
          ${L.presets.map((p, i) => `<span class="sim-preset-chip" data-preset="${i}">${p}</span>`).join('')}
        </div>
      </div>

      <!-- Stats -->
      <div class="sim-stats-row">
        <div class="sim-stat-box">
          <div class="sim-stat-value" id="sim-stat-tokens">0</div>
          <div class="sim-stat-label">${L.statTokens}</div>
        </div>
        <div class="sim-stat-box">
          <div class="sim-stat-value" id="sim-stat-chars" style="color:#f472b6">0</div>
          <div class="sim-stat-label">${L.statChars}</div>
        </div>
        <div class="sim-stat-box">
          <div class="sim-stat-value" id="sim-stat-ratio" style="color:#10b981">0</div>
          <div class="sim-stat-label">${L.statRatio}</div>
        </div>
        <div class="sim-stat-box">
          <div class="sim-stat-value" id="sim-stat-unique" style="color:#fbbf24">0</div>
          <div class="sim-stat-label">${L.statUnique}</div>
        </div>
      </div>

      <!-- Pipeline SVG -->
      <div class="sim-viz-card">
        <div class="sim-section-label">${L.pipelineLabel}</div>
        <div class="sim-pipeline-wrap">
          <svg id="sim-svg-pipeline" height="90" style="min-width:600px"></svg>
        </div>
      </div>

      <!-- Token stream + highlighted text -->
      <div class="sim-two-col">
        <div class="sim-viz-card">
          <div class="sim-section-label">${L.tokensLabel}</div>
          <div id="sim-token-stream">
            <div class="sim-empty-state">⚡ ${L.emptyPress}</div>
          </div>
        </div>
        <div class="sim-viz-card">
          <div class="sim-section-label">${L.segmentedLabel}</div>
          <div id="sim-highlighted-text">
            <div class="sim-empty-state">${L.emptyViz}</div>
          </div>
        </div>
      </div>

      <!-- Subword card -->
      <div class="sim-viz-card" id="sim-subword-card" style="display:none">
        <div class="sim-section-label">${L.bpeLabel}</div>
        <div id="sim-subword-view"></div>
      </div>

      <!-- Frequency chart -->
      <div class="sim-viz-card">
        <div class="sim-section-label">${L.freqLabel}</div>
        <div id="sim-freq-chart">
          <div class="sim-empty-state">${L.noData}</div>
        </div>
      </div>

      <!-- Mode info -->
      <div class="sim-viz-card">
        <div class="sim-section-label">${L.aboutLabel}</div>
        <div id="sim-mode-info" class="sim-mode-info-text"></div>
      </div>
    `

        // Wire events
        const tokenizeBtn = container.querySelector('#sim-tokenize-btn')
        tokenizeBtn?.addEventListener('click', () => doTokenize(L))

        container.querySelectorAll('.sim-mode-tab').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = btn.id.replace('sim-tab-', '')
                setMode(id, L)
            })
        })

        container.querySelectorAll('.sim-preset-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const i = parseInt((chip as HTMLElement).dataset.preset ?? '0')
                const ta = container.querySelector('#sim-input-text') as HTMLTextAreaElement
                if (ta) { ta.value = L.presets[i]; doTokenize(L) }
            })
        })

        // Tooltip follow mouse
        const mouseHandler = (e: Event) => {
            const tip = container.querySelector('#sim-tooltip') as HTMLElement
            if (tip && tip.style.opacity === '1') {
                tip.style.left = ((e as MouseEvent).clientX + 12) + 'px'
                tip.style.top = ((e as MouseEvent).clientY - 10) + 'px'
            }
        }
        document.addEventListener('mousemove', mouseHandler)

        // Run initial tokenize
        modeRef.current = 'word'
        doTokenize(L)

        return () => {
            document.removeEventListener('mousemove', mouseHandler)
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [language])   // re-run when language changes to rebuild labels

    // ── Styles (scoped via sim- prefix) ─────────────────────────────────────
    const simStyles = `
    .sim-container {
      font-family: 'Space Grotesk', 'Inter', sans-serif;
      color: #e2e8f0;
      padding: 32px 28px 48px;
      max-width: 1060px;
      margin: 0 auto;
      position: relative;
    }
    #sim-tooltip {
      position: fixed;
      background: #111d2d;
      border: 1px solid #00e5ff;
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.15s;
      z-index: 9999;
      color: #e2e8f0;
      max-width: 220px;
    }
    .sim-header { text-align: center; margin-bottom: 36px; }
    .sim-badge {
      display: inline-block;
      background: rgba(0,229,255,0.08);
      border: 1px solid rgba(0,229,255,0.25);
      color: #00e5ff;
      font-size: 11px;
      letter-spacing: 3px;
      text-transform: uppercase;
      padding: 6px 16px;
      border-radius: 2px;
      margin-bottom: 14px;
      font-family: 'JetBrains Mono', monospace;
    }
    .sim-h1 {
      font-size: clamp(24px, 4vw, 44px);
      font-weight: 700;
      line-height: 1.1;
      margin-bottom: 10px;
      background: linear-gradient(135deg, #fff 30%, #00e5ff 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .sim-subtitle { color: #4a6080; font-size: 14px; max-width: 480px; margin: 0 auto; }
    .sim-mode-tabs { display: flex; gap: 8px; justify-content: center; margin-bottom: 28px; flex-wrap: wrap; }
    .sim-mode-tab {
      background: #0d1520;
      border: 1px solid #1a2a3a;
      color: #4a6080;
      padding: 8px 18px;
      border-radius: 4px;
      cursor: pointer;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      letter-spacing: 1px;
      transition: all 0.2s;
      text-transform: uppercase;
    }
    .sim-mode-tab:hover { border-color: #00e5ff; color: #00e5ff; }
    .sim-mode-tab.active {
      background: rgba(0,229,255,0.10);
      border-color: #00e5ff;
      color: #00e5ff;
      box-shadow: 0 0 20px rgba(0,229,255,0.15);
    }
    .sim-input-card { background: #0d1520; border: 1px solid #1a2a3a; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    .sim-input-label { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #4a6080; font-family: 'JetBrains Mono', monospace; margin-bottom: 10px; }
    .sim-input-row { display: flex; gap: 12px; }
    .sim-input-card textarea {
      flex: 1;
      background: rgba(0,0,0,0.4);
      border: 1px solid #1a2a3a;
      border-radius: 6px;
      color: #e2e8f0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      padding: 12px;
      resize: vertical;
      min-height: 70px;
      outline: none;
      transition: border-color 0.2s;
    }
    .sim-input-card textarea:focus { border-color: #00e5ff; }
    .sim-tokenize-btn {
      background: linear-gradient(135deg, #00e5ff, #0090a8);
      border: none;
      color: #000;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      letter-spacing: 2px;
      padding: 12px 22px;
      border-radius: 6px;
      cursor: pointer;
      text-transform: uppercase;
      transition: transform 0.15s, box-shadow 0.2s;
      white-space: nowrap;
      align-self: flex-start;
    }
    .sim-tokenize-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,229,255,0.35); }
    .sim-tokenize-btn:active { transform: translateY(0); }
    .sim-presets { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; align-items: center; }
    .sim-presets-label { font-size: 11px; color: #4a6080; font-family: monospace; }
    .sim-preset-chip {
      background: rgba(255,107,53,0.08); border: 1px solid rgba(255,107,53,0.2); color: #ff6b35;
      font-size: 11px; font-family: 'JetBrains Mono', monospace; padding: 4px 10px; border-radius: 3px;
      cursor: pointer; transition: all 0.2s;
    }
    .sim-preset-chip:hover { background: rgba(255,107,53,0.15); border-color: #ff6b35; }
    .sim-stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 20px; }
    .sim-stat-box { background: #0d1520; border: 1px solid #1a2a3a; border-radius: 6px; padding: 16px; text-align: center; }
    .sim-stat-value { font-size: 30px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #00e5ff; line-height: 1; margin-bottom: 4px; transition: color 0.3s; }
    .sim-stat-label { font-size: 11px; color: #4a6080; letter-spacing: 1px; text-transform: uppercase; }
    .sim-viz-card { background: #0d1520; border: 1px solid #1a2a3a; border-radius: 8px; padding: 22px; margin-bottom: 20px; min-height: 160px; }
    .sim-section-label {
      font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #00e5ff;
      font-family: 'JetBrains Mono', monospace; margin-bottom: 14px;
      display: flex; align-items: center; gap: 8px;
    }
    .sim-section-label::after { content: ''; flex: 1; height: 1px; background: #1a2a3a; }
    .sim-pipeline-wrap { overflow-x: auto; padding-bottom: 8px; }
    #sim-svg-pipeline { width: 100%; overflow: visible; }
    .sim-two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
    @media (max-width: 640px) { .sim-two-col { grid-template-columns: 1fr; } }
    #sim-token-stream { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start; min-height: 80px; }
    #sim-highlighted-text { font-family: 'JetBrains Mono', monospace; font-size: 14px; line-height: 2; word-break: break-word; }
    .sim-token-chip {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 6px 10px; border-radius: 4px;
      font-family: 'JetBrains Mono', monospace; font-size: 13px;
      border: 1px solid; opacity: 0;
      transform: translateY(12px) scale(0.9);
      animation: simTokenAppear 0.4s forwards;
      cursor: pointer; transition: transform 0.15s;
      position: relative;
    }
    .sim-token-chip:hover { transform: translateY(-3px) scale(1.05) !important; z-index: 10; }
    .sim-tid { font-size: 9px; opacity: 0.6; position: absolute; top: -14px; left: 50%; transform: translateX(-50%); white-space: nowrap; pointer-events: none; color: #4a6080; font-family: 'JetBrains Mono', monospace; }
    @keyframes simTokenAppear { to { opacity: 1; transform: translateY(0) scale(1); } }
    .tc0 { background: rgba(0,229,255,0.08); border-color: rgba(0,229,255,0.35); color: #00e5ff; }
    .tc1 { background: rgba(255,107,53,0.08); border-color: rgba(255,107,53,0.35); color: #ff6b35; }
    .tc2 { background: rgba(124,58,237,0.08); border-color: rgba(124,58,237,0.35); color: #a78bfa; }
    .tc3 { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.35); color: #10b981; }
    .tc4 { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.35); color: #fbbf24; }
    .tc5 { background: rgba(236,72,153,0.08); border-color: rgba(236,72,153,0.35); color: #f472b6; }
    .sim-ht-span { padding: 2px 4px; border-radius: 3px; display: inline; cursor: pointer; transition: all 0.2s; }
    .sim-ht-span:hover { filter: brightness(1.4); }
    .sim-freq-chart, #sim-freq-chart { display: flex; gap: 8px; align-items: flex-end; min-height: 100px; flex-wrap: wrap; }
    .sim-freq-bar-wrap { display: flex; flex-direction: column; align-items: center; gap: 4px; }
    .sim-freq-bar { width: 28px; background: linear-gradient(to top, #00e5ff, rgba(0,229,255,0.3)); border-radius: 3px 3px 0 0; transition: height 0.5s cubic-bezier(0.34,1.56,0.64,1); min-height: 4px; }
    .sim-freq-label { font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #4a6080; text-align: center; max-width: 36px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .sim-freq-count { font-size: 9px; color: #00e5ff; font-family: 'JetBrains Mono', monospace; }
    #sim-subword-view { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
    .sim-sw-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
    .sim-sw-original { color: #4a6080; min-width: 140px; }
    .sim-sw-arrow { color: #1a2a3a; }
    .sim-sw-parts { display: flex; gap: 6px; flex-wrap: wrap; }
    .sim-sw-part { padding: 4px 10px; border-radius: 3px; font-size: 12px; border: 1px solid rgba(124,58,237,0.4); background: rgba(124,58,237,0.08); color: #a78bfa; }
    .sim-sw-part.prefix { border-color: rgba(0,229,255,0.4); background: rgba(0,229,255,0.08); color: #00e5ff; }
    .sim-sw-part.suffix { border-color: rgba(255,107,53,0.4); background: rgba(255,107,53,0.08); color: #ff6b35; }
    .sim-empty-state { display: flex; align-items: center; justify-content: center; height: 80px; color: #4a6080; font-family: 'JetBrains Mono', monospace; font-size: 13px; gap: 8px; }
    .sim-mode-info-text { font-size: 13px; color: #4a6080; line-height: 1.8; font-family: 'JetBrains Mono', monospace; }
    @keyframes simPulseGlow { 0%,100% { box-shadow: 0 0 0 0 rgba(0,229,255,0); } 50% { box-shadow: 0 0 0 6px rgba(0,229,255,0.10); } }
    .sim-pulse { animation: simPulseGlow 2s infinite; }
  `

    return (
        <>
            <style>{simStyles}</style>
            <Box
                ref={containerRef}
                className="sim-container"
                sx={{
                    // Light mode overrides: give the dark sim a slight tinted dark bg
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
