// Interface do Assistente de RH.
import { useEffect, useRef, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EXEMPLOS = [
  { titulo: 'Férias', pergunta: 'Quantos dias de férias eu tenho?', icone: 'sun' },
  { titulo: 'Home Office', pergunta: 'Posso fazer home office?', icone: 'home' },
  { titulo: 'Benefícios', pergunta: 'Quais são os benefícios da empresa?', icone: 'gift' },
  { titulo: 'Reembolso', pergunta: 'Qual o limite de reembolso de viagem?', icone: 'receipt' },
  { titulo: 'Horário', pergunta: 'Como funciona o horário flexível?', icone: 'clock' },
  { titulo: 'Licenças', pergunta: 'Como funciona a licença-paternidade?', icone: 'heart' },
]

export default function App() {
  const [mensagens, setMensagens] = useState([])
  const [pergunta, setPergunta] = useState('')
  const [loading, setLoading] = useState(false)
  const [solicitacoes, setSolicitacoes] = useState([])
  const [painelAberto, setPainelAberto] = useState(false)
  const [painelBaseAberto, setPainelBaseAberto] = useState(false)
  const [documentosBase, setDocumentosBase] = useState([])
  const [carregandoBase, setCarregandoBase] = useState(false)
  const [validarTeto, setValidarTeto] = useState(false)
  const [autoCorrigir, setAutoCorrigir] = useState(false)
  const listaRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    if (listaRef.current) {
      listaRef.current.scrollTop = listaRef.current.scrollHeight
    }
  }, [mensagens, loading])

  async function carregarSolicitacoes() {
    try {
      const r = await fetch(`${API_URL}/api/solicitacoes`)
      setSolicitacoes(await r.json())
    } catch {
      // silencioso: a listagem é auxiliar e não deve quebrar o chat.
    }
  }

  useEffect(() => {
    carregarSolicitacoes()
  }, [])

  // Carrega a base sob demanda (ao abrir o painel), espelhando o estado real do
  // backend a cada abertura/operação.
  async function carregarBase() {
    setCarregandoBase(true)
    try {
      const r = await fetch(`${API_URL}/api/base`)
      setDocumentosBase(await r.json())
    } catch {
      // silencioso: a listagem é auxiliar e não deve quebrar o chat.
    } finally {
      setCarregandoBase(false)
    }
  }

  function abrirPainelBase() {
    setPainelBaseAberto(true)
    carregarBase()
  }

  async function perguntar(texto) {
    const q = texto.trim()
    if (!q || loading) return
    setMensagens((m) => [...m, { autor: 'user', texto: q, ts: Date.now() }])
    setPergunta('')
    setLoading(true)
    try {
      const r = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pergunta: q,
          validar_teto: validarTeto,
          auto_corrigir: autoCorrigir,
        }),
      })
      const data = await r.json()
      setMensagens((m) => [
        ...m,
        {
          autor: 'assistente',
          texto: data.resposta,
          fontes: data.fontes || [],
          categoria: data.categoria,
          confianca: data.confianca,
          ts: Date.now(),
        },
      ])
    } catch (err) {
      setMensagens((m) => [
        ...m,
        {
          autor: 'assistente',
          texto: 'Não foi possível obter a resposta agora. Tente novamente em instantes.',
          fontes: [],
          erro: true,
          ts: Date.now(),
        },
      ])
    } finally {
      setLoading(false)
      // Atualiza a listagem: se a resposta registrou uma solicitação, ela
      // aparece sem ação manual. O botão "Atualizar" no painel é o reforço.
      carregarSolicitacoes()
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }

  function onSubmit(e) {
    e.preventDefault()
    perguntar(pergunta)
  }

  function novaConversa() {
    setMensagens([])
    setPergunta('')
    inputRef.current?.focus()
  }

  const vazio = mensagens.length === 0

  return (
    <div className="flex h-full flex-col bg-gradient-to-b from-slate-50 to-white text-slate-900">
      <Header
        onReset={novaConversa}
        hasMessages={!vazio}
        onAbrirPainel={() => setPainelAberto(true)}
        onAbrirPainelBase={abrirPainelBase}
        totalSolicitacoes={solicitacoes.length}
        validarTeto={validarTeto}
        onToggleValidarTeto={() => setValidarTeto((v) => !v)}
        autoCorrigir={autoCorrigir}
        onToggleAutoCorrigir={() => setAutoCorrigir((v) => !v)}
      />

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col overflow-hidden px-4 sm:px-6">
        <div
          ref={listaRef}
          className="scroll-soft flex-1 space-y-6 overflow-y-auto py-8"
        >
          {vazio ? (
            <EstadoVazio onPick={perguntar} />
          ) : (
            mensagens.map((m, i) => <Mensagem key={i} m={m} />)
          )}
          {loading && <LoadingBubble />}
        </div>

        <Composer
          inputRef={inputRef}
          value={pergunta}
          onChange={setPergunta}
          onSubmit={onSubmit}
          loading={loading}
        />
      </main>

      <PainelSolicitacoes
        aberto={painelAberto}
        onFechar={() => setPainelAberto(false)}
        solicitacoes={solicitacoes}
        onAtualizar={carregarSolicitacoes}
      />

      <PainelBase
        aberto={painelBaseAberto}
        onFechar={() => setPainelBaseAberto(false)}
        documentos={documentosBase}
        carregando={carregandoBase}
        onAtualizar={carregarBase}
      />
    </div>
  )
}

function Header({
  onReset,
  hasMessages,
  onAbrirPainel,
  onAbrirPainelBase,
  totalSolicitacoes,
  validarTeto,
  onToggleValidarTeto,
  autoCorrigir,
  onToggleAutoCorrigir,
}) {
  return (
    <header className="border-b border-slate-200/70 bg-white/70 backdrop-blur">
      <div className="mx-auto flex w-full max-w-3xl items-center justify-between px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <Logo />
          <div className="leading-tight">
            <div className="text-sm font-semibold text-slate-900">RH Assistant</div>
            <div className="text-[11px] text-slate-500">
              Tire dúvidas sobre as políticas internas
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onToggleValidarTeto}
            role="switch"
            aria-checked={validarTeto}
            className="inline-flex shrink-0 items-center gap-2 whitespace-nowrap rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
            title="Validar o saldo contra o teto da política"
          >
            <span
              className={`relative inline-block h-4 w-7 shrink-0 rounded-full transition-colors ${
                validarTeto ? 'bg-violet-500' : 'bg-slate-300'
              }`}
            >
              <span
                className={`absolute left-0.5 top-0.5 h-3 w-3 rounded-full bg-white shadow transition-transform ${
                  validarTeto ? 'translate-x-3' : 'translate-x-0'
                }`}
              />
            </span>
            <span>Validar política</span>
          </button>
          <button
            type="button"
            onClick={onToggleAutoCorrigir}
            role="switch"
            aria-checked={autoCorrigir}
            className="inline-flex shrink-0 items-center gap-2 whitespace-nowrap rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
            title="Se a busca falhar, o sistema reescreve a pergunta e tenta novamente"
          >
            <span
              className={`relative inline-block h-4 w-7 shrink-0 rounded-full transition-colors ${
                autoCorrigir ? 'bg-violet-500' : 'bg-slate-300'
              }`}
            >
              <span
                className={`absolute left-0.5 top-0.5 h-3 w-3 rounded-full bg-white shadow transition-transform ${
                  autoCorrigir ? 'translate-x-3' : 'translate-x-0'
                }`}
              />
            </span>
            <span>Auto-correção</span>
          </button>
          <button
            type="button"
            onClick={onAbrirPainelBase}
            className="inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
            title="Base de conhecimento"
          >
            <Icone nome="book" tiny />
            Base
          </button>
          <button
            type="button"
            onClick={onAbrirPainel}
            className="inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
            title="Solicitações registradas"
          >
            <Icone nome="list" tiny />
            Solicitações
            {totalSolicitacoes > 0 && (
              <span className="ml-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-violet-100 px-1 text-[10px] font-semibold text-violet-700">
                {totalSolicitacoes}
              </span>
            )}
          </button>
          {hasMessages && (
            <button
              type="button"
              onClick={onReset}
              className="shrink-0 whitespace-nowrap rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
              title="Nova conversa"
            >
              Nova conversa
            </button>
          )}
        </div>
      </div>
    </header>
  )
}

function Logo() {
  return (
    <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 text-[11px] font-bold tracking-tight text-white shadow-sm shadow-violet-500/30">
      RH
    </div>
  )
}

function EstadoVazio({ onPick }) {
  return (
    <div className="flex flex-col items-center pt-6 text-center animate-fade-up">
      <Logo />
      <h1 className="mt-4 text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
        Olá, sou seu Assistente de RH
      </h1>
      <p className="mt-2 max-w-md text-sm text-slate-500">
        Pergunte sobre férias, benefícios, home office, reembolso, horário ou
        licenças. Toda resposta vem com a política de origem citada.
      </p>

      <div className="mt-8 grid w-full max-w-2xl grid-cols-1 gap-2.5 sm:grid-cols-2">
        {EXEMPLOS.map((ex) => (
          <button
            key={ex.pergunta}
            type="button"
            onClick={() => onPick(ex.pergunta)}
            className="group flex items-start gap-3 rounded-xl border border-slate-200 bg-white p-3.5 text-left transition hover:-translate-y-px hover:border-violet-300 hover:shadow-sm hover:shadow-violet-100"
          >
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-50 text-violet-600 group-hover:bg-violet-100">
              <Icone nome={ex.icone} />
            </div>
            <div className="min-w-0">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                {ex.titulo}
              </div>
              <div className="mt-0.5 text-sm text-slate-800">{ex.pergunta}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

function Mensagem({ m }) {
  if (m.autor === 'user') {
    return (
      <div className="flex justify-end animate-fade-up">
        <div className="flex max-w-[80%] items-end gap-2">
          <div className="rounded-2xl rounded-br-md bg-gradient-to-br from-violet-600 to-indigo-600 px-4 py-2.5 text-sm leading-relaxed text-white shadow-sm shadow-indigo-500/20">
            {m.texto}
          </div>
          <Avatar autor="user" />
        </div>
      </div>
    )
  }
  return (
    <div className="flex animate-fade-up items-start gap-2.5">
      <Avatar autor="bot" />
      <div className="min-w-0 max-w-[85%] flex-1">
        <div
          className={`rounded-2xl rounded-tl-md border px-4 py-3 text-sm leading-relaxed shadow-sm ${
            m.erro
              ? 'border-red-200 bg-red-50 text-red-800'
              : 'border-slate-200 bg-white text-slate-800'
          }`}
        >
          <RenderTexto texto={m.texto} />
          {m.fontes && m.fontes.length > 0 && <Fontes fontes={m.fontes} />}
          {!m.erro && (m.categoria || typeof m.confianca === 'number') && (
            <MetaResposta categoria={m.categoria} confianca={m.confianca} />
          )}
        </div>
      </div>
    </div>
  )
}

function Avatar({ autor }) {
  if (autor === 'user') {
    return (
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-200 text-[10px] font-semibold text-slate-600">
        Você
      </div>
    )
  }
  return (
    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 text-[10px] font-bold text-white shadow-sm shadow-violet-500/30">
      RH
    </div>
  )
}

function Fontes({ fontes }) {
  return (
    <div className="mt-3 border-t border-slate-100 pt-3">
      <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-slate-400">
        <Icone nome="doc" tiny />
        Fontes
      </div>
      <div className="flex flex-wrap gap-1.5">
        {fontes.map((f) => (
          <span
            key={f.arquivo}
            className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-slate-50 px-2 py-1 text-[11px] text-slate-700"
            title={f.arquivo}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-violet-500" />
            <span className="font-medium">{f.titulo || f.arquivo}</span>
            <span className="font-mono text-[10px] text-slate-400">
              {f.arquivo}
            </span>
          </span>
        ))}
      </div>
    </div>
  )
}

// "home-office" -> "Home office"
function formatarCategoria(c) {
  const texto = String(c).replace(/-/g, ' ').trim()
  return texto.charAt(0).toUpperCase() + texto.slice(1)
}

function MetaResposta({ categoria, confianca }) {
  const temConfianca = typeof confianca === 'number' && !Number.isNaN(confianca)
  const pct = temConfianca ? Math.round(Math.max(0, Math.min(1, confianca)) * 100) : null
  // Cor da barra por faixa de confiança (neutra, sem alarde).
  const cor =
    pct === null
      ? 'bg-slate-300'
      : pct >= 70
        ? 'bg-emerald-500'
        : pct >= 40
          ? 'bg-amber-500'
          : 'bg-rose-400'

  return (
    <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-slate-100 pt-3">
      {categoria && (
        <span className="inline-flex items-center rounded-md bg-violet-50 px-2 py-0.5 text-[11px] font-medium text-violet-700">
          {formatarCategoria(categoria)}
        </span>
      )}
      {temConfianca && (
        <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
          <span className="uppercase tracking-wide">Confiança</span>
          <span className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-100">
            <span
              className={`block h-full rounded-full ${cor}`}
              style={{ width: `${pct}%` }}
            />
          </span>
          <span className="font-medium text-slate-500">{pct}%</span>
        </div>
      )}
    </div>
  )
}

function formatarData(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('pt-BR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function PainelSolicitacoes({ aberto, onFechar, solicitacoes, onAtualizar }) {
  return (
    <>
      {/* Overlay */}
      <div
        onClick={onFechar}
        className={`fixed inset-0 z-20 bg-slate-900/20 backdrop-blur-sm transition-opacity ${
          aberto ? 'opacity-100' : 'pointer-events-none opacity-0'
        }`}
      />
      {/* Drawer */}
      <aside
        className={`fixed inset-y-0 right-0 z-30 flex w-full max-w-sm flex-col border-l border-slate-200 bg-white shadow-xl transition-transform duration-300 ${
          aberto ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
          <div className="leading-tight">
            <div className="text-sm font-semibold text-slate-900">
              Solicitações de férias
            </div>
            <div className="text-[11px] text-slate-500">
              {solicitacoes.length}{' '}
              {solicitacoes.length === 1 ? 'registro' : 'registros'}
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={onAtualizar}
              className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
              title="Atualizar lista"
            >
              <Icone nome="refresh" tiny />
              Atualizar
            </button>
            <button
              type="button"
              onClick={onFechar}
              className="flex h-7 w-7 items-center justify-center rounded-full text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
              title="Fechar"
              aria-label="Fechar"
            >
              <Icone nome="close" tiny />
            </button>
          </div>
        </div>

        <div className="scroll-soft flex-1 space-y-2 overflow-y-auto p-4">
          {solicitacoes.length === 0 ? (
            <div className="mt-10 text-center text-sm text-slate-400">
              Nenhuma solicitação registrada ainda.
            </div>
          ) : (
            solicitacoes.map((s, i) => (
              <div
                key={i}
                className="rounded-xl border border-slate-200 bg-slate-50/60 p-3"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-semibold text-slate-800">
                    {s.funcionario}
                  </span>
                  <span className="inline-flex items-center rounded-md bg-violet-50 px-2 py-0.5 text-[11px] font-medium text-violet-700">
                    {s.dias} {s.dias === 1 ? 'dia' : 'dias'}
                  </span>
                </div>
                <div className="mt-1 text-xs text-slate-600">
                  Período: <span className="font-medium">{s.periodo}</span>
                </div>
                <div className="mt-1 text-[11px] text-slate-400">
                  {formatarData(s.timestamp)}
                </div>
              </div>
            ))
          )}
        </div>
      </aside>
    </>
  )
}

function PainelBase({ aberto, onFechar, documentos, carregando, onAtualizar }) {
  const [arquivo, setArquivo] = useState(null)
  const [enviando, setEnviando] = useState(false)
  const [resultado, setResultado] = useState(null)
  const [resultadoOn, setResultadoOn] = useState(false)
  const [erro, setErro] = useState(null)
  const [paraRemover, setParaRemover] = useState(null) // doc aguardando confirmação
  const [removendo, setRemovendo] = useState(null) // arquivo em animação de saída
  const fileRef = useRef(null)

  // Faixa de sucesso: entra suave e some sozinha após alguns segundos. O erro
  // não usa isto — permanece até o usuário agir (trocar arquivo, reenviar).
  useEffect(() => {
    if (!resultado) return
    const raf = requestAnimationFrame(() => setResultadoOn(true))
    const tFade = setTimeout(() => setResultadoOn(false), 3000)
    const tLimpa = setTimeout(() => setResultado(null), 3300)
    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(tFade)
      clearTimeout(tLimpa)
    }
  }, [resultado])

  async function enviar() {
    if (!arquivo || enviando) return
    setEnviando(true)
    setErro(null)
    setResultado(null)
    setResultadoOn(false)
    try {
      const fd = new FormData()
      fd.append('file', arquivo)
      const r = await fetch(`${API_URL}/api/base/upload`, { method: 'POST', body: fd })
      const data = await r.json()
      if (!r.ok) {
        // O backend devolve 400 com {detail} para formato inválido; mostramos
        // a mensagem dele de forma amigável, sem quebrar a UI.
        setErro(data.detail || 'Não foi possível indexar o documento.')
      } else {
        const n = data.chunks
        setResultado(`"${data.titulo}" indexado: ${n} ${n === 1 ? 'chunk' : 'chunks'}.`)
        setArquivo(null)
        if (fileRef.current) fileRef.current.value = ''
        onAtualizar()
      }
    } catch {
      setErro('Falha de conexão ao enviar o documento.')
    } finally {
      setEnviando(false)
    }
  }

  // Confirmada no modal. Animamos a saída do item (curta) antes de efetivar o
  // DELETE e recarregar a lista, para a remoção não "piscar".
  async function confirmarRemocao() {
    const arq = paraRemover.arquivo
    setParaRemover(null)
    setErro(null)
    setRemovendo(arq)
    await new Promise((res) => setTimeout(res, 200)) // duração da animação de saída
    try {
      await fetch(`${API_URL}/api/base/${encodeURIComponent(arq)}`, { method: 'DELETE' })
      await onAtualizar()
    } catch {
      setErro('Falha ao remover o documento.')
    } finally {
      setRemovendo(null)
    }
  }

  return (
    <>
      {/* Overlay */}
      <div
        onClick={onFechar}
        className={`fixed inset-0 z-20 bg-slate-900/20 backdrop-blur-sm transition-opacity ${
          aberto ? 'opacity-100' : 'pointer-events-none opacity-0'
        }`}
      />
      {/* Drawer */}
      <aside
        className={`fixed inset-y-0 right-0 z-30 flex w-full max-w-sm flex-col border-l border-slate-200 bg-white shadow-xl transition-transform duration-300 ${
          aberto ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
          <div className="leading-tight">
            <div className="text-sm font-semibold text-slate-900">
              Base de conhecimento
            </div>
            <div className="text-[11px] text-slate-500">
              {documentos.length}{' '}
              {documentos.length === 1 ? 'documento' : 'documentos'}
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={onAtualizar}
              className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
              title="Atualizar lista"
            >
              <Icone nome="refresh" tiny />
              Atualizar
            </button>
            <button
              type="button"
              onClick={onFechar}
              className="flex h-7 w-7 items-center justify-center rounded-full text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
              title="Fechar"
              aria-label="Fechar"
            >
              <Icone nome="close" tiny />
            </button>
          </div>
        </div>

        {/* Upload de novo documento */}
        <div className="border-b border-slate-200 bg-slate-50/60 p-4">
          <input
            ref={fileRef}
            type="file"
            accept=".md"
            onChange={(e) => {
              setArquivo(e.target.files?.[0] || null)
              setErro(null)
              setResultado(null)
              setResultadoOn(false)
            }}
            className="block w-full text-xs text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-violet-50 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-violet-700 hover:file:bg-violet-100"
          />
          <button
            type="button"
            onClick={enviar}
            disabled={!arquivo || enviando}
            className="mt-2.5 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-gradient-to-br from-violet-600 to-indigo-600 px-3 py-2 text-xs font-medium text-white shadow-sm shadow-violet-500/30 transition hover:brightness-110 disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-300 disabled:shadow-none"
          >
            {enviando ? (
              <>
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                Indexando documento...
              </>
            ) : (
              <>
                <Icone nome="upload" tiny />
                Adicionar documento
              </>
            )}
          </button>
          {resultado && (
            <div
              className={`mt-2.5 flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-[11px] text-emerald-700 transition-all duration-300 ${
                resultadoOn ? 'translate-y-0 opacity-100' : '-translate-y-1 opacity-0'
              }`}
            >
              <span className="text-emerald-600">
                <Icone nome="check" tiny />
              </span>
              <span>{resultado}</span>
            </div>
          )}
          {erro && (
            <div className="animate-fade-up mt-2.5 flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-[11px] text-rose-700">
              <span className="text-rose-500">
                <Icone nome="alert" tiny />
              </span>
              <span>{erro}</span>
            </div>
          )}
        </div>

        <div className="scroll-soft flex-1 space-y-2 overflow-y-auto p-4">
          {carregando ? (
            <BaseSkeleton />
          ) : documentos.length === 0 ? (
            <BaseVazia />
          ) : (
            documentos.map((d) => (
              <div
                key={d.arquivo}
                className={`transition-all duration-200 ease-out ${
                  removendo === d.arquivo
                    ? 'translate-x-3 opacity-0'
                    : 'translate-x-0 opacity-100'
                }`}
              >
                <div className="animate-fade-up flex items-start justify-between gap-2 rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="shrink-0 text-violet-500">
                        <Icone nome="doc" tiny />
                      </span>
                      <span className="truncate text-sm font-semibold text-slate-800">
                        {d.titulo || d.arquivo}
                      </span>
                    </div>
                    <div className="mt-1 truncate font-mono text-[10px] text-slate-400">
                      {d.arquivo}
                    </div>
                    <span className="mt-2 inline-flex items-center gap-1 rounded-md bg-violet-50 px-2 py-0.5 text-[11px] font-medium text-violet-700">
                      <span className="h-1 w-1 rounded-full bg-violet-500" />
                      {d.chunks} {d.chunks === 1 ? 'chunk' : 'chunks'}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      setParaRemover({ arquivo: d.arquivo, titulo: d.titulo || d.arquivo })
                    }
                    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-slate-400 transition hover:bg-rose-50 hover:text-rose-600"
                    title="Remover documento"
                    aria-label={`Remover ${d.arquivo}`}
                  >
                    <Icone nome="trash" tiny />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </aside>

      <ModalConfirmacao
        item={paraRemover}
        onCancelar={() => setParaRemover(null)}
        onConfirmar={confirmarRemocao}
      />
    </>
  )
}

function BaseSkeleton() {
  // Placeholder sóbrio enquanto a lista carrega, no formato dos itens reais.
  return (
    <div className="space-y-2">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="rounded-xl border border-slate-200 bg-white p-3">
          <div className="h-3.5 w-2/3 animate-pulse rounded bg-slate-200" />
          <div className="mt-2 h-2.5 w-2/5 animate-pulse rounded bg-slate-100" />
          <div className="mt-2.5 h-4 w-16 animate-pulse rounded-md bg-slate-100" />
        </div>
      ))}
    </div>
  )
}

function BaseVazia() {
  return (
    <div className="animate-fade-up flex flex-col items-center pt-16 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-violet-50 text-violet-500">
        <Icone nome="book" />
      </div>
      <div className="mt-3 text-sm font-medium text-slate-700">
        Nenhum documento na base
      </div>
      <p className="mt-1 max-w-[15rem] text-xs text-slate-400">
        Adicione um arquivo .md acima para indexá-lo e usá-lo nas respostas.
      </p>
    </div>
  )
}

function ModalConfirmacao({ item, onCancelar, onConfirmar }) {
  const cancelarRef = useRef(null)

  // Foco inicial no "Cancelar" (opção segura para ação destrutiva) e fecha no Esc.
  useEffect(() => {
    if (!item) return
    cancelarRef.current?.focus()
    function onKey(e) {
      if (e.key === 'Escape') onCancelar()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [item, onCancelar])

  if (!item) return null
  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
    >
      <div
        onClick={onCancelar}
        className="animate-fade-in absolute inset-0 bg-slate-900/40 backdrop-blur-sm"
      />
      <div className="animate-pop-in relative w-full max-w-sm rounded-2xl border border-slate-200 bg-white p-5 shadow-xl">
        <div className="flex items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-rose-50 text-rose-600">
            <Icone nome="trash" />
          </div>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-slate-900">Remover documento</h2>
            <p className="mt-1 text-sm leading-relaxed text-slate-600">
              Remover{' '}
              <span className="font-semibold text-slate-800">{item.titulo}</span> da
              base? Os trechos indexados deixam de ser usados nas respostas.
            </p>
          </div>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <button
            ref={cancelarRef}
            type="button"
            onClick={onCancelar}
            className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
          >
            Cancelar
          </button>
          <button
            type="button"
            onClick={onConfirmar}
            className="inline-flex items-center gap-1.5 rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-medium text-white shadow-sm shadow-rose-500/30 transition hover:bg-rose-700"
          >
            <Icone nome="trash" tiny />
            Remover
          </button>
        </div>
      </div>
    </div>
  )
}

function LoadingBubble() {
  return (
    <div className="flex items-start gap-2.5">
      <Avatar autor="bot" />
      <div className="flex items-center gap-1 rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 shadow-sm">
        <span className="dot h-1.5 w-1.5 rounded-full bg-slate-400" />
        <span className="dot h-1.5 w-1.5 rounded-full bg-slate-400" />
        <span className="dot h-1.5 w-1.5 rounded-full bg-slate-400" />
      </div>
    </div>
  )
}

function Composer({ inputRef, value, onChange, onSubmit, loading }) {
  const podeEnviar = value.trim().length > 0 && !loading
  return (
    <form
      onSubmit={onSubmit}
      className="sticky bottom-0 mt-2 border-t border-slate-200/70 bg-gradient-to-t from-white via-white to-white/60 pb-6 pt-4 backdrop-blur"
    >
      <div className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2 shadow-sm focus-within:border-violet-400 focus-within:ring-2 focus-within:ring-violet-100">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Pergunte algo sobre as políticas de RH..."
          disabled={loading}
          autoFocus
          className="flex-1 bg-transparent px-1 py-1.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!podeEnviar}
          className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-600 to-indigo-600 text-white shadow-sm shadow-violet-500/30 transition hover:brightness-110 disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-300 disabled:shadow-none"
          aria-label="Enviar"
          title="Enviar"
        >
          <Icone nome="send" />
        </button>
      </div>
      <p className="mt-2 text-center text-[11px] text-slate-400">
        As respostas são informativas. Em caso de dúvida, confirme com o RH.
      </p>
    </form>
  )
}

// Render bem simples de markdown: **bold**, listas com "-", parágrafos.
function RenderTexto({ texto }) {
  const blocos = texto.split(/\n{2,}/)
  return (
    <div className="space-y-2">
      {blocos.map((bloco, i) => {
        const linhas = bloco.split('\n')
        const ehLista = linhas.every((l) => l.trim().startsWith('-'))
        if (ehLista && linhas.length > 1) {
          return (
            <ul key={i} className="ml-4 list-disc space-y-1">
              {linhas.map((l, j) => (
                <li key={j}>
                  <Inline texto={l.replace(/^\s*-\s*/, '')} />
                </li>
              ))}
            </ul>
          )
        }
        return (
          <p key={i} className="whitespace-pre-line">
            <Inline texto={bloco} />
          </p>
        )
      })}
    </div>
  )
}

function Inline({ texto }) {
  // alterna trechos normais e **negritos**
  const partes = texto.split(/(\*\*[^*]+\*\*)/g)
  return (
    <>
      {partes.map((p, i) => {
        if (p.startsWith('**') && p.endsWith('**')) {
          return (
            <strong key={i} className="font-semibold text-slate-900">
              {p.slice(2, -2)}
            </strong>
          )
        }
        return <span key={i}>{p}</span>
      })}
    </>
  )
}

function Icone({ nome, tiny = false }) {
  const size = tiny ? 12 : 16
  const props = {
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 2,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
  }
  switch (nome) {
    case 'send':
      return (
        <svg {...props}>
          <path d="M5 12l14-7-4 14-3-6-7-1z" />
        </svg>
      )
    case 'sun':
      return (
        <svg {...props}>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      )
    case 'home':
      return (
        <svg {...props}>
          <path d="M3 11l9-8 9 8" />
          <path d="M5 10v10h14V10" />
        </svg>
      )
    case 'gift':
      return (
        <svg {...props}>
          <rect x="3" y="8" width="18" height="13" rx="1" />
          <path d="M3 12h18M12 8v13M7 8a2 2 0 110-4c2 0 5 4 5 4M17 8a2 2 0 100-4c-2 0-5 4-5 4" />
        </svg>
      )
    case 'receipt':
      return (
        <svg {...props}>
          <path d="M5 3h14v18l-3-2-2 2-2-2-2 2-2-2-3 2V3z" />
          <path d="M9 8h6M9 12h6M9 16h4" />
        </svg>
      )
    case 'clock':
      return (
        <svg {...props}>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 7v5l3 2" />
        </svg>
      )
    case 'heart':
      return (
        <svg {...props}>
          <path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z" />
        </svg>
      )
    case 'doc':
      return (
        <svg {...props}>
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
          <path d="M14 2v6h6" />
        </svg>
      )
    case 'list':
      return (
        <svg {...props}>
          <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
        </svg>
      )
    case 'refresh':
      return (
        <svg {...props}>
          <path d="M21 12a9 9 0 11-2.64-6.36M21 3v6h-6" />
        </svg>
      )
    case 'book':
      return (
        <svg {...props}>
          <path d="M4 19.5A2.5 2.5 0 016.5 17H20" />
          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" />
        </svg>
      )
    case 'upload':
      return (
        <svg {...props}>
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
          <path d="M17 8l-5-5-5 5M12 3v12" />
        </svg>
      )
    case 'trash':
      return (
        <svg {...props}>
          <path d="M3 6h18M8 6V4a1 1 0 011-1h6a1 1 0 011 1v2m2 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6" />
          <path d="M10 11v6M14 11v6" />
        </svg>
      )
    case 'check':
      return (
        <svg {...props}>
          <path d="M20 6L9 17l-5-5" />
        </svg>
      )
    case 'alert':
      return (
        <svg {...props}>
          <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          <path d="M12 9v4M12 17h.01" />
        </svg>
      )
    case 'close':
      return (
        <svg {...props}>
          <path d="M18 6L6 18M6 6l12 12" />
        </svg>
      )
    default:
      return null
  }
}
