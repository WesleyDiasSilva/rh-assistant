// BRANCH DEMONSTRACAO — interface completa.
// O backend desta branch responde com matching por palavra-chave sobre
// documentos fake em backend/fake_data/. Não há LLM nem chamada externa.
import { useEffect, useRef, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EXEMPLOS = [
  'Quantos dias de férias eu tenho?',
  'Posso fazer home office?',
  'Quais são os benefícios?',
  'Qual o limite de reembolso de viagem?',
  'Como funciona a licença-paternidade?',
]

export default function App() {
  const [mensagens, setMensagens] = useState([])
  const [pergunta, setPergunta] = useState('')
  const [loading, setLoading] = useState(false)
  const listaRef = useRef(null)

  useEffect(() => {
    if (listaRef.current) {
      listaRef.current.scrollTop = listaRef.current.scrollHeight
    }
  }, [mensagens, loading])

  async function perguntar(texto) {
    const q = texto.trim()
    if (!q || loading) return
    setMensagens((m) => [...m, { autor: 'user', texto: q }])
    setPergunta('')
    setLoading(true)
    try {
      const r = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pergunta: q }),
      })
      const data = await r.json()
      setMensagens((m) => [
        ...m,
        { autor: 'assistente', texto: data.resposta, fontes: data.fontes || [] },
      ])
    } catch (err) {
      setMensagens((m) => [
        ...m,
        {
          autor: 'assistente',
          texto: `Falha ao consultar o backend: ${err.message || err}`,
          fontes: [],
          erro: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function onSubmit(e) {
    e.preventDefault()
    perguntar(pergunta)
  }

  const vazio = mensagens.length === 0

  return (
    <div className="min-h-full bg-slate-50 text-slate-900">
      <div className="mx-auto flex h-screen max-w-3xl flex-col px-4 py-6">
        <header className="mb-4 flex items-center justify-between border-b border-slate-200 pb-4">
          <div>
            <h1 className="text-xl font-semibold">Assistente de RH</h1>
            <p className="text-xs text-slate-500">
              Demonstração offline · dados simulados de políticas internas
            </p>
          </div>
          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
            Demo
          </span>
        </header>

        <div ref={listaRef} className="flex-1 space-y-4 overflow-y-auto pr-1">
          {vazio && (
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <p className="mb-3 text-sm text-slate-600">
                Olá! Sou seu assistente de RH. Posso ajudar com dúvidas sobre férias,
                home office, benefícios, reembolso, horário e licenças. Comece com
                uma destas perguntas:
              </p>
              <div className="flex flex-wrap gap-2">
                {EXEMPLOS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => perguntar(q)}
                    className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-700 hover:border-slate-400 hover:bg-slate-100"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {mensagens.map((m, i) => (
            <Mensagem key={i} m={m} />
          ))}

          {loading && (
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-slate-400" />
              Pensando...
            </div>
          )}
        </div>

        <form onSubmit={onSubmit} className="mt-4 flex gap-2">
          <input
            type="text"
            value={pergunta}
            onChange={(e) => setPergunta(e.target.value)}
            placeholder="Faça uma pergunta sobre as políticas de RH..."
            disabled={loading}
            className="flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm focus:border-slate-500 focus:outline-none disabled:bg-slate-100"
          />
          <button
            type="submit"
            disabled={loading || !pergunta.trim()}
            className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            Enviar
          </button>
        </form>
      </div>
    </div>
  )
}

function Mensagem({ m }) {
  if (m.autor === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-slate-900 px-4 py-2 text-sm text-white">
          {m.texto}
        </div>
      </div>
    )
  }
  return (
    <div className="flex justify-start">
      <div
        className={`max-w-[85%] rounded-2xl rounded-bl-sm border px-4 py-3 text-sm whitespace-pre-line ${
          m.erro
            ? 'border-red-200 bg-red-50 text-red-800'
            : 'border-slate-200 bg-white text-slate-800'
        }`}
      >
        <div>{m.texto}</div>
        {m.fontes && m.fontes.length > 0 && (
          <div className="mt-3 border-t border-slate-100 pt-2 text-xs text-slate-500">
            <div className="mb-1 font-medium text-slate-600">Fontes</div>
            <ul className="space-y-0.5">
              {m.fontes.map((f) => (
                <li key={f.arquivo}>
                  <span className="font-mono text-slate-700">{f.arquivo}</span>
                  {f.titulo && <span className="text-slate-500"> · {f.titulo}</span>}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
