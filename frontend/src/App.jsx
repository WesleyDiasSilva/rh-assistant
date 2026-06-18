// Aula 1 — ponto de partida.
// A interface existe mas o backend ainda devolve apenas um mock fixo.
// Ao longo do curso, ligaremos isto a uma cadeia LangChain real.
import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [pergunta, setPergunta] = useState('')
  const [resposta, setResposta] = useState(null)
  const [loading, setLoading] = useState(false)
  const [erro, setErro] = useState(null)

  async function enviar(e) {
    e.preventDefault()
    if (!pergunta.trim()) return
    setLoading(true)
    setErro(null)
    setResposta(null)
    try {
      const r = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pergunta }),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const data = await r.json()
      setResposta(data)
    } catch (err) {
      setErro(err.message || 'Falha ao chamar o backend')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-full bg-slate-50 text-slate-900">
      <div className="mx-auto max-w-2xl px-4 py-10">
        <header className="mb-8">
          <h1 className="text-2xl font-semibold">Assistente de RH</h1>
          <p className="text-sm text-slate-500">
            Aula 1 — ponto de partida. O backend ainda responde com um mock fixo.
          </p>
        </header>

        <div className="mb-6 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          Ainda não implementado. Esta interface será conectada a um pipeline LangChain ao longo do curso.
        </div>

        <form onSubmit={enviar} className="flex gap-2">
          <input
            type="text"
            value={pergunta}
            onChange={(e) => setPergunta(e.target.value)}
            placeholder="Faça uma pergunta..."
            className="flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={loading}
            className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {loading ? 'Enviando...' : 'Enviar'}
          </button>
        </form>

        {erro && (
          <div className="mt-6 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {erro}
          </div>
        )}

        {resposta && (
          <div className="mt-6 rounded-md border border-slate-200 bg-white px-4 py-3 text-sm">
            <div className="text-slate-800">{resposta.resposta}</div>
            {resposta.fontes && resposta.fontes.length > 0 && (
              <div className="mt-3 text-xs text-slate-500">
                Fontes: {resposta.fontes.map((f) => f.arquivo).join(', ')}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
