import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Ensemble } from '../api/client'

const ensembles = signal<Ensemble[]>([])
const loading = signal(true)
const error = signal<string | null>(null)
const selectedEnsemble = signal<Ensemble | null>(null)
const executeInput = signal('')
const executeResult = signal<string | null>(null)
const executing = signal(false)

const styles = {
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1.5rem',
  },
  title: {
    fontSize: '1.5rem',
    fontWeight: 'bold',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
    gap: '1rem',
  },
  card: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: '8px',
    padding: '1rem',
    cursor: 'pointer',
    transition: 'border-color 0.2s',
  },
  cardName: {
    fontSize: '1.1rem',
    fontWeight: '600',
    color: '#58a6ff',
    marginBottom: '0.5rem',
  },
  cardDesc: {
    color: '#8b949e',
    fontSize: '0.9rem',
  },
  cardSource: {
    marginTop: '0.5rem',
    fontSize: '0.75rem',
    color: '#6e7681',
  },
  detail: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: '8px',
    padding: '1.5rem',
    marginTop: '1rem',
  },
  input: {
    width: '100%',
    padding: '0.75rem',
    background: '#0d1117',
    border: '1px solid #30363d',
    borderRadius: '6px',
    color: '#c9d1d9',
    marginBottom: '0.5rem',
  },
  button: {
    padding: '0.5rem 1rem',
    background: '#238636',
    border: 'none',
    borderRadius: '6px',
    color: 'white',
    cursor: 'pointer',
    fontWeight: '500',
  },
  buttonDisabled: {
    background: '#21262d',
    cursor: 'not-allowed',
  },
  result: {
    marginTop: '1rem',
    padding: '1rem',
    background: '#0d1117',
    borderRadius: '6px',
    fontFamily: 'monospace',
    fontSize: '0.85rem',
    whiteSpace: 'pre-wrap' as const,
    overflow: 'auto',
    maxHeight: '300px',
  },
}

async function loadEnsembles() {
  loading.value = true
  error.value = null
  try {
    ensembles.value = await api.ensembles.list()
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load ensembles'
  } finally {
    loading.value = false
  }
}

async function executeEnsemble() {
  if (!selectedEnsemble.value || !executeInput.value.trim()) return
  executing.value = true
  executeResult.value = null
  try {
    const result = await api.ensembles.execute(
      selectedEnsemble.value.name,
      executeInput.value
    )
    executeResult.value = JSON.stringify(result, null, 2)
  } catch (e) {
    executeResult.value = `Error: ${e instanceof Error ? e.message : 'Unknown error'}`
  } finally {
    executing.value = false
  }
}

function EnsembleCard({ ensemble }: { ensemble: Ensemble }) {
  return (
    <div
      style={styles.card}
      onClick={() => (selectedEnsemble.value = ensemble)}
    >
      <div style={styles.cardName}>{ensemble.name}</div>
      <div style={styles.cardDesc}>{ensemble.description}</div>
      <div style={styles.cardSource}>{ensemble.source}</div>
    </div>
  )
}

function EnsembleDetail() {
  const ens = selectedEnsemble.value
  if (!ens) return null

  return (
    <div style={styles.detail}>
      <h3 style={{ marginBottom: '0.5rem' }}>{ens.name}</h3>
      <p style={{ color: '#8b949e', marginBottom: '1rem' }}>{ens.description}</p>

      <h4 style={{ marginBottom: '0.5rem' }}>Execute</h4>
      <textarea
        style={styles.input}
        placeholder="Enter input for the ensemble..."
        value={executeInput.value}
        onInput={(e) => (executeInput.value = (e.target as HTMLTextAreaElement).value)}
        rows={3}
      />
      <button
        style={{
          ...styles.button,
          ...(executing.value ? styles.buttonDisabled : {}),
        }}
        onClick={executeEnsemble}
        disabled={executing.value}
      >
        {executing.value ? 'Executing...' : 'Execute'}
      </button>

      {executeResult.value && (
        <div style={styles.result}>{executeResult.value}</div>
      )}
    </div>
  )
}

export function EnsemblesPage() {
  useEffect(() => {
    loadEnsembles()
  }, [])

  if (loading.value) {
    return <div>Loading ensembles...</div>
  }

  if (error.value) {
    return <div style={{ color: '#f85149' }}>Error: {error.value}</div>
  }

  return (
    <div>
      <div style={styles.header}>
        <h2 style={styles.title}>Ensembles</h2>
        <span style={{ color: '#8b949e' }}>{ensembles.value.length} available</span>
      </div>

      <div style={styles.grid}>
        {ensembles.value.map((e) => (
          <EnsembleCard key={e.name} ensemble={e} />
        ))}
      </div>

      <EnsembleDetail />
    </div>
  )
}
