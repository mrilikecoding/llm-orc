import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Artifact } from '../api/client'

const artifacts = signal<Artifact[]>([])
const loading = signal(true)
const error = signal<string | null>(null)

const styles = {
  title: {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    marginBottom: '1.5rem',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
    gap: '1rem',
  },
  card: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: '8px',
    padding: '1rem',
  },
  name: {
    fontSize: '1.1rem',
    fontWeight: '600',
    color: '#58a6ff',
    marginBottom: '0.5rem',
  },
  stat: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '0.25rem',
    color: '#8b949e',
    fontSize: '0.9rem',
  },
  empty: {
    textAlign: 'center' as const,
    padding: '3rem',
    color: '#8b949e',
  },
}

async function loadArtifacts() {
  loading.value = true
  error.value = null
  try {
    artifacts.value = await api.artifacts.list()
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load artifacts'
  } finally {
    loading.value = false
  }
}

function ArtifactCard({ artifact }: { artifact: Artifact }) {
  return (
    <div style={styles.card}>
      <div style={styles.name}>{artifact.name}</div>
      <div style={styles.stat}>
        <span>Executions:</span>
        <span>{artifact.executions_count}</span>
      </div>
      <div style={styles.stat}>
        <span>Latest:</span>
        <span>{artifact.latest_execution}</span>
      </div>
    </div>
  )
}

export function ArtifactsPage() {
  useEffect(() => {
    loadArtifacts()
  }, [])

  if (loading.value) {
    return <div>Loading artifacts...</div>
  }

  if (error.value) {
    return <div style={{ color: '#f85149' }}>Error: {error.value}</div>
  }

  if (artifacts.value.length === 0) {
    return (
      <div>
        <h2 style={styles.title}>Execution Artifacts</h2>
        <div style={styles.empty}>
          <p>No execution artifacts yet.</p>
          <p style={{ marginTop: '0.5rem' }}>
            Run an ensemble to create artifacts.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div>
      <h2 style={styles.title}>Execution Artifacts</h2>
      <div style={styles.grid}>
        {artifacts.value.map((a) => (
          <ArtifactCard key={a.name} artifact={a} />
        ))}
      </div>
    </div>
  )
}
