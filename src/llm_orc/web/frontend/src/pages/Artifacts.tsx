import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Artifact, ArtifactDetail as ArtifactDetailType } from '../api/client'
import { SlidePanel } from '../components/SlidePanel'

const artifacts = signal<Artifact[]>([])
const loading = signal(true)
const error = signal<string | null>(null)
const selectedArtifact = signal<Artifact | null>(null)
const artifactExecutions = signal<ArtifactDetailType[]>([])
const loadingExecutions = signal(false)
const selectedExecution = signal<ArtifactDetailType | null>(null)

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

async function selectArtifact(artifact: Artifact) {
  selectedArtifact.value = artifact
  selectedExecution.value = null
  loadingExecutions.value = true
  try {
    artifactExecutions.value = await api.artifacts.getForEnsemble(artifact.name)
    if (artifactExecutions.value.length > 0) {
      selectedExecution.value = artifactExecutions.value[0]
    }
  } catch {
    artifactExecutions.value = []
  } finally {
    loadingExecutions.value = false
  }
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}m`
}

function formatTimestamp(ts: string): string {
  try {
    const date = new Date(ts)
    return date.toLocaleString()
  } catch {
    return ts
  }
}

function ArtifactCard({ artifact }: { artifact: Artifact }) {
  const isSelected = selectedArtifact.value?.name === artifact.name

  return (
    <article
      className={`card${isSelected ? ' selected' : ''}`}
      onClick={() => selectArtifact(artifact)}
    >
      <header>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <strong>{artifact.name}</strong>
          <span className="badge badge-primary">
            {artifact.executions_count} run{artifact.executions_count !== 1 ? 's' : ''}
          </span>
        </div>
      </header>
      <p><small>Latest: {artifact.latest_execution}</small></p>
    </article>
  )
}

function ExecutionMetrics({ execution }: { execution: ArtifactDetailType }) {
  const successCount = execution.agents.filter((a) => a.status === 'completed').length
  const totalAgents = execution.agents.length

  return (
    <div className="metrics-grid">
      <div className="metric">
        <div className="value">{formatDuration(execution.total_duration_ms)}</div>
        <div className="label">Duration</div>
      </div>
      <div className="metric">
        <div className="value">{successCount}/{totalAgents}</div>
        <div className="label">Succeeded</div>
      </div>
      <div className="metric">
        <div className="value" style={{ color: execution.status === 'success' ? '#3fb950' : '#f85149' }}>
          {execution.status === 'success' ? 'Pass' : 'Fail'}
        </div>
        <div className="label">Status</div>
      </div>
    </div>
  )
}

function AgentResults({ execution }: { execution: ArtifactDetailType }) {
  return (
    <div>
      <p className="muted-label">Agent Results</p>
      <div className="spaced-sm">
        {execution.agents.map((agent) => (
          <details key={agent.name}>
            <summary>
              <span style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong>{agent.name}</strong>
                <span className={`badge ${agent.status === 'completed' ? 'badge-success' : 'badge-error'}`}>
                  {agent.status}
                </span>
              </span>
            </summary>
            <pre><code>{agent.result || agent.error || 'No output'}</code></pre>
          </details>
        ))}
      </div>
    </div>
  )
}

function SynthesisResult({ execution }: { execution: ArtifactDetailType }) {
  if (!execution.synthesis) return null

  return (
    <div style={{ marginTop: '1rem' }}>
      <p className="muted-label">Synthesis</p>
      <article style={{ borderColor: 'var(--pico-primary)' }}>
        <pre style={{ whiteSpace: 'pre-wrap' }}><code>{execution.synthesis}</code></pre>
      </article>
    </div>
  )
}

function ArtifactDetailPanel() {
  const artifact = selectedArtifact.value

  return (
    <SlidePanel
      open={artifact !== null}
      onClose={() => (selectedArtifact.value = null)}
      title={artifact?.name || ''}
      subtitle={`${artifact?.executions_count || 0} execution${artifact?.executions_count !== 1 ? 's' : ''}`}
      width="xl"
    >
      {loadingExecutions.value ? (
        <p aria-busy="true">Loading executions...</p>
      ) : artifactExecutions.value.length === 0 ? (
        <p>No executions found</p>
      ) : (
        <>
          <div style={{ marginBottom: '1rem' }}>
            <p className="muted-label">Select Execution</p>
            <div role="group">
              {artifactExecutions.value.map((exec, idx) => (
                <button
                  key={exec.timestamp}
                  className={selectedExecution.value?.timestamp === exec.timestamp ? '' : 'outline secondary'}
                  onClick={() => (selectedExecution.value = exec)}
                  style={{ padding: '0.25rem 0.75rem', fontSize: '0.75rem' }}
                >
                  #{artifactExecutions.value.length - idx}{' '}
                  <span className={`status-dot ${exec.status === 'success' ? 'success' : 'error'}`} />
                </button>
              ))}
            </div>
          </div>

          {selectedExecution.value && (
            <>
              <p><small>{formatTimestamp(selectedExecution.value.timestamp)}</small></p>
              <ExecutionMetrics execution={selectedExecution.value} />
              <AgentResults execution={selectedExecution.value} />
              <SynthesisResult execution={selectedExecution.value} />
            </>
          )}
        </>
      )}
    </SlidePanel>
  )
}

export function ArtifactsPage() {
  useEffect(() => {
    loadArtifacts()
  }, [])

  if (loading.value) {
    return <p aria-busy="true">Loading artifacts...</p>
  }

  if (error.value) {
    return <p style={{ color: '#f85149' }}>Error: {error.value}</p>
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Execution Artifacts</h1>
          <p>{artifacts.value.length} ensemble{artifacts.value.length !== 1 ? 's' : ''} with artifacts</p>
        </div>
      </div>

      {artifacts.value.length === 0 ? (
        <div className="empty-state">
          <p>No execution artifacts yet.</p>
          <p><small>Run an ensemble to create artifacts.</small></p>
        </div>
      ) : (
        <div className="card-grid">
          {artifacts.value.map((a) => (
            <ArtifactCard key={a.name} artifact={a} />
          ))}
        </div>
      )}

      <ArtifactDetailPanel />
    </div>
  )
}
