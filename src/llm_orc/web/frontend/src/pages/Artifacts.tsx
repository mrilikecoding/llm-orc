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
    <div
      className={`bg-bg-secondary border rounded-xl p-6 cursor-pointer transition-all
        hover:shadow-xl hover:-translate-y-1
        ${isSelected ? 'border-accent ring-2 ring-accent/20' : 'border-border hover:border-accent/50'}`}
      onClick={() => selectArtifact(artifact)}
    >
      <div className="flex items-start justify-between mb-3">
        <h3 className="text-lg font-semibold text-text-primary">{artifact.name}</h3>
        <span className="text-xs py-1 px-2.5 bg-accent/15 text-accent rounded-full font-medium">
          {artifact.executions_count} run{artifact.executions_count !== 1 ? 's' : ''}
        </span>
      </div>
      <p className="text-sm text-text-muted">
        Latest: {artifact.latest_execution}
      </p>
    </div>
  )
}

function ExecutionMetrics({ execution }: { execution: ArtifactDetailType }) {
  const successCount = execution.agents.filter((a) => a.status === 'completed').length
  const totalAgents = execution.agents.length

  return (
    <div className="grid grid-cols-3 gap-3 mb-4">
      <div className="bg-bg-primary border border-border rounded-lg p-3 text-center">
        <div className="text-xl font-bold text-text-primary">
          {formatDuration(execution.total_duration_ms)}
        </div>
        <div className="text-xs text-text-muted mt-1">Duration</div>
      </div>
      <div className="bg-bg-primary border border-border rounded-lg p-3 text-center">
        <div className="text-xl font-bold text-text-primary">
          {successCount}/{totalAgents}
        </div>
        <div className="text-xs text-text-muted mt-1">Succeeded</div>
      </div>
      <div className="bg-bg-primary border border-border rounded-lg p-3 text-center">
        <div className={`text-xl font-bold ${
          execution.status === 'success' ? 'text-success' : 'text-error'
        }`}>
          {execution.status === 'success' ? 'Pass' : 'Fail'}
        </div>
        <div className="text-xs text-text-muted mt-1">Status</div>
      </div>
    </div>
  )
}

function AgentResults({ execution }: { execution: ArtifactDetailType }) {
  return (
    <div className="space-y-2">
      <div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
        Agent Results
      </div>
      {execution.agents.map((agent) => (
        <details key={agent.name} className="bg-bg-primary border border-border rounded-lg">
          <summary className="px-3 py-2 cursor-pointer flex items-center justify-between">
            <span className="font-medium text-sm">{agent.name}</span>
            <span className={`text-xs py-0.5 px-2 rounded text-white ${
              agent.status === 'completed' ? 'bg-success-bg' : 'bg-error-bg'
            }`}>
              {agent.status}
            </span>
          </summary>
          <div className="px-3 pb-3 pt-1 border-t border-border">
            <pre className="text-xs whitespace-pre-wrap text-text-secondary">
              {agent.result || agent.error || 'No output'}
            </pre>
          </div>
        </details>
      ))}
    </div>
  )
}

function SynthesisResult({ execution }: { execution: ArtifactDetailType }) {
  if (!execution.synthesis) return null

  return (
    <div className="mt-4">
      <div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
        Synthesis
      </div>
      <div className="bg-accent/5 border border-accent/25 rounded-lg p-3">
        <pre className="text-sm whitespace-pre-wrap">{execution.synthesis}</pre>
      </div>
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
        <div className="text-text-secondary py-8 text-center">Loading executions...</div>
      ) : artifactExecutions.value.length === 0 ? (
        <div className="text-text-secondary py-8 text-center">No executions found</div>
      ) : (
        <>
          {/* Execution selector */}
          <div className="mb-4">
            <div className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
              Select Execution
            </div>
            <div className="flex gap-2 flex-wrap">
              {artifactExecutions.value.map((exec, idx) => (
                <button
                  key={exec.timestamp}
                  className={`text-xs px-3 py-1.5 rounded-full border transition-colors
                    ${selectedExecution.value?.timestamp === exec.timestamp
                      ? 'border-accent bg-accent/10 text-accent'
                      : 'border-border text-text-secondary hover:border-text-secondary'}`}
                  onClick={() => (selectedExecution.value = exec)}
                >
                  #{artifactExecutions.value.length - idx}
                  <span className={`ml-1.5 w-1.5 h-1.5 rounded-full inline-block ${
                    exec.status === 'success' ? 'bg-success' : 'bg-error'
                  }`} />
                </button>
              ))}
            </div>
          </div>

          {selectedExecution.value && (
            <>
              <div className="text-xs text-text-muted mb-4">
                {formatTimestamp(selectedExecution.value.timestamp)}
              </div>
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
    return <div className="text-text-secondary">Loading artifacts...</div>
  }

  if (error.value) {
    return <div className="text-error">Error: {error.value}</div>
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold">Execution Artifacts</h1>
          <p className="text-text-secondary text-sm mt-1">
            {artifacts.value.length} ensemble{artifacts.value.length !== 1 ? 's' : ''} with artifacts
          </p>
        </div>
      </div>

      {artifacts.value.length === 0 ? (
        <div className="text-center py-16 text-text-secondary">
          <div className="text-4xl mb-4 opacity-50">📊</div>
          <p>No execution artifacts yet.</p>
          <p className="text-sm mt-1">Run an ensemble to create artifacts.</p>
        </div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-6">
          {artifacts.value.map((a) => (
            <ArtifactCard key={a.name} artifact={a} />
          ))}
        </div>
      )}

      <ArtifactDetailPanel />
    </div>
  )
}
