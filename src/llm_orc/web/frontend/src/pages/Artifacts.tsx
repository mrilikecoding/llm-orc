import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Artifact, ArtifactDetail as ArtifactDetailType } from '../api/client'

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
      className={`bg-bg-secondary border rounded-lg p-4 cursor-pointer transition-colors
        ${isSelected ? 'border-accent' : 'border-border hover:border-text-secondary'}`}
      onClick={() => selectArtifact(artifact)}
    >
      <div className="text-lg font-semibold text-accent mb-2">{artifact.name}</div>
      <div className="flex justify-between mb-1 text-text-secondary text-sm">
        <span>Executions:</span>
        <span>{artifact.executions_count}</span>
      </div>
      <div className="flex justify-between mb-1 text-text-secondary text-sm">
        <span>Latest:</span>
        <span>{artifact.latest_execution}</span>
      </div>
    </div>
  )
}

function ExecutionMetrics({ execution }: { execution: ArtifactDetailType }) {
  const successCount = execution.agents.filter((a) => a.status === 'completed').length
  const totalAgents = execution.agents.length

  return (
    <div className="mb-6">
      <div className="text-sm font-semibold text-text-secondary mb-3 uppercase tracking-wider">
        Metrics
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-bg-primary border border-border rounded-md p-4 text-center">
          <div className="text-2xl font-bold text-text-primary">
            {formatDuration(execution.total_duration_ms)}
          </div>
          <div className="text-sm text-text-secondary mt-1">Duration</div>
        </div>
        <div className="bg-bg-primary border border-border rounded-md p-4 text-center">
          <div className="text-2xl font-bold text-text-primary">
            {successCount}/{totalAgents}
          </div>
          <div className="text-sm text-text-secondary mt-1">Agents Succeeded</div>
        </div>
        <div className="bg-bg-primary border border-border rounded-md p-4 text-center">
          <div className={`text-2xl font-bold ${
            execution.status === 'success' ? 'text-success' : 'text-error'
          }`}>
            {execution.status === 'success' ? 'Success' : 'Failed'}
          </div>
          <div className="text-sm text-text-secondary mt-1">Status</div>
        </div>
      </div>
    </div>
  )
}

function AgentResults({ execution }: { execution: ArtifactDetailType }) {
  return (
    <div className="mb-6">
      <div className="text-sm font-semibold text-text-secondary mb-3 uppercase tracking-wider">
        Agent Results
      </div>
      <div className="flex flex-col gap-3">
        {execution.agents.map((agent) => (
          <div key={agent.name} className="bg-bg-primary border border-border rounded-md overflow-hidden">
            <div className="py-3 px-4 bg-bg-secondary flex justify-between items-center">
              <span className="font-semibold text-text-primary">{agent.name}</span>
              <span className={`py-0.5 px-2 rounded text-xs text-white ${
                agent.status === 'completed' ? 'bg-success-bg' : 'bg-error-bg'
              }`}>
                {agent.status}
              </span>
            </div>
            <div className="p-4 font-mono text-sm whitespace-pre-wrap max-h-[200px] overflow-auto">
              {agent.result || agent.error || 'No output'}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function SynthesisResult({ execution }: { execution: ArtifactDetailType }) {
  if (!execution.synthesis) return null

  return (
    <div className="mb-6">
      <div className="text-sm font-semibold text-text-secondary mb-3 uppercase tracking-wider">
        Synthesis
      </div>
      <div className="bg-accent/5 border border-accent/25 rounded-md p-4">
        <div className="font-mono text-sm whitespace-pre-wrap">{execution.synthesis}</div>
      </div>
    </div>
  )
}

function ArtifactDetailPanel() {
  const artifact = selectedArtifact.value
  if (!artifact) return null

  return (
    <div className="bg-bg-secondary border border-border rounded-lg mt-6 overflow-hidden">
      <div className="p-4 border-b border-border flex justify-between items-center">
        <div>
          <h3 className="m-0">{artifact.name}</h3>
          <p className="m-0 mt-1 text-text-secondary text-sm">
            {artifact.executions_count} execution{artifact.executions_count !== 1 ? 's' : ''}
          </p>
        </div>
        <button
          className="bg-transparent border-none text-text-secondary cursor-pointer text-xl p-1
            hover:text-text-primary"
          onClick={() => (selectedArtifact.value = null)}
        >
          ×
        </button>
      </div>

      {loadingExecutions.value ? (
        <div className="p-6 text-text-secondary">Loading executions...</div>
      ) : artifactExecutions.value.length === 0 ? (
        <div className="p-6 text-text-secondary">No executions found</div>
      ) : (
        <>
          <div className="max-h-[200px] overflow-auto border-b border-border">
            {artifactExecutions.value.map((exec) => (
              <div
                key={exec.timestamp}
                className={`py-3 px-6 flex justify-between items-center cursor-pointer
                  border-b border-border-light last:border-0
                  ${selectedExecution.value?.timestamp === exec.timestamp ? 'bg-border-light' : 'hover:bg-border-light/50'}`}
                onClick={() => (selectedExecution.value = exec)}
              >
                <span className="text-sm text-text-primary">{formatTimestamp(exec.timestamp)}</span>
                <div className="flex gap-3 text-sm text-text-secondary">
                  <span>{formatDuration(exec.total_duration_ms)}</span>
                  <span className={`py-0.5 px-2 rounded text-xs text-white ${
                    exec.status === 'success' ? 'bg-success-bg' : 'bg-error-bg'
                  }`}>
                    {exec.status}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {selectedExecution.value && (
            <div className="p-6">
              <ExecutionMetrics execution={selectedExecution.value} />
              <AgentResults execution={selectedExecution.value} />
              <SynthesisResult execution={selectedExecution.value} />
            </div>
          )}
        </>
      )}
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
    return <div className="text-error">Error: {error.value}</div>
  }

  if (artifacts.value.length === 0) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-6">Execution Artifacts</h2>
        <div className="text-center p-12 text-text-secondary">
          <p>No execution artifacts yet.</p>
          <p className="mt-2">Run an ensemble to create artifacts.</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Execution Artifacts</h2>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-4">
        {artifacts.value.map((a) => (
          <ArtifactCard key={a.name} artifact={a} />
        ))}
      </div>
      <ArtifactDetailPanel />
    </div>
  )
}
