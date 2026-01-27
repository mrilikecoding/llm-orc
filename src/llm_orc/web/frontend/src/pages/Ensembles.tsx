import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import {
  api,
  Ensemble,
  EnsembleDetail as EnsembleDetailType,
  ExecutionResult,
  RunnableStatus,
  AgentRunnableStatus,
} from '../api/client'
import { SlidePanel } from '../components/SlidePanel'

const ensembles = signal<Ensemble[]>([])
const loading = signal(true)
const error = signal<string | null>(null)
const selectedEnsemble = signal<EnsembleDetailType | null>(null)
const loadingDetail = signal(false)
const executeInput = signal('')
const executeResult = signal<ExecutionResult | null>(null)
const executing = signal(false)
const activeTab = signal<'agents' | 'execute' | 'config'>('agents')
const runnableStatus = signal<RunnableStatus | null>(null)
const loadingRunnable = signal(false)

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

async function selectEnsemble(ensemble: Ensemble) {
  loadingDetail.value = true
  loadingRunnable.value = true
  executeResult.value = null
  runnableStatus.value = null
  activeTab.value = 'agents'

  try {
    const [detail, runnable] = await Promise.all([
      api.ensembles.get(ensemble.name),
      api.ensembles.checkRunnable(ensemble.name),
    ])
    selectedEnsemble.value = detail
    runnableStatus.value = runnable
  } catch {
    selectedEnsemble.value = { ...ensemble, agents: ensemble.agents || [] }
  } finally {
    loadingDetail.value = false
    loadingRunnable.value = false
  }
}

async function executeEnsemble() {
  if (!selectedEnsemble.value || !executeInput.value.trim()) return
  executing.value = true
  executeResult.value = null
  try {
    executeResult.value = await api.ensembles.execute(
      selectedEnsemble.value.name,
      executeInput.value
    )
  } catch (e) {
    executeResult.value = {
      status: 'error',
      results: {},
      synthesis: e instanceof Error ? e.message : 'Unknown error',
    }
  } finally {
    executing.value = false
  }
}

function EnsembleCard({ ensemble }: { ensemble: Ensemble }) {
  const isSelected = selectedEnsemble.value?.name === ensemble.name
  const agentCount = ensemble.agents?.length || 0

  return (
    <article
      className={`card${isSelected ? ' selected' : ''}`}
      onClick={() => selectEnsemble(ensemble)}
    >
      <header>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <strong>{ensemble.name}</strong>
          {agentCount > 0 && (
            <span className="badge badge-primary">
              {agentCount} agent{agentCount !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </header>
      <p><small>{ensemble.description || 'No description'}</small></p>
    </article>
  )
}

function RunnableStatusBanner() {
  if (loadingRunnable.value) {
    return <p><small>Checking availability...</small></p>
  }

  const status = runnableStatus.value
  if (!status) return null

  const unavailableAgents = status.agents.filter((a) => a.status !== 'available')

  if (status.runnable) {
    return (
      <div role="group" style={{ marginBottom: '1.5rem' }}>
        <ins style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="status-dot success" />
          <small><strong>Ready to run</strong> &mdash; All {status.agents.length} agents available</small>
        </ins>
      </div>
    )
  }

  return (
    <div style={{ marginBottom: '1.5rem', padding: '1rem', border: '1px solid #d29922', borderRadius: 'var(--pico-border-radius)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
        <span className="status-dot warning" />
        <small><strong style={{ color: '#d29922' }}>
          {unavailableAgents.length} agent{unavailableAgents.length !== 1 ? 's' : ''} unavailable
        </strong></small>
      </div>
      <div className="spaced-sm" style={{ paddingLeft: '1.25rem' }}>
        {unavailableAgents.map((agent) => (
          <div key={agent.name}>
            <small>
              <strong>{agent.name}</strong>
              {': '}
              {agent.status === 'provider_unavailable' && `${agent.provider} not available`}
              {agent.status === 'missing_profile' && 'profile not found'}
              {agent.status === 'model_not_found' && 'model not found'}
              {agent.alternatives.length > 0 && (
                <span style={{ opacity: 0.6, marginLeft: '0.5rem' }}>
                  (alternatives: {agent.alternatives.slice(0, 3).join(', ')})
                </span>
              )}
            </small>
          </div>
        ))}
      </div>
    </div>
  )
}

function getAgentStatus(agentName: string): AgentRunnableStatus | undefined {
  return runnableStatus.value?.agents.find((a) => a.name === agentName)
}

function AgentsTab() {
  const ens = selectedEnsemble.value
  if (!ens?.agents?.length) {
    return <p>No agents configured</p>
  }

  return (
    <div className="spaced">
      {ens.agents.map((agent, idx) => {
        const status = getAgentStatus(agent.name)
        const isAvailable = !status || status.status === 'available'

        return (
          <article
            key={agent.name}
            style={!isAvailable ? { borderColor: '#d29922' } : undefined}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <span className="agent-number">{idx + 1}</span>
              <strong>{agent.name}</strong>
              {status && <span className={`status-dot ${isAvailable ? 'success' : 'warning'}`} title={isAvailable ? 'Available' : status.status} />}
              <small style={{ marginLeft: 'auto' }}>
                <kbd>{agent.model_profile || agent.script || 'no profile'}</kbd>
              </small>
            </div>
            {agent.role && <p><small>{agent.role}</small></p>}
            {agent.script && (
              <p><small>Script: <code>{agent.script}</code></small></p>
            )}
            {agent.depends_on && agent.depends_on.length > 0 && (
              <p>
                <small>
                  Depends on:{' '}
                  {agent.depends_on.map((dep) => (
                    <kbd key={dep} style={{ marginRight: '0.25rem' }}>{dep}</kbd>
                  ))}
                </small>
              </p>
            )}
            {status && !isAvailable && (
              <p>
                <small style={{ color: '#d29922' }}>
                  {status.status === 'provider_unavailable' && `${status.provider} not available`}
                  {status.status === 'missing_profile' && 'Profile not found'}
                  {status.status === 'model_not_found' && 'Model not found'}
                  {status.alternatives.length > 0 && (
                    <span style={{ opacity: 0.6, marginLeft: '0.5rem' }}>
                      Alternatives: {status.alternatives.slice(0, 3).join(', ')}
                    </span>
                  )}
                </small>
              </p>
            )}
          </article>
        )
      })}
    </div>
  )
}

function ExecuteTab() {
  return (
    <div>
      <label>
        Input
        <textarea
          placeholder="Enter input for the ensemble..."
          value={executeInput.value}
          onInput={(e) => (executeInput.value = (e.target as HTMLTextAreaElement).value)}
          rows={4}
        />
      </label>
      <button
        onClick={executeEnsemble}
        disabled={executing.value}
        aria-busy={executing.value}
      >
        {executing.value ? 'Executing...' : 'Execute Ensemble'}
      </button>

      {executeResult.value && (
        <div style={{ marginTop: '1rem' }}>
          <p>
            <span className={`badge ${executeResult.value.status === 'success' ? 'badge-success' : 'badge-error'}`}>
              <span className={`status-dot ${executeResult.value.status === 'success' ? 'success' : 'error'}`} />{' '}
              {executeResult.value.status === 'success' ? 'Completed' : 'Failed'}
            </span>
          </p>

          <div className="spaced-sm">
            {Object.entries(executeResult.value.results).map(([name, result]) => (
              <article key={name}>
                <header><strong style={{ color: 'var(--pico-primary)' }}>{name}</strong></header>
                <pre><code>{result.response || result.error || 'No output'}</code></pre>
              </article>
            ))}
          </div>

          {executeResult.value.synthesis && (
            <article style={{ borderColor: 'var(--pico-primary)', marginTop: '0.75rem' }}>
              <header><small className="muted-label">Synthesis</small></header>
              <p>{executeResult.value.synthesis}</p>
            </article>
          )}
        </div>
      )}
    </div>
  )
}

function ConfigTab() {
  const ens = selectedEnsemble.value
  if (!ens) return null

  return (
    <pre><code>{JSON.stringify(ens, null, 2)}</code></pre>
  )
}

function EnsembleDetailPanel() {
  const ens = selectedEnsemble.value

  return (
    <SlidePanel
      open={ens !== null}
      onClose={() => {
        selectedEnsemble.value = null
        runnableStatus.value = null
      }}
      title={ens?.name || ''}
      subtitle={ens?.description}
      width="xl"
    >
      {loadingDetail.value ? (
        <p aria-busy="true">Loading...</p>
      ) : ens ? (
        <>
          <RunnableStatusBanner />

          <div className="inner-tabs">
            <button className={activeTab.value === 'agents' ? 'active' : ''} onClick={() => (activeTab.value = 'agents')}>
              Agents ({ens.agents?.length || 0})
            </button>
            <button className={activeTab.value === 'execute' ? 'active' : ''} onClick={() => (activeTab.value = 'execute')}>
              Execute
            </button>
            <button className={activeTab.value === 'config' ? 'active' : ''} onClick={() => (activeTab.value = 'config')}>
              Config
            </button>
          </div>

          {activeTab.value === 'agents' && <AgentsTab />}
          {activeTab.value === 'execute' && <ExecuteTab />}
          {activeTab.value === 'config' && <ConfigTab />}
        </>
      ) : null}
    </SlidePanel>
  )
}

function getDirectory(relativePath: string | undefined): string {
  if (!relativePath) return '(root)'
  const lastSlash = relativePath.lastIndexOf('/')
  if (lastSlash === -1) return '(root)'
  return relativePath.substring(0, lastSlash)
}

function formatSourceLabel(source: string): string {
  if (source === 'local') return 'Project Ensembles'
  if (source === 'library') return 'Library'
  if (source === 'global') return 'Global'
  return source
}

export function EnsemblesPage() {
  useEffect(() => {
    loadEnsembles()
  }, [])

  if (loading.value) {
    return <p aria-busy="true">Loading ensembles...</p>
  }

  if (error.value) {
    return <p style={{ color: '#f85149' }}>Error: {error.value}</p>
  }

  // Group ensembles by source then by directory
  const bySource = ensembles.value.reduce(
    (acc, ens) => {
      const source = ens.source || 'unknown'
      if (!acc[source]) acc[source] = {}
      const dir = getDirectory(ens.relative_path)
      if (!acc[source][dir]) acc[source][dir] = []
      acc[source][dir].push(ens)
      return acc
    },
    {} as Record<string, Record<string, Ensemble[]>>
  )

  const sortedSources = Object.keys(bySource).sort((a, b) => {
    const order: Record<string, number> = { local: 0, library: 1, global: 2 }
    return (order[a] ?? 3) - (order[b] ?? 3)
  })

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Ensembles</h1>
          <p>{ensembles.value.length} ensemble{ensembles.value.length !== 1 ? 's' : ''} available</p>
        </div>
      </div>

      {ensembles.value.length === 0 ? (
        <div className="empty-state">
          <p>No ensembles found.</p>
          <p><small>Create ensembles in .llm-orc/ensembles/</small></p>
        </div>
      ) : (
        <div className="spaced" style={{ gap: '3rem' }}>
          {sortedSources.map((source) => {
            const directories = bySource[source]
            const sortedDirs = Object.keys(directories).sort((a, b) => {
              if (a === '(root)') return 1
              if (b === '(root)') return -1
              return a.localeCompare(b)
            })
            const totalInSource = Object.values(directories).flat().length

            return (
              <details key={source}>
                <summary>
                  <strong>{formatSourceLabel(source)}</strong>{' '}
                  <span className="badge badge-primary">{totalInSource}</span>
                </summary>
                <div className="spaced" style={{ paddingLeft: '1rem' }}>
                  {sortedDirs.map((dir) => (
                    <details key={dir}>
                      <summary>
                        <code>{dir === '(root)' ? '/' : `${dir}/`}</code>{' '}
                        <small>({directories[dir].length})</small>
                      </summary>
                      <div className="card-grid" style={{ marginTop: '0.75rem' }}>
                        {directories[dir].map((e) => (
                          <EnsembleCard key={e.name} ensemble={e} />
                        ))}
                      </div>
                    </details>
                  ))}
                </div>
              </details>
            )
          })}
        </div>
      )}

      <EnsembleDetailPanel />
    </div>
  )
}
