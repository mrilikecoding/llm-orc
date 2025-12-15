import { signal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { api, Ensemble, EnsembleDetail as EnsembleDetailType, ExecutionResult } from '../api/client'
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
  executeResult.value = null
  activeTab.value = 'agents'
  try {
    selectedEnsemble.value = await api.ensembles.get(ensemble.name)
  } catch {
    selectedEnsemble.value = { ...ensemble, agents: ensemble.agents || [] }
  } finally {
    loadingDetail.value = false
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
    <div
      className={`bg-bg-secondary border rounded-lg p-4 cursor-pointer transition-all
        hover:shadow-lg hover:-translate-y-0.5
        ${isSelected ? 'border-accent ring-2 ring-accent/20' : 'border-border hover:border-text-secondary'}`}
      onClick={() => selectEnsemble(ensemble)}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="text-base font-semibold text-accent">{ensemble.name}</div>
        {agentCount > 0 && (
          <span className="text-xs py-0.5 px-2 bg-accent/10 text-accent rounded-full">
            {agentCount} agent{agentCount !== 1 ? 's' : ''}
          </span>
        )}
      </div>
      <div className="text-text-secondary text-sm line-clamp-2">{ensemble.description}</div>
      <div className="mt-3 text-xs text-text-muted">
        <span className="py-0.5 px-2 bg-border-light rounded">{ensemble.source}</span>
      </div>
    </div>
  )
}

function TabButton({ tab, label }: { tab: 'agents' | 'execute' | 'config'; label: string }) {
  const isActive = activeTab.value === tab
  return (
    <button
      className={`px-4 py-2 text-sm font-medium rounded-t border-b-2 -mb-px transition-colors
        ${isActive
          ? 'text-accent border-accent bg-bg-primary'
          : 'text-text-secondary border-transparent hover:text-text-primary hover:bg-border-light/50'}`}
      onClick={() => (activeTab.value = tab)}
    >
      {label}
    </button>
  )
}

function AgentsTab() {
  const ens = selectedEnsemble.value
  if (!ens?.agents?.length) {
    return <div className="text-text-secondary py-4">No agents configured</div>
  }

  return (
    <div className="space-y-3">
      {ens.agents.map((agent, idx) => (
        <div key={agent.name} className="bg-bg-primary border border-border rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-6 h-6 rounded-full bg-accent/20 text-accent text-xs
              flex items-center justify-center font-medium">
              {idx + 1}
            </span>
            <span className="font-medium text-text-primary">{agent.name}</span>
            <span className="ml-auto text-xs text-text-muted bg-border-light py-0.5 px-2 rounded">
              {agent.model_profile}
            </span>
          </div>
          {agent.role && (
            <p className="text-sm text-text-secondary mb-2 pl-8">{agent.role}</p>
          )}
          {agent.script && (
            <div className="text-xs text-text-muted pl-8">
              Script: <code className="text-success">{agent.script}</code>
            </div>
          )}
          {agent.depends_on && agent.depends_on.length > 0 && (
            <div className="text-xs text-text-muted pl-8 mt-1 flex items-center gap-1 flex-wrap">
              <span>Depends on:</span>
              {agent.depends_on.map((dep) => (
                <span key={dep} className="bg-accent/10 text-accent py-0.5 px-1.5 rounded">
                  {dep}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function ExecuteTab() {
  return (
    <div>
      <label className="block text-sm font-medium text-text-secondary mb-2">Input</label>
      <textarea
        className="w-full p-3 bg-bg-primary border border-border rounded-lg text-text-primary
          font-sans text-sm resize-y focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/50"
        placeholder="Enter input for the ensemble..."
        value={executeInput.value}
        onInput={(e) => (executeInput.value = (e.target as HTMLTextAreaElement).value)}
        rows={4}
      />
      <button
        className={`mt-3 w-full py-2.5 px-4 rounded-lg text-white font-medium transition-colors
          ${executing.value
            ? 'bg-border-light cursor-not-allowed'
            : 'bg-success-bg hover:bg-success-bg/90'}`}
        onClick={executeEnsemble}
        disabled={executing.value}
      >
        {executing.value ? 'Executing...' : 'Execute Ensemble'}
      </button>

      {executeResult.value && (
        <div className="mt-4">
          <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium mb-3
            ${executeResult.value.status === 'success'
              ? 'bg-success/20 text-success'
              : 'bg-error/20 text-error'}`}>
            <span className={`w-2 h-2 rounded-full ${
              executeResult.value.status === 'success' ? 'bg-success' : 'bg-error'
            }`} />
            {executeResult.value.status === 'success' ? 'Completed' : 'Failed'}
          </div>

          <div className="space-y-3">
            {Object.entries(executeResult.value.results).map(([name, result]) => (
              <div key={name} className="bg-bg-primary border border-border rounded-lg p-3">
                <div className="text-sm font-medium text-accent mb-2">{name}</div>
                <pre className="text-sm text-text-primary whitespace-pre-wrap">
                  {result.response || result.error || 'No output'}
                </pre>
              </div>
            ))}
          </div>

          {executeResult.value.synthesis && (
            <div className="mt-3 bg-accent/5 border border-accent/25 rounded-lg p-3">
              <div className="text-xs font-medium text-accent mb-2 uppercase tracking-wider">
                Synthesis
              </div>
              <div className="text-sm">{executeResult.value.synthesis}</div>
            </div>
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
    <pre className="text-sm bg-bg-primary border border-border rounded-lg p-4 overflow-auto">
      {JSON.stringify(ens, null, 2)}
    </pre>
  )
}

function EnsembleDetailPanel() {
  const ens = selectedEnsemble.value

  return (
    <SlidePanel
      open={ens !== null}
      onClose={() => (selectedEnsemble.value = null)}
      title={ens?.name || ''}
      subtitle={ens?.description}
      width="xl"
    >
      {loadingDetail.value ? (
        <div className="text-text-secondary py-8 text-center">Loading...</div>
      ) : ens ? (
        <>
          <div className="flex gap-1 border-b border-border mb-4">
            <TabButton tab="agents" label={`Agents (${ens.agents?.length || 0})`} />
            <TabButton tab="execute" label="Execute" />
            <TabButton tab="config" label="Config" />
          </div>

          {activeTab.value === 'agents' && <AgentsTab />}
          {activeTab.value === 'execute' && <ExecuteTab />}
          {activeTab.value === 'config' && <ConfigTab />}
        </>
      ) : null}
    </SlidePanel>
  )
}

export function EnsemblesPage() {
  useEffect(() => {
    loadEnsembles()
  }, [])

  if (loading.value) {
    return <div className="text-text-secondary">Loading ensembles...</div>
  }

  if (error.value) {
    return <div className="text-error">Error: {error.value}</div>
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold">Ensembles</h1>
          <p className="text-text-secondary text-sm mt-1">
            {ensembles.value.length} ensemble{ensembles.value.length !== 1 ? 's' : ''} available
          </p>
        </div>
      </div>

      {ensembles.value.length === 0 ? (
        <div className="text-center py-16 text-text-secondary">
          <div className="text-4xl mb-4 opacity-50">📦</div>
          <p>No ensembles found.</p>
          <p className="text-sm mt-1">Create ensembles in .llm-orc/ensembles/</p>
        </div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-4">
          {ensembles.value.map((e) => (
            <EnsembleCard key={e.name} ensemble={e} />
          ))}
        </div>
      )}

      <EnsembleDetailPanel />
    </div>
  )
}
